from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.merge import merge as raster_merge
from rasterstats import zonal_stats
from shapely.geometry import box

from ssa_urban_deprivation_benchmark.io_utils import ensure_parent_dir
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.io_utils import write_table
from ssa_urban_deprivation_benchmark.study import load_study_config
from ssa_urban_deprivation_benchmark.study import resolve_city_worldcover_tiles
from ssa_urban_deprivation_benchmark.study import slugify


WORLD_COVER_BUILT_UP = 50
WORLD_COVER_OPEN = {10, 20, 30, 60, 90, 95}
AMENITY_BLACKLIST = {
    "bench",
    "bicycle_parking",
    "clock",
    "drinking_water",
    "fountain",
    "parking",
    "parking_space",
    "post_box",
    "recycling",
    "shelter",
    "telephone",
    "toilets",
    "vending_machine",
    "waste_basket",
    "waste_disposal",
}
SCHOOL_AMENITIES = {"school", "college", "university", "kindergarten"}
CLINIC_AMENITIES = {"clinic", "doctors", "hospital", "pharmacy", "health_post"}


def _utm_crs_for_gdf(frame: gpd.GeoDataFrame) -> pyproj.CRS:
    return frame.estimate_utm_crs()


def _create_grid(boundary_proj: gpd.GeoDataFrame, cell_size_m: int, city_slug: str) -> gpd.GeoDataFrame:
    boundary_geom = boundary_proj.unary_union
    minx, miny, maxx, maxy = boundary_geom.bounds

    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    boxes = []
    for x0 in xs:
        for y0 in ys:
            candidate = box(x0, y0, x0 + cell_size_m, y0 + cell_size_m)
            if not candidate.intersects(boundary_geom):
                continue
            clipped = candidate.intersection(boundary_geom)
            if clipped.is_empty:
                continue
            boxes.append(clipped)

    grid = gpd.GeoDataFrame({"geometry": boxes}, crs=boundary_proj.crs)
    grid["cell_id"] = [
        "{slug}_{idx:05d}".format(slug=city_slug, idx=index + 1)
        for index in range(len(grid))
    ]
    grid["area_m2"] = grid.geometry.area
    grid["area_km2"] = grid["area_m2"] / 1_000_000.0
    return grid


def _representative_points(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.copy()
    points = frame.copy()
    points["geometry"] = points.geometry.representative_point()
    return points


def _filter_service_amenities(amenities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "amenity" not in amenities.columns:
        return amenities.iloc[0:0].copy()
    amenity_series = amenities["amenity"].astype(str).str.lower()
    keep_mask = ~amenity_series.isin(AMENITY_BLACKLIST)
    return amenities.loc[keep_mask].copy()


def _filter_school_amenities(amenities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "amenity" not in amenities.columns:
        return amenities.iloc[0:0].copy()
    mask = amenities["amenity"].astype(str).str.lower().isin(SCHOOL_AMENITIES)
    return amenities.loc[mask].copy()


def _filter_clinic_amenities(amenities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if amenities.empty:
        return amenities.copy()

    amenity_mask = False
    healthcare_mask = False
    if "amenity" in amenities.columns:
        amenity_mask = amenities["amenity"].astype(str).str.lower().isin(CLINIC_AMENITIES)
    if "healthcare" in amenities.columns:
        healthcare_values = amenities["healthcare"].astype(str).str.lower()
        healthcare_mask = healthcare_values.isin(CLINIC_AMENITIES.union({"yes", "centre", "center"}))

    return amenities.loc[amenity_mask | healthcare_mask].copy()


def _nearest_distance(
    centroids_proj: gpd.GeoDataFrame,
    targets_proj: gpd.GeoDataFrame,
    column_name: str,
) -> pd.Series:
    if targets_proj.empty:
        return pd.Series(np.nan, index=centroids_proj.index, name=column_name)

    joined = gpd.sjoin_nearest(
        centroids_proj[["cell_id", "geometry"]],
        targets_proj[["geometry"]],
        how="left",
        distance_col=column_name,
    )
    reduced = joined.groupby("cell_id", as_index=True)[column_name].min()
    return centroids_proj["cell_id"].map(reduced).reset_index(drop=True)


def _count_points_within_distance(
    centroids_proj: gpd.GeoDataFrame,
    targets_proj: gpd.GeoDataFrame,
    distance_m: float,
) -> pd.Series:
    if targets_proj.empty:
        return pd.Series(np.zeros(len(centroids_proj), dtype=int), index=centroids_proj.index)

    buffers = centroids_proj[["cell_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(distance_m)

    joined = gpd.sjoin(
        buffers,
        targets_proj[["geometry"]],
        how="left",
        predicate="contains",
    )
    valid = joined.loc[~joined.index_right.isna()].copy()
    counts = valid.groupby("cell_id").size()
    return centroids_proj["cell_id"].map(counts).fillna(0).astype(int)


def _intersection_nodes(nodes_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if nodes_proj.empty:
        return nodes_proj.copy()
    if "street_count" in nodes_proj.columns:
        mask = pd.to_numeric(nodes_proj["street_count"], errors="coerce").fillna(0) >= 3
        return nodes_proj.loc[mask].copy()
    return nodes_proj.copy()


def _combine_targets(frames: Iterable[gpd.GeoDataFrame], crs) -> gpd.GeoDataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    combined = pd.concat(non_empty, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)


def _population_sum(grid_wgs84: gpd.GeoDataFrame, raster_path: Path) -> pd.Series:
    stats = zonal_stats(
        vectors=grid_wgs84.geometry,
        raster=str(raster_path),
        stats=["sum"],
        all_touched=True,
        nodata=-99999,
    )
    values = [0.0 if item["sum"] is None else float(item["sum"]) for item in stats]
    return pd.Series(values)


def _load_city_raw_layers(city_slug: str) -> Dict[str, gpd.GeoDataFrame]:
    boundary = gpd.read_file(Path("data/raw/osm") / "{slug}_boundary.gpkg".format(slug=city_slug))
    roads = gpd.read_file(Path("data/raw/osm") / "{slug}_roads.gpkg".format(slug=city_slug))
    road_nodes = gpd.read_file(Path("data/raw/osm") / "{slug}_road_nodes.gpkg".format(slug=city_slug))
    amenities = gpd.read_file(Path("data/raw/osm") / "{slug}_amenities.gpkg".format(slug=city_slug))
    return {
        "boundary": boundary,
        "roads": roads,
        "road_nodes": road_nodes,
        "amenities": amenities,
    }


def _worldpop_path_for_iso(iso3: str) -> Path:
    matches = sorted((Path("data/raw/worldpop") / iso3).glob("*.tif"))
    if not matches:
        raise FileNotFoundError("No WorldPop raster found for {iso3}".format(iso3=iso3))
    return matches[-1]


def _worldcover_path_for_tile(tile: str) -> Path:
    path = Path("data/raw/worldcover") / (
        "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif".format(tile=tile)
    )
    if not path.exists():
        raise FileNotFoundError("No WorldCover tile found for {tile}".format(tile=tile))
    return path


def _worldcover_paths_for_city(city: Dict, boundary: Optional[gpd.GeoDataFrame] = None) -> List[Path]:
    return [_worldcover_path_for_tile(tile) for tile in resolve_city_worldcover_tiles(city=city, boundary=boundary)]


def _worldcover_shares(grid_wgs84: gpd.GeoDataFrame, raster_paths: Iterable[Path]) -> pd.DataFrame:
    raster_paths = [Path(path) for path in raster_paths]
    if len(raster_paths) == 1:
        stats = zonal_stats(
            vectors=grid_wgs84.geometry,
            raster=str(raster_paths[0]),
            categorical=True,
            all_touched=True,
            nodata=0,
        )
    else:
        sources = [rasterio.open(path) for path in raster_paths]
        try:
            mosaic, transform = raster_merge(sources, method="first")
        finally:
            for source in sources:
                source.close()
        stats = zonal_stats(
            vectors=grid_wgs84.geometry,
            raster=mosaic[0],
            affine=transform,
            categorical=True,
            all_touched=True,
            nodata=0,
        )

    built_up_share = []
    open_space_share = []
    for record in stats:
        record = record or {}
        total = float(sum(record.values()))
        if total == 0:
            built_up_share.append(np.nan)
            open_space_share.append(np.nan)
            continue
        built = float(record.get(WORLD_COVER_BUILT_UP, 0.0))
        open_count = float(sum(record.get(code, 0.0) for code in WORLD_COVER_OPEN))
        built_up_share.append(built / total)
        open_space_share.append(open_count / total)

    return pd.DataFrame(
        {
            "building_coverage_ratio": built_up_share,
            "open_space_share": open_space_share,
        }
    )


def build_city_feature_table(city: Dict, cell_size_m: int) -> gpd.GeoDataFrame:
    city_slug = city.get("slug", slugify(city["name"]))
    layers = _load_city_raw_layers(city_slug)

    boundary = layers["boundary"]
    utm_crs = _utm_crs_for_gdf(boundary)
    boundary_proj = boundary.to_crs(utm_crs)
    roads_proj = layers["roads"].to_crs(utm_crs)
    road_nodes_proj = layers["road_nodes"].to_crs(utm_crs)
    amenities_proj = layers["amenities"].to_crs(utm_crs)

    service_amenities_proj = _representative_points(_filter_service_amenities(amenities_proj))
    school_amenities_proj = _representative_points(_filter_school_amenities(amenities_proj))
    clinic_amenities_proj = _representative_points(_filter_clinic_amenities(amenities_proj))
    intersections_proj = _intersection_nodes(road_nodes_proj)
    core_services_proj = _combine_targets([school_amenities_proj, clinic_amenities_proj], crs=utm_crs)

    grid_proj = _create_grid(boundary_proj, cell_size_m=cell_size_m, city_slug=city_slug)
    centroids_proj = grid_proj[["cell_id"]].copy()
    centroids_proj = gpd.GeoDataFrame(centroids_proj, geometry=grid_proj.geometry.centroid, crs=utm_crs)

    road_distance = _nearest_distance(centroids_proj, roads_proj, "road_distance_m")
    school_distance = _nearest_distance(centroids_proj, school_amenities_proj, "school_distance_m")
    clinic_distance = _nearest_distance(centroids_proj, clinic_amenities_proj, "clinic_distance_m")
    amenity_count = _count_points_within_distance(centroids_proj, service_amenities_proj, distance_m=1000.0)
    service_count = _count_points_within_distance(
        centroids_proj,
        core_services_proj,
        distance_m=1000.0,
    )

    intersection_join = gpd.sjoin(
        grid_proj[["cell_id", "geometry"]],
        intersections_proj[["geometry"]],
        how="left",
        predicate="contains",
    )
    valid_intersections = intersection_join.loc[~intersection_join.index_right.isna()]
    intersection_counts = valid_intersections.groupby("cell_id").size()
    intersection_density = (
        grid_proj["cell_id"].map(intersection_counts).fillna(0).astype(float) / grid_proj["area_km2"]
    )

    grid_wgs84 = grid_proj.to_crs("EPSG:4326")
    worldpop_path = _worldpop_path_for_iso(city["country_iso"])
    worldcover_paths = _worldcover_paths_for_city(city=city, boundary=boundary)

    population = _population_sum(grid_wgs84, worldpop_path)
    cover_shares = _worldcover_shares(grid_wgs84, worldcover_paths)

    feature_frame = grid_proj.copy()
    centroids_wgs84 = centroids_proj.to_crs("EPSG:4326")
    feature_frame["city"] = city["name"]
    feature_frame["country_iso"] = city["country_iso"]
    feature_frame["lon"] = centroids_wgs84.geometry.x.values
    feature_frame["lat"] = centroids_wgs84.geometry.y.values
    feature_frame["population"] = population.values
    feature_frame["road_distance_m"] = road_distance.values
    feature_frame["school_distance_m"] = school_distance.values
    feature_frame["clinic_distance_m"] = clinic_distance.values
    feature_frame["amenity_count_1km"] = amenity_count.values
    feature_frame["service_count_1km"] = service_count.values
    feature_frame["population_per_service"] = feature_frame["population"] / feature_frame["service_count_1km"].clip(
        lower=1
    )
    feature_frame["building_coverage_ratio"] = cover_shares["building_coverage_ratio"].values
    feature_frame["open_space_share"] = cover_shares["open_space_share"].values
    feature_frame["intersection_density_km2"] = intersection_density.values
    feature_frame["grid_cell_size_m"] = cell_size_m

    return feature_frame.to_crs("EPSG:4326")


def build_study_feature_tables(
    study_config_path: Path,
    output_path: Path,
    metadata_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    study = load_study_config(Path(study_config_path))
    scope = study["scope"]
    cell_size_m = int(scope.get("grid_cell_size_m", 500))

    frames: List[gpd.GeoDataFrame] = []
    city_summaries = []
    for city in study["city_details"]:
        city_frame = build_city_feature_table(city, cell_size_m=cell_size_m)
        boundary = _load_city_raw_layers(city.get("slug", slugify(city["name"])))["boundary"]
        frames.append(city_frame)
        city_summaries.append(
            {
                "city": city["name"],
                "country_iso": city["country_iso"],
                "grid_rows": int(len(city_frame)),
                "worldcover_tiles": resolve_city_worldcover_tiles(city, boundary=boundary),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    ensure_parent_dir(Path(output_path))
    write_table(combined, Path(output_path))

    if metadata_path:
        write_json(
            {
                "study_id": study["study_id"],
                "cell_size_m": cell_size_m,
                "cities": city_summaries,
                "output_path": str(Path(output_path)),
            },
            Path(metadata_path),
        )

    return combined
