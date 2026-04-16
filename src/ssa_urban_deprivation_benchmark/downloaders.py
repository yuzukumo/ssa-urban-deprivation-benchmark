from pathlib import Path
from typing import Dict
from typing import Iterable

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests

from ssa_urban_deprivation_benchmark.io_utils import ensure_parent_dir
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.study import load_study_config
from ssa_urban_deprivation_benchmark.study import resolve_city_worldcover_tiles
from ssa_urban_deprivation_benchmark.study import slugify


WORLDPOP_API = "https://hub.worldpop.org/rest/data/pop/wpgp?iso3={iso3}"
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{adm}/"
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)


def _download_file(url: str, output_path: Path, overwrite: bool = False) -> Dict:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return {
            "path": str(output_path),
            "url": url,
            "status": "exists",
            "bytes": output_path.stat().st_size,
        }

    ensure_parent_dir(output_path)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    return {
        "path": str(output_path),
        "url": url,
        "status": "downloaded",
        "bytes": output_path.stat().st_size,
    }


def _get_worldpop_latest_record(iso3: str) -> Dict:
    response = requests.get(WORLDPOP_API.format(iso3=iso3), timeout=60)
    response.raise_for_status()
    payload = response.json()["data"]
    latest = sorted(payload, key=lambda record: int(record.get("popyear") or 0))[-1]
    return latest


def _download_geoboundaries(iso3: str, adm: str = "ADM2", overwrite: bool = False) -> Dict:
    response = requests.get(GEOBOUNDARIES_API.format(iso3=iso3, adm=adm), timeout=60)
    response.raise_for_status()
    payload = response.json()
    url = payload["gjDownloadURL"]
    output_path = Path("data/raw/boundaries/geoboundaries") / "{iso3}_{adm}.geojson".format(
        iso3=iso3,
        adm=adm.lower(),
    )
    file_result = _download_file(url, output_path, overwrite=overwrite)
    file_result["metadata"] = {
        "boundary_id": payload["boundaryID"],
        "boundary_type": payload["boundaryType"],
        "source": payload["boundarySource"],
    }
    return file_result


def _download_worldpop(iso3: str, overwrite: bool = False) -> Dict:
    record = _get_worldpop_latest_record(iso3)
    url = record["files"][0]
    output_path = Path("data/raw/worldpop") / iso3 / Path(url).name
    result = _download_file(url, output_path, overwrite=overwrite)
    result["metadata"] = {
        "title": record["title"],
        "year": record["popyear"],
        "category": record["category"],
        "license": record["license"],
    }
    return result


def _download_worldcover_tiles(tiles: Iterable[str], overwrite: bool = False) -> Dict:
    results = {}
    for tile in sorted(set(tiles)):
        url = WORLDCOVER_URL.format(tile=tile)
        output_path = Path("data/raw/worldcover") / Path(url).name
        results[tile] = _download_file(url, output_path, overwrite=overwrite)
    return results


def _sanitize_gdf_for_file(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if column == cleaned.geometry.name:
            continue
        if cleaned[column].dtype != "object":
            continue

        cleaned[column] = cleaned[column].map(_normalize_value_for_file)

    return cleaned


def _normalize_value_for_file(value):
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _safe_write_gpkg(frame: gpd.GeoDataFrame, path: Path) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    if path.exists():
        path.unlink()
    _sanitize_gdf_for_file(frame).to_file(path, driver="GPKG", engine="pyogrio")


def _select_columns(frame: gpd.GeoDataFrame, columns: Iterable[str]) -> gpd.GeoDataFrame:
    available = [column for column in columns if column in frame.columns]
    if frame.geometry.name not in available:
        available.append(frame.geometry.name)
    return frame[available].copy()


def _save_osm_city_layers(city: Dict, overwrite: bool = False) -> Dict:
    place_query = city["place_query"]
    slug = city.get("slug", slugify(city["name"]))

    boundary_path = Path("data/raw/osm") / "{slug}_boundary.gpkg".format(slug=slug)
    roads_path = Path("data/raw/osm") / "{slug}_roads.gpkg".format(slug=slug)
    nodes_path = Path("data/raw/osm") / "{slug}_road_nodes.gpkg".format(slug=slug)
    amenities_path = Path("data/raw/osm") / "{slug}_amenities.gpkg".format(slug=slug)

    if all(path.exists() for path in [boundary_path, roads_path, nodes_path, amenities_path]) and not overwrite:
        return {
            "boundary": str(boundary_path),
            "roads": str(roads_path),
            "road_nodes": str(nodes_path),
            "amenities": str(amenities_path),
            "status": "exists",
        }

    ox.settings.use_cache = True
    ox.settings.cache_folder = "data/raw/osm/cache"

    boundary = ox.geocode_to_gdf(place_query)
    polygon = boundary.geometry.iloc[0]
    graph = ox.graph_from_polygon(polygon, network_type="all", simplify=True)
    nodes, edges = ox.graph_to_gdfs(graph)
    amenities = ox.features_from_polygon(polygon, tags={"amenity": True})

    boundary = _select_columns(
        boundary.reset_index(drop=True),
        ["display_name", "osm_type", "osm_id", "geometry"],
    )
    nodes = _select_columns(
        nodes.reset_index(),
        ["osmid", "street_count", "highway", "geometry"],
    )
    edges = _select_columns(
        edges.reset_index(),
        ["u", "v", "key", "osmid", "name", "highway", "oneway", "length", "geometry"],
    )
    amenities = _select_columns(
        amenities.reset_index(),
        ["element_type", "osmid", "amenity", "healthcare", "name", "geometry"],
    )

    _safe_write_gpkg(boundary, boundary_path)
    _safe_write_gpkg(edges, roads_path)
    _safe_write_gpkg(nodes, nodes_path)
    _safe_write_gpkg(amenities, amenities_path)

    return {
        "boundary": str(boundary_path),
        "roads": str(roads_path),
        "road_nodes": str(nodes_path),
        "amenities": str(amenities_path),
        "status": "downloaded",
        "counts": {
            "boundary_rows": int(len(boundary)),
            "road_edges": int(len(edges)),
            "road_nodes": int(len(nodes)),
            "amenities": int(len(amenities)),
        },
    }


def download_study_assets(
    study_config_path: Path,
    overwrite: bool = False,
    manifest_output: Path = None,
) -> Dict:
    study = load_study_config(Path(study_config_path))
    scope = study["scope"]
    city_details = study.get("city_details", [])

    manifest = {
        "study_id": study["study_id"],
        "countries": {},
        "worldcover_tiles": {},
        "cities": {},
    }

    for iso3 in scope["countries"]:
        manifest["countries"][iso3] = {
            "geoboundaries_adm2": _download_geoboundaries(iso3, overwrite=overwrite),
            "worldpop_latest": _download_worldpop(iso3, overwrite=overwrite),
        }

    tiles = set()
    for city in city_details:
        city_assets = _save_osm_city_layers(city, overwrite=overwrite)
        boundary = gpd.read_file(city_assets["boundary"])
        city_tiles = resolve_city_worldcover_tiles(city=city, boundary=boundary)
        tiles.update(city_tiles)
        city_assets["worldcover_tiles"] = city_tiles
        manifest["cities"][city["name"]] = city_assets

    manifest["worldcover_tiles"] = _download_worldcover_tiles(sorted(tiles), overwrite=overwrite)

    if manifest_output:
        write_json(manifest, Path(manifest_output))

    return manifest
