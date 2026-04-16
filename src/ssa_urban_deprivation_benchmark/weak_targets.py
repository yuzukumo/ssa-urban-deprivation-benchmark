from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors

from ssa_urban_deprivation_benchmark.io_utils import ensure_parent_dir
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.io_utils import write_table
from ssa_urban_deprivation_benchmark.io_utils import read_table


HDX_RWI_DATASET_ID = "relative-wealth-index"
HDX_RWI_PACKAGE_SHOW_URL = "https://data.humdata.org/api/3/action/package_show?id={dataset_id}"
COUNTRY_NAME_LOOKUP = {
    "KEN": "Kenya",
    "TZA": "Tanzania",
    "UGA": "Uganda",
    "GHA": "Ghana",
    "ETH": "Ethiopia",
    "NGA": "Nigeria",
}


def _resource_match_score(
    resource: dict[str, Any],
    country_name: Optional[str],
    country_iso: str,
) -> tuple[int, int]:
    name = str(resource.get("name") or "").lower()
    url = str(resource.get("download_url") or resource.get("url") or "").lower()
    score = 0
    if "/{iso}_relative_wealth_index.csv".format(iso=country_iso.lower()) in url:
        score += 5
    if name == "{iso}_relative_wealth_index.csv".format(iso=country_iso.lower()):
        score += 3
    if country_name:
        if name == "{country}_relative_wealth_index.csv".format(country=country_name.lower()):
            score += 4
        if name.startswith(country_name.lower()):
            score += 2
    if url.endswith(".csv"):
        score += 1
    return score, int(resource.get("size") or 0)


def _resolve_rwi_resource(resources: list[dict[str, Any]], country_iso: str) -> dict[str, Any]:
    country_name = COUNTRY_NAME_LOOKUP.get(country_iso)
    candidates = []
    for resource in resources:
        score, size = _resource_match_score(resource, country_name=country_name, country_iso=country_iso)
        if score > 0:
            candidates.append((score, size, resource))

    if not candidates:
        raise ValueError("Could not find an RWI CSV resource for {iso}.".format(iso=country_iso))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def download_rwi_country_files(
    countries: Iterable[str],
    output_dir: Path,
    manifest_path: Optional[Path] = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        HDX_RWI_PACKAGE_SHOW_URL.format(dataset_id=HDX_RWI_DATASET_ID),
        timeout=60,
    )
    response.raise_for_status()
    package = response.json()["result"]
    resources = package.get("resources", [])

    manifest: dict[str, Any] = {
        "dataset_id": HDX_RWI_DATASET_ID,
        "title": package.get("title"),
        "source_url": "https://data.humdata.org/dataset/{dataset_id}".format(
            dataset_id=HDX_RWI_DATASET_ID
        ),
        "countries": {},
    }

    for country_iso in countries:
        resource = _resolve_rwi_resource(resources=resources, country_iso=str(country_iso))
        download_url = resource.get("download_url") or resource.get("url")
        if not download_url:
            raise ValueError("RWI resource for {iso} has no download URL.".format(iso=country_iso))

        output_path = output_dir / "{iso}_relative_wealth_index.csv".format(iso=country_iso)
        asset = requests.get(download_url, timeout=180)
        asset.raise_for_status()
        output_path.write_bytes(asset.content)

        manifest["countries"][str(country_iso)] = {
            "country_name": COUNTRY_NAME_LOOKUP.get(str(country_iso), str(country_iso)),
            "resource_name": resource.get("name"),
            "download_url": download_url,
            "path": str(output_path),
            "bytes": int(output_path.stat().st_size),
        }

    if manifest_path:
        write_json(manifest, Path(manifest_path))
    return manifest


def _load_rwi_points(path: Path, country_iso: str) -> gpd.GeoDataFrame:
    frame = pd.read_csv(path)
    required = ["latitude", "longitude", "rwi", "error"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "RWI file {path} is missing columns: {columns}".format(
                path=path,
                columns=", ".join(missing),
            )
        )
    result = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs="EPSG:4326",
    )
    result["country_iso"] = str(country_iso)
    return result


def _weighted_knn_targets(
    centroids_proj: gpd.GeoDataFrame,
    rwi_proj: gpd.GeoDataFrame,
    neighbors: int,
    max_distance_m: float,
) -> pd.DataFrame:
    if rwi_proj.empty:
        raise ValueError("No RWI points were available for the requested country.")

    point_coords = np.column_stack([rwi_proj.geometry.x.to_numpy(), rwi_proj.geometry.y.to_numpy()])
    query_coords = np.column_stack([centroids_proj.geometry.x.to_numpy(), centroids_proj.geometry.y.to_numpy()])
    model = NearestNeighbors(
        n_neighbors=min(int(neighbors), len(rwi_proj)),
        metric="euclidean",
    )
    model.fit(point_coords)
    distances, indices = model.kneighbors(query_coords)

    values = rwi_proj["rwi"].to_numpy(dtype=float)
    errors = rwi_proj["error"].to_numpy(dtype=float)

    target_values = []
    target_errors = []
    target_distances = []
    target_neighbors = []

    for row_distances, row_indices in zip(distances, indices):
        valid_mask = row_distances <= float(max_distance_m)
        if not valid_mask.any():
            valid_mask[np.argmin(row_distances)] = True

        selected_distances = row_distances[valid_mask]
        selected_indices = row_indices[valid_mask]
        selected_values = values[selected_indices]
        selected_errors = errors[selected_indices]
        weights = 1.0 / np.maximum(selected_distances, 25.0)
        weights = weights / weights.sum()

        target_values.append(float(np.sum(selected_values * weights)))
        target_errors.append(float(np.sum(selected_errors * weights)))
        target_distances.append(float(np.max(selected_distances)))
        target_neighbors.append(int(len(selected_indices)))

    return pd.DataFrame(
        {
            "rwi_mean": target_values,
            "rwi_error_weighted_mean": target_errors,
            "rwi_max_neighbor_distance_m": target_distances,
            "rwi_neighbor_count": target_neighbors,
        }
    )


def build_rwi_grid_targets(
    input_path: Path,
    output_path: Path,
    rwi_dir: Path,
    metadata_path: Optional[Path] = None,
    group_col: str = "city",
    country_col: str = "country_iso",
    score_col: str = "deprivation_index_0_100",
    neighbors: int = 4,
    max_distance_m: float = 4000.0,
    low_wealth_quantile: float = 0.2,
) -> None:
    frame = read_table(Path(input_path))
    if not isinstance(frame, gpd.GeoDataFrame):
        raise ValueError("build-rwi-grid-targets requires a geospatial input file.")

    required = [group_col, country_col, score_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))
    if low_wealth_quantile <= 0 or low_wealth_quantile >= 1:
        raise ValueError("low_wealth_quantile must be in the interval (0, 1).")

    result_frames = []
    metadata: dict[str, Any] = {
        "input_path": str(Path(input_path)),
        "output_path": str(Path(output_path)),
        "rwi_dir": str(Path(rwi_dir)),
        "group_col": group_col,
        "country_col": country_col,
        "score_col": score_col,
        "neighbors": int(neighbors),
        "max_distance_m": float(max_distance_m),
        "low_wealth_quantile": float(low_wealth_quantile),
        "cities": [],
    }

    for city, subset in frame.groupby(group_col):
        country_iso = str(subset[country_col].iloc[0])
        rwi_path = Path(rwi_dir) / "{iso}_relative_wealth_index.csv".format(iso=country_iso)
        if not rwi_path.exists():
            raise FileNotFoundError("Missing RWI file for {iso}: {path}".format(iso=country_iso, path=rwi_path))

        centroid_proj = subset[[group_col, country_col, score_col, "geometry"]].copy()
        centroid_proj = gpd.GeoDataFrame(centroid_proj, geometry="geometry", crs=frame.crs)
        centroid_proj["geometry"] = centroid_proj.geometry.representative_point()
        city_crs = subset.estimate_utm_crs()
        centroid_proj = centroid_proj.to_crs(city_crs)

        rwi_proj = _load_rwi_points(rwi_path, country_iso=country_iso).to_crs(city_crs)
        targets = _weighted_knn_targets(
            centroids_proj=centroid_proj,
            rwi_proj=rwi_proj,
            neighbors=int(neighbors),
            max_distance_m=float(max_distance_m),
        )

        city_result = subset.copy().reset_index(drop=True)
        city_result["rwi_mean"] = targets["rwi_mean"]
        city_result["rwi_error_weighted_mean"] = targets["rwi_error_weighted_mean"]
        city_result["rwi_max_neighbor_distance_m"] = targets["rwi_max_neighbor_distance_m"]
        city_result["rwi_neighbor_count"] = targets["rwi_neighbor_count"]
        city_result["rwi_label_available"] = city_result["rwi_neighbor_count"] > 0
        city_result["rwi_within_city_percentile"] = city_result["rwi_mean"].rank(
            method="average",
            pct=True,
            ascending=True,
        )
        city_result["rwi_bottom_quantile_flag"] = (
            city_result["rwi_within_city_percentile"] <= float(low_wealth_quantile)
        )
        city_result["rwi_deprivation_proxy_0_100"] = (
            (1.0 - city_result["rwi_within_city_percentile"]) * 100.0
        )

        score = pd.to_numeric(city_result[score_col], errors="coerce")
        wealth = pd.to_numeric(city_result["rwi_mean"], errors="coerce")
        metadata["cities"].append(
            {
                "city": str(city),
                "country_iso": country_iso,
                "n_rows": int(len(city_result)),
                "rwi_file": str(rwi_path),
                "rwi_mean_min": float(wealth.min()),
                "rwi_mean_max": float(wealth.max()),
                "spearman_corr_rwi_vs_score": float(wealth.corr(score, method="spearman")),
                "spearman_corr_rwi_vs_negative_score": float(wealth.corr(-score, method="spearman")),
                "mean_max_neighbor_distance_m": float(city_result["rwi_max_neighbor_distance_m"].mean()),
                "label_coverage_share": float(city_result["rwi_label_available"].mean()),
            }
        )
        result_frames.append(city_result)

    result = gpd.GeoDataFrame(
        pd.concat(result_frames, ignore_index=True),
        geometry="geometry",
        crs=frame.crs,
    )
    write_table(result, Path(output_path))

    if metadata_path:
        write_json(metadata, Path(metadata_path))
