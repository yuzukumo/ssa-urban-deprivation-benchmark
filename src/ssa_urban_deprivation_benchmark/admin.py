from math import ceil
from pathlib import Path
from typing import Any
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.io_utils import write_table
from ssa_urban_deprivation_benchmark.io_utils import read_table


ADMIN_LEVEL_SUFFIX = {
    1: "adm1",
    2: "adm2",
    3: "adm3",
}


def _boundary_path(boundary_dir: Path, country_iso: str, admin_level: int) -> Path:
    if admin_level not in ADMIN_LEVEL_SUFFIX:
        raise ValueError("Unsupported admin level: {level}".format(level=admin_level))
    suffix = ADMIN_LEVEL_SUFFIX[admin_level]
    path = Path(boundary_dir) / "{country}_{suffix}.geojson".format(
        country=country_iso,
        suffix=suffix,
    )
    if not path.exists():
        raise FileNotFoundError("Boundary file not found: {path}".format(path=path))
    return path


def _load_boundaries(
    boundary_dir: Path,
    country_iso: str,
    admin_level: int,
    target_crs,
    admin_prefix: str,
) -> gpd.GeoDataFrame:
    path = _boundary_path(boundary_dir=Path(boundary_dir), country_iso=country_iso, admin_level=admin_level)
    boundaries = gpd.read_file(path)
    required = ["shapeName", "shapeID", "geometry"]
    missing = [column for column in required if column not in boundaries.columns]
    if missing:
        raise ValueError(
            "Boundary file {path} is missing columns: {columns}".format(
                path=path,
                columns=", ".join(missing),
            )
        )

    iso_column = "shapeISO" if "shapeISO" in boundaries.columns else None
    selected = boundaries[["shapeName", "shapeID", "geometry"] + ([iso_column] if iso_column else [])].copy()
    selected = selected.rename(
        columns={
            "shapeName": "{prefix}_name".format(prefix=admin_prefix),
            "shapeID": "{prefix}_id".format(prefix=admin_prefix),
            iso_column: "{prefix}_iso".format(prefix=admin_prefix),
        }
    )
    if "{prefix}_iso".format(prefix=admin_prefix) not in selected.columns:
        selected["{prefix}_iso".format(prefix=admin_prefix)] = None
    selected["country_iso"] = country_iso
    return selected.to_crs(target_crs)


def attach_admin_units(
    input_path: Path,
    output_path: Path,
    boundary_dir: Path,
    admin_level: int = 2,
    country_col: str = "country_iso",
    admin_prefix: str = "admin2",
    metadata_path: Optional[Path] = None,
) -> None:
    frame = read_table(Path(input_path))
    if not isinstance(frame, gpd.GeoDataFrame):
        raise ValueError("attach-admin-units requires a geospatial input file.")
    if country_col not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=country_col))

    result = frame.copy().reset_index(drop=True)
    result["_row_id"] = np.arange(len(result))
    points = result[["_row_id", country_col, "geometry"]].copy()
    points["geometry"] = points.geometry.representative_point()
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=result.crs)

    admin_name_col = "{prefix}_name".format(prefix=admin_prefix)
    admin_id_col = "{prefix}_id".format(prefix=admin_prefix)
    admin_iso_col = "{prefix}_iso".format(prefix=admin_prefix)

    matched_frames = []
    metadata: dict[str, Any] = {
        "input_path": str(Path(input_path)),
        "output_path": str(Path(output_path)),
        "boundary_dir": str(Path(boundary_dir)),
        "admin_level": int(admin_level),
        "country_col": country_col,
        "admin_prefix": admin_prefix,
        "countries": [],
    }

    for country_iso, subset in points.groupby(country_col):
        boundaries = _load_boundaries(
            boundary_dir=Path(boundary_dir),
            country_iso=str(country_iso),
            admin_level=admin_level,
            target_crs=result.crs,
            admin_prefix=admin_prefix,
        )
        attach_cols = [admin_name_col, admin_id_col, admin_iso_col, "geometry"]
        working = subset[["_row_id", "geometry"]].copy()
        within = gpd.sjoin(
            gpd.GeoDataFrame(working, geometry="geometry", crs=result.crs),
            boundaries[attach_cols],
            how="left",
            predicate="within",
        )
        within = within.drop(columns=["index_right"], errors="ignore")
        within = within.sort_values("_row_id").drop_duplicates("_row_id", keep="first")

        missing_ids = set(subset["_row_id"]) - set(within.loc[within[admin_id_col].notna(), "_row_id"])
        nearest = pd.DataFrame(columns=["_row_id", admin_name_col, admin_id_col, admin_iso_col])
        if missing_ids:
            missing_points = working.loc[working["_row_id"].isin(missing_ids)].copy()
            projected_crs = boundaries.estimate_utm_crs()
            boundaries_proj = boundaries[attach_cols].to_crs(projected_crs)
            missing_points_proj = gpd.GeoDataFrame(
                missing_points,
                geometry="geometry",
                crs=result.crs,
            ).to_crs(projected_crs)
            nearest = gpd.sjoin_nearest(
                missing_points_proj,
                boundaries_proj,
                how="left",
            )
            nearest = nearest.drop(columns=["index_right"], errors="ignore")
            nearest = nearest.sort_values("_row_id").drop_duplicates("_row_id", keep="first")

        combined = pd.concat(
            [
                within.loc[within[admin_id_col].notna(), ["_row_id", admin_name_col, admin_id_col, admin_iso_col]],
                nearest[["_row_id", admin_name_col, admin_id_col, admin_iso_col]],
            ],
            ignore_index=True,
        )
        combined = combined.sort_values("_row_id").drop_duplicates("_row_id", keep="first")
        matched_frames.append(combined)

        metadata["countries"].append(
            {
                "country_iso": str(country_iso),
                "boundary_path": str(_boundary_path(Path(boundary_dir), str(country_iso), admin_level)),
                "n_rows": int(len(subset)),
                "matched_within_count": int(within[admin_id_col].notna().sum()),
                "matched_nearest_count": int(len(nearest)),
                "unmatched_count": int(len(subset) - len(combined)),
            }
        )

    matched = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame(columns=["_row_id"])
    result = result.merge(matched, on="_row_id", how="left")
    result[admin_name_col] = result[admin_name_col].fillna("unassigned")
    result[admin_id_col] = result[admin_id_col].fillna("unassigned")
    result[admin_iso_col] = result[admin_iso_col].where(result[admin_iso_col].notna(), None)
    result = gpd.GeoDataFrame(result.drop(columns=["_row_id"]), geometry="geometry", crs=frame.crs)
    write_table(result, Path(output_path))

    if metadata_path:
        metadata["unassigned_total"] = int((result[admin_id_col] == "unassigned").sum())
        write_json(metadata, Path(metadata_path))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> Optional[float]:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    valid = values.notna()
    if not valid.any():
        return None
    values = values.loc[valid]
    weights = weights.loc[valid]
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float(values.mean())
    return float((values * weights).sum() / total_weight)


def _weighted_mode(values: pd.Series, weights: pd.Series) -> tuple[str, Optional[float]]:
    series = values.fillna("missing").astype(str)
    weight_series = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    if series.empty:
        return "missing", None
    grouped = (
        pd.DataFrame({"category": series, "weight": weight_series})
        .groupby("category", as_index=False)["weight"]
        .sum()
        .sort_values(["weight", "category"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if grouped.empty:
        return "missing", None
    total_weight = float(grouped["weight"].sum())
    top_weight = float(grouped.loc[0, "weight"])
    return str(grouped.loc[0, "category"]), (top_weight / total_weight if total_weight > 0 else None)


def _classify_district_priority(metric_count: int, eligible: bool) -> str:
    if not eligible:
        return "small_footprint"
    if metric_count >= 2:
        return "high_priority"
    if metric_count == 1:
        return "emerging_priority"
    return "lower_priority"


def summarize_admin_units(
    input_path: Path,
    output_path: Path,
    group_col: str,
    country_col: str,
    admin_name_col: str,
    admin_id_col: str,
    score_col: str,
    population_col: str,
    admin_iso_col: Optional[str] = None,
    hotspot_col: Optional[str] = None,
    hotspot_value: str = "high_high",
    dominant_dimension_col: Optional[str] = None,
    priority_col: Optional[str] = None,
    top_fraction: float = 0.1,
    priority_fraction: float = 0.25,
    min_cells: int = 10,
    min_city_population_share: float = 0.01,
    metadata_path: Optional[Path] = None,
) -> None:
    frame = read_table(Path(input_path))
    if not isinstance(frame, gpd.GeoDataFrame):
        raise ValueError("summarize-admin-units requires a geospatial input file.")

    required = [group_col, country_col, admin_name_col, admin_id_col, score_col, population_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: {columns}".format(columns=", ".join(missing)))
    if top_fraction <= 0 or top_fraction >= 1:
        raise ValueError("top_fraction must be between 0 and 1.")
    if priority_fraction <= 0 or priority_fraction > 1:
        raise ValueError("priority_fraction must be in the interval (0, 1].")

    working = frame.copy()
    working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
    working[population_col] = pd.to_numeric(working[population_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    working["_is_city_top_fraction"] = False
    if hotspot_col and hotspot_col in working.columns:
        working["_is_hotspot"] = working[hotspot_col].fillna("missing").astype(str) == str(hotspot_value)
    else:
        working["_is_hotspot"] = False

    city_score_thresholds = (
        working.groupby(group_col)[score_col]
        .quantile(1.0 - float(top_fraction))
        .to_dict()
    )
    working["_city_score_threshold"] = working[group_col].map(city_score_thresholds)
    working["_is_city_top_fraction"] = working[score_col] >= working["_city_score_threshold"]

    group_cols = [group_col, country_col, admin_name_col, admin_id_col]
    if admin_iso_col and admin_iso_col in working.columns:
        group_cols.append(admin_iso_col)
    else:
        working["_admin_iso_placeholder"] = None
        group_cols.append("_admin_iso_placeholder")

    city_population_totals = working.groupby(group_col)[population_col].sum().to_dict()
    city_cell_totals = working.groupby(group_col).size().to_dict()

    records = []
    for keys, subset in working.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        city = key_map[group_col]
        population_total = float(subset[population_col].sum())
        city_population_total = float(city_population_totals.get(city, 0.0))
        hotspot_mask = subset["_is_hotspot"].fillna(False)
        top_fraction_mask = subset["_is_city_top_fraction"].fillna(False)

        hotspot_dimension = "missing"
        hotspot_dimension_weight_share = None
        if dominant_dimension_col and dominant_dimension_col in subset.columns and hotspot_mask.any():
            hotspot_dimension, hotspot_dimension_weight_share = _weighted_mode(
                subset.loc[hotspot_mask, dominant_dimension_col],
                subset.loc[hotspot_mask, population_col],
            )

        hotspot_priority_mode = "missing"
        hotspot_priority_mode_weight_share = None
        if priority_col and priority_col in subset.columns and hotspot_mask.any():
            hotspot_priority_mode, hotspot_priority_mode_weight_share = _weighted_mode(
                subset.loc[hotspot_mask, priority_col],
                subset.loc[hotspot_mask, population_col],
            )

        records.append(
            {
                group_col: city,
                country_col: key_map[country_col],
                admin_name_col: key_map[admin_name_col],
                admin_id_col: key_map[admin_id_col],
                admin_iso_col or "_admin_iso_placeholder": key_map[group_cols[-1]],
                "n_cells": int(len(subset)),
                "city_cell_share": float(len(subset) / max(city_cell_totals.get(city, 1), 1)),
                "population_total": population_total,
                "city_population_share": float(population_total / city_population_total)
                if city_population_total > 0
                else 0.0,
                "mean_score": float(subset[score_col].mean()),
                "median_score": float(subset[score_col].median()),
                "population_weighted_mean_score": _weighted_mean(subset[score_col], subset[population_col]),
                "city_score_top_fraction_threshold": float(city_score_thresholds.get(city, np.nan)),
                "citywide_q90_population_share": float(
                    subset.loc[top_fraction_mask, population_col].sum() / population_total
                )
                if population_total > 0
                else 0.0,
                "citywide_q90_cell_share": float(top_fraction_mask.mean()) if len(subset) > 0 else 0.0,
                "hotspot_population_total": float(subset.loc[hotspot_mask, population_col].sum()),
                "hotspot_population_share": float(
                    subset.loc[hotspot_mask, population_col].sum() / population_total
                )
                if population_total > 0
                else 0.0,
                "hotspot_cell_count": int(hotspot_mask.sum()),
                "hotspot_cell_share": float(hotspot_mask.mean()) if len(subset) > 0 else 0.0,
                "hotspot_dominant_dimension": hotspot_dimension,
                "hotspot_dominant_dimension_weight_share": hotspot_dimension_weight_share,
                "hotspot_priority_mode": hotspot_priority_mode,
                "hotspot_priority_mode_weight_share": hotspot_priority_mode_weight_share,
            }
        )

    summary = pd.DataFrame(records)
    if summary.empty:
        raise ValueError("No admin-unit summaries could be computed.")

    summary["eligible_priority_flag"] = (
        (summary["n_cells"] >= int(min_cells))
        & (summary["city_population_share"] >= float(min_city_population_share))
    )

    metric_columns = [
        "population_weighted_mean_score",
        "citywide_q90_population_share",
        "hotspot_population_share",
    ]
    metric_flag_columns = []
    for metric in metric_columns:
        flag_col = "{metric}_top_priority_flag".format(metric=metric)
        metric_flag_columns.append(flag_col)
        summary[flag_col] = False

    for city, subset in summary.loc[summary["eligible_priority_flag"]].groupby(group_col):
        if subset.empty:
            continue
        top_n = max(1, int(ceil(len(subset) * float(priority_fraction))))
        for metric, flag_col in zip(metric_columns, metric_flag_columns):
            valid = subset.loc[subset[metric].notna()]
            if valid.empty:
                continue
            top_index = valid.nlargest(top_n, metric).index
            summary.loc[top_index, flag_col] = True

    summary["district_priority_metric_count"] = summary[metric_flag_columns].sum(axis=1).astype(int)
    summary["district_priority_class"] = [
        _classify_district_priority(metric_count=int(metric_count), eligible=bool(eligible))
        for metric_count, eligible in zip(
            summary["district_priority_metric_count"],
            summary["eligible_priority_flag"],
        )
    ]

    summary["district_priority_rank"] = np.nan
    for city, subset in summary.loc[summary["eligible_priority_flag"]].groupby(group_col):
        ordered = subset.sort_values(
            [
                "district_priority_metric_count",
                "hotspot_population_share",
                "citywide_q90_population_share",
                "population_weighted_mean_score",
                "population_total",
            ],
            ascending=[False, False, False, False, False],
        )
        summary.loc[ordered.index, "district_priority_rank"] = np.arange(1, len(ordered) + 1)

    geometry_summary = (
        working[group_cols + ["geometry"]]
        .dissolve(by=group_cols, as_index=False)
        .rename(columns={"_admin_iso_placeholder": admin_iso_col or "_admin_iso_placeholder"})
    )
    result = geometry_summary.merge(summary, on=group_cols, how="left")
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=working.crs)
    result = result.sort_values([group_col, "district_priority_rank", "population_total"], ascending=[True, True, False])
    write_table(result, Path(output_path))

    if metadata_path:
        metadata = {
            "input_path": str(Path(input_path)),
            "output_path": str(Path(output_path)),
            "group_col": group_col,
            "country_col": country_col,
            "admin_name_col": admin_name_col,
            "admin_id_col": admin_id_col,
            "admin_iso_col": admin_iso_col,
            "score_col": score_col,
            "population_col": population_col,
            "hotspot_col": hotspot_col,
            "hotspot_value": hotspot_value,
            "dominant_dimension_col": dominant_dimension_col,
            "priority_col": priority_col,
            "top_fraction": float(top_fraction),
            "priority_fraction": float(priority_fraction),
            "min_cells": int(min_cells),
            "min_city_population_share": float(min_city_population_share),
            "city_score_thresholds": {
                str(key): float(value)
                for key, value in city_score_thresholds.items()
            },
        }
        write_json(metadata, Path(metadata_path))
