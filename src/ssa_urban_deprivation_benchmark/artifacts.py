from datetime import datetime
from datetime import timezone
from pathlib import Path
import shutil
from typing import Any
from typing import Optional

import numpy as np
import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import ensure_parent_dir
from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import read_yaml
from ssa_urban_deprivation_benchmark.io_utils import write_json


def _resolve_config_path(path_value: str, config_path: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    config_candidate = config_path.parent / candidate
    return config_candidate


def _filter_scenario(frame: pd.DataFrame, scenario: Optional[str]) -> pd.DataFrame:
    if scenario and "scenario" in frame.columns:
        return frame.loc[frame["scenario"].astype(str) == str(scenario)].copy()
    return frame.copy()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _row_as_dict(frame: pd.DataFrame, key_col: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = str(row[key_col])
        result[key] = {
            str(column): _to_builtin(row[column])
            for column in frame.columns
            if column != key_col
        }
    return result


def _ranking_summary(frame: pd.DataFrame, value_col: str) -> dict[str, Any]:
    ordered = frame.sort_values(value_col, ascending=False).reset_index(drop=True)
    ranking = [
        {
            "city": str(row["city"]),
            "value": _to_builtin(row[value_col]),
            "rank": index + 1,
        }
        for index, (_, row) in enumerate(ordered.iterrows())
    ]
    gap = None
    if len(ordered) >= 2:
        gap = float(ordered.iloc[0][value_col] - ordered.iloc[-1][value_col])
    return {
        "metric": value_col,
        "ranking": ranking,
        "top_minus_bottom_gap": gap,
    }


def _top_categories(
    frame: pd.DataFrame,
    category_col: str,
    share_col: str,
    top_n: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for city, subset in frame.groupby("city"):
        ordered = subset.sort_values(share_col, ascending=False).head(top_n)
        result[str(city)] = [
            {
                "category": str(row[category_col]),
                share_col: _to_builtin(row[share_col]),
                "rank": index + 1,
            }
            for index, (_, row) in enumerate(ordered.iterrows())
        ]
    return result


def _top_admin_units(
    frame: pd.DataFrame,
    top_n: int = 3,
    only_high_priority: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for city, subset in frame.groupby("city"):
        working = subset.copy()
        if only_high_priority:
            working = working.loc[working["district_priority_class"].astype(str) == "high_priority"].copy()
        working = working.loc[working["district_priority_rank"].notna()].copy()
        working = working.sort_values(
            ["district_priority_rank", "population_total"],
            ascending=[True, False],
        ).head(top_n)
        result[str(city)] = [
            {
                "admin_unit": str(row["admin2_name"]),
                "district_priority_rank": _to_builtin(row["district_priority_rank"]),
                "district_priority_class": str(row["district_priority_class"]),
                "district_priority_metric_count": _to_builtin(row["district_priority_metric_count"]),
                "population_weighted_mean_score": _to_builtin(row["population_weighted_mean_score"]),
                "citywide_q90_population_share": _to_builtin(row["citywide_q90_population_share"]),
                "hotspot_population_share": _to_builtin(row["hotspot_population_share"]),
                "hotspot_dominant_dimension": _to_builtin(row["hotspot_dominant_dimension"]),
            }
            for _, row in working.iterrows()
        ]
    return result


def _category_share_lookup(
    frame: pd.DataFrame,
    category_col: str,
    value_col: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for city, subset in frame.groupby("city"):
        result[str(city)] = {
            str(row[category_col]): _to_builtin(row[value_col])
            for _, row in subset.iterrows()
        }
    return result


def build_core_findings_artifact(config_path: Path, output_path: Path) -> None:
    config_path = Path(config_path)
    config = read_yaml(config_path)
    paths = config.get("paths", {})
    primary_scenario = config.get("primary_scenario")
    sensitivity_scenario = config.get("sensitivity_scenario")
    top_fraction = float(config.get("top_fraction", 0.1))

    city_score_summary = _filter_scenario(
        read_table(_resolve_config_path(paths["city_score_summary"], config_path)),
        primary_scenario,
    )
    population_exposure = _filter_scenario(
        read_table(_resolve_config_path(paths["population_exposure"], config_path)),
        primary_scenario,
    )
    inequality_summary = _filter_scenario(
        read_table(_resolve_config_path(paths["inequality_summary"], config_path)),
        primary_scenario,
    )
    absolute_relative_alignment = _filter_scenario(
        read_table(_resolve_config_path(paths["absolute_relative_alignment"], config_path)),
        primary_scenario,
    )
    relative_shift_summary = _filter_scenario(
        read_table(_resolve_config_path(paths["relative_shift_summary"], config_path)),
        primary_scenario,
    )
    hotspot_dominant_population_share = _filter_scenario(
        read_table(_resolve_config_path(paths["hotspot_dominant_population_share"], config_path)),
        primary_scenario,
    )
    hotspot_priority_population_share = _filter_scenario(
        read_table(_resolve_config_path(paths["hotspot_priority_population_share"], config_path)),
        primary_scenario,
    )
    hotspot_typology_population_share = _filter_scenario(
        read_table(_resolve_config_path(paths["hotspot_typology_population_share"], config_path)),
        primary_scenario,
    )
    admin_summary = None
    if paths.get("admin_summary"):
        admin_summary = _filter_scenario(
            read_table(_resolve_config_path(paths["admin_summary"], config_path)),
            primary_scenario,
        )

    grid_size_score_summary = _filter_scenario(
        read_table(_resolve_config_path(paths["grid_size_score_summary"], config_path)),
        None,
    )
    grid_size_population_exposure = _filter_scenario(
        read_table(_resolve_config_path(paths["grid_size_population_exposure"], config_path)),
        None,
    )
    grid_size_inequality_summary = _filter_scenario(
        read_table(_resolve_config_path(paths["grid_size_inequality_summary"], config_path)),
        None,
    )

    city_score_lookup = _row_as_dict(city_score_summary, "city")
    exposure_lookup = _row_as_dict(population_exposure, "city")
    inequality_lookup = _row_as_dict(inequality_summary, "city")
    alignment_lookup = _row_as_dict(absolute_relative_alignment, "city")
    shift_lookup = _row_as_dict(relative_shift_summary, "city")
    hotspot_dominant_lookup = _top_categories(
        hotspot_dominant_population_share,
        category_col="dominant_dimension",
        share_col="weight_share",
        top_n=2,
    )
    hotspot_priority_lookup = _top_categories(
        hotspot_priority_population_share,
        category_col="priority_quadrant",
        share_col="weight_share",
        top_n=2,
    )
    hotspot_typology_lookup = _top_categories(
        hotspot_typology_population_share,
        category_col="hotspot_typology",
        share_col="weight_share",
        top_n=3,
    )
    admin_priority_lookup = _top_admin_units(admin_summary, top_n=3) if admin_summary is not None else {}
    admin_high_priority_lookup = (
        _top_admin_units(admin_summary, top_n=5, only_high_priority=True)
        if admin_summary is not None
        else {}
    )
    admin_priority_population_share_lookup = {}
    if admin_summary is not None:
        admin_priority_population_share = (
            admin_summary.groupby(["city", "district_priority_class"], as_index=False)["population_total"]
            .sum()
            .rename(columns={"population_total": "population_total_by_class"})
        )
        city_totals = admin_summary.groupby("city")["population_total"].sum().to_dict()
        admin_priority_population_share_lookup = _category_share_lookup(
            admin_priority_population_share.assign(
                population_share=lambda frame: frame["population_total_by_class"]
                / frame["city"].map(city_totals)
            ),
            category_col="district_priority_class",
            value_col="population_share",
        )

    cities = sorted(city_score_lookup.keys())
    score_by_scenario = {
        str(scenario): _row_as_dict(subset, "city")
        for scenario, subset in grid_size_score_summary.groupby("scenario")
    }
    exposure_by_scenario = {
        str(scenario): _row_as_dict(subset, "city")
        for scenario, subset in grid_size_population_exposure.groupby("scenario")
    }
    inequality_by_scenario = {
        str(scenario): _row_as_dict(subset, "city")
        for scenario, subset in grid_size_inequality_summary.groupby("scenario")
    }

    city_results = {}
    for city in cities:
        sensitivity = {}
        if primary_scenario in score_by_scenario and sensitivity_scenario in score_by_scenario:
            primary_score = score_by_scenario[primary_scenario][city]
            secondary_score = score_by_scenario[sensitivity_scenario][city]
            primary_exposure = exposure_by_scenario[primary_scenario][city]
            secondary_exposure = exposure_by_scenario[sensitivity_scenario][city]
            primary_inequality = inequality_by_scenario[primary_scenario][city]
            secondary_inequality = inequality_by_scenario[sensitivity_scenario][city]
            sensitivity = {
                "mean_score": {
                    primary_scenario: primary_score["mean"],
                    sensitivity_scenario: secondary_score["mean"],
                    "delta_{secondary}_minus_{primary}".format(
                        secondary=sensitivity_scenario,
                        primary=primary_scenario,
                    ): _to_builtin(secondary_score["mean"] - primary_score["mean"]),
                },
                "population_share_at_or_above_q90": {
                    primary_scenario: primary_exposure["population_share_at_or_above_q90"],
                    sensitivity_scenario: secondary_exposure["population_share_at_or_above_q90"],
                    "delta_{secondary}_minus_{primary}".format(
                        secondary=sensitivity_scenario,
                        primary=primary_scenario,
                    ): _to_builtin(
                        secondary_exposure["population_share_at_or_above_q90"]
                        - primary_exposure["population_share_at_or_above_q90"]
                    ),
                },
                "score_gini_population_weighted": {
                    primary_scenario: primary_inequality["score_gini_population_weighted"],
                    sensitivity_scenario: secondary_inequality["score_gini_population_weighted"],
                    "delta_{secondary}_minus_{primary}".format(
                        secondary=sensitivity_scenario,
                        primary=primary_scenario,
                    ): _to_builtin(
                        secondary_inequality["score_gini_population_weighted"]
                        - primary_inequality["score_gini_population_weighted"]
                    ),
                },
            }

        city_results[city] = {
            "primary_summary": {
                "score": city_score_lookup[city],
                "exposure": exposure_lookup[city],
                "inequality": inequality_lookup[city],
            },
            "hotspots": {
                "dominant_dimension_by_population": hotspot_dominant_lookup.get(city, []),
                "priority_quadrant_by_population": hotspot_priority_lookup.get(city, []),
                "top_typologies_by_population": hotspot_typology_lookup.get(city, []),
            },
            "absolute_vs_relative": {
                "alignment": alignment_lookup.get(city, {}),
                "shift": shift_lookup.get(city, {}),
            },
            "districts": {
                "top_priority_units": admin_priority_lookup.get(city, []),
                "high_priority_units": admin_high_priority_lookup.get(city, []),
                "priority_class_population_share": admin_priority_population_share_lookup.get(city, {}),
            },
            "sensitivity": sensitivity,
        }

    sensitivity_order_stability = {
        "mean_score": _ranking_summary(
            grid_size_score_summary.loc[grid_size_score_summary["scenario"].astype(str) == str(primary_scenario)],
            "mean",
        )["ranking"][0]["city"]
        == _ranking_summary(
            grid_size_score_summary.loc[
                grid_size_score_summary["scenario"].astype(str) == str(sensitivity_scenario)
            ],
            "mean",
        )["ranking"][0]["city"],
        "population_share_at_or_above_q90": _ranking_summary(
            grid_size_population_exposure.loc[
                grid_size_population_exposure["scenario"].astype(str) == str(primary_scenario)
            ],
            "population_share_at_or_above_q90",
        )["ranking"][0]["city"]
        == _ranking_summary(
            grid_size_population_exposure.loc[
                grid_size_population_exposure["scenario"].astype(str) == str(sensitivity_scenario)
            ],
            "population_share_at_or_above_q90",
        )["ranking"][0]["city"],
        "score_gini_population_weighted": _ranking_summary(
            grid_size_inequality_summary.loc[
                grid_size_inequality_summary["scenario"].astype(str) == str(primary_scenario)
            ],
            "score_gini_population_weighted",
        )["ranking"][0]["city"]
        == _ranking_summary(
            grid_size_inequality_summary.loc[
                grid_size_inequality_summary["scenario"].astype(str) == str(sensitivity_scenario)
            ],
            "score_gini_population_weighted",
        )["ranking"][0]["city"],
    }

    claims = [
        {
            "claim_id": "primary_mean_score_order",
            "topic": "cross_city_ordering",
            "scenario": primary_scenario,
            **_ranking_summary(city_score_summary, "mean"),
        },
        {
            "claim_id": "primary_q90_exposure_order",
            "topic": "cross_city_ordering",
            "scenario": primary_scenario,
            **_ranking_summary(population_exposure, "population_share_at_or_above_q90"),
        },
        {
            "claim_id": "primary_weighted_gini_order",
            "topic": "cross_city_ordering",
            "scenario": primary_scenario,
            **_ranking_summary(inequality_summary, "score_gini_population_weighted"),
        },
    ]

    for city in cities:
        top_dimension = hotspot_dominant_lookup.get(city, [])
        if top_dimension:
            claims.append(
                {
                    "claim_id": "hotspot_population_dominant_dimension__{city}".format(
                        city=city.lower().replace(" ", "_")
                    ),
                    "topic": "hotspot_mechanism",
                    "city": city,
                    "scenario": primary_scenario,
                    "category": top_dimension[0]["category"],
                    "weight_share": top_dimension[0]["weight_share"],
                }
            )

        top_typology = hotspot_typology_lookup.get(city, [])
        if top_typology:
            claims.append(
                {
                    "claim_id": "hotspot_population_top_typology__{city}".format(
                        city=city.lower().replace(" ", "_")
                    ),
                    "topic": "hotspot_typology",
                    "city": city,
                    "scenario": primary_scenario,
                    "category": top_typology[0]["category"],
                    "weight_share": top_typology[0]["weight_share"],
                }
            )

        alignment = alignment_lookup.get(city, {})
        shift = shift_lookup.get(city, {})
        if alignment and shift:
            claims.append(
                {
                    "claim_id": "absolute_relative_reranking__{city}".format(
                        city=city.lower().replace(" ", "_")
                    ),
                    "topic": "robustness_absolute_vs_relative",
                    "city": city,
                    "scenario": primary_scenario,
                    "pearson_corr": alignment.get("pearson_corr"),
                    "top_overlap_share": alignment.get("top_overlap_share"),
                    "top_fraction_reclassification_within_top_sets": _to_builtin(
                        1.0 - float(alignment.get("top_overlap_share", 0.0))
                    ),
                    "mean_abs_percentile_shift": shift.get("mean_abs_percentile_shift"),
                    "p90_abs_percentile_shift": shift.get("p90_abs_percentile_shift"),
                }
            )

        top_priority_units = admin_priority_lookup.get(city, [])
        if top_priority_units:
            claims.append(
                {
                    "claim_id": "top_priority_admin_unit__{city}".format(
                        city=city.lower().replace(" ", "_")
                    ),
                    "topic": "district_targeting",
                    "city": city,
                    "scenario": primary_scenario,
                    "admin_unit": top_priority_units[0]["admin_unit"],
                    "district_priority_class": top_priority_units[0]["district_priority_class"],
                    "district_priority_metric_count": top_priority_units[0]["district_priority_metric_count"],
                    "population_weighted_mean_score": top_priority_units[0]["population_weighted_mean_score"],
                    "citywide_q90_population_share": top_priority_units[0]["citywide_q90_population_share"],
                    "hotspot_population_share": top_priority_units[0]["hotspot_population_share"],
                    "hotspot_dominant_dimension": top_priority_units[0]["hotspot_dominant_dimension"],
                }
            )

        high_priority_population_share = admin_priority_population_share_lookup.get(city, {}).get("high_priority")
        if high_priority_population_share is not None:
            claims.append(
                {
                    "claim_id": "high_priority_district_population_share__{city}".format(
                        city=city.lower().replace(" ", "_")
                    ),
                    "topic": "district_targeting",
                    "city": city,
                    "scenario": primary_scenario,
                    "district_priority_class": "high_priority",
                    "population_share": high_priority_population_share,
                }
            )

    claims.append(
        {
            "claim_id": "grid_size_order_stability",
            "topic": "sensitivity",
            "primary_scenario": primary_scenario,
            "sensitivity_scenario": sensitivity_scenario,
            "order_stability": sensitivity_order_stability,
        }
    )

    payload = {
        "study_id": config.get("study_id", config_path.stem),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_scenario": primary_scenario,
        "sensitivity_scenario": sensitivity_scenario,
        "top_fraction": top_fraction,
        "source_paths": {key: str(_resolve_config_path(value, config_path)) for key, value in paths.items()},
        "cross_city": {
            "mean_score": _ranking_summary(city_score_summary, "mean"),
            "population_weighted_mean_score": _ranking_summary(
                population_exposure,
                "population_weighted_mean_score",
            ),
            "population_share_at_or_above_q90": _ranking_summary(
                population_exposure,
                "population_share_at_or_above_q90",
            ),
            "score_gini_population_weighted": _ranking_summary(
                inequality_summary,
                "score_gini_population_weighted",
            ),
        },
        "cities": city_results,
        "sensitivity": {
            "order_stability": sensitivity_order_stability,
            "score_rankings": {
                str(scenario): _ranking_summary(subset, "mean")
                for scenario, subset in grid_size_score_summary.groupby("scenario")
            },
            "exposure_rankings": {
                str(scenario): _ranking_summary(subset, "population_share_at_or_above_q90")
                for scenario, subset in grid_size_population_exposure.groupby("scenario")
            },
            "inequality_rankings": {
                str(scenario): _ranking_summary(subset, "score_gini_population_weighted")
                for scenario, subset in grid_size_inequality_summary.groupby("scenario")
            },
        },
        "claims": claims,
    }
    write_json(_to_builtin(payload), Path(output_path))


def stage_figure_set(config_path: Path, output_dir: Path, manifest_output: Path) -> None:
    config_path = Path(config_path)
    config = read_yaml(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    staged_items = []
    for index, item in enumerate(config.get("items", []), start=1):
        source_path = _resolve_config_path(item["source"], config_path)
        if not source_path.exists():
            raise FileNotFoundError("Figure source not found: {path}".format(path=source_path))

        destination_name = item.get("filename") or source_path.name
        destination_path = output_dir / destination_name
        ensure_parent_dir(destination_path)
        shutil.copy2(source_path, destination_path)
        staged_items.append(
            {
                "order": index,
                "id": item.get("id", destination_path.stem),
                "title": item.get("title"),
                "role": item.get("role"),
                "source": str(source_path),
                "destination": str(destination_path),
                "filename": destination_name,
            }
        )

    manifest = {
        "figure_set_id": config.get("figure_set_id", config_path.stem),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "items": staged_items,
    }
    write_json(manifest, Path(manifest_output))
