from pathlib import Path
from typing import Iterable
from typing import Optional

import numpy as np
import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import write_table


DEFAULT_FEATURE_COLUMNS = [
    "road_distance_m",
    "school_distance_m",
    "clinic_distance_m",
    "amenity_count_1km",
    "population_per_service",
    "building_coverage_ratio",
    "open_space_share",
    "intersection_density_km2",
]


def _maybe_add_scenario(frame: pd.DataFrame, scenario: Optional[str]) -> pd.DataFrame:
    if scenario:
        frame = frame.copy()
        frame.insert(0, "scenario", scenario)
    return frame


def _apply_optional_filter(
    frame: pd.DataFrame,
    filter_col: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> pd.DataFrame:
    if not filter_col:
        return frame
    if filter_col not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=filter_col))
    subset = frame.loc[frame[filter_col].astype(str) == str(filter_value)].copy()
    return subset


def _prepare_comparison_frame(
    frame: pd.DataFrame,
    group_col: str,
    left_score_name: str,
    right_score_name: str,
    left_label: str,
    right_label: str,
    top_fraction: float,
) -> pd.DataFrame:
    if top_fraction <= 0 or top_fraction > 1:
        raise ValueError("top_fraction must be in the interval (0, 1].")

    working = frame.copy()
    working[left_score_name] = pd.to_numeric(working[left_score_name], errors="coerce")
    working[right_score_name] = pd.to_numeric(working[right_score_name], errors="coerce")

    shift_name = "{right}_minus_{left}".format(right=right_label, left=left_label)
    left_rank_name = "{label}_rank_desc".format(label=left_label)
    right_rank_name = "{label}_rank_desc".format(label=right_label)
    left_percentile_name = "{label}_percentile".format(label=left_label)
    right_percentile_name = "{label}_percentile".format(label=right_label)
    rank_shift_name = "{right}_rank_minus_{left}_rank".format(right=right_label, left=left_label)
    percentile_shift_name = "{right}_percentile_minus_{left}_percentile".format(
        right=right_label,
        left=left_label,
    )

    working[shift_name] = working[right_score_name] - working[left_score_name]
    working[left_rank_name] = working.groupby(group_col)[left_score_name].rank(
        ascending=False,
        method="average",
        na_option="keep",
    )
    working[right_rank_name] = working.groupby(group_col)[right_score_name].rank(
        ascending=False,
        method="average",
        na_option="keep",
    )

    left_counts = working.groupby(group_col)[left_score_name].transform("count").astype(float)
    right_counts = working.groupby(group_col)[right_score_name].transform("count").astype(float)
    working[left_percentile_name] = np.where(
        working[left_rank_name].isna(),
        np.nan,
        np.where(
            left_counts <= 1,
            1.0,
            1.0 - ((working[left_rank_name] - 1.0) / (left_counts - 1.0)),
        ),
    )
    working[right_percentile_name] = np.where(
        working[right_rank_name].isna(),
        np.nan,
        np.where(
            right_counts <= 1,
            1.0,
            1.0 - ((working[right_rank_name] - 1.0) / (right_counts - 1.0)),
        ),
    )

    working[rank_shift_name] = working[right_rank_name] - working[left_rank_name]
    working["abs_rank_shift"] = working[rank_shift_name].abs()
    working[percentile_shift_name] = working[right_percentile_name] - working[left_percentile_name]
    working["abs_percentile_shift"] = working[percentile_shift_name].abs()
    working["abs_score_shift"] = working[shift_name].abs()

    working["left_top_fraction_flag"] = False
    working["right_top_fraction_flag"] = False

    for _, subset in working.groupby(group_col):
        left_valid = subset[left_score_name].dropna()
        right_valid = subset[right_score_name].dropna()

        if not left_valid.empty:
            left_cutoff = max(1, int(len(left_valid) * top_fraction))
            working.loc[left_valid.nlargest(left_cutoff).index, "left_top_fraction_flag"] = True

        if not right_valid.empty:
            right_cutoff = max(1, int(len(right_valid) * top_fraction))
            working.loc[right_valid.nlargest(right_cutoff).index, "right_top_fraction_flag"] = True

    working["top_fraction_status"] = np.select(
        [
            working["left_top_fraction_flag"] & working["right_top_fraction_flag"],
            (~working["left_top_fraction_flag"]) & working["right_top_fraction_flag"],
            working["left_top_fraction_flag"] & (~working["right_top_fraction_flag"]),
        ],
        [
            "stable_top_fraction",
            "entered_top_fraction",
            "exited_top_fraction",
        ],
        default="stable_non_top_fraction",
    )
    return working


def summarize_index(
    input_path: Path,
    group_col: str,
    score_col: str,
    score_summary_output: Path,
    feature_summary_output: Optional[Path] = None,
    feature_columns: Optional[Iterable[str]] = None,
    scenario: Optional[str] = None,
    filter_col: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    frame = _apply_optional_filter(frame, filter_col=filter_col, filter_value=filter_value)
    if group_col not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=group_col))
    if score_col not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=score_col))

    score_summary = (
        frame.groupby(group_col)[score_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    score_summary = _maybe_add_scenario(score_summary, scenario)
    write_table(score_summary, Path(score_summary_output))

    if feature_summary_output:
        selected = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
        available = [column for column in selected if column in frame.columns]
        if not available:
            raise ValueError("No requested feature columns were found in the input table.")

        feature_summary = frame.groupby(group_col)[available].median().reset_index()
        feature_summary = _maybe_add_scenario(feature_summary, scenario)
        write_table(feature_summary, Path(feature_summary_output))


def summarize_category_shares(
    input_path: Path,
    group_col: str,
    category_col: str,
    output_path: Path,
    scenario: Optional[str] = None,
    filter_col: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    frame = _apply_optional_filter(frame, filter_col=filter_col, filter_value=filter_value)
    if group_col not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=group_col))
    if category_col not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=category_col))

    counts = (
        frame.groupby([group_col, category_col])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(group_col)["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    counts = _maybe_add_scenario(counts, scenario)
    write_table(counts, Path(output_path))


def _summarize_alignment_frame(
    frame: pd.DataFrame,
    group_col: str,
    score_a_col: str,
    score_b_col: str,
    top_fraction: float,
) -> pd.DataFrame:
    required = [group_col, score_a_col, score_b_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    records = []
    for label, subset in frame.groupby(group_col):
        ranked = subset[[score_a_col, score_b_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if ranked.empty:
            continue

        cutoff = max(1, int(len(ranked) * top_fraction))
        score_a_top = set(ranked.nlargest(cutoff, score_a_col).index)
        score_b_top = set(ranked.nlargest(cutoff, score_b_col).index)
        overlap = len(score_a_top & score_b_top) / cutoff

        records.append(
            {
                group_col: label,
                "n_rows": int(len(ranked)),
                "pearson_corr": float(ranked[score_a_col].corr(ranked[score_b_col], method="pearson")),
                "spearman_corr": float(ranked[score_a_col].corr(ranked[score_b_col], method="spearman")),
                "top_fraction": float(top_fraction),
                "top_overlap_share": float(overlap),
            }
        )

    return pd.DataFrame(records)


def _pooled_std(left: pd.Series, right: pd.Series) -> float:
    left_std = float(left.std(ddof=0))
    right_std = float(right.std(ddof=0))
    pooled = float(np.sqrt((left_std ** 2 + right_std ** 2) / 2.0))
    return pooled


def summarize_binary_contrast(
    input_path: Path,
    group_col: str,
    binary_col: str,
    target_value: str,
    value_columns: Iterable[str],
    output_path: Path,
    scenario: Optional[str] = None,
    reference_value: Optional[str] = None,
    filter_col: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    frame = _apply_optional_filter(frame, filter_col=filter_col, filter_value=filter_value)

    required = [group_col, binary_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    value_columns = [column for column in value_columns if column in frame.columns]
    if not value_columns:
        raise ValueError("No requested value columns were found in the input table.")

    records = []
    for label, subset in frame.groupby(group_col):
        if reference_value is None:
            target_mask = subset[binary_col].astype(str) == str(target_value)
            reference_mask = subset[binary_col].astype(str) != str(target_value)
            reference_label = "other"
        else:
            target_mask = subset[binary_col].astype(str) == str(target_value)
            reference_mask = subset[binary_col].astype(str) == str(reference_value)
            reference_label = str(reference_value)

        target_count = int(target_mask.sum())
        reference_count = int(reference_mask.sum())
        if target_count == 0 or reference_count == 0:
            continue

        for column in value_columns:
            target = pd.to_numeric(subset.loc[target_mask, column], errors="coerce").dropna()
            reference = pd.to_numeric(subset.loc[reference_mask, column], errors="coerce").dropna()
            if target.empty or reference.empty:
                continue

            target_mean = float(target.mean())
            reference_mean = float(reference.mean())
            target_median = float(target.median())
            reference_median = float(reference.median())
            pooled = _pooled_std(target, reference)
            standardized_mean_diff = (
                float((target_mean - reference_mean) / pooled)
                if pooled > 0
                else 0.0
            )

            records.append(
                {
                    group_col: label,
                    "binary_col": binary_col,
                    "target_value": str(target_value),
                    "reference_value": reference_label,
                    "feature": column,
                    "target_n": int(len(target)),
                    "reference_n": int(len(reference)),
                    "target_share": float(target_count / max(len(subset), 1)),
                    "target_mean": target_mean,
                    "reference_mean": reference_mean,
                    "mean_difference": float(target_mean - reference_mean),
                    "target_median": target_median,
                    "reference_median": reference_median,
                    "median_difference": float(target_median - reference_median),
                    "standardized_mean_diff": standardized_mean_diff,
                    "abs_standardized_mean_diff": float(abs(standardized_mean_diff)),
                    "contrast_direction": "target_higher"
                    if standardized_mean_diff > 0
                    else "target_lower"
                    if standardized_mean_diff < 0
                    else "no_difference",
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No contrast rows could be computed from the provided inputs.")

    result["contrast_rank"] = (
        result.groupby(group_col)["abs_standardized_mean_diff"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    result = result.sort_values([group_col, "contrast_rank", "feature"]).reset_index(drop=True)
    result = _maybe_add_scenario(result, scenario)
    write_table(result, Path(output_path))


def summarize_category_feature_profiles(
    input_path: Path,
    group_col: str,
    category_col: str,
    value_columns: Iterable[str],
    output_path: Path,
    scenario: Optional[str] = None,
    filter_col: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    frame = _apply_optional_filter(frame, filter_col=filter_col, filter_value=filter_value)

    required = [group_col, category_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    value_columns = [column for column in value_columns if column in frame.columns]
    if not value_columns:
        raise ValueError("No requested value columns were found in the input table.")

    records = []
    for label, subset in frame.groupby(group_col):
        subset_size = int(len(subset))
        if subset_size == 0:
            continue

        category_counts = subset[category_col].fillna("missing").astype(str).value_counts()
        for column in value_columns:
            full_series = pd.to_numeric(subset[column], errors="coerce").dropna()
            if full_series.empty:
                continue

            group_mean = float(full_series.mean())
            group_median = float(full_series.median())
            group_std = float(full_series.std(ddof=0))

            for category_value, category_subset in subset.groupby(category_col, dropna=False):
                category_series = pd.to_numeric(category_subset[column], errors="coerce").dropna()
                if category_series.empty:
                    continue

                category_name = "missing" if pd.isna(category_value) else str(category_value)
                category_mean = float(category_series.mean())
                category_median = float(category_series.median())
                standardized_profile_diff = (
                    float((category_mean - group_mean) / group_std)
                    if group_std > 0
                    else 0.0
                )

                records.append(
                    {
                        group_col: label,
                        category_col: category_name,
                        "feature": column,
                        "group_n": subset_size,
                        "category_n": int(category_counts.get(category_name, len(category_subset))),
                        "category_share": float(category_counts.get(category_name, len(category_subset)) / subset_size),
                        "category_mean": category_mean,
                        "category_median": category_median,
                        "group_mean": group_mean,
                        "group_median": group_median,
                        "mean_difference_from_group": float(category_mean - group_mean),
                        "median_difference_from_group": float(category_median - group_median),
                        "standardized_profile_diff": standardized_profile_diff,
                        "abs_standardized_profile_diff": float(abs(standardized_profile_diff)),
                        "profile_direction": "above_group_mean"
                        if standardized_profile_diff > 0
                        else "below_group_mean"
                        if standardized_profile_diff < 0
                        else "at_group_mean",
                    }
                )

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No category feature profiles could be computed from the provided inputs.")

    result["profile_rank"] = (
        result.groupby([group_col, category_col])["abs_standardized_profile_diff"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    result = result.sort_values([group_col, category_col, "profile_rank", "feature"]).reset_index(drop=True)
    result = _maybe_add_scenario(result, scenario)
    write_table(result, Path(output_path))


def summarize_pca_alignment(
    input_path: Path,
    group_col: str,
    composite_col: str,
    pca_col: str,
    output_path: Path,
    top_fraction: float = 0.1,
    scenario: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    result = _summarize_alignment_frame(
        frame=frame,
        group_col=group_col,
        score_a_col=composite_col,
        score_b_col=pca_col,
        top_fraction=top_fraction,
    )
    result = _maybe_add_scenario(result, scenario)
    write_table(result, Path(output_path))


def compare_index_scores(
    left_input_path: Path,
    right_input_path: Path,
    join_columns: Iterable[str],
    group_col: str,
    left_score_col: str,
    right_score_col: str,
    output_path: Path,
    merged_output_path: Optional[Path] = None,
    left_label: str = "left",
    right_label: str = "right",
    top_fraction: float = 0.1,
    scenario: Optional[str] = None,
) -> None:
    left = read_table(Path(left_input_path))
    right = read_table(Path(right_input_path))

    join_columns = list(join_columns)
    for column in join_columns:
        if column not in left.columns:
            raise ValueError("Join column '{column}' not found in left input.".format(column=column))
        if column not in right.columns:
            raise ValueError("Join column '{column}' not found in right input.".format(column=column))

    for column_name, frame_name, frame in [
        (left_score_col, "left", left),
        (right_score_col, "right", right),
    ]:
        if column_name not in frame.columns:
            raise ValueError(
                "Score column '{column}' not found in {frame_name} input.".format(
                    column=column_name,
                    frame_name=frame_name,
                )
            )

    left_score_name = "{label}_score".format(label=left_label)
    right_score_name = "{label}_score".format(label=right_label)

    left_renamed = left.copy().rename(columns={left_score_col: left_score_name})
    right_subset = right[join_columns + [right_score_col]].copy().rename(
        columns={right_score_col: right_score_name}
    )

    merged = left_renamed.merge(right_subset, on=join_columns, how="inner")
    merged = _prepare_comparison_frame(
        frame=merged,
        group_col=group_col,
        left_score_name=left_score_name,
        right_score_name=right_score_name,
        left_label=left_label,
        right_label=right_label,
        top_fraction=top_fraction,
    )

    result = _summarize_alignment_frame(
        frame=merged,
        group_col=group_col,
        score_a_col=left_score_name,
        score_b_col=right_score_name,
        top_fraction=top_fraction,
    )
    result.insert(1, "left_label", left_label)
    result.insert(2, "right_label", right_label)
    shift_summary = summarize_comparison_shift_frame(
        frame=merged,
        group_col=group_col,
        left_label=left_label,
        right_label=right_label,
        top_fraction=top_fraction,
    )
    if not shift_summary.empty:
        shift_summary = shift_summary.drop(columns=["n_rows"], errors="ignore")
        result = result.merge(
            shift_summary,
            on=[group_col, "left_label", "right_label", "top_fraction"],
            how="left",
        )
    result = _maybe_add_scenario(result, scenario)
    write_table(result, Path(output_path))

    if merged_output_path:
        write_table(merged, Path(merged_output_path))


def summarize_comparison_shift_frame(
    frame: pd.DataFrame,
    group_col: str,
    left_label: str,
    right_label: str,
    top_fraction: float,
) -> pd.DataFrame:
    left_score_name = "{label}_score".format(label=left_label)
    right_score_name = "{label}_score".format(label=right_label)
    shift_name = "{right}_minus_{left}".format(right=right_label, left=left_label)
    rank_shift_name = "{right}_rank_minus_{left}_rank".format(right=right_label, left=left_label)
    percentile_shift_name = "{right}_percentile_minus_{left}_percentile".format(
        right=right_label,
        left=left_label,
    )
    required = [
        group_col,
        left_score_name,
        right_score_name,
        shift_name,
        rank_shift_name,
        percentile_shift_name,
        "abs_score_shift",
        "abs_rank_shift",
        "abs_percentile_shift",
        "top_fraction_status",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing comparison-shift columns: {columns}".format(columns=", ".join(missing)))

    records = []
    for label, subset in frame.groupby(group_col):
        valid = subset.loc[
            subset[left_score_name].notna()
            & subset[right_score_name].notna()
            & subset[shift_name].notna()
        ].copy()
        if valid.empty:
            continue

        status_share = valid["top_fraction_status"].value_counts(normalize=True)
        records.append(
            {
                group_col: label,
                "left_label": left_label,
                "right_label": right_label,
                "top_fraction": float(top_fraction),
                "n_rows": int(len(valid)),
                "mean_score_shift": float(valid[shift_name].mean()),
                "median_score_shift": float(valid[shift_name].median()),
                "mean_abs_score_shift": float(valid["abs_score_shift"].mean()),
                "median_abs_score_shift": float(valid["abs_score_shift"].median()),
                "p90_abs_score_shift": float(valid["abs_score_shift"].quantile(0.90)),
                "mean_abs_rank_shift": float(valid["abs_rank_shift"].mean()),
                "median_abs_rank_shift": float(valid["abs_rank_shift"].median()),
                "p90_abs_rank_shift": float(valid["abs_rank_shift"].quantile(0.90)),
                "mean_abs_percentile_shift": float(valid["abs_percentile_shift"].mean()),
                "median_abs_percentile_shift": float(valid["abs_percentile_shift"].median()),
                "p90_abs_percentile_shift": float(valid["abs_percentile_shift"].quantile(0.90)),
                "stable_top_fraction_share": float(status_share.get("stable_top_fraction", 0.0)),
                "entered_top_fraction_share": float(status_share.get("entered_top_fraction", 0.0)),
                "exited_top_fraction_share": float(status_share.get("exited_top_fraction", 0.0)),
                "top_fraction_reclassification_share": float(
                    status_share.get("entered_top_fraction", 0.0)
                    + status_share.get("exited_top_fraction", 0.0)
                ),
            }
        )

    return pd.DataFrame(records)


def summarize_comparison_shift(
    input_path: Path,
    group_col: str,
    left_label: str,
    right_label: str,
    output_path: Path,
    top_fraction: float = 0.1,
    scenario: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    summary = summarize_comparison_shift_frame(
        frame=frame,
        group_col=group_col,
        left_label=left_label,
        right_label=right_label,
        top_fraction=top_fraction,
    )
    summary = _maybe_add_scenario(summary, scenario)
    write_table(summary, Path(output_path))


def export_top_cells(
    input_path: Path,
    group_col: str,
    score_col: str,
    output_path: Path,
    top_n: int = 25,
    columns: Optional[Iterable[str]] = None,
    scenario: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    required = [group_col, score_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    default_columns = [
        "cell_id",
        group_col,
        "country_iso",
        "lon",
        "lat",
        "population",
        score_col,
        "road_distance_m",
        "school_distance_m",
        "clinic_distance_m",
        "amenity_count_1km",
        "population_per_service",
        "building_coverage_ratio",
        "open_space_share",
        "intersection_density_km2",
    ]
    selected = list(columns or default_columns)
    available = [column for column in selected if column in frame.columns]

    exports = []
    for label, subset in frame.groupby(group_col):
        top = subset.sort_values(score_col, ascending=False).head(top_n)
        exports.append(top[available].copy())

    result = pd.concat(exports, ignore_index=True)
    result = _maybe_add_scenario(result, scenario)
    write_table(result, Path(output_path))


def summarize_population_exposure(
    input_path: Path,
    group_col: str,
    score_col: str,
    population_col: str,
    output_path: Path,
    quantiles: Optional[Iterable[float]] = None,
    scenario: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    required = [group_col, score_col, population_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    quantiles = list(quantiles or [0.8, 0.9])
    records = []

    for label, subset in frame.groupby(group_col):
        score = pd.to_numeric(subset[score_col], errors="coerce")
        population = pd.to_numeric(subset[population_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        total_population = float(population.sum())

        row = {
            group_col: label,
            "n_cells": int(len(subset)),
            "total_population": total_population,
            "population_weighted_mean_score": float((score * population).sum() / total_population)
            if total_population > 0
            else None,
        }

        for quantile in quantiles:
            threshold = float(score.quantile(quantile))
            exposed = float(population.loc[score >= threshold].sum())
            suffix = int(round(quantile * 100))
            row["score_q{suffix}".format(suffix=suffix)] = threshold
            row["population_share_at_or_above_q{suffix}".format(suffix=suffix)] = (
                exposed / total_population if total_population > 0 else None
            )

        records.append(row)

    result = pd.DataFrame(records)
    result = _maybe_add_scenario(result, scenario)
    write_table(result, Path(output_path))


def summarize_weighted_categories(
    input_path: Path,
    group_col: str,
    category_col: str,
    weight_col: str,
    output_path: Path,
    scenario: Optional[str] = None,
    filter_col: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    frame = _apply_optional_filter(frame, filter_col=filter_col, filter_value=filter_value)
    required = [group_col, category_col, weight_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    working = frame.copy()
    working[weight_col] = pd.to_numeric(working[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    grouped = (
        working.groupby([group_col, category_col])[weight_col]
        .sum()
        .rename("weight_sum")
        .reset_index()
    )
    totals = grouped.groupby(group_col)["weight_sum"].transform("sum")
    grouped["weight_share"] = grouped["weight_sum"] / totals
    grouped = _maybe_add_scenario(grouped, scenario)
    write_table(grouped, Path(output_path))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cumulative = np.cumsum(weights)
    if len(cumulative) == 0 or cumulative[-1] == 0:
        return float("nan")
    threshold = quantile * cumulative[-1]
    index = np.searchsorted(cumulative, threshold, side="left")
    index = min(index, len(values) - 1)
    return float(values[index])


def _weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    total_weight = float(weights.sum())
    total_value = float(np.sum(values * weights))
    if total_weight == 0 or total_value == 0:
        return float("nan")

    cumw = np.cumsum(weights)
    cumxw = np.cumsum(values * weights)
    lorenz_x = np.insert(cumw / total_weight, 0, 0.0)
    lorenz_y = np.insert(cumxw / total_value, 0, 0.0)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = trapezoid(lorenz_y, lorenz_x)
    return float(1.0 - 2.0 * area)


def summarize_inequality(
    input_path: Path,
    group_col: str,
    score_col: str,
    population_col: str,
    output_path: Path,
    scenario: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    required = [group_col, score_col, population_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    records = []
    for label, subset in frame.groupby(group_col):
        score = pd.to_numeric(subset[score_col], errors="coerce")
        population = pd.to_numeric(subset[population_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        valid = score.notna()
        score_values = score.loc[valid].to_numpy(dtype=float)
        pop_values = population.loc[valid].to_numpy(dtype=float)
        if len(score_values) == 0:
            continue

        p10 = float(np.quantile(score_values, 0.10))
        p50 = float(np.quantile(score_values, 0.50))
        p90 = float(np.quantile(score_values, 0.90))
        records.append(
            {
                group_col: label,
                "n_cells": int(len(score_values)),
                "population_total": float(pop_values.sum()),
                "score_mean": float(np.mean(score_values)),
                "score_gini_unweighted": _weighted_gini(score_values, np.ones_like(score_values)),
                "score_gini_population_weighted": _weighted_gini(score_values, pop_values),
                "score_p10": p10,
                "score_p50": p50,
                "score_p90": p90,
                "score_p90_p10_ratio": float(p90 / max(p10, 1e-9)),
                "score_wp10": _weighted_quantile(score_values, pop_values, 0.10),
                "score_wp50": _weighted_quantile(score_values, pop_values, 0.50),
                "score_wp90": _weighted_quantile(score_values, pop_values, 0.90),
            }
        )

    result = pd.DataFrame(records)
    result = _maybe_add_scenario(result, scenario)
    write_table(result, Path(output_path))
