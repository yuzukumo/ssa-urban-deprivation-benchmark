from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.io_utils import write_table


DEFAULT_MARGIN_THRESHOLDS = [0.25, 0.75]
DEFAULT_PRIORITY_TOP_FRACTION = 0.1


def _clean_dimension_name(column_name: str) -> str:
    return column_name.replace("__score", "")


def _resolve_margin_thresholds(margin_thresholds: Optional[List[float]]) -> List[float]:
    if margin_thresholds is None:
        return DEFAULT_MARGIN_THRESHOLDS
    if len(margin_thresholds) != 2:
        raise ValueError("Expected exactly two margin thresholds: [mixed_upper, strong_lower].")

    thresholds = [float(value) for value in margin_thresholds]
    if thresholds[0] < 0 or thresholds[1] < thresholds[0]:
        raise ValueError("Margin thresholds must satisfy 0 <= mixed_upper <= strong_lower.")
    return thresholds


def _classify_margin_strength(margin: float, mixed_upper: float, strong_lower: float) -> str:
    if pd.isna(margin):
        return "missing"
    if margin < mixed_upper:
        return "mixed"
    if margin < strong_lower:
        return "moderate"
    return "strong"


def _priority_quadrant_label(absolute_flag: bool, relative_flag: bool) -> str:
    if absolute_flag and relative_flag:
        return "joint_priority"
    if absolute_flag:
        return "absolute_only"
    if relative_flag:
        return "relative_only"
    return "lower_priority"


def annotate_dominant_dimension(
    input_path: Path,
    dimension_cols: List[str],
    output_path: Path,
    metadata_path: Optional[Path] = None,
    margin_thresholds: Optional[List[float]] = None,
) -> None:
    frame = read_table(Path(input_path))
    missing = [column for column in dimension_cols if column not in frame.columns]
    if missing:
        raise ValueError("Missing dimension columns: {columns}".format(columns=", ".join(missing)))

    mixed_upper, strong_lower = _resolve_margin_thresholds(margin_thresholds)
    scores = frame[dimension_cols].apply(pd.to_numeric, errors="coerce")
    values = scores.to_numpy(dtype=float)
    valid_rows = np.isfinite(values).any(axis=1)

    top_indices = np.full(len(values), -1, dtype=int)
    dominant_scores = np.full(len(values), np.nan, dtype=float)
    second_scores = np.full(len(values), np.nan, dtype=float)

    if valid_rows.any():
        safe_values = np.where(np.isfinite(values), values, -np.inf)
        valid_row_indices = np.where(valid_rows)[0]
        valid_values = safe_values[valid_rows]
        valid_top_indices = np.argmax(valid_values, axis=1)

        top_indices[valid_row_indices] = valid_top_indices
        dominant_scores[valid_row_indices] = valid_values[
            np.arange(len(valid_values)),
            valid_top_indices,
        ]

        if values.shape[1] >= 2:
            sorted_values = np.sort(valid_values, axis=1)
            second_scores[valid_row_indices] = sorted_values[:, -2]
            second_scores[np.isneginf(second_scores)] = np.nan
        else:
            second_scores[valid_row_indices] = 0.0

    dominant_names = [
        "missing" if index < 0 else _clean_dimension_name(dimension_cols[index])
        for index in top_indices
    ]
    margin = dominant_scores - second_scores

    result = frame.copy()
    result["dominant_dimension"] = dominant_names
    result["dominant_dimension_score"] = dominant_scores
    result["dominant_dimension_margin"] = margin
    result["dominant_dimension_strength"] = [
        _classify_margin_strength(value, mixed_upper=mixed_upper, strong_lower=strong_lower)
        for value in margin
    ]

    write_table(result, Path(output_path))

    if metadata_path:
        metadata: Dict = {
            "input_path": str(Path(input_path)),
            "output_path": str(Path(output_path)),
            "dimension_cols": dimension_cols,
            "dimension_names": [_clean_dimension_name(column) for column in dimension_cols],
            "margin_thresholds": {
                "mixed_upper": mixed_upper,
                "strong_lower": strong_lower,
            },
        }
        write_json(metadata, Path(metadata_path))


def annotate_priority_quadrants(
    input_path: Path,
    absolute_score_col: str,
    relative_score_col: str,
    output_path: Path,
    group_col: str,
    metadata_path: Optional[Path] = None,
    absolute_top_fraction: float = DEFAULT_PRIORITY_TOP_FRACTION,
    relative_top_fraction: float = DEFAULT_PRIORITY_TOP_FRACTION,
) -> None:
    frame = read_table(Path(input_path))
    required = [absolute_score_col, relative_score_col, group_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: {columns}".format(columns=", ".join(missing)))

    if not (0.0 < absolute_top_fraction < 1.0):
        raise ValueError("absolute_top_fraction must be between 0 and 1.")
    if not (0.0 < relative_top_fraction < 1.0):
        raise ValueError("relative_top_fraction must be between 0 and 1.")

    result = frame.copy()
    absolute_scores = pd.to_numeric(result[absolute_score_col], errors="coerce")
    relative_scores = pd.to_numeric(result[relative_score_col], errors="coerce")

    absolute_quantile = 1.0 - float(absolute_top_fraction)
    relative_quantile = 1.0 - float(relative_top_fraction)

    absolute_threshold = float(absolute_scores.quantile(absolute_quantile))
    relative_thresholds = (
        result.assign(_relative_score=relative_scores)
        .groupby(group_col)["_relative_score"]
        .quantile(relative_quantile)
        .to_dict()
    )

    result["absolute_priority_threshold"] = absolute_threshold
    result["relative_priority_threshold"] = result[group_col].map(relative_thresholds)
    result["absolute_priority_flag"] = absolute_scores >= absolute_threshold
    result["relative_priority_flag"] = relative_scores >= result["relative_priority_threshold"]
    result["priority_quadrant"] = [
        _priority_quadrant_label(bool(absolute_flag), bool(relative_flag))
        for absolute_flag, relative_flag in zip(
            result["absolute_priority_flag"].fillna(False),
            result["relative_priority_flag"].fillna(False),
        )
    ]

    write_table(result, Path(output_path))

    if metadata_path:
        metadata: Dict = {
            "input_path": str(Path(input_path)),
            "output_path": str(Path(output_path)),
            "absolute_score_col": absolute_score_col,
            "relative_score_col": relative_score_col,
            "group_col": group_col,
            "absolute_top_fraction": float(absolute_top_fraction),
            "relative_top_fraction": float(relative_top_fraction),
            "absolute_priority_threshold": absolute_threshold,
            "relative_priority_thresholds_by_group": {
                str(key): float(value)
                for key, value in relative_thresholds.items()
            },
        }
        write_json(metadata, Path(metadata_path))
