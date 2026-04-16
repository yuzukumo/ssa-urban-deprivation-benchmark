from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import read_yaml
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.io_utils import write_table


VALID_OPERATORS = {">=", ">", "<=", "<", "==", "!="}


def _evaluate_condition(frame: pd.DataFrame, condition: Dict) -> pd.Series:
    column = condition["column"]
    operator = condition["op"]
    value = condition["value"]

    if column not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=column))
    if operator not in VALID_OPERATORS:
        raise ValueError("Unsupported operator '{operator}'.".format(operator=operator))

    series = pd.to_numeric(frame[column], errors="coerce")
    if operator == ">=":
        return series >= value
    if operator == ">":
        return series > value
    if operator == "<=":
        return series <= value
    if operator == "<":
        return series < value
    if operator == "==":
        return series == value
    return series != value


def _combine_masks(masks: List[pd.Series], mode: str) -> pd.Series:
    if not masks:
        raise ValueError("At least one analysis-mask condition is required.")
    if mode == "all":
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined &= mask
        return combined.fillna(False)
    if mode == "any":
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined |= mask
        return combined.fillna(False)
    raise ValueError("analysis_mask.mode must be 'all' or 'any'.")


def load_analysis_mask(config_path: Path) -> Dict:
    payload = read_yaml(Path(config_path))
    if "analysis_mask" in payload:
        return payload["analysis_mask"]
    return payload


def apply_analysis_mask(input_path: Path, config_path: Path) -> Tuple[pd.DataFrame, Dict]:
    frame = read_table(Path(input_path))
    config = load_analysis_mask(Path(config_path))
    mode = config.get("mode", "any")
    conditions = config.get("conditions", [])

    masks = [_evaluate_condition(frame, condition) for condition in conditions]
    keep_mask = _combine_masks(masks, mode=mode)

    filtered = frame.loc[keep_mask].copy()
    metadata = {
        "input_path": str(Path(input_path)),
        "config_path": str(Path(config_path)),
        "rows_before": int(len(frame)),
        "rows_after": int(len(filtered)),
        "rows_removed": int(len(frame) - len(filtered)),
        "share_kept": float(len(filtered) / len(frame)) if len(frame) else 0.0,
        "mode": mode,
        "conditions": conditions,
    }

    if "city" in frame.columns:
        before = frame.groupby("city").size().rename("rows_before")
        after = filtered.groupby("city").size().rename("rows_after")
        city_summary = before.to_frame().join(after, how="left").fillna(0)
        city_summary["rows_after"] = city_summary["rows_after"].astype(int)
        city_summary["rows_removed"] = city_summary["rows_before"] - city_summary["rows_after"]
        city_summary["share_kept"] = city_summary["rows_after"] / city_summary["rows_before"]
        metadata["by_city"] = city_summary.reset_index().to_dict(orient="records")

    return filtered, metadata


def run_analysis_mask(
    input_path: Path,
    config_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> None:
    filtered, metadata = apply_analysis_mask(Path(input_path), Path(config_path))
    write_table(filtered, Path(output_path))
    write_json(metadata, Path(metadata_path))
