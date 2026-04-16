from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import read_table


def _example_values(series: pd.Series, limit: int = 3) -> List[Any]:
    values = series.dropna().head(limit).tolist()
    cleaned = []
    for value in values:
        if hasattr(value, "isoformat"):
            cleaned.append(value.isoformat())
        else:
            cleaned.append(value)
    return cleaned


def build_profile(path: Path) -> Dict[str, Any]:
    frame = read_table(Path(path))

    column_summaries = []
    for column in frame.columns:
        series = frame[column]
        column_summaries.append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "nulls": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
                "examples": _example_values(series),
            }
        )

    profile = {
        "path": str(Path(path)),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_summaries": column_summaries,
        "preview": frame.head(5).to_dict(orient="records"),
    }

    if hasattr(frame, "geometry"):
        profile["is_geospatial"] = True
        profile["crs"] = str(frame.crs)
        profile["bounds"] = list(frame.total_bounds)
    else:
        profile["is_geospatial"] = False

    return profile
