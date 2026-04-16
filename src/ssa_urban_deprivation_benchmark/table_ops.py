from pathlib import Path
from typing import Iterable

import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import write_table


def concat_tables(inputs: Iterable[Path], output_path: Path) -> None:
    frames = [read_table(Path(path)) for path in inputs]
    if not frames:
        raise ValueError("At least one input table is required.")

    first = frames[0]
    if hasattr(first, "geometry"):
        import geopandas as gpd

        combined = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry=first.geometry.name,
            crs=first.crs,
        )
        write_table(combined, Path(output_path))
        return

    combined = pd.concat(frames, ignore_index=True)
    write_table(combined, Path(output_path))


def filter_table(
    input_path: Path,
    output_path: Path,
    filter_col: str,
    filter_value: str,
) -> None:
    frame = read_table(Path(input_path))
    if filter_col not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=filter_col))

    subset = frame.loc[frame[filter_col].astype(str) == str(filter_value)].copy()
    write_table(subset, Path(output_path))


def add_composite_column(
    input_path: Path,
    output_path: Path,
    source_columns: Iterable[str],
    output_column: str,
    separator: str = "__",
) -> None:
    frame = read_table(Path(input_path))
    source_columns = list(source_columns)
    missing = [column for column in source_columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing source columns: {columns}".format(columns=", ".join(missing)))

    result = frame.copy()
    result[output_column] = result[source_columns].fillna("missing").astype(str).agg(separator.join, axis=1)
    write_table(result, Path(output_path))
