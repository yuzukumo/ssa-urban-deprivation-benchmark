from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Iterable
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.io_utils import write_table


def attach_external_raster_signal(
    input_path: Path,
    raster_path: Path,
    output_path: Path,
    metadata_path: Optional[Path] = None,
    prefix: str = "external_signal",
    stats: Optional[Iterable[str]] = None,
    all_touched: bool = True,
) -> None:
    frame = read_table(Path(input_path))
    if not isinstance(frame, gpd.GeoDataFrame):
        raise ValueError("attach-external-raster requires a geospatial input file.")

    stats = list(stats or ["mean"])
    if not stats:
        raise ValueError("At least one raster summary statistic must be requested.")

    with rasterio.open(Path(raster_path)) as src:
        nodata = src.nodata
        raster_crs = src.crs

    working = frame.to_crs(raster_crs) if raster_crs else frame.copy()
    records = zonal_stats(
        vectors=working.geometry,
        raster=str(Path(raster_path)),
        stats=stats,
        all_touched=bool(all_touched),
        nodata=nodata,
    )

    result = frame.copy()
    attached_columns = []
    for stat in stats:
        column = "{prefix}_{stat}".format(prefix=prefix, stat=stat)
        result[column] = [record.get(stat) if record else np.nan for record in records]
        result[column] = pd.to_numeric(result[column], errors="coerce")
        attached_columns.append(column)

    result["{prefix}_available".format(prefix=prefix)] = result[attached_columns].notna().any(axis=1)
    write_table(result, Path(output_path))

    if metadata_path:
        write_json(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_path": str(Path(input_path)),
                "raster_path": str(Path(raster_path)),
                "output_path": str(Path(output_path)),
                "prefix": prefix,
                "stats": stats,
                "all_touched": bool(all_touched),
                "n_rows": int(len(result)),
                "n_available_rows": int(result["{prefix}_available".format(prefix=prefix)].sum()),
                "coverage_share": float(result["{prefix}_available".format(prefix=prefix)].mean()),
            },
            Path(metadata_path),
        )


def summarize_external_validation(
    input_path: Path,
    group_col: str,
    external_col: str,
    score_columns: Iterable[str],
    output_path: Path,
    top_fraction: float = 0.1,
    expected_relation: str = "negative",
) -> None:
    frame = read_table(Path(input_path))
    score_columns = list(score_columns)
    if not score_columns:
        raise ValueError("At least one score column is required.")
    if top_fraction <= 0 or top_fraction >= 0.5:
        raise ValueError("top_fraction must be in the interval (0, 0.5).")

    required = [group_col, external_col, *score_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    direction_sign = -1.0 if str(expected_relation) == "negative" else 1.0
    records = []

    for group_value, subset in frame.groupby(group_col):
        for score_col in score_columns:
            working = subset[[score_col, external_col]].copy()
            working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
            working[external_col] = pd.to_numeric(working[external_col], errors="coerce")
            working = working.dropna()
            if working.empty:
                continue

            cutoff = max(1, int(len(working) * float(top_fraction)))
            top_slice = working.nlargest(cutoff, score_col)
            bottom_slice = working.nsmallest(cutoff, score_col)

            pearson = working[score_col].corr(working[external_col], method="pearson")
            spearman = working[score_col].corr(working[external_col], method="spearman")
            top_mean = float(top_slice[external_col].mean())
            bottom_mean = float(bottom_slice[external_col].mean())
            std = float(working[external_col].std(ddof=0))
            standardized_gap = np.nan if std <= 1e-9 else (top_mean - bottom_mean) / std

            records.append(
                {
                    group_col: str(group_value),
                    "score_column": str(score_col),
                    "external_column": str(external_col),
                    "n_rows": int(len(working)),
                    "expected_relation": str(expected_relation),
                    "pearson_corr": float(pearson),
                    "spearman_corr": float(spearman),
                    "expected_signed_pearson": float(direction_sign * pearson),
                    "expected_signed_spearman": float(direction_sign * spearman),
                    "top_fraction": float(top_fraction),
                    "top_score_external_mean": top_mean,
                    "bottom_score_external_mean": bottom_mean,
                    "top_minus_bottom_external_mean": float(top_mean - bottom_mean),
                    "bottom_minus_top_external_mean": float(bottom_mean - top_mean),
                    "top_over_bottom_external_ratio": float(top_mean / bottom_mean) if abs(bottom_mean) > 1e-9 else np.nan,
                    "standardized_top_minus_bottom_gap": float(standardized_gap),
                }
            )

    summary = pd.DataFrame(records)
    write_table(summary, Path(output_path))


def build_validation_findings_artifact(
    summary_input_path: Path,
    output_path: Path,
) -> None:
    summary = pd.read_csv(Path(summary_input_path))
    if summary.empty:
        raise ValueError("Validation summary is empty: {path}".format(path=summary_input_path))

    strongest_by_score = {}
    for score_col, subset in summary.groupby("score_column"):
        ordered = subset.sort_values("expected_signed_spearman", ascending=False).reset_index(drop=True)
        strongest_by_score[str(score_col)] = {
            "best_group": str(ordered.iloc[0][summary.columns[0]]),
            "expected_signed_spearman": float(ordered.iloc[0]["expected_signed_spearman"]),
            "expected_signed_pearson": float(ordered.iloc[0]["expected_signed_pearson"]),
            "external_column": str(ordered.iloc[0]["external_column"]),
        }

    weakest_overall = summary.sort_values("expected_signed_spearman", ascending=True).iloc[0]
    strongest_overall = summary.sort_values("expected_signed_spearman", ascending=False).iloc[0]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_input_path": str(Path(summary_input_path)),
        "n_rows": int(len(summary)),
        "score_columns": sorted(summary["score_column"].astype(str).unique().tolist()),
        "groups": sorted(summary.iloc[:, 0].astype(str).unique().tolist()),
        "strongest_by_score": strongest_by_score,
        "overall": {
            "strongest_expected_alignment": strongest_overall.to_dict(),
            "weakest_expected_alignment": weakest_overall.to_dict(),
        },
    }
    write_json(payload, Path(output_path))
