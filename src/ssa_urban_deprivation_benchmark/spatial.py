from pathlib import Path
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import require_optional_dependency
from ssa_urban_deprivation_benchmark.io_utils import write_json


def _cluster_label(quadrant: int, p_value: float) -> str:
    if p_value >= 0.05:
        return "not_significant"

    mapping = {
        1: "high_high",
        2: "low_high",
        3: "low_low",
        4: "high_low",
    }
    return mapping.get(int(quadrant), "unknown")


def compute_spatial_autocorrelation(input_path: Path, score_col: str, k: int = 8) -> Tuple[Dict, object]:
    geopandas = require_optional_dependency("geopandas", "geo")
    libpysal = require_optional_dependency("libpysal", "spatial")
    esda = require_optional_dependency("esda", "spatial")

    geodata = geopandas.read_file(Path(input_path))
    if score_col not in geodata.columns:
        raise ValueError("Column '{column}' not found.".format(column=score_col))
    if len(geodata) < 3:
        raise ValueError("Need at least three geometries for spatial autocorrelation.")

    y = pd.to_numeric(geodata[score_col], errors="coerce")
    median_value = float(y.median()) if y.notna().any() else 0.0
    y = y.fillna(median_value)

    projected = geodata.to_crs(geodata.estimate_utm_crs())
    centroids = projected.geometry.centroid
    coordinates = np.column_stack([centroids.x.values, centroids.y.values])
    neighbor_count = min(max(1, k), len(geodata) - 1)

    weights = libpysal.weights.KNN.from_array(coordinates, k=neighbor_count)
    weights.transform = "r"

    global_moran = esda.Moran(y.to_numpy(), weights)
    local_moran = esda.Moran_Local(y.to_numpy(), weights)

    geodata["local_moran_i"] = local_moran.Is
    geodata["local_moran_p"] = local_moran.p_sim
    geodata["local_moran_quadrant"] = local_moran.q
    geodata["local_moran_cluster"] = [
        _cluster_label(quadrant, p_value)
        for quadrant, p_value in zip(local_moran.q, local_moran.p_sim)
    ]

    summary = {
        "input_path": str(Path(input_path)),
        "score_col": score_col,
        "n_observations": int(len(geodata)),
        "k_neighbors": int(neighbor_count),
        "moran_i": float(global_moran.I),
        "moran_p_sim": float(global_moran.p_sim),
        "score_mean": float(y.mean()),
        "score_std": float(y.std(ddof=0)),
    }

    return summary, geodata


def run_spatial_autocorrelation(
    input_path: Path,
    score_col: str,
    summary_output: Path,
    local_output: Path,
    k: int = 8,
) -> None:
    summary, geodata = compute_spatial_autocorrelation(Path(input_path), score_col, k=k)
    write_json(summary, Path(summary_output))

    suffix = Path(local_output).suffix.lower()
    if suffix == ".csv":
        export_frame = geodata.copy()
        export_frame["geometry"] = export_frame.geometry.astype(str)
        export_frame.to_csv(local_output, index=False)
    else:
        geodata.to_file(local_output)
