from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.io_utils import write_table


def cluster_cells(input_path: Path, columns: List[str], k: int = 4) -> Tuple[pd.DataFrame, Dict]:
    frame = read_table(Path(input_path)).copy()

    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError("Missing clustering columns: {cols}".format(cols=", ".join(missing_columns)))
    if k < 2:
        raise ValueError("k must be at least 2.")
    if len(frame) < k:
        raise ValueError("The number of rows must be at least k.")

    matrix = frame[columns].apply(pd.to_numeric, errors="coerce")
    medians = matrix.median()
    matrix = matrix.fillna(medians)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(scaled)

    result = frame.copy()
    result["cluster_id"] = labels.astype(int)

    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=columns,
    )
    centers.insert(0, "cluster_id", range(k))

    cluster_sizes = result["cluster_id"].value_counts().sort_index()
    summary = {
        "input_path": str(Path(input_path)),
        "k": int(k),
        "columns": columns,
        "cluster_sizes": {str(index): int(value) for index, value in cluster_sizes.items()},
        "cluster_centers": centers.to_dict(orient="records"),
    }

    return result, summary


def run_clustering(
    input_path: Path,
    columns: List[str],
    k: int,
    output_path: Path,
    summary_path: Path,
) -> None:
    result, summary = cluster_cells(Path(input_path), columns=columns, k=k)
    write_table(result, Path(output_path))
    write_json(summary, Path(summary_path))
