import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ssa_urban_deprivation_benchmark.validation import attach_external_raster_signal
from ssa_urban_deprivation_benchmark.validation import build_validation_findings_artifact
from ssa_urban_deprivation_benchmark.validation import summarize_external_validation


def test_attach_external_raster_signal_adds_mean_column(tmp_path):
    frame = gpd.GeoDataFrame(
        {
            "cell_id": ["a", "b"],
            "city": ["A", "A"],
            "geometry": [box(0.0, 0.0, 0.5, 1.0), box(0.5, 0.0, 1.0, 1.0)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    input_path = tmp_path / "cells.gpkg"
    output_path = tmp_path / "cells_with_signal.gpkg"
    raster_path = tmp_path / "signal.tif"
    frame.to_file(input_path, driver="GPKG", engine="pyogrio")

    values = np.array([[10.0, 20.0]], dtype=np.float32)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=1,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 0.5, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(values, 1)

    attach_external_raster_signal(
        input_path=input_path,
        raster_path=raster_path,
        output_path=output_path,
        prefix="viirs",
        stats=["mean"],
    )

    result = gpd.read_file(output_path)
    assert result["viirs_mean"].round(6).tolist() == [10.0, 20.0]
    assert result["viirs_available"].tolist() == [True, True]


def test_summarize_external_validation_and_findings(tmp_path):
    input_path = tmp_path / "validation.csv"
    summary_path = tmp_path / "summary.csv"
    findings_path = tmp_path / "findings.json"
    pd.DataFrame(
        {
            "city": ["A", "A", "A", "B", "B", "B"],
            "deprivation_index_0_100": [90, 60, 10, 80, 40, 5],
            "rwi_deprivation_proxy_0_100": [85, 55, 15, 75, 35, 10],
            "viirs_mean": [1.0, 2.0, 6.0, 2.0, 4.0, 8.0],
        }
    ).to_csv(input_path, index=False)

    summarize_external_validation(
        input_path=input_path,
        group_col="city",
        external_col="viirs_mean",
        score_columns=["deprivation_index_0_100", "rwi_deprivation_proxy_0_100"],
        output_path=summary_path,
        top_fraction=1 / 3,
        expected_relation="negative",
    )
    build_validation_findings_artifact(summary_input_path=summary_path, output_path=findings_path)

    summary = pd.read_csv(summary_path)
    findings = json.loads(findings_path.read_text())
    subset = summary.loc[
        (summary["city"] == "A") & (summary["score_column"] == "deprivation_index_0_100")
    ].iloc[0]

    assert subset["expected_signed_spearman"] > 0
    assert findings["strongest_by_score"]["deprivation_index_0_100"]["best_group"] in {"A", "B"}
