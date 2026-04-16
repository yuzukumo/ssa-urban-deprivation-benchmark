import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ssa_urban_deprivation_benchmark.multimodal_ml import _build_graph_edge_index
from ssa_urban_deprivation_benchmark.multimodal_ml import _make_protocol_splits
from ssa_urban_deprivation_benchmark.multimodal_ml import build_benchmark_findings_artifact
from ssa_urban_deprivation_benchmark.multimodal_ml import summarize_multimodal_benchmark


def test_build_graph_edge_index_respects_city_blocks():
    frame = gpd.GeoDataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "city": ["Nairobi", "Nairobi", "Dar es Salaam", "Dar es Salaam"],
            "geometry": [
                Point(36.81, -1.29),
                Point(36.82, -1.30),
                Point(39.24, -6.80),
                Point(39.25, -6.81),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    edge_index, _edge_weight = _build_graph_edge_index(frame, city_col="city", k=1)
    edges = {(int(source), int(target)) for source, target in edge_index.T.tolist()}

    assert (0, 1) in edges
    assert (1, 0) in edges
    assert (2, 3) in edges
    assert (3, 2) in edges
    assert (0, 2) not in edges
    assert (1, 3) not in edges


def test_summarize_multimodal_benchmark_includes_average_ranks(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    summary_path = tmp_path / "summary.json"
    pd.DataFrame(
        [
            {"protocol": "pooled_random", "model": "xgboost_tabular", "metric": "rmse", "value": 0.20},
            {"protocol": "pooled_random", "model": "graph_fusion", "metric": "rmse", "value": 0.30},
            {"protocol": "city_a_to_city_b", "model": "xgboost_tabular", "metric": "rmse", "value": 0.40},
            {"protocol": "city_a_to_city_b", "model": "graph_fusion", "metric": "rmse", "value": 0.25},
            {"protocol": "pooled_random", "model": "xgboost_tabular", "metric": "roc_auc", "value": 0.80},
            {"protocol": "pooled_random", "model": "graph_fusion", "metric": "roc_auc", "value": 0.75},
            {"protocol": "city_a_to_city_b", "model": "xgboost_tabular", "metric": "roc_auc", "value": 0.70},
            {"protocol": "city_a_to_city_b", "model": "graph_fusion", "metric": "roc_auc", "value": 0.85},
        ]
    ).to_csv(metrics_path, index=False)

    summarize_multimodal_benchmark(metrics_input_path=metrics_path, output_path=summary_path)

    summary = json.loads(summary_path.read_text())
    transfer_ranks = summary["average_ranks"]["transfer_only"]["rmse"]
    assert transfer_ranks[0]["model"] == "graph_fusion"
    assert "pooled_random" in summary["by_protocol"]


def test_make_protocol_splits_uses_leave_one_city_out_for_multicity():
    city_values = pd.Series(
        ["Nairobi", "Nairobi", "Dar es Salaam", "Dar es Salaam", "Kampala", "Kampala"]
    ).to_numpy()
    class_target = pd.Series([0, 1, 0, 1, 0, 1]).to_numpy()
    label_mask = pd.Series([True, True, True, True, True, True]).to_numpy()

    protocols = _make_protocol_splits(
        city_values=city_values,
        class_target=class_target,
        label_mask=label_mask,
        random_state=42,
        strategy="auto",
    )
    names = [protocol["protocol"] for protocol in protocols]

    assert names[0] == "pooled_random"
    assert "holdout_nairobi" in names
    assert "holdout_dar_es_salaam" in names
    assert "holdout_kampala" in names


def test_build_benchmark_findings_artifact_writes_holdout_summary(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    summary_path = tmp_path / "summary.json"
    findings_path = tmp_path / "findings.json"
    pd.DataFrame(
        [
            {"protocol": "pooled_random", "model": "resnet_fusion_pretrained", "metric": "rmse", "value": 0.2},
            {"protocol": "pooled_random", "model": "graph_fusion", "metric": "rmse", "value": 0.3},
            {"protocol": "holdout_accra", "model": "atlas_linear_baseline", "metric": "rmse", "value": 0.4},
            {"protocol": "holdout_accra", "model": "graph_fusion", "metric": "rmse", "value": 0.25},
            {"protocol": "holdout_accra", "model": "atlas_linear_baseline", "metric": "average_precision", "value": 0.2},
            {"protocol": "holdout_accra", "model": "graph_fusion", "metric": "average_precision", "value": 0.5},
            {"protocol": "holdout_accra", "model": "atlas_linear_baseline", "metric": "spearman_corr", "value": 0.1},
            {"protocol": "holdout_accra", "model": "graph_fusion", "metric": "spearman_corr", "value": 0.3},
        ]
    ).to_csv(metrics_path, index=False)

    summarize_multimodal_benchmark(metrics_input_path=metrics_path, output_path=summary_path)
    build_benchmark_findings_artifact(
        metrics_input_path=metrics_path,
        summary_input_path=summary_path,
        output_path=findings_path,
    )

    findings = json.loads(findings_path.read_text())
    assert findings["hardest_holdouts"]["rmse"]["protocol"] == "holdout_accra"
    assert findings["per_holdout_recommendations"]["holdout_accra"]["screening_recommendation"]["winner_model"] == "graph_fusion"
