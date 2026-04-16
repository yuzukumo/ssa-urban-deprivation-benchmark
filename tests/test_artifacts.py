import json

import pandas as pd

from ssa_urban_deprivation_benchmark.artifacts import build_core_findings_artifact
from ssa_urban_deprivation_benchmark.artifacts import stage_figure_set


def test_build_core_findings_artifact_writes_nested_summary(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()

    pd.DataFrame(
        [
            {"scenario": "500m", "city": "Dar es Salaam", "mean": 45.0, "median": 44.0},
            {"scenario": "500m", "city": "Nairobi", "mean": 40.0, "median": 39.0},
        ]
    ).to_csv(tables_dir / "city_score_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "500m",
                "city": "Dar es Salaam",
                "population_weighted_mean_score": 42.0,
                "population_share_at_or_above_q90": 0.14,
            },
            {
                "scenario": "500m",
                "city": "Nairobi",
                "population_weighted_mean_score": 38.0,
                "population_share_at_or_above_q90": 0.10,
            },
            {
                "scenario": "1km",
                "city": "Dar es Salaam",
                "population_weighted_mean_score": 41.0,
                "population_share_at_or_above_q90": 0.13,
            },
            {
                "scenario": "1km",
                "city": "Nairobi",
                "population_weighted_mean_score": 37.0,
                "population_share_at_or_above_q90": 0.09,
            },
        ]
    ).to_csv(tables_dir / "population_exposure.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "500m",
                "city": "Dar es Salaam",
                "score_gini_population_weighted": 0.38,
            },
            {
                "scenario": "500m",
                "city": "Nairobi",
                "score_gini_population_weighted": 0.35,
            },
            {
                "scenario": "1km",
                "city": "Dar es Salaam",
                "score_gini_population_weighted": 0.40,
            },
            {
                "scenario": "1km",
                "city": "Nairobi",
                "score_gini_population_weighted": 0.36,
            },
        ]
    ).to_csv(tables_dir / "inequality_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "500m",
                "city": "Dar es Salaam",
                "left_label": "absolute",
                "right_label": "relative",
                "n_rows": 10,
                "pearson_corr": 0.99,
                "spearman_corr": 0.98,
                "top_fraction": 0.1,
                "top_overlap_share": 0.95,
            },
            {
                "scenario": "500m",
                "city": "Nairobi",
                "left_label": "absolute",
                "right_label": "relative",
                "n_rows": 10,
                "pearson_corr": 0.98,
                "spearman_corr": 0.97,
                "top_fraction": 0.1,
                "top_overlap_share": 0.90,
            },
        ]
    ).to_csv(tables_dir / "absolute_relative_alignment.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "500m",
                "city": "Dar es Salaam",
                "left_label": "absolute",
                "right_label": "relative",
                "top_fraction": 0.1,
                "mean_abs_percentile_shift": 0.04,
                "p90_abs_percentile_shift": 0.12,
            },
            {
                "scenario": "500m",
                "city": "Nairobi",
                "left_label": "absolute",
                "right_label": "relative",
                "top_fraction": 0.1,
                "mean_abs_percentile_shift": 0.06,
                "p90_abs_percentile_shift": 0.18,
            },
        ]
    ).to_csv(tables_dir / "relative_shift_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": "500m", "city": "Dar es Salaam", "dominant_dimension": "services", "weight_share": 0.55},
            {"scenario": "500m", "city": "Dar es Salaam", "dominant_dimension": "urban_form", "weight_share": 0.35},
            {"scenario": "500m", "city": "Nairobi", "dominant_dimension": "services", "weight_share": 0.72},
            {"scenario": "500m", "city": "Nairobi", "dominant_dimension": "access", "weight_share": 0.06},
        ]
    ).to_csv(tables_dir / "hotspot_dominant_population_share.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": "500m", "city": "Dar es Salaam", "priority_quadrant": "joint_priority", "weight_share": 0.53},
            {"scenario": "500m", "city": "Nairobi", "priority_quadrant": "lower_priority", "weight_share": 0.55},
        ]
    ).to_csv(tables_dir / "hotspot_priority_population_share.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": "500m", "city": "Dar es Salaam", "hotspot_typology": "joint_priority__services", "weight_share": 0.39},
            {"scenario": "500m", "city": "Dar es Salaam", "hotspot_typology": "lower_priority__urban_form", "weight_share": 0.27},
            {"scenario": "500m", "city": "Nairobi", "hotspot_typology": "joint_priority__services", "weight_share": 0.41},
            {"scenario": "500m", "city": "Nairobi", "hotspot_typology": "lower_priority__services", "weight_share": 0.31},
        ]
    ).to_csv(tables_dir / "hotspot_typology_population_share.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": "500m", "city": "Dar es Salaam", "mean": 45.0},
            {"scenario": "500m", "city": "Nairobi", "mean": 40.0},
            {"scenario": "1km", "city": "Dar es Salaam", "mean": 44.0},
            {"scenario": "1km", "city": "Nairobi", "mean": 41.0},
        ]
    ).to_csv(tables_dir / "grid_size_score_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": "500m", "city": "Dar es Salaam", "population_share_at_or_above_q90": 0.14},
            {"scenario": "500m", "city": "Nairobi", "population_share_at_or_above_q90": 0.10},
            {"scenario": "1km", "city": "Dar es Salaam", "population_share_at_or_above_q90": 0.13},
            {"scenario": "1km", "city": "Nairobi", "population_share_at_or_above_q90": 0.09},
        ]
    ).to_csv(tables_dir / "grid_size_population_exposure.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": "500m", "city": "Dar es Salaam", "score_gini_population_weighted": 0.38},
            {"scenario": "500m", "city": "Nairobi", "score_gini_population_weighted": 0.35},
            {"scenario": "1km", "city": "Dar es Salaam", "score_gini_population_weighted": 0.40},
            {"scenario": "1km", "city": "Nairobi", "score_gini_population_weighted": 0.36},
        ]
    ).to_csv(tables_dir / "grid_size_inequality_summary.csv", index=False)

    config_path = tmp_path / "core_findings.yaml"
    config_path.write_text(
        "\n".join(
            [
                "study_id: demo_study",
                "primary_scenario: 500m",
                "sensitivity_scenario: 1km",
                "top_fraction: 0.1",
                "paths:",
                "  city_score_summary: {path}".format(path=tables_dir / "city_score_summary.csv"),
                "  population_exposure: {path}".format(path=tables_dir / "population_exposure.csv"),
                "  inequality_summary: {path}".format(path=tables_dir / "inequality_summary.csv"),
                "  absolute_relative_alignment: {path}".format(path=tables_dir / "absolute_relative_alignment.csv"),
                "  relative_shift_summary: {path}".format(path=tables_dir / "relative_shift_summary.csv"),
                "  hotspot_dominant_population_share: {path}".format(path=tables_dir / "hotspot_dominant_population_share.csv"),
                "  hotspot_priority_population_share: {path}".format(path=tables_dir / "hotspot_priority_population_share.csv"),
                "  hotspot_typology_population_share: {path}".format(path=tables_dir / "hotspot_typology_population_share.csv"),
                "  grid_size_score_summary: {path}".format(path=tables_dir / "grid_size_score_summary.csv"),
                "  grid_size_population_exposure: {path}".format(path=tables_dir / "grid_size_population_exposure.csv"),
                "  grid_size_inequality_summary: {path}".format(path=tables_dir / "grid_size_inequality_summary.csv"),
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "core_findings.json"
    build_core_findings_artifact(config_path=config_path, output_path=output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["study_id"] == "demo_study"
    assert payload["cross_city"]["mean_score"]["ranking"][0]["city"] == "Dar es Salaam"
    assert payload["cities"]["Nairobi"]["hotspots"]["dominant_dimension_by_population"][0]["category"] == "services"
    assert payload["cities"]["Nairobi"]["absolute_vs_relative"]["shift"]["mean_abs_percentile_shift"] == 0.06
    assert payload["sensitivity"]["order_stability"]["mean_score"] is True


def test_stage_figure_set_copies_files_and_writes_manifest(tmp_path):
    figure_source_dir = tmp_path / "source_figures"
    figure_source_dir.mkdir()
    (figure_source_dir / "a.png").write_bytes(b"png-a")
    (figure_source_dir / "b.png").write_bytes(b"png-b")

    config_path = tmp_path / "figure_set.yaml"
    config_path.write_text(
        "\n".join(
            [
                "figure_set_id: demo_figures",
                "items:",
                "  - id: fig_a",
                "    source: {path}".format(path=figure_source_dir / "a.png"),
                "    filename: figure_01.png",
                "  - id: fig_b",
                "    source: {path}".format(path=figure_source_dir / "b.png"),
                "    filename: figure_02.png",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "paper_core"
    manifest_output = tmp_path / "figure_manifest.json"
    stage_figure_set(
        config_path=config_path,
        output_dir=output_dir,
        manifest_output=manifest_output,
    )

    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert manifest["figure_set_id"] == "demo_figures"
    assert len(manifest["items"]) == 2
    assert (output_dir / "figure_01.png").read_bytes() == b"png-a"
    assert (output_dir / "figure_02.png").read_bytes() == b"png-b"
