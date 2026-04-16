import pandas as pd

from ssa_urban_deprivation_benchmark.reporting import compare_index_scores
from ssa_urban_deprivation_benchmark.reporting import summarize_binary_contrast
from ssa_urban_deprivation_benchmark.reporting import summarize_category_feature_profiles
from ssa_urban_deprivation_benchmark.reporting import summarize_comparison_shift
from ssa_urban_deprivation_benchmark.reporting import summarize_inequality
from ssa_urban_deprivation_benchmark.reporting import summarize_index
from ssa_urban_deprivation_benchmark.table_ops import add_composite_column
from ssa_urban_deprivation_benchmark.table_ops import filter_table


def test_filter_table_keeps_only_matching_rows(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "filtered.csv"

    pd.DataFrame(
        [
            {"cell_id": "a", "cluster": "high_high", "score": 1.0},
            {"cell_id": "b", "cluster": "low_low", "score": 2.0},
            {"cell_id": "c", "cluster": "high_high", "score": 3.0},
        ]
    ).to_csv(input_path, index=False)

    filter_table(
        input_path=input_path,
        output_path=output_path,
        filter_col="cluster",
        filter_value="high_high",
    )

    result = pd.read_csv(output_path)
    assert result["cell_id"].tolist() == ["a", "c"]


def test_summarize_index_supports_row_filtering(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "summary.csv"

    pd.DataFrame(
        [
            {"city": "Nairobi", "cluster": "high_high", "score": 4.0},
            {"city": "Nairobi", "cluster": "not_significant", "score": 10.0},
            {"city": "Dar es Salaam", "cluster": "high_high", "score": 6.0},
        ]
    ).to_csv(input_path, index=False)

    summarize_index(
        input_path=input_path,
        group_col="city",
        score_col="score",
        score_summary_output=output_path,
        filter_col="cluster",
        filter_value="high_high",
    )

    result = pd.read_csv(output_path).set_index("city")
    assert result.loc["Nairobi", "count"] == 1
    assert result.loc["Nairobi", "mean"] == 4.0
    assert result.loc["Dar es Salaam", "count"] == 1
    assert result.loc["Dar es Salaam", "mean"] == 6.0


def test_summarize_inequality_writes_gini_outputs(tmp_path):
    input_path = tmp_path / "inequality_input.csv"
    output_path = tmp_path / "inequality_summary.csv"

    pd.DataFrame(
        [
            {"city": "Nairobi", "population": 10, "score": 10.0},
            {"city": "Nairobi", "population": 30, "score": 30.0},
            {"city": "Dar es Salaam", "population": 20, "score": 20.0},
            {"city": "Dar es Salaam", "population": 20, "score": 40.0},
        ]
    ).to_csv(input_path, index=False)

    summarize_inequality(
        input_path=input_path,
        group_col="city",
        score_col="score",
        population_col="population",
        output_path=output_path,
    )

    result = pd.read_csv(output_path)
    assert set(result["city"]) == {"Nairobi", "Dar es Salaam"}
    assert result["score_gini_unweighted"].notna().all()
    assert result["score_gini_population_weighted"].notna().all()


def test_compare_index_scores_writes_alignment_and_shift(tmp_path):
    left_input = tmp_path / "left.csv"
    right_input = tmp_path / "right.csv"
    summary_output = tmp_path / "alignment.csv"
    merged_output = tmp_path / "merged.csv"

    pd.DataFrame(
        [
            {"cell_id": "a", "city": "Nairobi", "score": 10.0},
            {"cell_id": "b", "city": "Nairobi", "score": 20.0},
            {"cell_id": "c", "city": "Dar es Salaam", "score": 30.0},
            {"cell_id": "d", "city": "Dar es Salaam", "score": 40.0},
        ]
    ).to_csv(left_input, index=False)

    pd.DataFrame(
        [
            {"cell_id": "a", "city": "Nairobi", "score": 12.0},
            {"cell_id": "b", "city": "Nairobi", "score": 24.0},
            {"cell_id": "c", "city": "Dar es Salaam", "score": 33.0},
            {"cell_id": "d", "city": "Dar es Salaam", "score": 44.0},
        ]
    ).to_csv(right_input, index=False)

    compare_index_scores(
        left_input_path=left_input,
        right_input_path=right_input,
        join_columns=["cell_id", "city"],
        group_col="city",
        left_score_col="score",
        right_score_col="score",
        output_path=summary_output,
        merged_output_path=merged_output,
        left_label="absolute",
        right_label="relative",
        top_fraction=0.5,
    )

    summary = pd.read_csv(summary_output)
    merged = pd.read_csv(merged_output)

    assert set(summary["city"]) == {"Nairobi", "Dar es Salaam"}
    assert summary["top_overlap_share"].eq(1.0).all()
    assert "mean_abs_percentile_shift" in summary.columns
    assert "relative_minus_absolute" in merged.columns
    assert "relative_percentile_minus_absolute_percentile" in merged.columns
    assert "top_fraction_status" in merged.columns
    assert merged["relative_minus_absolute"].tolist() == [2.0, 4.0, 3.0, 4.0]


def test_summarize_comparison_shift_writes_reranking_metrics(tmp_path):
    input_path = tmp_path / "comparison.csv"
    output_path = tmp_path / "shift_summary.csv"

    pd.DataFrame(
        [
            {
                "cell_id": "a",
                "city": "Nairobi",
                "absolute_score": 10.0,
                "relative_score": 30.0,
                "relative_minus_absolute": 20.0,
                "relative_rank_minus_absolute_rank": -1.0,
                "relative_percentile_minus_absolute_percentile": 1.0,
                "abs_score_shift": 20.0,
                "abs_rank_shift": 1.0,
                "abs_percentile_shift": 1.0,
                "top_fraction_status": "entered_top_fraction",
            },
            {
                "cell_id": "b",
                "city": "Nairobi",
                "absolute_score": 20.0,
                "relative_score": 10.0,
                "relative_minus_absolute": -10.0,
                "relative_rank_minus_absolute_rank": 1.0,
                "relative_percentile_minus_absolute_percentile": -1.0,
                "abs_score_shift": 10.0,
                "abs_rank_shift": 1.0,
                "abs_percentile_shift": 1.0,
                "top_fraction_status": "exited_top_fraction",
            },
        ]
    ).to_csv(input_path, index=False)

    summarize_comparison_shift(
        input_path=input_path,
        group_col="city",
        left_label="absolute",
        right_label="relative",
        output_path=output_path,
        top_fraction=0.5,
    )

    result = pd.read_csv(output_path)
    assert result.loc[0, "city"] == "Nairobi"
    assert result.loc[0, "mean_abs_rank_shift"] == 1.0
    assert result.loc[0, "mean_abs_percentile_shift"] == 1.0
    assert result.loc[0, "top_fraction_reclassification_share"] == 1.0


def test_summarize_binary_contrast_writes_effect_sizes(tmp_path):
    input_path = tmp_path / "contrast_input.csv"
    output_path = tmp_path / "contrast_output.csv"

    pd.DataFrame(
        [
            {"city": "Nairobi", "cluster": "high_high", "feature_a": 10.0, "feature_b": 2.0},
            {"city": "Nairobi", "cluster": "high_high", "feature_a": 12.0, "feature_b": 1.0},
            {"city": "Nairobi", "cluster": "other", "feature_a": 4.0, "feature_b": 5.0},
            {"city": "Nairobi", "cluster": "other", "feature_a": 6.0, "feature_b": 4.0},
            {"city": "Dar es Salaam", "cluster": "high_high", "feature_a": 9.0, "feature_b": 7.0},
            {"city": "Dar es Salaam", "cluster": "other", "feature_a": 3.0, "feature_b": 9.0},
        ]
    ).to_csv(input_path, index=False)

    summarize_binary_contrast(
        input_path=input_path,
        group_col="city",
        binary_col="cluster",
        target_value="high_high",
        value_columns=["feature_a", "feature_b"],
        output_path=output_path,
    )

    result = pd.read_csv(output_path)
    nairobi = result.loc[result["city"] == "Nairobi"].set_index("feature")
    assert set(result["city"]) == {"Nairobi", "Dar es Salaam"}
    assert nairobi.loc["feature_a", "standardized_mean_diff"] > 0
    assert nairobi.loc["feature_b", "standardized_mean_diff"] < 0
    assert nairobi.loc["feature_a", "contrast_rank"] == 1


def test_summarize_category_feature_profiles_writes_group_relative_profiles(tmp_path):
    input_path = tmp_path / "profiles_input.csv"
    output_path = tmp_path / "profiles_output.csv"

    pd.DataFrame(
        [
            {"city": "Nairobi", "mechanism": "access", "feature_a": 10.0, "feature_b": 2.0},
            {"city": "Nairobi", "mechanism": "access", "feature_a": 12.0, "feature_b": 4.0},
            {"city": "Nairobi", "mechanism": "services", "feature_a": 4.0, "feature_b": 8.0},
            {"city": "Nairobi", "mechanism": "services", "feature_a": 6.0, "feature_b": 10.0},
            {"city": "Dar es Salaam", "mechanism": "access", "feature_a": 9.0, "feature_b": 3.0},
            {"city": "Dar es Salaam", "mechanism": "services", "feature_a": 3.0, "feature_b": 9.0},
        ]
    ).to_csv(input_path, index=False)

    summarize_category_feature_profiles(
        input_path=input_path,
        group_col="city",
        category_col="mechanism",
        value_columns=["feature_a", "feature_b"],
        output_path=output_path,
    )

    result = pd.read_csv(output_path)
    nairobi = result.loc[(result["city"] == "Nairobi") & (result["mechanism"] == "access")].set_index("feature")
    assert set(result["city"]) == {"Nairobi", "Dar es Salaam"}
    assert nairobi.loc["feature_a", "standardized_profile_diff"] > 0
    assert nairobi.loc["feature_b", "standardized_profile_diff"] < 0
    assert nairobi.loc["feature_a", "profile_rank"] == 1


def test_add_composite_column_joins_source_values(tmp_path):
    input_path = tmp_path / "composite_input.csv"
    output_path = tmp_path / "composite_output.csv"

    pd.DataFrame(
        [
            {"quadrant": "joint_priority", "dimension": "services"},
            {"quadrant": "lower_priority", "dimension": "access"},
        ]
    ).to_csv(input_path, index=False)

    add_composite_column(
        input_path=input_path,
        output_path=output_path,
        source_columns=["quadrant", "dimension"],
        output_column="typology",
        separator="__",
    )

    result = pd.read_csv(output_path)
    assert result["typology"].tolist() == [
        "joint_priority__services",
        "lower_priority__access",
    ]
