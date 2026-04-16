import pandas as pd

from ssa_urban_deprivation_benchmark.interpretation import annotate_dominant_dimension
from ssa_urban_deprivation_benchmark.interpretation import annotate_priority_quadrants


def test_annotate_dominant_dimension_adds_margin_and_strength(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    pd.DataFrame(
        [
            {"cell_id": "a", "access__score": 1.0, "services__score": 0.6, "urban_form__score": 0.2},
            {"cell_id": "b", "access__score": 0.1, "services__score": 1.6, "urban_form__score": 0.3},
            {"cell_id": "c", "access__score": 0.4, "services__score": 0.3, "urban_form__score": 0.2},
            {"cell_id": "d", "access__score": None, "services__score": None, "urban_form__score": None},
        ]
    ).to_csv(input_path, index=False)

    annotate_dominant_dimension(
        input_path=input_path,
        dimension_cols=["access__score", "services__score", "urban_form__score"],
        output_path=output_path,
        margin_thresholds=[0.25, 0.75],
    )

    result = pd.read_csv(output_path)

    assert result["dominant_dimension"].tolist() == ["access", "services", "access", "missing"]
    assert result["dominant_dimension_strength"].tolist() == ["moderate", "strong", "mixed", "missing"]
    assert result["dominant_dimension_margin"].round(3).tolist()[:3] == [0.4, 1.3, 0.1]
    assert pd.isna(result.loc[3, "dominant_dimension_margin"])


def test_annotate_priority_quadrants_combines_absolute_and_relative_flags(tmp_path):
    input_path = tmp_path / "priority_input.csv"
    output_path = tmp_path / "priority_output.csv"

    pd.DataFrame(
        [
            {"cell_id": "a", "city": "Nairobi", "absolute": 95.0, "relative": 92.0},
            {"cell_id": "b", "city": "Nairobi", "absolute": 93.0, "relative": 10.0},
            {"cell_id": "c", "city": "Nairobi", "absolute": 20.0, "relative": 91.0},
            {"cell_id": "d", "city": "Nairobi", "absolute": 10.0, "relative": 5.0},
            {"cell_id": "e", "city": "Dar es Salaam", "absolute": 85.0, "relative": 88.0},
            {"cell_id": "f", "city": "Dar es Salaam", "absolute": 5.0, "relative": 3.0},
        ]
    ).to_csv(input_path, index=False)

    annotate_priority_quadrants(
        input_path=input_path,
        absolute_score_col="absolute",
        relative_score_col="relative",
        group_col="city",
        output_path=output_path,
        absolute_top_fraction=0.34,
        relative_top_fraction=0.34,
    )

    result = pd.read_csv(output_path).set_index("cell_id")

    assert result.loc["a", "priority_quadrant"] == "joint_priority"
    assert result.loc["b", "priority_quadrant"] == "absolute_only"
    assert result.loc["c", "priority_quadrant"] == "relative_only"
    assert result.loc["d", "priority_quadrant"] == "lower_priority"
