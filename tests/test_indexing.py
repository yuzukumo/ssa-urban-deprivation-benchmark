from pathlib import Path

import pandas as pd

from ssa_urban_deprivation_benchmark.indexing import build_index_table


def test_build_index_table_supports_within_group_standardization(tmp_path):
    input_path = tmp_path / "features.csv"
    config_path = tmp_path / "config.yaml"

    pd.DataFrame(
        [
            {"cell_id": "a", "city": "Nairobi", "road_distance_m": 0.0},
            {"cell_id": "b", "city": "Nairobi", "road_distance_m": 10.0},
            {"cell_id": "c", "city": "Dar es Salaam", "road_distance_m": 100.0},
            {"cell_id": "d", "city": "Dar es Salaam", "road_distance_m": 110.0},
        ]
    ).to_csv(input_path, index=False)

    config_path.write_text(
        "\n".join(
            [
                "index_name: grouped_test",
                "keep_debug_columns: true",
                "winsorize_quantiles: [0.0, 1.0]",
                "winsorize_group_col: city",
                "feature_standardization_group_col: city",
                "output_scaling:",
                "  lower_quantile: 0.0",
                "  upper_quantile: 1.0",
                "  group_col: city",
                "dimension_weights:",
                "  access: 1.0",
                "features:",
                "  - name: road_distance_m",
                "    dimension: access",
                "    higher_is_more_deprived: true",
                "    feature_weight: 1.0",
                "pca:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    result, metadata = build_index_table(Path(input_path), Path(config_path))

    assert result["road_distance_m__z"].round(6).tolist() == [-1.0, 1.0, -1.0, 1.0]
    assert result["deprivation_index_0_100"].tolist() == [0.0, 100.0, 0.0, 100.0]
    assert metadata["feature_standardization_group_col"] == "city"
    assert metadata["winsorize_group_col"] == "city"
