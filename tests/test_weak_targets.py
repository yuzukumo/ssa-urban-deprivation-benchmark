import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from ssa_urban_deprivation_benchmark.weak_targets import build_rwi_grid_targets


def test_build_rwi_grid_targets_assigns_weighted_targets_and_quantiles(tmp_path):
    grid = gpd.GeoDataFrame(
        [
            {
                "cell_id": "a",
                "city": "Nairobi",
                "country_iso": "KEN",
                "deprivation_index_0_100": 80.0,
                "geometry": Polygon([(36.80, -1.30), (36.81, -1.30), (36.81, -1.29), (36.80, -1.29)]),
            },
            {
                "cell_id": "b",
                "city": "Nairobi",
                "country_iso": "KEN",
                "deprivation_index_0_100": 20.0,
                "geometry": Polygon([(36.82, -1.30), (36.83, -1.30), (36.83, -1.29), (36.82, -1.29)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    input_path = tmp_path / "grid.geojson"
    output_path = tmp_path / "targets.geojson"
    rwi_dir = tmp_path / "rwi"
    rwi_dir.mkdir()
    grid.to_file(input_path, driver="GeoJSON", engine="pyogrio")

    pd.DataFrame(
        [
            {"latitude": -1.295, "longitude": 36.805, "rwi": -0.8, "error": 0.2},
            {"latitude": -1.295, "longitude": 36.825, "rwi": 0.5, "error": 0.3},
        ]
    ).to_csv(rwi_dir / "KEN_relative_wealth_index.csv", index=False)

    build_rwi_grid_targets(
        input_path=input_path,
        output_path=output_path,
        rwi_dir=rwi_dir,
        neighbors=1,
        max_distance_m=5000.0,
        low_wealth_quantile=0.5,
    )

    result = gpd.read_file(output_path).sort_values("cell_id").reset_index(drop=True)
    assert result["rwi_label_available"].tolist() == [True, True]
    assert result.loc[0, "rwi_mean"] < result.loc[1, "rwi_mean"]
    assert bool(result.loc[0, "rwi_bottom_quantile_flag"]) is True
    assert bool(result.loc[1, "rwi_bottom_quantile_flag"]) is False
    assert result.loc[0, "rwi_deprivation_proxy_0_100"] > result.loc[1, "rwi_deprivation_proxy_0_100"]
