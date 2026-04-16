import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import Polygon

from ssa_urban_deprivation_benchmark.admin import attach_admin_units
from ssa_urban_deprivation_benchmark.admin import summarize_admin_units


def test_attach_admin_units_assigns_admin_labels(tmp_path):
    boundary_dir = tmp_path / "boundaries"
    boundary_dir.mkdir()

    boundaries = gpd.GeoDataFrame(
        [
            {
                "shapeName": "Alpha",
                "shapeID": "KEN.alpha",
                "shapeISO": "KE-ALPHA",
                "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            },
            {
                "shapeName": "Beta",
                "shapeID": "KEN.beta",
                "shapeISO": "KE-BETA",
                "geometry": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    boundaries.to_file(boundary_dir / "KEN_adm2.geojson", driver="GeoJSON", engine="pyogrio")

    cells = gpd.GeoDataFrame(
        [
            {"cell_id": "a", "country_iso": "KEN", "geometry": Point(0.5, 0.5).buffer(0.1)},
            {"cell_id": "b", "country_iso": "KEN", "geometry": Point(1.5, 0.5).buffer(0.1)},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    input_path = tmp_path / "cells.geojson"
    output_path = tmp_path / "cells_with_admin.geojson"
    cells.to_file(input_path, driver="GeoJSON", engine="pyogrio")

    attach_admin_units(
        input_path=input_path,
        output_path=output_path,
        boundary_dir=boundary_dir,
        admin_level=2,
        country_col="country_iso",
        admin_prefix="admin2",
    )

    result = gpd.read_file(output_path).sort_values("cell_id").reset_index(drop=True)
    assert result["admin2_name"].tolist() == ["Alpha", "Beta"]
    assert result["admin2_id"].tolist() == ["KEN.alpha", "KEN.beta"]


def test_summarize_admin_units_writes_priority_metrics(tmp_path):
    rows = [
        {
            "cell_id": "a1",
            "city": "Nairobi",
            "country_iso": "KEN",
            "admin2_name": "Alpha",
            "admin2_id": "alpha",
            "admin2_iso": "KE-ALPHA",
            "score": 95.0,
            "population": 100.0,
            "local_moran_cluster": "high_high",
            "dominant_dimension": "services",
            "priority_quadrant": "joint_priority",
            "geometry": Point(0.1, 0.1).buffer(0.05),
        },
        {
            "cell_id": "a2",
            "city": "Nairobi",
            "country_iso": "KEN",
            "admin2_name": "Alpha",
            "admin2_id": "alpha",
            "admin2_iso": "KE-ALPHA",
            "score": 90.0,
            "population": 80.0,
            "local_moran_cluster": "high_high",
            "dominant_dimension": "services",
            "priority_quadrant": "joint_priority",
            "geometry": Point(0.2, 0.1).buffer(0.05),
        },
        {
            "cell_id": "b1",
            "city": "Nairobi",
            "country_iso": "KEN",
            "admin2_name": "Beta",
            "admin2_id": "beta",
            "admin2_iso": "KE-BETA",
            "score": 70.0,
            "population": 70.0,
            "local_moran_cluster": "not_significant",
            "dominant_dimension": "access",
            "priority_quadrant": "lower_priority",
            "geometry": Point(1.1, 0.1).buffer(0.05),
        },
        {
            "cell_id": "b2",
            "city": "Nairobi",
            "country_iso": "KEN",
            "admin2_name": "Beta",
            "admin2_id": "beta",
            "admin2_iso": "KE-BETA",
            "score": 60.0,
            "population": 60.0,
            "local_moran_cluster": "not_significant",
            "dominant_dimension": "access",
            "priority_quadrant": "lower_priority",
            "geometry": Point(1.2, 0.1).buffer(0.05),
        },
        {
            "cell_id": "c1",
            "city": "Nairobi",
            "country_iso": "KEN",
            "admin2_name": "Gamma",
            "admin2_id": "gamma",
            "admin2_iso": "KE-GAMMA",
            "score": 55.0,
            "population": 90.0,
            "local_moran_cluster": "high_high",
            "dominant_dimension": "urban_form",
            "priority_quadrant": "relative_only",
            "geometry": Point(2.1, 0.1).buffer(0.05),
        },
        {
            "cell_id": "c2",
            "city": "Nairobi",
            "country_iso": "KEN",
            "admin2_name": "Gamma",
            "admin2_id": "gamma",
            "admin2_iso": "KE-GAMMA",
            "score": 50.0,
            "population": 80.0,
            "local_moran_cluster": "high_high",
            "dominant_dimension": "urban_form",
            "priority_quadrant": "relative_only",
            "geometry": Point(2.2, 0.1).buffer(0.05),
        },
        {
            "cell_id": "d1",
            "city": "Nairobi",
            "country_iso": "KEN",
            "admin2_name": "Delta",
            "admin2_id": "delta",
            "admin2_iso": "KE-DELTA",
            "score": 30.0,
            "population": 40.0,
            "local_moran_cluster": "not_significant",
            "dominant_dimension": "access",
            "priority_quadrant": "lower_priority",
            "geometry": Point(3.1, 0.1).buffer(0.05),
        },
        {
            "cell_id": "d2",
            "city": "Nairobi",
            "country_iso": "KEN",
            "admin2_name": "Delta",
            "admin2_id": "delta",
            "admin2_iso": "KE-DELTA",
            "score": 25.0,
            "population": 30.0,
            "local_moran_cluster": "not_significant",
            "dominant_dimension": "access",
            "priority_quadrant": "lower_priority",
            "geometry": Point(3.2, 0.1).buffer(0.05),
        },
    ]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    input_path = tmp_path / "admin_input.geojson"
    output_path = tmp_path / "admin_summary.geojson"
    gdf.to_file(input_path, driver="GeoJSON", engine="pyogrio")

    summarize_admin_units(
        input_path=input_path,
        output_path=output_path,
        group_col="city",
        country_col="country_iso",
        admin_name_col="admin2_name",
        admin_id_col="admin2_id",
        admin_iso_col="admin2_iso",
        score_col="score",
        population_col="population",
        hotspot_col="local_moran_cluster",
        hotspot_value="high_high",
        dominant_dimension_col="dominant_dimension",
        priority_col="priority_quadrant",
        top_fraction=0.25,
        priority_fraction=0.25,
        min_cells=2,
        min_city_population_share=0.0,
    )

    result = gpd.read_file(output_path).set_index("admin2_name")
    assert result.loc["Alpha", "population_total"] == 180.0
    assert result.loc["Alpha", "hotspot_population_share"] == 1.0
    assert result.loc["Alpha", "hotspot_dominant_dimension"] == "services"
    assert result.loc["Alpha", "district_priority_class"] == "high_priority"
    assert result.loc["Alpha", "district_priority_metric_count"] >= 2
    assert result.loc["Alpha", "district_priority_rank"] == 1.0
    assert result.loc["Beta", "district_priority_class"] == "lower_priority"
