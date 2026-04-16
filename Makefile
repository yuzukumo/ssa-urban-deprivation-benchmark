PYTHONPATH := src
ML_ENV := LD_LIBRARY_PATH=$$CONDA_PREFIX/lib:$$LD_LIBRARY_PATH PYTHONPATH=$(PYTHONPATH)
VIIRS_RASTER ?= data/raw/viirs/VNL_v21_npp_2020_global_vcmslcfg_c202205302300.average_masked.dat.tif
GHSL_BUILT_RASTER ?= data/raw/ghsl/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif

catalog:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark catalog --catalog metadata/data_sources.yaml

catalog-pending:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark catalog --catalog metadata/data_sources.yaml --status pending_confirmation

profile-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark profile examples/mock_grid_cells.csv --output outputs/tables/mock_grid_profile.json

download-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark download-study-assets --study-config configs/studies/mvp_nairobi_dar.yaml --manifest-output outputs/tables/mvp_download_manifest.json

features-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-study-features --study-config configs/studies/mvp_nairobi_dar.yaml --output data/processed/mvp_nairobi_dar_features.gpkg --metadata outputs/tables/mvp_feature_metadata.json

mask-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark apply-analysis-mask --input data/processed/mvp_nairobi_dar_features.gpkg --config configs/studies/mvp_nairobi_dar.yaml --output data/processed/mvp_nairobi_dar_analysis_features.gpkg --metadata outputs/tables/mvp_analysis_mask_metadata.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark apply-analysis-mask --input data/processed/nairobi_features.gpkg --config configs/studies/mvp_nairobi_dar.yaml --output data/processed/nairobi_analysis_features.gpkg --metadata outputs/tables/nairobi_analysis_mask_metadata.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark apply-analysis-mask --input data/processed/dar_es_salaam_features.gpkg --config configs/studies/mvp_nairobi_dar.yaml --output data/processed/dar_es_salaam_analysis_features.gpkg --metadata outputs/tables/dar_es_salaam_analysis_mask_metadata.json

index-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-index --input data/processed/mvp_nairobi_dar_analysis_features.gpkg --config configs/methods/baseline_index_real_v1.yaml --output data/processed/mvp_nairobi_dar_index.gpkg --metadata outputs/tables/mvp_index_metadata.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-index --input data/processed/nairobi_analysis_features.gpkg --config configs/methods/baseline_index_real_v1.yaml --output data/processed/nairobi_index.gpkg --metadata outputs/tables/nairobi_index_metadata.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-index --input data/processed/dar_es_salaam_analysis_features.gpkg --config configs/methods/baseline_index_real_v1.yaml --output data/processed/dar_es_salaam_index.gpkg --metadata outputs/tables/dar_es_salaam_index_metadata.json

cluster-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark cluster-cells --input data/processed/mvp_nairobi_dar_index.gpkg --columns access__score services__score urban_form__score --k 4 --output outputs/tables/mvp_clusters.geojson --summary outputs/tables/mvp_clusters_summary.json

plot-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-quicklook --input data/processed/mvp_nairobi_dar_index.gpkg --score-col deprivation_index_0_100 --id-col cell_id --group-col city --lon-col lon --lat-col lat --output-dir outputs/figures/mvp_index

summarize-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-index --input data/processed/mvp_nairobi_dar_index.gpkg --group-col city --score-col deprivation_index_0_100 --score-summary-output outputs/tables/mvp_city_score_summary.csv --feature-summary-output outputs/tables/mvp_city_feature_medians.csv --scenario 500m

moran-nairobi:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark spatial-autocorr --input data/processed/nairobi_index.gpkg --score-col deprivation_index_0_100 --summary-output outputs/tables/nairobi_moran_summary.json --local-output outputs/tables/nairobi_moran_local.geojson --k 8

moran-dar:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark spatial-autocorr --input data/processed/dar_es_salaam_index.gpkg --score-col deprivation_index_0_100 --summary-output outputs/tables/dar_es_salaam_moran_summary.json --local-output outputs/tables/dar_es_salaam_moran_local.geojson --k 8

cluster-nairobi:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark cluster-cells --input data/processed/nairobi_index.gpkg --columns access__score services__score urban_form__score --k 4 --output outputs/tables/nairobi_clusters.geojson --summary outputs/tables/nairobi_clusters_summary.json

cluster-dar:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark cluster-cells --input data/processed/dar_es_salaam_index.gpkg --columns access__score services__score urban_form__score --k 4 --output outputs/tables/dar_es_salaam_clusters.geojson --summary outputs/tables/dar_es_salaam_clusters_summary.json

robustness-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-pca --input data/processed/mvp_nairobi_dar_index.gpkg --group-col city --output outputs/tables/mvp_pca_alignment.csv --top-fraction 0.1 --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-scatter --input data/processed/mvp_nairobi_dar_index.gpkg --x-col deprivation_index_0_100 --y-col pca1_index_0_100 --group-col city --output outputs/figures/atlas/mvp_pca_vs_composite.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input outputs/tables/mvp_clusters.geojson --group-col city --category-col cluster_id --output outputs/tables/mvp_cluster_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-weighted-categories --input outputs/tables/mvp_clusters.geojson --group-col city --category-col cluster_id --weight-col population --output outputs/tables/mvp_cluster_population_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark export-top-cells --input data/processed/mvp_nairobi_dar_index.gpkg --group-col city --score-col deprivation_index_z --output outputs/tables/mvp_top_deprived_cells.csv --top-n 25 --columns cell_id city country_iso lon lat population deprivation_index_z deprivation_index_0_100 road_distance_m school_distance_m clinic_distance_m amenity_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2 --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-exposure --input data/processed/mvp_nairobi_dar_index.gpkg --group-col city --score-col deprivation_index_0_100 --population-col population --output outputs/tables/mvp_population_exposure.csv --quantiles 0.8 0.9 --scenario 500m

hotspots-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark concat-tables --inputs outputs/tables/nairobi_moran_local.geojson outputs/tables/dar_es_salaam_moran_local.geojson --output outputs/tables/mvp_moran_local.geojson
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input outputs/tables/mvp_moran_local.geojson --group-col city --category-col local_moran_cluster --output outputs/tables/mvp_hotspot_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-weighted-categories --input outputs/tables/mvp_moran_local.geojson --group-col city --category-col local_moran_cluster --weight-col population --output outputs/tables/mvp_hotspot_population_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-hotspot-map --input outputs/tables/mvp_moran_local.geojson --hotspot-col local_moran_cluster --group-col city --output outputs/figures/atlas/mvp_hotspots_by_city.png

interpret-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark annotate-dominant-dimension --input data/processed/mvp_nairobi_dar_index.gpkg --dimension-cols access__score services__score urban_form__score --margin-thresholds 0.25 0.75 --output data/processed/mvp_nairobi_dar_interpretation.gpkg --metadata outputs/tables/mvp_dominant_dimension_metadata.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark annotate-dominant-dimension --input outputs/tables/mvp_moran_local.geojson --dimension-cols access__score services__score urban_form__score --margin-thresholds 0.25 0.75 --output outputs/tables/mvp_moran_with_dominant.geojson
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/mvp_moran_with_dominant.geojson --filter-col local_moran_cluster --filter-value high_high --output outputs/tables/mvp_high_high_hotspots_with_dominant.geojson
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input data/processed/mvp_nairobi_dar_interpretation.gpkg --group-col city --category-col dominant_dimension --output outputs/tables/mvp_dominant_dimension_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-weighted-categories --input data/processed/mvp_nairobi_dar_interpretation.gpkg --group-col city --category-col dominant_dimension --weight-col population --output outputs/tables/mvp_dominant_dimension_population_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input data/processed/mvp_nairobi_dar_interpretation.gpkg --group-col city --category-col dominant_dimension_strength --output outputs/tables/mvp_dominant_strength_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-index --input data/processed/mvp_nairobi_dar_interpretation.gpkg --group-col city --score-col dominant_dimension_margin --score-summary-output outputs/tables/mvp_dominant_dimension_margin_summary.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input outputs/tables/mvp_moran_with_dominant.geojson --group-col city --category-col dominant_dimension --filter-col local_moran_cluster --filter-value high_high --output outputs/tables/mvp_hotspot_dominant_dimension_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-weighted-categories --input outputs/tables/mvp_moran_with_dominant.geojson --group-col city --category-col dominant_dimension --weight-col population --filter-col local_moran_cluster --filter-value high_high --output outputs/tables/mvp_hotspot_dominant_dimension_population_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input outputs/tables/mvp_moran_with_dominant.geojson --group-col city --category-col dominant_dimension_strength --filter-col local_moran_cluster --filter-value high_high --output outputs/tables/mvp_hotspot_dominant_strength_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-index --input outputs/tables/mvp_moran_with_dominant.geojson --group-col city --score-col dominant_dimension_margin --score-summary-output outputs/tables/mvp_hotspot_dominant_dimension_margin_summary.csv --scenario 500m --filter-col local_moran_cluster --filter-value high_high
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-category-map --input data/processed/mvp_nairobi_dar_interpretation.gpkg --category-col dominant_dimension --group-col city --output outputs/figures/atlas/mvp_dominant_dimension_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-category-map --input outputs/tables/mvp_high_high_hotspots_with_dominant.geojson --category-col dominant_dimension --group-col city --output outputs/figures/atlas/mvp_hotspot_dominant_dimension_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_dominant_dimension_share_by_city.csv --x-col city --y-col share --hue-col dominant_dimension --output outputs/figures/atlas/mvp_dominant_dimension_share_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_dominant_dimension_population_share_by_city.csv --x-col city --y-col weight_share --hue-col dominant_dimension --output outputs/figures/atlas/mvp_dominant_dimension_population_share_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_hotspot_dominant_dimension_share_by_city.csv --x-col city --y-col share --hue-col dominant_dimension --output outputs/figures/atlas/mvp_hotspot_dominant_dimension_share_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_hotspot_dominant_dimension_population_share_by_city.csv --x-col city --y-col weight_share --hue-col dominant_dimension --output outputs/figures/atlas/mvp_hotspot_dominant_dimension_population_share_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_hotspot_dominant_strength_share_by_city.csv --x-col city --y-col share --hue-col dominant_dimension_strength --output outputs/figures/atlas/mvp_hotspot_dominant_strength_share_by_city.png

relative-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-index --input data/processed/mvp_nairobi_dar_analysis_features.gpkg --config configs/methods/baseline_index_within_city_v1.yaml --output data/processed/mvp_nairobi_dar_within_city_index.gpkg --metadata outputs/tables/mvp_within_city_index_metadata.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark compare-index-scores --left-input data/processed/mvp_nairobi_dar_index.gpkg --right-input data/processed/mvp_nairobi_dar_within_city_index.gpkg --join-columns cell_id city --group-col city --left-score-col deprivation_index_0_100 --right-score-col deprivation_index_0_100 --left-label pooled_absolute --right-label within_city_relative --output outputs/tables/mvp_absolute_vs_relative_alignment.csv --merged-output data/processed/mvp_absolute_vs_relative_scores.gpkg --top-fraction 0.1 --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-comparison-shift --input data/processed/mvp_absolute_vs_relative_scores.gpkg --group-col city --left-label pooled_absolute --right-label within_city_relative --output outputs/tables/mvp_relative_shift_summary.csv --top-fraction 0.1 --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-score-map --input data/processed/mvp_nairobi_dar_within_city_index.gpkg --score-col deprivation_index_0_100 --group-col city --output outputs/figures/atlas/mvp_within_city_relative_score_map.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-score-map --input data/processed/mvp_absolute_vs_relative_scores.gpkg --score-col within_city_relative_minus_pooled_absolute --group-col city --output outputs/figures/atlas/mvp_relative_minus_absolute_score_map.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-scatter --input data/processed/mvp_absolute_vs_relative_scores.gpkg --x-col pooled_absolute_score --y-col within_city_relative_score --group-col city --output outputs/figures/atlas/mvp_absolute_vs_relative_scatter.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_absolute_vs_relative_alignment.csv --x-col city --y-col top_overlap_share --output outputs/figures/atlas/mvp_absolute_vs_relative_top_overlap.png

hotspot-profile-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-binary-contrast --input outputs/tables/mvp_moran_local.geojson --group-col city --binary-col local_moran_cluster --target-value high_high --value-columns access__score services__score urban_form__score --output outputs/tables/mvp_hotspot_dimension_contrast.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-binary-contrast --input outputs/tables/mvp_moran_local.geojson --group-col city --binary-col local_moran_cluster --target-value high_high --value-columns road_distance_m school_distance_m clinic_distance_m amenity_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2 --output outputs/tables/mvp_hotspot_feature_contrast.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-contrast-heatmap --input outputs/tables/mvp_hotspot_dimension_contrast.csv --feature-col feature --group-col city --value-col standardized_mean_diff --output outputs/figures/atlas/mvp_hotspot_dimension_contrast_heatmap.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-contrast-heatmap --input outputs/tables/mvp_hotspot_feature_contrast.csv --feature-col feature --group-col city --value-col standardized_mean_diff --output outputs/figures/atlas/mvp_hotspot_feature_contrast_heatmap.png

hotspot-mechanism-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-category-feature-profiles --input outputs/tables/mvp_high_high_hotspots_with_dominant.geojson --group-col city --category-col dominant_dimension --value-columns access__score services__score urban_form__score dominant_dimension_margin --output outputs/tables/mvp_hotspot_mechanism_dimension_profiles.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-category-feature-profiles --input outputs/tables/mvp_high_high_hotspots_with_dominant.geojson --group-col city --category-col dominant_dimension --value-columns road_distance_m school_distance_m clinic_distance_m amenity_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2 --output outputs/tables/mvp_hotspot_mechanism_feature_profiles.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-faceted-heatmap --input outputs/tables/mvp_hotspot_mechanism_dimension_profiles.csv --facet-col city --x-col dominant_dimension --y-col feature --value-col standardized_profile_diff --output outputs/figures/atlas/mvp_hotspot_mechanism_dimension_profiles.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-faceted-heatmap --input outputs/tables/mvp_hotspot_mechanism_feature_profiles.csv --facet-col city --x-col dominant_dimension --y-col feature --value-col standardized_profile_diff --output outputs/figures/atlas/mvp_hotspot_mechanism_feature_profiles.png

priority-hotspots-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark compare-index-scores --left-input outputs/tables/mvp_moran_with_dominant.geojson --right-input data/processed/mvp_nairobi_dar_within_city_index.gpkg --join-columns cell_id city --group-col city --left-score-col deprivation_index_0_100 --right-score-col deprivation_index_0_100 --left-label pooled_absolute --right-label within_city_relative --output outputs/tables/mvp_moran_absolute_relative_alignment.csv --merged-output outputs/tables/mvp_moran_with_absolute_relative.geojson --top-fraction 0.1 --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark annotate-priority-quadrants --input outputs/tables/mvp_moran_with_absolute_relative.geojson --absolute-score-col pooled_absolute_score --relative-score-col within_city_relative_score --group-col city --absolute-top-fraction 0.1 --relative-top-fraction 0.1 --output outputs/tables/mvp_moran_priority_quadrants.geojson --metadata outputs/tables/mvp_priority_quadrants_metadata.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/mvp_moran_priority_quadrants.geojson --filter-col local_moran_cluster --filter-value high_high --output outputs/tables/mvp_high_high_priority_quadrants.geojson
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input outputs/tables/mvp_high_high_priority_quadrants.geojson --group-col city --category-col priority_quadrant --output outputs/tables/mvp_hotspot_priority_quadrant_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-weighted-categories --input outputs/tables/mvp_high_high_priority_quadrants.geojson --group-col city --category-col priority_quadrant --weight-col population --output outputs/tables/mvp_hotspot_priority_quadrant_population_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-category-feature-profiles --input outputs/tables/mvp_high_high_priority_quadrants.geojson --group-col city --category-col priority_quadrant --value-columns access__score services__score urban_form__score dominant_dimension_margin pooled_absolute_score within_city_relative_score within_city_relative_minus_pooled_absolute --output outputs/tables/mvp_hotspot_priority_dimension_profiles.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-category-feature-profiles --input outputs/tables/mvp_high_high_priority_quadrants.geojson --group-col city --category-col priority_quadrant --value-columns road_distance_m school_distance_m clinic_distance_m amenity_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2 --output outputs/tables/mvp_hotspot_priority_feature_profiles.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-category-map --input outputs/tables/mvp_high_high_priority_quadrants.geojson --category-col priority_quadrant --group-col city --output outputs/figures/atlas/mvp_hotspot_priority_quadrants_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_hotspot_priority_quadrant_share_by_city.csv --x-col city --y-col share --hue-col priority_quadrant --output outputs/figures/atlas/mvp_hotspot_priority_quadrant_share_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_hotspot_priority_quadrant_population_share_by_city.csv --x-col city --y-col weight_share --hue-col priority_quadrant --output outputs/figures/atlas/mvp_hotspot_priority_quadrant_population_share_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-faceted-heatmap --input outputs/tables/mvp_hotspot_priority_dimension_profiles.csv --facet-col city --x-col priority_quadrant --y-col feature --value-col standardized_profile_diff --output outputs/figures/atlas/mvp_hotspot_priority_dimension_profiles.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-faceted-heatmap --input outputs/tables/mvp_hotspot_priority_feature_profiles.csv --facet-col city --x-col priority_quadrant --y-col feature --value-col standardized_profile_diff --output outputs/figures/atlas/mvp_hotspot_priority_feature_profiles.png

admin-atlas-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark attach-admin-units --input outputs/tables/mvp_moran_priority_quadrants.geojson --output outputs/tables/mvp_moran_priority_admin2.geojson --boundary-dir data/raw/boundaries/geoboundaries --admin-level 2 --country-col country_iso --admin-prefix admin2 --metadata outputs/tables/mvp_admin2_attachment_metadata.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-admin-units --input outputs/tables/mvp_moran_priority_admin2.geojson --output outputs/tables/mvp_admin2_summary.geojson --metadata outputs/tables/mvp_admin2_summary_metadata.json --group-col city --country-col country_iso --admin-name-col admin2_name --admin-id-col admin2_id --admin-iso-col admin2_iso --score-col pooled_absolute_score --population-col population --hotspot-col local_moran_cluster --hotspot-value high_high --dominant-dimension-col dominant_dimension --priority-col priority_quadrant --top-fraction 0.1 --priority-fraction 0.25 --min-cells 10 --min-city-population-share 0.01
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input outputs/tables/mvp_admin2_summary.geojson --group-col city --category-col district_priority_class --output outputs/tables/mvp_admin2_priority_class_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-weighted-categories --input outputs/tables/mvp_admin2_summary.geojson --group-col city --category-col district_priority_class --weight-col population_total --output outputs/tables/mvp_admin2_priority_class_population_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-score-map --input outputs/tables/mvp_admin2_summary.geojson --score-col population_weighted_mean_score --group-col city --output outputs/figures/atlas/mvp_admin2_weighted_mean_score.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-score-map --input outputs/tables/mvp_admin2_summary.geojson --score-col hotspot_population_share --group-col city --output outputs/figures/atlas/mvp_admin2_hotspot_population_share.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-category-map --input outputs/tables/mvp_admin2_summary.geojson --category-col district_priority_class --group-col city --output outputs/figures/atlas/mvp_admin2_priority_class.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_admin2_priority_class_population_share_by_city.csv --x-col city --y-col weight_share --hue-col district_priority_class --output outputs/figures/atlas/mvp_admin2_priority_class_population_share_by_city.png

typology-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark add-composite-column --input outputs/tables/mvp_high_high_priority_quadrants.geojson --source-columns priority_quadrant dominant_dimension --output-column hotspot_typology --output outputs/tables/mvp_high_high_hotspot_typologies.geojson
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-categories --input outputs/tables/mvp_high_high_hotspot_typologies.geojson --group-col city --category-col hotspot_typology --output outputs/tables/mvp_hotspot_typology_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-weighted-categories --input outputs/tables/mvp_high_high_hotspot_typologies.geojson --group-col city --category-col hotspot_typology --weight-col population --output outputs/tables/mvp_hotspot_typology_population_share_by_city.csv --scenario 500m
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-category-map --input outputs/tables/mvp_high_high_hotspot_typologies.geojson --category-col hotspot_typology --group-col city --output outputs/figures/atlas/mvp_hotspot_typologies_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_hotspot_typology_share_by_city.csv --x-col city --y-col share --hue-col hotspot_typology --output outputs/figures/atlas/mvp_hotspot_typology_share_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_hotspot_typology_population_share_by_city.csv --x-col city --y-col weight_share --hue-col hotspot_typology --output outputs/figures/atlas/mvp_hotspot_typology_population_share_by_city.png

atlas-nairobi:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-score-map --input data/processed/nairobi_index.gpkg --score-col deprivation_index_0_100 --output outputs/figures/atlas/nairobi_score_map.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-hotspot-map --input outputs/tables/nairobi_moran_local.geojson --hotspot-col local_moran_cluster --output outputs/figures/atlas/nairobi_hotspots.png

atlas-dar:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-score-map --input data/processed/dar_es_salaam_index.gpkg --score-col deprivation_index_0_100 --output outputs/figures/atlas/dar_es_salaam_score_map.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-hotspot-map --input outputs/tables/dar_es_salaam_moran_local.geojson --hotspot-col local_moran_cluster --output outputs/figures/atlas/dar_es_salaam_hotspots.png

atlas-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-score-map --input data/processed/mvp_nairobi_dar_index.gpkg --score-col deprivation_index_0_100 --group-col city --output outputs/figures/atlas/mvp_score_map_by_city.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-category-map --input outputs/tables/mvp_clusters.geojson --category-col cluster_id --group-col city --output outputs/figures/atlas/mvp_clusters_by_city.png

results-all: cluster-nairobi cluster-dar cluster-mvp moran-nairobi moran-dar atlas-nairobi atlas-dar atlas-mvp summarize-mvp robustness-mvp hotspots-mvp interpret-mvp relative-mvp hotspot-profile-mvp hotspot-mechanism-mvp priority-hotspots-mvp admin-atlas-mvp typology-mvp

mvp-all: download-mvp features-mvp mask-mvp index-mvp plot-mvp summarize-mvp

build-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-index --input examples/mock_grid_cells.csv --config configs/methods/baseline_index_v1.yaml --output outputs/tables/mock_grid_index.csv --metadata outputs/tables/mock_grid_index_metadata.json

cluster-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark cluster-cells --input outputs/tables/mock_grid_index.csv --columns access__score services__score urban_form__score economic_proxy__score --k 3 --output outputs/tables/mock_grid_clusters.csv --summary outputs/tables/mock_grid_clusters_summary.json

plot-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-quicklook --input outputs/tables/mock_grid_index.csv --score-col deprivation_index_0_100 --id-col cell_id --group-col city --lon-col lon --lat-col lat --output-dir outputs/figures/mock_grid

sample-all: profile-sample build-sample cluster-sample plot-sample

inequality-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-inequality --input data/processed/mvp_nairobi_dar_index.gpkg --group-col city --score-col deprivation_index_0_100 --population-col population --output outputs/tables/mvp_inequality_summary.csv --scenario 500m

features-mvp-1km:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-study-features --study-config configs/studies/mvp_nairobi_dar_1km.yaml --output data/processed/mvp_nairobi_dar_1km_features.gpkg --metadata outputs/tables/mvp_1km_feature_metadata.json

mask-mvp-1km:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark apply-analysis-mask --input data/processed/mvp_nairobi_dar_1km_features.gpkg --config configs/studies/mvp_nairobi_dar_1km.yaml --output data/processed/mvp_nairobi_dar_1km_analysis_features.gpkg --metadata outputs/tables/mvp_1km_analysis_mask_metadata.json

index-mvp-1km:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-index --input data/processed/mvp_nairobi_dar_1km_analysis_features.gpkg --config configs/methods/baseline_index_real_v1.yaml --output data/processed/mvp_nairobi_dar_1km_index.gpkg --metadata outputs/tables/mvp_1km_index_metadata.json

sensitivity-mvp-1km:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-index --input data/processed/mvp_nairobi_dar_1km_index.gpkg --group-col city --score-col deprivation_index_0_100 --score-summary-output outputs/tables/mvp_1km_city_score_summary.csv --feature-summary-output outputs/tables/mvp_1km_city_feature_medians.csv --scenario 1km
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-exposure --input data/processed/mvp_nairobi_dar_1km_index.gpkg --group-col city --score-col deprivation_index_0_100 --population-col population --output outputs/tables/mvp_1km_population_exposure.csv --quantiles 0.8 0.9 --scenario 1km
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-inequality --input data/processed/mvp_nairobi_dar_1km_index.gpkg --group-col city --score-col deprivation_index_0_100 --population-col population --output outputs/tables/mvp_1km_inequality_summary.csv --scenario 1km

compare-sensitivity:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark concat-tables --inputs outputs/tables/mvp_city_score_summary.csv outputs/tables/mvp_1km_city_score_summary.csv --output outputs/tables/mvp_grid_size_score_summary.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark concat-tables --inputs outputs/tables/mvp_population_exposure.csv outputs/tables/mvp_1km_population_exposure.csv --output outputs/tables/mvp_grid_size_population_exposure.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark concat-tables --inputs outputs/tables/mvp_inequality_summary.csv outputs/tables/mvp_1km_inequality_summary.csv --output outputs/tables/mvp_grid_size_inequality_summary.csv

plot-sensitivity:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_grid_size_score_summary.csv --x-col city --y-col mean --hue-col scenario --output outputs/figures/atlas/mvp_grid_size_mean_score.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_grid_size_population_exposure.csv --x-col city --y-col population_share_at_or_above_q90 --hue-col scenario --output outputs/figures/atlas/mvp_grid_size_q90_exposure.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/mvp_grid_size_inequality_summary.csv --x-col city --y-col score_gini_population_weighted --hue-col scenario --output outputs/figures/atlas/mvp_grid_size_weighted_gini.png

findings-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-core-findings --config configs/results/mvp_core_findings_v1.yaml --output outputs/tables/mvp_core_findings.json

paper-figures-mvp:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark stage-figure-set --config configs/figure_sets/mvp_core_v1.yaml --output-dir outputs/figures/paper_core --manifest-output outputs/tables/mvp_core_figure_set_manifest.json

paper-artifacts-mvp: findings-mvp paper-figures-mvp

download-rwi-mvp:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark download-rwi --countries KEN TZA --output-dir data/raw/rwi --manifest-output outputs/tables/mvp_rwi_download_manifest.json

rwi-targets-mvp:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark build-rwi-grid-targets --input data/processed/mvp_nairobi_dar_index.gpkg --output data/processed/mvp_nairobi_dar_rwi_targets.gpkg --rwi-dir data/raw/rwi --metadata outputs/tables/mvp_rwi_target_metadata.json --neighbors 4 --max-distance-m 4000 --low-wealth-quantile 0.2

multimodal-dataset-mvp:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark build-multimodal-patch-dataset --input data/processed/mvp_nairobi_dar_rwi_targets.gpkg --study-config configs/studies/mvp_nairobi_dar.yaml --output data/processed/mvp_rwi_multimodal_dataset.npz --metadata outputs/tables/mvp_rwi_multimodal_dataset_metadata.json --patch-size 64 --context-m 1500 --feature-columns population road_distance_m school_distance_m clinic_distance_m amenity_count_1km service_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2

pretrain-multimodal-mvp:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark pretrain-patch-autoencoder --dataset data/processed/mvp_rwi_multimodal_dataset.npz --checkpoint outputs/models/mvp_patch_autoencoder.pt --metrics outputs/tables/ml/mvp_patch_autoencoder_metrics.json --epochs 8 --batch-size 512 --learning-rate 0.001 --random-state 42

benchmark-multimodal-mvp:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark run-multimodal-rwi-benchmark --input data/processed/mvp_nairobi_dar_rwi_targets.gpkg --dataset data/processed/mvp_rwi_multimodal_dataset.npz --metrics-output outputs/tables/ml/mvp_rwi_benchmark_metrics.csv --predictions-output outputs/tables/ml/mvp_rwi_benchmark_predictions.csv --metadata outputs/tables/ml/mvp_rwi_benchmark_metadata.json --pretrained-encoder outputs/models/mvp_patch_autoencoder.pt --feature-columns population road_distance_m school_distance_m clinic_distance_m amenity_count_1km service_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2 --models atlas_linear_baseline xgboost_tabular cnn_image cnn_fusion resnet_fusion_pretrained graph_fusion --graph-k 8 --epochs 10 --batch-size 256 --learning-rate 0.001 --random-state 42

benchmark-summary-mvp:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark summarize-multimodal-benchmark --input outputs/tables/ml/mvp_rwi_benchmark_metrics.csv --output outputs/tables/ml/mvp_rwi_benchmark_summary.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/mvp_rwi_benchmark_metrics.csv --filter-col metric --filter-value rmse --output outputs/tables/ml/mvp_rwi_benchmark_rmse.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/mvp_rwi_benchmark_metrics.csv --filter-col metric --filter-value spearman_corr --output outputs/tables/ml/mvp_rwi_benchmark_spearman.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/mvp_rwi_benchmark_metrics.csv --filter-col metric --filter-value roc_auc --output outputs/tables/ml/mvp_rwi_benchmark_roc_auc.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/mvp_rwi_benchmark_metrics.csv --filter-col metric --filter-value average_precision --output outputs/tables/ml/mvp_rwi_benchmark_average_precision.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/mvp_rwi_benchmark_metrics.csv --filter-col metric --filter-value balanced_accuracy --output outputs/tables/ml/mvp_rwi_benchmark_balanced_accuracy.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/mvp_rwi_benchmark_rmse.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/mvp_rwi_benchmark_rmse.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/mvp_rwi_benchmark_spearman.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/mvp_rwi_benchmark_spearman.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/mvp_rwi_benchmark_roc_auc.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/mvp_rwi_benchmark_roc_auc.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/mvp_rwi_benchmark_average_precision.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/mvp_rwi_benchmark_average_precision.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/mvp_rwi_benchmark_balanced_accuracy.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/mvp_rwi_benchmark_balanced_accuracy.png

ml-mvp: download-rwi-mvp rwi-targets-mvp multimodal-dataset-mvp pretrain-multimodal-mvp benchmark-multimodal-mvp benchmark-summary-mvp

download-core4:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark download-study-assets --study-config configs/studies/ssa_multicity_core4.yaml --manifest-output outputs/tables/core4_download_manifest.json

features-core4:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-study-features --study-config configs/studies/ssa_multicity_core4.yaml --output data/processed/core4_features.gpkg --metadata outputs/tables/core4_feature_metadata.json

mask-core4:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark apply-analysis-mask --input data/processed/core4_features.gpkg --config configs/studies/ssa_multicity_core4.yaml --output data/processed/core4_analysis_features.gpkg --metadata outputs/tables/core4_analysis_mask_metadata.json

index-core4:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-index --input data/processed/core4_analysis_features.gpkg --config configs/methods/baseline_index_real_v1.yaml --output data/processed/core4_index.gpkg --metadata outputs/tables/core4_index_metadata.json

download-rwi-core4:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark download-rwi --countries KEN TZA UGA GHA --output-dir data/raw/rwi --manifest-output outputs/tables/core4_rwi_download_manifest.json

rwi-targets-core4:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark build-rwi-grid-targets --input data/processed/core4_index.gpkg --output data/processed/core4_rwi_targets.gpkg --rwi-dir data/raw/rwi --metadata outputs/tables/core4_rwi_target_metadata.json --neighbors 4 --max-distance-m 4000 --low-wealth-quantile 0.2

multimodal-dataset-core4:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark build-multimodal-patch-dataset --input data/processed/core4_rwi_targets.gpkg --study-config configs/studies/ssa_multicity_core4.yaml --output data/processed/core4_multimodal_dataset.npz --metadata outputs/tables/core4_multimodal_dataset_metadata.json --patch-size 64 --context-m 1500 --feature-columns population road_distance_m school_distance_m clinic_distance_m amenity_count_1km service_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2

pretrain-multimodal-core4:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark pretrain-patch-autoencoder --dataset data/processed/core4_multimodal_dataset.npz --checkpoint outputs/models/core4_patch_autoencoder.pt --metrics outputs/tables/ml/core4_patch_autoencoder_metrics.json --epochs 8 --batch-size 512 --learning-rate 0.001 --random-state 42

benchmark-multimodal-core4:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark run-multimodal-rwi-benchmark --input data/processed/core4_rwi_targets.gpkg --dataset data/processed/core4_multimodal_dataset.npz --metrics-output outputs/tables/ml/core4_rwi_benchmark_metrics.csv --predictions-output outputs/tables/ml/core4_rwi_benchmark_predictions.csv --metadata outputs/tables/ml/core4_rwi_benchmark_metadata.json --pretrained-encoder outputs/models/core4_patch_autoencoder.pt --feature-columns population road_distance_m school_distance_m clinic_distance_m amenity_count_1km service_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2 --models atlas_linear_baseline xgboost_tabular cnn_image cnn_fusion resnet_fusion_pretrained graph_fusion --protocol-strategy leave_one_city_out --graph-k 8 --epochs 8 --batch-size 256 --learning-rate 0.001 --random-state 42

benchmark-summary-core4:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark summarize-multimodal-benchmark --input outputs/tables/ml/core4_rwi_benchmark_metrics.csv --output outputs/tables/ml/core4_rwi_benchmark_summary.json
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark build-benchmark-findings --metrics-input outputs/tables/ml/core4_rwi_benchmark_metrics.csv --summary-input outputs/tables/ml/core4_rwi_benchmark_summary.json --output outputs/tables/ml/core4_rwi_benchmark_findings.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core4_rwi_benchmark_metrics.csv --filter-col metric --filter-value rmse --output outputs/tables/ml/core4_rwi_benchmark_rmse.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core4_rwi_benchmark_metrics.csv --filter-col metric --filter-value spearman_corr --output outputs/tables/ml/core4_rwi_benchmark_spearman.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core4_rwi_benchmark_metrics.csv --filter-col metric --filter-value roc_auc --output outputs/tables/ml/core4_rwi_benchmark_roc_auc.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core4_rwi_benchmark_metrics.csv --filter-col metric --filter-value average_precision --output outputs/tables/ml/core4_rwi_benchmark_average_precision.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core4_rwi_benchmark_metrics.csv --filter-col metric --filter-value balanced_accuracy --output outputs/tables/ml/core4_rwi_benchmark_balanced_accuracy.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core4_rwi_benchmark_rmse.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core4_rwi_benchmark_rmse.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core4_rwi_benchmark_spearman.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core4_rwi_benchmark_spearman.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core4_rwi_benchmark_roc_auc.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core4_rwi_benchmark_roc_auc.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core4_rwi_benchmark_average_precision.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core4_rwi_benchmark_average_precision.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core4_rwi_benchmark_balanced_accuracy.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core4_rwi_benchmark_balanced_accuracy.png

ml-core4: download-core4 features-core4 mask-core4 index-core4 download-rwi-core4 rwi-targets-core4 multimodal-dataset-core4 pretrain-multimodal-core4 benchmark-multimodal-core4 benchmark-summary-core4

download-core6:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark download-study-assets --study-config configs/studies/ssa_multicity_core6.yaml --manifest-output outputs/tables/core6_download_manifest.json

features-core6:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-study-features --study-config configs/studies/ssa_multicity_core6.yaml --output data/processed/core6_features.gpkg --metadata outputs/tables/core6_feature_metadata.json

mask-core6:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark apply-analysis-mask --input data/processed/core6_features.gpkg --config configs/studies/ssa_multicity_core6.yaml --output data/processed/core6_analysis_features.gpkg --metadata outputs/tables/core6_analysis_mask_metadata.json

index-core6:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-index --input data/processed/core6_analysis_features.gpkg --config configs/methods/baseline_index_real_v1.yaml --output data/processed/core6_index.gpkg --metadata outputs/tables/core6_index_metadata.json

download-rwi-core6:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark download-rwi --countries KEN TZA UGA GHA NGA ETH --output-dir data/raw/rwi --manifest-output outputs/tables/core6_rwi_download_manifest.json

rwi-targets-core6:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark build-rwi-grid-targets --input data/processed/core6_index.gpkg --output data/processed/core6_rwi_targets.gpkg --rwi-dir data/raw/rwi --metadata outputs/tables/core6_rwi_target_metadata.json --neighbors 4 --max-distance-m 4000 --low-wealth-quantile 0.2

multimodal-dataset-core6:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark build-multimodal-patch-dataset --input data/processed/core6_rwi_targets.gpkg --study-config configs/studies/ssa_multicity_core6.yaml --output data/processed/core6_multimodal_dataset.npz --metadata outputs/tables/core6_multimodal_dataset_metadata.json --patch-size 64 --context-m 1500 --feature-columns population road_distance_m school_distance_m clinic_distance_m amenity_count_1km service_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2

pretrain-multimodal-core6:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark pretrain-patch-autoencoder --dataset data/processed/core6_multimodal_dataset.npz --checkpoint outputs/models/core6_patch_autoencoder.pt --metrics outputs/tables/ml/core6_patch_autoencoder_metrics.json --epochs 8 --batch-size 512 --learning-rate 0.001 --random-state 42

benchmark-multimodal-core6:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark run-multimodal-rwi-benchmark --input data/processed/core6_rwi_targets.gpkg --dataset data/processed/core6_multimodal_dataset.npz --metrics-output outputs/tables/ml/core6_rwi_benchmark_metrics.csv --predictions-output outputs/tables/ml/core6_rwi_benchmark_predictions.csv --metadata outputs/tables/ml/core6_rwi_benchmark_metadata.json --pretrained-encoder outputs/models/core6_patch_autoencoder.pt --feature-columns population road_distance_m school_distance_m clinic_distance_m amenity_count_1km service_count_1km population_per_service building_coverage_ratio open_space_share intersection_density_km2 --models atlas_linear_baseline xgboost_tabular cnn_image cnn_fusion resnet_fusion_pretrained graph_fusion --protocol-strategy leave_one_city_out --graph-k 8 --epochs 8 --batch-size 256 --learning-rate 0.001 --random-state 42

benchmark-summary-core6:
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark summarize-multimodal-benchmark --input outputs/tables/ml/core6_rwi_benchmark_metrics.csv --output outputs/tables/ml/core6_rwi_benchmark_summary.json
	$(ML_ENV) python3 -m ssa_urban_deprivation_benchmark build-benchmark-findings --metrics-input outputs/tables/ml/core6_rwi_benchmark_metrics.csv --summary-input outputs/tables/ml/core6_rwi_benchmark_summary.json --output outputs/tables/ml/core6_rwi_benchmark_findings.json
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core6_rwi_benchmark_metrics.csv --filter-col metric --filter-value rmse --output outputs/tables/ml/core6_rwi_benchmark_rmse.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core6_rwi_benchmark_metrics.csv --filter-col metric --filter-value spearman_corr --output outputs/tables/ml/core6_rwi_benchmark_spearman.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core6_rwi_benchmark_metrics.csv --filter-col metric --filter-value roc_auc --output outputs/tables/ml/core6_rwi_benchmark_roc_auc.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core6_rwi_benchmark_metrics.csv --filter-col metric --filter-value average_precision --output outputs/tables/ml/core6_rwi_benchmark_average_precision.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark filter-table --input outputs/tables/ml/core6_rwi_benchmark_metrics.csv --filter-col metric --filter-value balanced_accuracy --output outputs/tables/ml/core6_rwi_benchmark_balanced_accuracy.csv
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core6_rwi_benchmark_rmse.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core6_rwi_benchmark_rmse.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core6_rwi_benchmark_spearman.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core6_rwi_benchmark_spearman.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core6_rwi_benchmark_roc_auc.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core6_rwi_benchmark_roc_auc.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core6_rwi_benchmark_average_precision.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core6_rwi_benchmark_average_precision.png
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark plot-summary-bars --input outputs/tables/ml/core6_rwi_benchmark_balanced_accuracy.csv --x-col protocol --y-col value --hue-col model --output outputs/figures/ml/core6_rwi_benchmark_balanced_accuracy.png

attach-viirs-core6:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark attach-external-raster --input data/processed/core6_rwi_targets.gpkg --raster $(VIIRS_RASTER) --output data/processed/core6_viirs_validation.gpkg --metadata outputs/tables/core6_viirs_validation_metadata.json --prefix viirs --stats mean --all-touched

summarize-viirs-core6:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-external-validation --input data/processed/core6_viirs_validation.gpkg --group-col city --external-col viirs_mean --score-columns deprivation_index_0_100 rwi_deprivation_proxy_0_100 --output outputs/tables/core6_viirs_validation_summary.csv --top-fraction 0.1 --expected-relation negative
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-validation-findings --input outputs/tables/core6_viirs_validation_summary.csv --output outputs/tables/core6_viirs_validation_findings.json

validate-core6-viirs: attach-viirs-core6 summarize-viirs-core6

download-ghsl-built:
	mkdir -p data/raw/ghsl
	cd data/raw/ghsl && wget -nv -O GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip
	cd data/raw/ghsl && unzip -o GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip

attach-ghsl-built-core6:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark attach-external-raster --input data/processed/core6_rwi_targets.gpkg --raster $(GHSL_BUILT_RASTER) --output data/processed/core6_ghsl_built_validation.gpkg --metadata outputs/tables/core6_ghsl_built_validation_metadata.json --prefix ghs_built --stats mean --all-touched

summarize-ghsl-built-core6:
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark summarize-external-validation --input data/processed/core6_ghsl_built_validation.gpkg --group-col city --external-col ghs_built_mean --score-columns building_coverage_ratio urban_form__score deprivation_index_0_100 --output outputs/tables/core6_ghsl_built_validation_summary.csv --top-fraction 0.1 --expected-relation positive
	PYTHONPATH=$(PYTHONPATH) python3 -m ssa_urban_deprivation_benchmark build-validation-findings --input outputs/tables/core6_ghsl_built_validation_summary.csv --output outputs/tables/core6_ghsl_built_validation_findings.json

validate-core6-ghsl-built: attach-ghsl-built-core6 summarize-ghsl-built-core6

ml-core6: download-core6 features-core6 mask-core6 index-core6 download-rwi-core6 rwi-targets-core6 multimodal-dataset-core6 pretrain-multimodal-core6 benchmark-multimodal-core6 benchmark-summary-core6

full-project: mvp-all results-all inequality-mvp features-mvp-1km mask-mvp-1km index-mvp-1km sensitivity-mvp-1km compare-sensitivity plot-sensitivity paper-artifacts-mvp
