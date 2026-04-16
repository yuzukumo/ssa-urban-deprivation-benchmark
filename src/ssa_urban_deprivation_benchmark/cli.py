import argparse
from pathlib import Path
from typing import Optional

from ssa_urban_deprivation_benchmark.admin import attach_admin_units
from ssa_urban_deprivation_benchmark.admin import summarize_admin_units
from ssa_urban_deprivation_benchmark.artifacts import build_core_findings_artifact
from ssa_urban_deprivation_benchmark.artifacts import stage_figure_set
from ssa_urban_deprivation_benchmark.catalog import filter_catalog
from ssa_urban_deprivation_benchmark.catalog import format_catalog_table
from ssa_urban_deprivation_benchmark.catalog import load_catalog
from ssa_urban_deprivation_benchmark.clustering import run_clustering
from ssa_urban_deprivation_benchmark.dataset_profile import build_profile
from ssa_urban_deprivation_benchmark.downloaders import download_study_assets
from ssa_urban_deprivation_benchmark.feature_pipeline import build_study_feature_tables
from ssa_urban_deprivation_benchmark.indexing import run_index_build
from ssa_urban_deprivation_benchmark.interpretation import annotate_dominant_dimension
from ssa_urban_deprivation_benchmark.interpretation import annotate_priority_quadrants
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.masking import run_analysis_mask
from ssa_urban_deprivation_benchmark.multimodal_ml import build_multimodal_patch_dataset
from ssa_urban_deprivation_benchmark.multimodal_ml import build_benchmark_findings_artifact
from ssa_urban_deprivation_benchmark.multimodal_ml import pretrain_patch_autoencoder
from ssa_urban_deprivation_benchmark.multimodal_ml import run_multimodal_rwi_benchmark
from ssa_urban_deprivation_benchmark.multimodal_ml import summarize_multimodal_benchmark
from ssa_urban_deprivation_benchmark.reporting import summarize_index
from ssa_urban_deprivation_benchmark.reporting import summarize_category_shares
from ssa_urban_deprivation_benchmark.reporting import compare_index_scores
from ssa_urban_deprivation_benchmark.reporting import export_top_cells
from ssa_urban_deprivation_benchmark.reporting import summarize_binary_contrast
from ssa_urban_deprivation_benchmark.reporting import summarize_category_feature_profiles
from ssa_urban_deprivation_benchmark.reporting import summarize_comparison_shift
from ssa_urban_deprivation_benchmark.reporting import summarize_inequality
from ssa_urban_deprivation_benchmark.reporting import summarize_pca_alignment
from ssa_urban_deprivation_benchmark.reporting import summarize_population_exposure
from ssa_urban_deprivation_benchmark.reporting import summarize_weighted_categories
from ssa_urban_deprivation_benchmark.spatial import run_spatial_autocorrelation
from ssa_urban_deprivation_benchmark.table_ops import add_composite_column
from ssa_urban_deprivation_benchmark.table_ops import concat_tables
from ssa_urban_deprivation_benchmark.table_ops import filter_table
from ssa_urban_deprivation_benchmark.validation import attach_external_raster_signal
from ssa_urban_deprivation_benchmark.validation import build_validation_findings_artifact
from ssa_urban_deprivation_benchmark.validation import summarize_external_validation
from ssa_urban_deprivation_benchmark.weak_targets import build_rwi_grid_targets
from ssa_urban_deprivation_benchmark.weak_targets import download_rwi_country_files
from ssa_urban_deprivation_benchmark.viz import create_quicklook_outputs
from ssa_urban_deprivation_benchmark.viz import plot_category_map
from ssa_urban_deprivation_benchmark.viz import plot_contrast_heatmap
from ssa_urban_deprivation_benchmark.viz import plot_faceted_heatmap
from ssa_urban_deprivation_benchmark.viz import plot_hotspot_map
from ssa_urban_deprivation_benchmark.viz import plot_scatter_by_group
from ssa_urban_deprivation_benchmark.viz import plot_summary_bars
from ssa_urban_deprivation_benchmark.viz import plot_score_map


def _maybe_bool_from_flag(flag: bool) -> Optional[bool]:
    return True if flag else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for the SSA Urban Deprivation Benchmark research pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    catalog_parser = subparsers.add_parser("catalog", help="List candidate data sources.")
    catalog_parser.add_argument("--catalog", default="metadata/data_sources.yaml")
    catalog_parser.add_argument("--route", default=None)
    catalog_parser.add_argument("--status", default=None)
    catalog_parser.add_argument(
        "--approval-required",
        action="store_true",
        help="Filter to sources that require approval or registration.",
    )

    profile_parser = subparsers.add_parser("profile", help="Profile a local dataset.")
    profile_parser.add_argument("input_path")
    profile_parser.add_argument("--output", default=None)

    concat_parser = subparsers.add_parser(
        "concat-tables",
        help="Concatenate multiple tabular or geospatial files.",
    )
    concat_parser.add_argument("--inputs", nargs="+", required=True)
    concat_parser.add_argument("--output", required=True)

    composite_parser = subparsers.add_parser(
        "add-composite-column",
        help="Create a composite categorical column from multiple source columns.",
    )
    composite_parser.add_argument("--input", required=True)
    composite_parser.add_argument("--source-columns", nargs="+", required=True)
    composite_parser.add_argument("--output-column", required=True)
    composite_parser.add_argument("--output", required=True)
    composite_parser.add_argument("--separator", default="__")

    filter_parser = subparsers.add_parser(
        "filter-table",
        help="Filter a table to rows matching a single value.",
    )
    filter_parser.add_argument("--input", required=True)
    filter_parser.add_argument("--filter-col", required=True)
    filter_parser.add_argument("--filter-value", required=True)
    filter_parser.add_argument("--output", required=True)

    mask_parser = subparsers.add_parser(
        "apply-analysis-mask",
        help="Filter a table using the analysis_mask config.",
    )
    mask_parser.add_argument("--input", required=True)
    mask_parser.add_argument("--config", required=True)
    mask_parser.add_argument("--output", required=True)
    mask_parser.add_argument("--metadata", required=True)

    attach_admin_parser = subparsers.add_parser(
        "attach-admin-units",
        help="Attach admin-unit labels to a geospatial file using geoBoundaries polygons.",
    )
    attach_admin_parser.add_argument("--input", required=True)
    attach_admin_parser.add_argument("--output", required=True)
    attach_admin_parser.add_argument("--boundary-dir", default="data/raw/boundaries/geoboundaries")
    attach_admin_parser.add_argument("--admin-level", type=int, default=2)
    attach_admin_parser.add_argument("--country-col", default="country_iso")
    attach_admin_parser.add_argument("--admin-prefix", default="admin2")
    attach_admin_parser.add_argument("--metadata", default=None)

    rwi_download_parser = subparsers.add_parser(
        "download-rwi",
        help="Download Relative Wealth Index country CSVs from HDX.",
    )
    rwi_download_parser.add_argument("--countries", nargs="+", required=True)
    rwi_download_parser.add_argument("--output-dir", required=True)
    rwi_download_parser.add_argument("--manifest-output", default=None)

    rwi_targets_parser = subparsers.add_parser(
        "build-rwi-grid-targets",
        help="Attach smoothed RWI weak-supervision targets to grid cells.",
    )
    rwi_targets_parser.add_argument("--input", required=True)
    rwi_targets_parser.add_argument("--output", required=True)
    rwi_targets_parser.add_argument("--rwi-dir", required=True)
    rwi_targets_parser.add_argument("--metadata", default=None)
    rwi_targets_parser.add_argument("--group-col", default="city")
    rwi_targets_parser.add_argument("--country-col", default="country_iso")
    rwi_targets_parser.add_argument("--score-col", default="deprivation_index_0_100")
    rwi_targets_parser.add_argument("--neighbors", type=int, default=4)
    rwi_targets_parser.add_argument("--max-distance-m", type=float, default=4000.0)
    rwi_targets_parser.add_argument("--low-wealth-quantile", type=float, default=0.2)

    download_parser = subparsers.add_parser(
        "download-study-assets",
        help="Download approved raw assets for a study configuration.",
    )
    download_parser.add_argument("--study-config", required=True)
    download_parser.add_argument("--manifest-output", required=True)
    download_parser.add_argument("--overwrite", action="store_true")

    features_parser = subparsers.add_parser(
        "build-study-features",
        help="Build the grid-level geospatial feature table for a study.",
    )
    features_parser.add_argument("--study-config", required=True)
    features_parser.add_argument("--output", required=True)
    features_parser.add_argument("--metadata", required=True)

    index_parser = subparsers.add_parser("build-index", help="Compute the baseline index.")
    index_parser.add_argument("--input", required=True)
    index_parser.add_argument("--config", required=True)
    index_parser.add_argument("--output", required=True)
    index_parser.add_argument("--metadata", required=True)

    dominant_parser = subparsers.add_parser(
        "annotate-dominant-dimension",
        help="Annotate each row with its dominant deprivation dimension.",
    )
    dominant_parser.add_argument("--input", required=True)
    dominant_parser.add_argument("--dimension-cols", nargs="+", required=True)
    dominant_parser.add_argument("--output", required=True)
    dominant_parser.add_argument("--metadata", default=None)
    dominant_parser.add_argument("--margin-thresholds", nargs=2, type=float, default=None)

    priority_parser = subparsers.add_parser(
        "annotate-priority-quadrants",
        help="Annotate rows using pooled absolute and within-group relative priority flags.",
    )
    priority_parser.add_argument("--input", required=True)
    priority_parser.add_argument("--absolute-score-col", required=True)
    priority_parser.add_argument("--relative-score-col", required=True)
    priority_parser.add_argument("--group-col", required=True)
    priority_parser.add_argument("--output", required=True)
    priority_parser.add_argument("--metadata", default=None)
    priority_parser.add_argument("--absolute-top-fraction", type=float, default=0.1)
    priority_parser.add_argument("--relative-top-fraction", type=float, default=0.1)

    cluster_parser = subparsers.add_parser(
        "cluster-cells",
        help="Cluster cells using selected numeric columns.",
    )
    cluster_parser.add_argument("--input", required=True)
    cluster_parser.add_argument("--columns", nargs="+", required=True)
    cluster_parser.add_argument("--k", type=int, default=4)
    cluster_parser.add_argument("--output", required=True)
    cluster_parser.add_argument("--summary", required=True)

    plot_parser = subparsers.add_parser("plot-quicklook", help="Generate quicklook figures.")
    plot_parser.add_argument("--input", required=True)
    plot_parser.add_argument("--score-col", required=True)
    plot_parser.add_argument("--output-dir", required=True)
    plot_parser.add_argument("--id-col", default=None)
    plot_parser.add_argument("--group-col", default=None)
    plot_parser.add_argument("--lon-col", default=None)
    plot_parser.add_argument("--lat-col", default=None)

    score_map_parser = subparsers.add_parser(
        "plot-score-map",
        help="Plot a choropleth map from a geospatial file.",
    )
    score_map_parser.add_argument("--input", required=True)
    score_map_parser.add_argument("--score-col", required=True)
    score_map_parser.add_argument("--output", required=True)
    score_map_parser.add_argument("--group-col", default=None)

    category_map_parser = subparsers.add_parser(
        "plot-category-map",
        help="Plot a categorical geospatial map.",
    )
    category_map_parser.add_argument("--input", required=True)
    category_map_parser.add_argument("--category-col", required=True)
    category_map_parser.add_argument("--output", required=True)
    category_map_parser.add_argument("--group-col", default=None)

    hotspot_map_parser = subparsers.add_parser(
        "plot-hotspot-map",
        help="Plot a hotspot cluster map from local Moran outputs.",
    )
    hotspot_map_parser.add_argument("--input", required=True)
    hotspot_map_parser.add_argument("--hotspot-col", required=True)
    hotspot_map_parser.add_argument("--output", required=True)
    hotspot_map_parser.add_argument("--group-col", default=None)

    scatter_parser = subparsers.add_parser(
        "plot-scatter",
        help="Plot a scatter comparison, optionally faceted by group.",
    )
    scatter_parser.add_argument("--input", required=True)
    scatter_parser.add_argument("--x-col", required=True)
    scatter_parser.add_argument("--y-col", required=True)
    scatter_parser.add_argument("--output", required=True)
    scatter_parser.add_argument("--group-col", default=None)

    summary_bars_parser = subparsers.add_parser(
        "plot-summary-bars",
        help="Plot a bar chart from a summary table.",
    )
    summary_bars_parser.add_argument("--input", required=True)
    summary_bars_parser.add_argument("--x-col", required=True)
    summary_bars_parser.add_argument("--y-col", required=True)
    summary_bars_parser.add_argument("--output", required=True)
    summary_bars_parser.add_argument("--hue-col", default=None)

    contrast_heatmap_parser = subparsers.add_parser(
        "plot-contrast-heatmap",
        help="Plot a feature-by-group contrast heatmap.",
    )
    contrast_heatmap_parser.add_argument("--input", required=True)
    contrast_heatmap_parser.add_argument("--feature-col", required=True)
    contrast_heatmap_parser.add_argument("--group-col", required=True)
    contrast_heatmap_parser.add_argument("--value-col", required=True)
    contrast_heatmap_parser.add_argument("--output", required=True)

    faceted_heatmap_parser = subparsers.add_parser(
        "plot-faceted-heatmap",
        help="Plot a faceted feature heatmap by category and group.",
    )
    faceted_heatmap_parser.add_argument("--input", required=True)
    faceted_heatmap_parser.add_argument("--facet-col", required=True)
    faceted_heatmap_parser.add_argument("--x-col", required=True)
    faceted_heatmap_parser.add_argument("--y-col", required=True)
    faceted_heatmap_parser.add_argument("--value-col", required=True)
    faceted_heatmap_parser.add_argument("--output", required=True)

    spatial_parser = subparsers.add_parser(
        "spatial-autocorr",
        help="Compute Moran statistics for a geospatial file.",
    )
    spatial_parser.add_argument("--input", required=True)
    spatial_parser.add_argument("--score-col", required=True)
    spatial_parser.add_argument("--summary-output", required=True)
    spatial_parser.add_argument("--local-output", required=True)
    spatial_parser.add_argument("--k", type=int, default=8)

    summary_parser = subparsers.add_parser(
        "summarize-index",
        help="Write grouped score summaries and feature medians.",
    )
    summary_parser.add_argument("--input", required=True)
    summary_parser.add_argument("--group-col", required=True)
    summary_parser.add_argument("--score-col", required=True)
    summary_parser.add_argument("--score-summary-output", required=True)
    summary_parser.add_argument("--feature-summary-output", default=None)
    summary_parser.add_argument("--feature-columns", nargs="*", default=None)
    summary_parser.add_argument("--scenario", default=None)
    summary_parser.add_argument("--filter-col", default=None)
    summary_parser.add_argument("--filter-value", default=None)

    category_summary_parser = subparsers.add_parser(
        "summarize-categories",
        help="Write grouped category counts and shares.",
    )
    category_summary_parser.add_argument("--input", required=True)
    category_summary_parser.add_argument("--group-col", required=True)
    category_summary_parser.add_argument("--category-col", required=True)
    category_summary_parser.add_argument("--output", required=True)
    category_summary_parser.add_argument("--scenario", default=None)
    category_summary_parser.add_argument("--filter-col", default=None)
    category_summary_parser.add_argument("--filter-value", default=None)

    contrast_parser = subparsers.add_parser(
        "summarize-binary-contrast",
        help="Compare a focal subset against a reference subset within each group.",
    )
    contrast_parser.add_argument("--input", required=True)
    contrast_parser.add_argument("--group-col", required=True)
    contrast_parser.add_argument("--binary-col", required=True)
    contrast_parser.add_argument("--target-value", required=True)
    contrast_parser.add_argument("--reference-value", default=None)
    contrast_parser.add_argument("--value-columns", nargs="+", required=True)
    contrast_parser.add_argument("--output", required=True)
    contrast_parser.add_argument("--scenario", default=None)
    contrast_parser.add_argument("--filter-col", default=None)
    contrast_parser.add_argument("--filter-value", default=None)

    category_profile_parser = subparsers.add_parser(
        "summarize-category-feature-profiles",
        help="Summarize feature profiles for each category within each group.",
    )
    category_profile_parser.add_argument("--input", required=True)
    category_profile_parser.add_argument("--group-col", required=True)
    category_profile_parser.add_argument("--category-col", required=True)
    category_profile_parser.add_argument("--value-columns", nargs="+", required=True)
    category_profile_parser.add_argument("--output", required=True)
    category_profile_parser.add_argument("--scenario", default=None)
    category_profile_parser.add_argument("--filter-col", default=None)
    category_profile_parser.add_argument("--filter-value", default=None)

    pca_summary_parser = subparsers.add_parser(
        "summarize-pca",
        help="Write PCA vs composite index alignment summaries.",
    )
    pca_summary_parser.add_argument("--input", required=True)
    pca_summary_parser.add_argument("--group-col", required=True)
    pca_summary_parser.add_argument("--composite-col", default="deprivation_index_0_100")
    pca_summary_parser.add_argument("--pca-col", default="pca1_index_0_100")
    pca_summary_parser.add_argument("--output", required=True)
    pca_summary_parser.add_argument("--top-fraction", type=float, default=0.1)
    pca_summary_parser.add_argument("--scenario", default=None)

    compare_scores_parser = subparsers.add_parser(
        "compare-index-scores",
        help="Compare two indexed tables on matched cells.",
    )
    compare_scores_parser.add_argument("--left-input", required=True)
    compare_scores_parser.add_argument("--right-input", required=True)
    compare_scores_parser.add_argument("--join-columns", nargs="+", required=True)
    compare_scores_parser.add_argument("--group-col", required=True)
    compare_scores_parser.add_argument("--left-score-col", required=True)
    compare_scores_parser.add_argument("--right-score-col", required=True)
    compare_scores_parser.add_argument("--output", required=True)
    compare_scores_parser.add_argument("--merged-output", default=None)
    compare_scores_parser.add_argument("--left-label", default="left")
    compare_scores_parser.add_argument("--right-label", default="right")
    compare_scores_parser.add_argument("--top-fraction", type=float, default=0.1)
    compare_scores_parser.add_argument("--scenario", default=None)

    comparison_shift_parser = subparsers.add_parser(
        "summarize-comparison-shift",
        help="Summarize reranking and shift magnitude from a merged comparison table.",
    )
    comparison_shift_parser.add_argument("--input", required=True)
    comparison_shift_parser.add_argument("--group-col", required=True)
    comparison_shift_parser.add_argument("--left-label", required=True)
    comparison_shift_parser.add_argument("--right-label", required=True)
    comparison_shift_parser.add_argument("--output", required=True)
    comparison_shift_parser.add_argument("--top-fraction", type=float, default=0.1)
    comparison_shift_parser.add_argument("--scenario", default=None)

    top_cells_parser = subparsers.add_parser(
        "export-top-cells",
        help="Export the top-ranked cells within each group.",
    )
    top_cells_parser.add_argument("--input", required=True)
    top_cells_parser.add_argument("--group-col", required=True)
    top_cells_parser.add_argument("--score-col", required=True)
    top_cells_parser.add_argument("--output", required=True)
    top_cells_parser.add_argument("--top-n", type=int, default=25)
    top_cells_parser.add_argument("--columns", nargs="*", default=None)
    top_cells_parser.add_argument("--scenario", default=None)

    exposure_parser = subparsers.add_parser(
        "summarize-exposure",
        help="Summarize population exposure to high-score cells.",
    )
    exposure_parser.add_argument("--input", required=True)
    exposure_parser.add_argument("--group-col", required=True)
    exposure_parser.add_argument("--score-col", required=True)
    exposure_parser.add_argument("--population-col", required=True)
    exposure_parser.add_argument("--output", required=True)
    exposure_parser.add_argument("--quantiles", nargs="*", type=float, default=None)
    exposure_parser.add_argument("--scenario", default=None)

    weighted_category_parser = subparsers.add_parser(
        "summarize-weighted-categories",
        help="Summarize category shares using a weight column.",
    )
    weighted_category_parser.add_argument("--input", required=True)
    weighted_category_parser.add_argument("--group-col", required=True)
    weighted_category_parser.add_argument("--category-col", required=True)
    weighted_category_parser.add_argument("--weight-col", required=True)
    weighted_category_parser.add_argument("--output", required=True)
    weighted_category_parser.add_argument("--scenario", default=None)
    weighted_category_parser.add_argument("--filter-col", default=None)
    weighted_category_parser.add_argument("--filter-value", default=None)

    inequality_parser = subparsers.add_parser(
        "summarize-inequality",
        help="Write city-level inequality summaries for a score.",
    )
    inequality_parser.add_argument("--input", required=True)
    inequality_parser.add_argument("--group-col", required=True)
    inequality_parser.add_argument("--score-col", required=True)
    inequality_parser.add_argument("--population-col", required=True)
    inequality_parser.add_argument("--output", required=True)
    inequality_parser.add_argument("--scenario", default=None)

    admin_summary_parser = subparsers.add_parser(
        "summarize-admin-units",
        help="Aggregate cell-level results to admin units within each city.",
    )
    admin_summary_parser.add_argument("--input", required=True)
    admin_summary_parser.add_argument("--output", required=True)
    admin_summary_parser.add_argument("--group-col", required=True)
    admin_summary_parser.add_argument("--country-col", default="country_iso")
    admin_summary_parser.add_argument("--admin-name-col", default="admin2_name")
    admin_summary_parser.add_argument("--admin-id-col", default="admin2_id")
    admin_summary_parser.add_argument("--admin-iso-col", default="admin2_iso")
    admin_summary_parser.add_argument("--score-col", required=True)
    admin_summary_parser.add_argument("--population-col", required=True)
    admin_summary_parser.add_argument("--hotspot-col", default=None)
    admin_summary_parser.add_argument("--hotspot-value", default="high_high")
    admin_summary_parser.add_argument("--dominant-dimension-col", default=None)
    admin_summary_parser.add_argument("--priority-col", default=None)
    admin_summary_parser.add_argument("--top-fraction", type=float, default=0.1)
    admin_summary_parser.add_argument("--priority-fraction", type=float, default=0.25)
    admin_summary_parser.add_argument("--min-cells", type=int, default=10)
    admin_summary_parser.add_argument("--min-city-population-share", type=float, default=0.01)
    admin_summary_parser.add_argument("--metadata", default=None)

    multimodal_dataset_parser = subparsers.add_parser(
        "build-multimodal-patch-dataset",
        help="Build a multi-channel patch dataset for weakly supervised ML.",
    )
    multimodal_dataset_parser.add_argument("--input", required=True)
    multimodal_dataset_parser.add_argument("--study-config", required=True)
    multimodal_dataset_parser.add_argument("--output", required=True)
    multimodal_dataset_parser.add_argument("--metadata", required=True)
    multimodal_dataset_parser.add_argument("--patch-size", type=int, default=64)
    multimodal_dataset_parser.add_argument("--context-m", type=float, default=1500.0)
    multimodal_dataset_parser.add_argument("--feature-columns", nargs="*", default=None)
    multimodal_dataset_parser.add_argument("--regression-target-col", default="rwi_mean")
    multimodal_dataset_parser.add_argument("--classification-target-col", default="rwi_bottom_quantile_flag")
    multimodal_dataset_parser.add_argument("--label-mask-col", default="rwi_label_available")

    autoencoder_parser = subparsers.add_parser(
        "pretrain-patch-autoencoder",
        help="Pretrain a convolutional autoencoder on the patch dataset.",
    )
    autoencoder_parser.add_argument("--dataset", required=True)
    autoencoder_parser.add_argument("--checkpoint", required=True)
    autoencoder_parser.add_argument("--metrics", required=True)
    autoencoder_parser.add_argument("--epochs", type=int, default=12)
    autoencoder_parser.add_argument("--batch-size", type=int, default=256)
    autoencoder_parser.add_argument("--learning-rate", type=float, default=1e-3)
    autoencoder_parser.add_argument("--random-state", type=int, default=42)

    multimodal_benchmark_parser = subparsers.add_parser(
        "run-multimodal-rwi-benchmark",
        help="Run atlas, tabular, image, and fusion benchmarks against RWI targets.",
    )
    multimodal_benchmark_parser.add_argument("--input", required=True)
    multimodal_benchmark_parser.add_argument("--dataset", required=True)
    multimodal_benchmark_parser.add_argument("--metrics-output", required=True)
    multimodal_benchmark_parser.add_argument("--predictions-output", required=True)
    multimodal_benchmark_parser.add_argument("--metadata", required=True)
    multimodal_benchmark_parser.add_argument("--pretrained-encoder", default=None)
    multimodal_benchmark_parser.add_argument("--feature-columns", nargs="*", default=None)
    multimodal_benchmark_parser.add_argument("--models", nargs="*", default=None)
    multimodal_benchmark_parser.add_argument("--score-col", default="deprivation_index_0_100")
    multimodal_benchmark_parser.add_argument("--regression-target-col", default="rwi_mean")
    multimodal_benchmark_parser.add_argument("--classification-target-col", default="rwi_bottom_quantile_flag")
    multimodal_benchmark_parser.add_argument("--epochs", type=int, default=10)
    multimodal_benchmark_parser.add_argument("--batch-size", type=int, default=256)
    multimodal_benchmark_parser.add_argument("--learning-rate", type=float, default=1e-3)
    multimodal_benchmark_parser.add_argument("--random-state", type=int, default=42)
    multimodal_benchmark_parser.add_argument("--protocol-strategy", default="auto")
    multimodal_benchmark_parser.add_argument("--graph-k", type=int, default=8)

    multimodal_summary_parser = subparsers.add_parser(
        "summarize-multimodal-benchmark",
        help="Summarize winning models across benchmark protocols and metrics.",
    )
    multimodal_summary_parser.add_argument("--input", required=True)
    multimodal_summary_parser.add_argument("--output", required=True)

    benchmark_findings_parser = subparsers.add_parser(
        "build-benchmark-findings",
        help="Build a paper-style summary artifact from benchmark metrics and summary tables.",
    )
    benchmark_findings_parser.add_argument("--metrics-input", required=True)
    benchmark_findings_parser.add_argument("--summary-input", required=True)
    benchmark_findings_parser.add_argument("--output", required=True)

    attach_validation_parser = subparsers.add_parser(
        "attach-external-raster",
        help="Attach external raster summary statistics to grid cells for convergent validation.",
    )
    attach_validation_parser.add_argument("--input", required=True)
    attach_validation_parser.add_argument("--raster", required=True)
    attach_validation_parser.add_argument("--output", required=True)
    attach_validation_parser.add_argument("--metadata", default=None)
    attach_validation_parser.add_argument("--prefix", default="external_signal")
    attach_validation_parser.add_argument("--stats", nargs="*", default=None)
    attach_validation_parser.add_argument("--all-touched", action="store_true")

    summarize_validation_parser = subparsers.add_parser(
        "summarize-external-validation",
        help="Summarize score alignment against an attached external validation signal.",
    )
    summarize_validation_parser.add_argument("--input", required=True)
    summarize_validation_parser.add_argument("--group-col", required=True)
    summarize_validation_parser.add_argument("--external-col", required=True)
    summarize_validation_parser.add_argument("--score-columns", nargs="+", required=True)
    summarize_validation_parser.add_argument("--output", required=True)
    summarize_validation_parser.add_argument("--top-fraction", type=float, default=0.1)
    summarize_validation_parser.add_argument("--expected-relation", default="negative")

    validation_findings_parser = subparsers.add_parser(
        "build-validation-findings",
        help="Build a machine-readable findings artifact from validation summaries.",
    )
    validation_findings_parser.add_argument("--input", required=True)
    validation_findings_parser.add_argument("--output", required=True)

    core_findings_parser = subparsers.add_parser(
        "build-core-findings",
        help="Assemble a machine-readable core findings artifact from result tables.",
    )
    core_findings_parser.add_argument("--config", required=True)
    core_findings_parser.add_argument("--output", required=True)

    figure_set_parser = subparsers.add_parser(
        "stage-figure-set",
        help="Copy a curated figure set into a stable output directory and write a manifest.",
    )
    figure_set_parser.add_argument("--config", required=True)
    figure_set_parser.add_argument("--output-dir", required=True)
    figure_set_parser.add_argument("--manifest-output", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "catalog":
        catalog = load_catalog(Path(args.catalog))
        filtered = filter_catalog(
            catalog,
            route=args.route,
            status=args.status,
            approval_required=_maybe_bool_from_flag(args.approval_required),
        )
        print(format_catalog_table(filtered))
        return

    if args.command == "profile":
        profile = build_profile(Path(args.input_path))
        if args.output:
            write_json(profile, Path(args.output))
            print("Wrote profile to {path}".format(path=args.output))
        else:
            import json

            print(json.dumps(profile, indent=2, ensure_ascii=False))
        return

    if args.command == "concat-tables":
        concat_tables(inputs=[Path(path) for path in args.inputs], output_path=Path(args.output))
        print("Wrote concatenated table to {path}".format(path=args.output))
        return

    if args.command == "add-composite-column":
        add_composite_column(
            input_path=Path(args.input),
            output_path=Path(args.output),
            source_columns=args.source_columns,
            output_column=args.output_column,
            separator=args.separator,
        )
        print("Wrote table with composite column to {path}".format(path=args.output))
        return

    if args.command == "filter-table":
        filter_table(
            input_path=Path(args.input),
            output_path=Path(args.output),
            filter_col=args.filter_col,
            filter_value=args.filter_value,
        )
        print("Wrote filtered table to {path}".format(path=args.output))
        return

    if args.command == "apply-analysis-mask":
        run_analysis_mask(
            input_path=Path(args.input),
            config_path=Path(args.config),
            output_path=Path(args.output),
            metadata_path=Path(args.metadata),
        )
        print("Wrote masked table to {path}".format(path=args.output))
        print("Wrote mask metadata to {path}".format(path=args.metadata))
        return

    if args.command == "attach-admin-units":
        attach_admin_units(
            input_path=Path(args.input),
            output_path=Path(args.output),
            boundary_dir=Path(args.boundary_dir),
            admin_level=args.admin_level,
            country_col=args.country_col,
            admin_prefix=args.admin_prefix,
            metadata_path=Path(args.metadata) if args.metadata else None,
        )
        print("Wrote admin-attached table to {path}".format(path=args.output))
        if args.metadata:
            print("Wrote admin-attachment metadata to {path}".format(path=args.metadata))
        return

    if args.command == "download-rwi":
        manifest = download_rwi_country_files(
            countries=args.countries,
            output_dir=Path(args.output_dir),
            manifest_path=Path(args.manifest_output) if args.manifest_output else None,
        )
        print("Downloaded RWI files for countries: {countries}".format(countries=", ".join(sorted(manifest["countries"].keys()))))
        if args.manifest_output:
            print("Wrote RWI manifest to {path}".format(path=args.manifest_output))
        return

    if args.command == "build-rwi-grid-targets":
        build_rwi_grid_targets(
            input_path=Path(args.input),
            output_path=Path(args.output),
            rwi_dir=Path(args.rwi_dir),
            metadata_path=Path(args.metadata) if args.metadata else None,
            group_col=args.group_col,
            country_col=args.country_col,
            score_col=args.score_col,
            neighbors=args.neighbors,
            max_distance_m=args.max_distance_m,
            low_wealth_quantile=args.low_wealth_quantile,
        )
        print("Wrote RWI target table to {path}".format(path=args.output))
        if args.metadata:
            print("Wrote RWI target metadata to {path}".format(path=args.metadata))
        return

    if args.command == "download-study-assets":
        manifest = download_study_assets(
            study_config_path=Path(args.study_config),
            overwrite=args.overwrite,
            manifest_output=Path(args.manifest_output),
        )
        print("Wrote download manifest to {path}".format(path=args.manifest_output))
        print("Downloaded assets for study {study_id}".format(study_id=manifest["study_id"]))
        return

    if args.command == "build-study-features":
        build_study_feature_tables(
            study_config_path=Path(args.study_config),
            output_path=Path(args.output),
            metadata_path=Path(args.metadata),
        )
        print("Wrote study feature table to {path}".format(path=args.output))
        print("Wrote feature metadata to {path}".format(path=args.metadata))
        return

    if args.command == "build-index":
        run_index_build(
            input_path=Path(args.input),
            config_path=Path(args.config),
            output_path=Path(args.output),
            metadata_path=Path(args.metadata),
        )
        print("Wrote indexed table to {path}".format(path=args.output))
        print("Wrote metadata to {path}".format(path=args.metadata))
        return

    if args.command == "annotate-dominant-dimension":
        annotate_dominant_dimension(
            input_path=Path(args.input),
            dimension_cols=args.dimension_cols,
            output_path=Path(args.output),
            metadata_path=Path(args.metadata) if args.metadata else None,
            margin_thresholds=args.margin_thresholds,
        )
        print("Wrote dominant-dimension table to {path}".format(path=args.output))
        if args.metadata:
            print("Wrote dominant-dimension metadata to {path}".format(path=args.metadata))
        return

    if args.command == "annotate-priority-quadrants":
        annotate_priority_quadrants(
            input_path=Path(args.input),
            absolute_score_col=args.absolute_score_col,
            relative_score_col=args.relative_score_col,
            output_path=Path(args.output),
            group_col=args.group_col,
            metadata_path=Path(args.metadata) if args.metadata else None,
            absolute_top_fraction=args.absolute_top_fraction,
            relative_top_fraction=args.relative_top_fraction,
        )
        print("Wrote priority-quadrant table to {path}".format(path=args.output))
        if args.metadata:
            print("Wrote priority-quadrant metadata to {path}".format(path=args.metadata))
        return

    if args.command == "cluster-cells":
        run_clustering(
            input_path=Path(args.input),
            columns=args.columns,
            k=args.k,
            output_path=Path(args.output),
            summary_path=Path(args.summary),
        )
        print("Wrote clustered table to {path}".format(path=args.output))
        print("Wrote clustering summary to {path}".format(path=args.summary))
        return

    if args.command == "plot-quicklook":
        create_quicklook_outputs(
            input_path=Path(args.input),
            score_col=args.score_col,
            output_dir=Path(args.output_dir),
            id_col=args.id_col,
            group_col=args.group_col,
            lon_col=args.lon_col,
            lat_col=args.lat_col,
        )
        print("Wrote quicklook outputs to {path}".format(path=args.output_dir))
        return

    if args.command == "plot-score-map":
        plot_score_map(
            input_path=Path(args.input),
            score_col=args.score_col,
            output_path=Path(args.output),
            group_col=args.group_col,
        )
        print("Wrote score map to {path}".format(path=args.output))
        return

    if args.command == "plot-category-map":
        plot_category_map(
            input_path=Path(args.input),
            category_col=args.category_col,
            output_path=Path(args.output),
            group_col=args.group_col,
        )
        print("Wrote category map to {path}".format(path=args.output))
        return

    if args.command == "plot-hotspot-map":
        plot_hotspot_map(
            input_path=Path(args.input),
            hotspot_col=args.hotspot_col,
            output_path=Path(args.output),
            group_col=args.group_col,
        )
        print("Wrote hotspot map to {path}".format(path=args.output))
        return

    if args.command == "plot-scatter":
        plot_scatter_by_group(
            input_path=Path(args.input),
            x_col=args.x_col,
            y_col=args.y_col,
            output_path=Path(args.output),
            group_col=args.group_col,
        )
        print("Wrote scatter plot to {path}".format(path=args.output))
        return

    if args.command == "plot-summary-bars":
        plot_summary_bars(
            input_path=Path(args.input),
            x_col=args.x_col,
            y_col=args.y_col,
            output_path=Path(args.output),
            hue_col=args.hue_col,
        )
        print("Wrote summary bar plot to {path}".format(path=args.output))
        return

    if args.command == "plot-contrast-heatmap":
        plot_contrast_heatmap(
            input_path=Path(args.input),
            feature_col=args.feature_col,
            group_col=args.group_col,
            value_col=args.value_col,
            output_path=Path(args.output),
        )
        print("Wrote contrast heatmap to {path}".format(path=args.output))
        return

    if args.command == "plot-faceted-heatmap":
        plot_faceted_heatmap(
            input_path=Path(args.input),
            facet_col=args.facet_col,
            x_col=args.x_col,
            y_col=args.y_col,
            value_col=args.value_col,
            output_path=Path(args.output),
        )
        print("Wrote faceted heatmap to {path}".format(path=args.output))
        return

    if args.command == "spatial-autocorr":
        run_spatial_autocorrelation(
            input_path=Path(args.input),
            score_col=args.score_col,
            summary_output=Path(args.summary_output),
            local_output=Path(args.local_output),
            k=args.k,
        )
        print("Wrote Moran summary to {path}".format(path=args.summary_output))
        print("Wrote local Moran output to {path}".format(path=args.local_output))
        return

    if args.command == "summarize-index":
        summarize_index(
            input_path=Path(args.input),
            group_col=args.group_col,
            score_col=args.score_col,
            score_summary_output=Path(args.score_summary_output),
            feature_summary_output=Path(args.feature_summary_output) if args.feature_summary_output else None,
            feature_columns=args.feature_columns,
            scenario=args.scenario,
            filter_col=args.filter_col,
            filter_value=args.filter_value,
        )
        print("Wrote score summary to {path}".format(path=args.score_summary_output))
        if args.feature_summary_output:
            print("Wrote feature summary to {path}".format(path=args.feature_summary_output))
        return

    if args.command == "summarize-categories":
        summarize_category_shares(
            input_path=Path(args.input),
            group_col=args.group_col,
            category_col=args.category_col,
            output_path=Path(args.output),
            scenario=args.scenario,
            filter_col=args.filter_col,
            filter_value=args.filter_value,
        )
        print("Wrote category summary to {path}".format(path=args.output))
        return

    if args.command == "summarize-binary-contrast":
        summarize_binary_contrast(
            input_path=Path(args.input),
            group_col=args.group_col,
            binary_col=args.binary_col,
            target_value=args.target_value,
            reference_value=args.reference_value,
            value_columns=args.value_columns,
            output_path=Path(args.output),
            scenario=args.scenario,
            filter_col=args.filter_col,
            filter_value=args.filter_value,
        )
        print("Wrote binary contrast summary to {path}".format(path=args.output))
        return

    if args.command == "summarize-category-feature-profiles":
        summarize_category_feature_profiles(
            input_path=Path(args.input),
            group_col=args.group_col,
            category_col=args.category_col,
            value_columns=args.value_columns,
            output_path=Path(args.output),
            scenario=args.scenario,
            filter_col=args.filter_col,
            filter_value=args.filter_value,
        )
        print("Wrote category feature profiles to {path}".format(path=args.output))
        return

    if args.command == "summarize-pca":
        summarize_pca_alignment(
            input_path=Path(args.input),
            group_col=args.group_col,
            composite_col=args.composite_col,
            pca_col=args.pca_col,
            output_path=Path(args.output),
            top_fraction=args.top_fraction,
            scenario=args.scenario,
        )
        print("Wrote PCA summary to {path}".format(path=args.output))
        return

    if args.command == "compare-index-scores":
        compare_index_scores(
            left_input_path=Path(args.left_input),
            right_input_path=Path(args.right_input),
            join_columns=args.join_columns,
            group_col=args.group_col,
            left_score_col=args.left_score_col,
            right_score_col=args.right_score_col,
            output_path=Path(args.output),
            merged_output_path=Path(args.merged_output) if args.merged_output else None,
            left_label=args.left_label,
            right_label=args.right_label,
            top_fraction=args.top_fraction,
            scenario=args.scenario,
        )
        print("Wrote score-comparison summary to {path}".format(path=args.output))
        if args.merged_output:
            print("Wrote merged comparison table to {path}".format(path=args.merged_output))
        return

    if args.command == "summarize-comparison-shift":
        summarize_comparison_shift(
            input_path=Path(args.input),
            group_col=args.group_col,
            left_label=args.left_label,
            right_label=args.right_label,
            output_path=Path(args.output),
            top_fraction=args.top_fraction,
            scenario=args.scenario,
        )
        print("Wrote comparison-shift summary to {path}".format(path=args.output))
        return

    if args.command == "export-top-cells":
        export_top_cells(
            input_path=Path(args.input),
            group_col=args.group_col,
            score_col=args.score_col,
            output_path=Path(args.output),
            top_n=args.top_n,
            columns=args.columns,
            scenario=args.scenario,
        )
        print("Wrote top-cell export to {path}".format(path=args.output))
        return

    if args.command == "summarize-exposure":
        summarize_population_exposure(
            input_path=Path(args.input),
            group_col=args.group_col,
            score_col=args.score_col,
            population_col=args.population_col,
            output_path=Path(args.output),
            quantiles=args.quantiles,
            scenario=args.scenario,
        )
        print("Wrote exposure summary to {path}".format(path=args.output))
        return

    if args.command == "summarize-weighted-categories":
        summarize_weighted_categories(
            input_path=Path(args.input),
            group_col=args.group_col,
            category_col=args.category_col,
            weight_col=args.weight_col,
            output_path=Path(args.output),
            scenario=args.scenario,
            filter_col=args.filter_col,
            filter_value=args.filter_value,
        )
        print("Wrote weighted category summary to {path}".format(path=args.output))
        return

    if args.command == "summarize-inequality":
        summarize_inequality(
            input_path=Path(args.input),
            group_col=args.group_col,
            score_col=args.score_col,
            population_col=args.population_col,
            output_path=Path(args.output),
            scenario=args.scenario,
        )
        print("Wrote inequality summary to {path}".format(path=args.output))
        return

    if args.command == "summarize-admin-units":
        summarize_admin_units(
            input_path=Path(args.input),
            output_path=Path(args.output),
            group_col=args.group_col,
            country_col=args.country_col,
            admin_name_col=args.admin_name_col,
            admin_id_col=args.admin_id_col,
            admin_iso_col=args.admin_iso_col,
            score_col=args.score_col,
            population_col=args.population_col,
            hotspot_col=args.hotspot_col,
            hotspot_value=args.hotspot_value,
            dominant_dimension_col=args.dominant_dimension_col,
            priority_col=args.priority_col,
            top_fraction=args.top_fraction,
            priority_fraction=args.priority_fraction,
            min_cells=args.min_cells,
            min_city_population_share=args.min_city_population_share,
            metadata_path=Path(args.metadata) if args.metadata else None,
        )
        print("Wrote admin summary to {path}".format(path=args.output))
        if args.metadata:
            print("Wrote admin summary metadata to {path}".format(path=args.metadata))
        return

    if args.command == "build-multimodal-patch-dataset":
        build_multimodal_patch_dataset(
            input_path=Path(args.input),
            study_config_path=Path(args.study_config),
            output_path=Path(args.output),
            metadata_path=Path(args.metadata),
            patch_size=args.patch_size,
            context_m=args.context_m,
            feature_columns=args.feature_columns,
            regression_target_col=args.regression_target_col,
            classification_target_col=args.classification_target_col,
            label_mask_col=args.label_mask_col,
        )
        print("Wrote multimodal patch dataset to {path}".format(path=args.output))
        print("Wrote multimodal dataset metadata to {path}".format(path=args.metadata))
        return

    if args.command == "pretrain-patch-autoencoder":
        pretrain_patch_autoencoder(
            dataset_path=Path(args.dataset),
            checkpoint_path=Path(args.checkpoint),
            metrics_path=Path(args.metrics),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
        print("Wrote autoencoder checkpoint to {path}".format(path=args.checkpoint))
        print("Wrote autoencoder metrics to {path}".format(path=args.metrics))
        return

    if args.command == "run-multimodal-rwi-benchmark":
        run_multimodal_rwi_benchmark(
            input_path=Path(args.input),
            dataset_path=Path(args.dataset),
            metrics_output_path=Path(args.metrics_output),
            predictions_output_path=Path(args.predictions_output),
            metadata_path=Path(args.metadata),
            pretrained_encoder_path=Path(args.pretrained_encoder) if args.pretrained_encoder else None,
            feature_columns=args.feature_columns,
            model_names=args.models,
            score_col=args.score_col,
            regression_target_col=args.regression_target_col,
            classification_target_col=args.classification_target_col,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            protocol_strategy=args.protocol_strategy,
            graph_k=args.graph_k,
        )
        print("Wrote multimodal benchmark metrics to {path}".format(path=args.metrics_output))
        print("Wrote multimodal benchmark predictions to {path}".format(path=args.predictions_output))
        print("Wrote multimodal benchmark metadata to {path}".format(path=args.metadata))
        return

    if args.command == "summarize-multimodal-benchmark":
        summarize_multimodal_benchmark(
            metrics_input_path=Path(args.input),
            output_path=Path(args.output),
        )
        print("Wrote multimodal benchmark summary to {path}".format(path=args.output))
        return

    if args.command == "build-benchmark-findings":
        build_benchmark_findings_artifact(
            metrics_input_path=Path(args.metrics_input),
            summary_input_path=Path(args.summary_input),
            output_path=Path(args.output),
        )
        print("Wrote benchmark findings artifact to {path}".format(path=args.output))
        return

    if args.command == "attach-external-raster":
        attach_external_raster_signal(
            input_path=Path(args.input),
            raster_path=Path(args.raster),
            output_path=Path(args.output),
            metadata_path=Path(args.metadata) if args.metadata else None,
            prefix=args.prefix,
            stats=args.stats,
            all_touched=args.all_touched,
        )
        print("Wrote validation-enriched table to {path}".format(path=args.output))
        if args.metadata:
            print("Wrote validation metadata to {path}".format(path=args.metadata))
        return

    if args.command == "summarize-external-validation":
        summarize_external_validation(
            input_path=Path(args.input),
            group_col=args.group_col,
            external_col=args.external_col,
            score_columns=args.score_columns,
            output_path=Path(args.output),
            top_fraction=args.top_fraction,
            expected_relation=args.expected_relation,
        )
        print("Wrote validation summary to {path}".format(path=args.output))
        return

    if args.command == "build-validation-findings":
        build_validation_findings_artifact(
            summary_input_path=Path(args.input),
            output_path=Path(args.output),
        )
        print("Wrote validation findings artifact to {path}".format(path=args.output))
        return

    if args.command == "build-core-findings":
        build_core_findings_artifact(
            config_path=Path(args.config),
            output_path=Path(args.output),
        )
        print("Wrote core findings artifact to {path}".format(path=args.output))
        return

    if args.command == "stage-figure-set":
        stage_figure_set(
            config_path=Path(args.config),
            output_dir=Path(args.output_dir),
            manifest_output=Path(args.manifest_output),
        )
        print("Staged figure set to {path}".format(path=args.output_dir))
        print("Wrote figure manifest to {path}".format(path=args.manifest_output))
        return

    parser.error("Unknown command.")
