from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import read_yaml
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.io_utils import write_table


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def _groupwise_transform(
    series: pd.Series,
    groups: Optional[pd.Series],
    transform_fn,
    **kwargs: Any,
) -> pd.Series:
    if groups is None:
        return transform_fn(series, **kwargs)
    return series.groupby(groups).transform(lambda subset: transform_fn(subset, **kwargs))


def _winsorize_series_basic(
    series: pd.Series,
    lower_quantile: float = None,
    upper_quantile: float = None,
) -> pd.Series:
    lower = float(series.quantile(lower_quantile)) if lower_quantile is not None else None
    upper = float(series.quantile(upper_quantile)) if upper_quantile is not None else None
    return series.clip(lower=lower, upper=upper)


def _winsorize_series(
    series: pd.Series,
    lower_quantile: float = None,
    upper_quantile: float = None,
    groups: Optional[pd.Series] = None,
) -> pd.Series:
    return _groupwise_transform(
        series,
        groups=groups,
        transform_fn=_winsorize_series_basic,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


def _safe_grouped_zscore(series: pd.Series, groups: Optional[pd.Series] = None) -> pd.Series:
    return _groupwise_transform(series, groups=groups, transform_fn=_safe_zscore)


def _minmax_0_100_basic(
    series: pd.Series,
    lower_quantile: float = None,
    upper_quantile: float = None,
) -> pd.Series:
    clipped = _winsorize_series_basic(series, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    lower = float(clipped.min())
    upper = float(clipped.max())
    if upper == lower:
        return pd.Series(np.repeat(50.0, len(series)), index=series.index)
    return 100.0 * (clipped - lower) / (upper - lower)


def _minmax_0_100(
    series: pd.Series,
    lower_quantile: float = None,
    upper_quantile: float = None,
    groups: Optional[pd.Series] = None,
) -> pd.Series:
    return _groupwise_transform(
        series,
        groups=groups,
        transform_fn=_minmax_0_100_basic,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


def _weighted_average(frame: pd.DataFrame, weighted_columns: List[Tuple[str, float]]) -> pd.Series:
    numerator = None
    denominator = 0.0

    for column, weight in weighted_columns:
        contribution = frame[column] * weight
        numerator = contribution if numerator is None else numerator + contribution
        denominator += weight

    if denominator == 0:
        raise ValueError("At least one positive weight is required.")
    return numerator / denominator


def load_index_config(config_path: Path) -> Dict[str, Any]:
    return read_yaml(Path(config_path))


def build_index_table(input_path: Path, config_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frame = read_table(Path(input_path)).copy()
    config = load_index_config(Path(config_path))
    features = config.get("features", [])
    winsorize_quantiles = config.get("winsorize_quantiles")
    winsorize_group_col = config.get("winsorize_group_col")
    feature_standardization_group_col = config.get("feature_standardization_group_col")
    zscore_clip = config.get("feature_zscore_clip")
    output_scaling = config.get("output_scaling", {})
    output_lower_quantile = output_scaling.get("lower_quantile")
    output_upper_quantile = output_scaling.get("upper_quantile")
    output_group_col = output_scaling.get("group_col")

    missing_features = [
        feature["name"]
        for feature in features
        if feature["name"] not in frame.columns
    ]
    if missing_features:
        raise ValueError("Missing feature columns: {cols}".format(cols=", ".join(missing_features)))

    for column_name in [
        winsorize_group_col,
        feature_standardization_group_col,
        output_group_col,
    ]:
        if column_name and column_name not in frame.columns:
            raise ValueError("Grouping column '{column}' not found.".format(column=column_name))

    result = frame.copy()
    winsorize_groups = result[winsorize_group_col] if winsorize_group_col else None
    feature_standardization_groups = (
        result[feature_standardization_group_col]
        if feature_standardization_group_col
        else None
    )
    output_groups = result[output_group_col] if output_group_col else None
    imputation_summary = {}
    dimension_columns = {}
    aligned_feature_names = []

    for feature in features:
        name = feature["name"]
        higher_is_more_deprived = bool(feature.get("higher_is_more_deprived", True))
        series = pd.to_numeric(result[name], errors="coerce")

        median_value = float(series.median()) if series.notna().any() else 0.0
        imputed = series.fillna(median_value)
        prepared = imputed
        if winsorize_quantiles:
            prepared = _winsorize_series(
                prepared,
                lower_quantile=float(winsorize_quantiles[0]),
                upper_quantile=float(winsorize_quantiles[1]),
                groups=winsorize_groups,
            )

        aligned = prepared if higher_is_more_deprived else -1.0 * prepared

        aligned_column = "{name}__aligned".format(name=name)
        z_column = "{name}__z".format(name=name)
        result[aligned_column] = aligned
        z_values = _safe_grouped_zscore(aligned, groups=feature_standardization_groups)
        if zscore_clip is not None:
            z_values = z_values.clip(lower=-float(zscore_clip), upper=float(zscore_clip))
        result[z_column] = z_values
        aligned_feature_names.append(aligned_column)

        dimension = feature["dimension"]
        dimension_columns.setdefault(dimension, [])
        dimension_columns[dimension].append((z_column, float(feature.get("feature_weight", 1.0))))

        imputation_summary[name] = {
            "median_used": median_value,
            "higher_is_more_deprived": higher_is_more_deprived,
            "dimension": dimension,
        }

    dimension_weights = config.get("dimension_weights", {})
    dimension_score_columns = []
    for dimension, weighted_columns in dimension_columns.items():
        score_column = "{dimension}__score".format(dimension=dimension)
        result[score_column] = _weighted_average(result, weighted_columns)
        dimension_score_columns.append((score_column, float(dimension_weights.get(dimension, 1.0))))

    result["deprivation_index_z"] = _weighted_average(result, dimension_score_columns)
    result["deprivation_index_0_100"] = _minmax_0_100(
        result["deprivation_index_z"],
        lower_quantile=output_lower_quantile,
        upper_quantile=output_upper_quantile,
        groups=output_groups,
    )
    result["deprivation_rank_desc"] = (
        result["deprivation_index_z"].rank(method="dense", ascending=False).astype(int)
    )

    pca_config = config.get("pca", {})
    pca_loadings = None
    if pca_config.get("enabled", False):
        scaler = StandardScaler()
        aligned_matrix = scaler.fit_transform(result[aligned_feature_names])
        pca = PCA(n_components=int(pca_config.get("n_components", 1)))
        component_scores = pca.fit_transform(aligned_matrix)[:, 0]

        component_series = pd.Series(component_scores, index=result.index)
        correlation = component_series.corr(result["deprivation_index_z"])
        if pd.isna(correlation):
            correlation = 1.0
        if correlation < 0:
            component_scores = -1.0 * component_scores
            pca.components_[0] = -1.0 * pca.components_[0]

        result["pca1_index_z"] = component_scores
        result["pca1_index_0_100"] = _minmax_0_100(
            result["pca1_index_z"],
            lower_quantile=output_lower_quantile,
            upper_quantile=output_upper_quantile,
            groups=output_groups,
        )
        pca_loadings = {
            feature["name"]: float(loading)
            for feature, loading in zip(features, pca.components_[0])
        }

    metadata = {
        "index_name": config.get("index_name", "deprivation_index"),
        "input_path": str(Path(input_path)),
        "config_path": str(Path(config_path)),
        "n_rows": int(len(result)),
        "id_columns": config.get("id_columns", []),
        "dimension_weights": dimension_weights,
        "feature_count": len(features),
        "imputation_summary": imputation_summary,
        "winsorize_quantiles": winsorize_quantiles,
        "winsorize_group_col": winsorize_group_col,
        "feature_zscore_clip": zscore_clip,
        "feature_standardization_group_col": feature_standardization_group_col,
        "output_scaling": output_scaling,
        "pca_enabled": bool(pca_config.get("enabled", False)),
        "pca_loadings": pca_loadings,
    }

    if not config.get("keep_debug_columns", True):
        debug_columns = [
            column
            for column in result.columns
            if column.endswith("__aligned") or column.endswith("__z")
        ]
        result = result.drop(columns=debug_columns)

    return result, metadata


def run_index_build(input_path: Path, config_path: Path, output_path: Path, metadata_path: Path) -> None:
    result, metadata = build_index_table(Path(input_path), Path(config_path))
    write_table(result, Path(output_path))
    write_json(metadata, Path(metadata_path))
