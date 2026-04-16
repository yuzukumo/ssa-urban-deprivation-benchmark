from pathlib import Path
from contextlib import ExitStack
from typing import Any
from typing import Iterable
from typing import Optional
from datetime import datetime
from datetime import timezone

import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.enums import MergeAlg
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from ssa_urban_deprivation_benchmark.feature_pipeline import _filter_clinic_amenities
from ssa_urban_deprivation_benchmark.feature_pipeline import _filter_school_amenities
from ssa_urban_deprivation_benchmark.feature_pipeline import _filter_service_amenities
from ssa_urban_deprivation_benchmark.feature_pipeline import _intersection_nodes
from ssa_urban_deprivation_benchmark.feature_pipeline import _load_city_raw_layers
from ssa_urban_deprivation_benchmark.feature_pipeline import _representative_points
from ssa_urban_deprivation_benchmark.feature_pipeline import _utm_crs_for_gdf
from ssa_urban_deprivation_benchmark.feature_pipeline import _worldcover_paths_for_city
from ssa_urban_deprivation_benchmark.feature_pipeline import _worldpop_path_for_iso
from ssa_urban_deprivation_benchmark.io_utils import ensure_parent_dir
from ssa_urban_deprivation_benchmark.io_utils import read_table
from ssa_urban_deprivation_benchmark.io_utils import write_json
from ssa_urban_deprivation_benchmark.study import load_study_config


DEFAULT_TABULAR_FEATURE_COLUMNS = [
    "population",
    "road_distance_m",
    "school_distance_m",
    "clinic_distance_m",
    "amenity_count_1km",
    "service_count_1km",
    "population_per_service",
    "building_coverage_ratio",
    "open_space_share",
    "intersection_density_km2",
]
DEFAULT_IMAGE_CHANNELS = [
    "worldpop_log1p",
    "worldcover_built_up",
    "worldcover_open_space",
    "roads_binary",
    "amenities_count",
    "schools_count",
    "clinics_count",
    "intersections_count",
]
DEFAULT_BENCHMARK_MODELS = [
    "atlas_linear_baseline",
    "xgboost_tabular",
    "cnn_image",
    "cnn_fusion",
    "resnet_fusion_pretrained",
    "graph_fusion",
]


def _torch_env_prefix() -> str:
    conda_prefix = Path(__import__("os").environ.get("CONDA_PREFIX", ""))
    if conda_prefix:
        return str(conda_prefix / "lib")
    return ""


def _load_numpy_dataset(dataset_path: Path) -> dict[str, Any]:
    with np.load(Path(dataset_path), allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _sample_raster_patch(
    src,
    bounds: tuple[float, float, float, float],
    patch_size: int,
    dst_crs: pyproj.CRS,
    resampling: Resampling,
    fill_value: float = 0.0,
) -> np.ndarray:
    destination = np.full((patch_size, patch_size), fill_value, dtype=np.float32)
    destination_transform = from_bounds(*bounds, width=patch_size, height=patch_size)
    reproject(
        source=rasterio.band(src, 1),
        destination=destination,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=destination_transform,
        dst_crs=dst_crs,
        src_nodata=src.nodata,
        dst_nodata=fill_value,
        resampling=resampling,
    )
    return destination.astype(np.float32)


def _sample_mosaic_patch(
    sources: Iterable[Any],
    bounds: tuple[float, float, float, float],
    patch_size: int,
    dst_crs: pyproj.CRS,
    resampling: Resampling,
    fill_value: float = 0.0,
) -> np.ndarray:
    combined = np.full((patch_size, patch_size), fill_value, dtype=np.float32)
    for source in sources:
        patch = _sample_raster_patch(
            source,
            bounds=bounds,
            patch_size=patch_size,
            dst_crs=dst_crs,
            resampling=resampling,
            fill_value=fill_value,
        )
        valid = patch != fill_value
        combined[valid] = patch[valid]
    return combined.astype(np.float32)


def _select_geometries(frame: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    index = list(frame.sindex.intersection(bounds))
    if not index:
        return frame.iloc[0:0].copy()
    return frame.iloc[index].copy()


def _rasterize_patch(
    frame: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    patch_size: int,
    add_mode: bool = True,
    binary: bool = False,
) -> np.ndarray:
    subset = _select_geometries(frame, bounds)
    if subset.empty:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    transform = from_bounds(*bounds, width=patch_size, height=patch_size)
    layer = rasterize(
        [(geometry, 1.0) for geometry in subset.geometry],
        out_shape=(patch_size, patch_size),
        transform=transform,
        fill=0.0,
        all_touched=True,
        merge_alg=MergeAlg.add if add_mode else MergeAlg.replace,
        dtype="float32",
    )
    if binary:
        layer = (layer > 0).astype(np.float32)
    else:
        layer = np.log1p(layer).astype(np.float32)
    return layer


def build_multimodal_patch_dataset(
    input_path: Path,
    study_config_path: Path,
    output_path: Path,
    metadata_path: Path,
    patch_size: int = 64,
    context_m: float = 1500.0,
    feature_columns: Optional[Iterable[str]] = None,
    regression_target_col: str = "rwi_mean",
    classification_target_col: str = "rwi_bottom_quantile_flag",
    label_mask_col: str = "rwi_label_available",
) -> None:
    frame = read_table(Path(input_path))
    if not isinstance(frame, gpd.GeoDataFrame):
        raise ValueError("build-multimodal-patch-dataset requires a geospatial input file.")

    feature_columns = list(feature_columns or DEFAULT_TABULAR_FEATURE_COLUMNS)
    required = ["cell_id", "city", "country_iso", "geometry", regression_target_col, classification_target_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    study = load_study_config(Path(study_config_path))
    city_lookup = {city["name"]: city for city in study["city_details"]}
    output_path = Path(output_path)
    ensure_parent_dir(output_path)

    images = []
    tabular = []
    cell_ids = []
    cities = []
    regression_targets = []
    classification_targets = []
    label_mask = []

    for city_name, subset in frame.groupby("city"):
        if city_name not in city_lookup:
            raise ValueError("City '{city}' was not found in study config.".format(city=city_name))

        city_meta = city_lookup[city_name]
        layers = _load_city_raw_layers(city_meta["slug"])
        boundary = layers["boundary"]
        utm_crs = _utm_crs_for_gdf(boundary)

        roads_proj = layers["roads"].to_crs(utm_crs)
        road_nodes_proj = layers["road_nodes"].to_crs(utm_crs)
        amenities_proj = layers["amenities"].to_crs(utm_crs)
        service_amenities_proj = _representative_points(_filter_service_amenities(amenities_proj))
        school_amenities_proj = _representative_points(_filter_school_amenities(amenities_proj))
        clinic_amenities_proj = _representative_points(_filter_clinic_amenities(amenities_proj))
        intersections_proj = _intersection_nodes(road_nodes_proj)

        subset_proj = subset.to_crs(utm_crs).reset_index(drop=True)
        centroids_proj = subset_proj.copy()
        centroids_proj["geometry"] = centroids_proj.geometry.representative_point()

        worldpop_path = _worldpop_path_for_iso(city_meta["country_iso"])
        worldcover_paths = _worldcover_paths_for_city(city=city_meta, boundary=boundary)

        with ExitStack() as stack:
            worldpop_src = stack.enter_context(rasterio.open(worldpop_path))
            worldcover_sources = [stack.enter_context(rasterio.open(path)) for path in worldcover_paths]
            for _, row in centroids_proj.iterrows():
                x = float(row.geometry.x)
                y = float(row.geometry.y)
                half = float(context_m) / 2.0
                bounds = (x - half, y - half, x + half, y + half)

                worldpop_patch = _sample_raster_patch(
                    worldpop_src,
                    bounds=bounds,
                    patch_size=int(patch_size),
                    dst_crs=utm_crs,
                    resampling=Resampling.bilinear,
                    fill_value=0.0,
                )
                worldpop_patch = np.log1p(np.clip(worldpop_patch, a_min=0.0, a_max=None))

                worldcover_patch = _sample_mosaic_patch(
                    worldcover_sources,
                    bounds=bounds,
                    patch_size=int(patch_size),
                    dst_crs=utm_crs,
                    resampling=Resampling.nearest,
                    fill_value=0.0,
                )
                built_up_patch = (worldcover_patch == 50).astype(np.float32)
                open_space_patch = np.isin(worldcover_patch, [10, 20, 30, 60, 90, 95]).astype(np.float32)

                road_patch = _rasterize_patch(
                    roads_proj,
                    bounds=bounds,
                    patch_size=int(patch_size),
                    add_mode=False,
                    binary=True,
                )
                amenities_patch = _rasterize_patch(
                    service_amenities_proj,
                    bounds=bounds,
                    patch_size=int(patch_size),
                    add_mode=True,
                    binary=False,
                )
                schools_patch = _rasterize_patch(
                    school_amenities_proj,
                    bounds=bounds,
                    patch_size=int(patch_size),
                    add_mode=True,
                    binary=False,
                )
                clinics_patch = _rasterize_patch(
                    clinic_amenities_proj,
                    bounds=bounds,
                    patch_size=int(patch_size),
                    add_mode=True,
                    binary=False,
                )
                intersections_patch = _rasterize_patch(
                    intersections_proj,
                    bounds=bounds,
                    patch_size=int(patch_size),
                    add_mode=True,
                    binary=False,
                )

                images.append(
                    np.stack(
                        [
                            worldpop_patch,
                            built_up_patch,
                            open_space_patch,
                            road_patch,
                            amenities_patch,
                            schools_patch,
                            clinics_patch,
                            intersections_patch,
                        ],
                        axis=0,
                    ).astype(np.float16)
                )
                tabular.append(
                    pd.to_numeric(row[feature_columns], errors="coerce")
                    .fillna(np.nan)
                    .to_numpy(dtype=np.float32)
                )
                cell_ids.append(str(row["cell_id"]))
                cities.append(str(city_name))
                regression_targets.append(float(row[regression_target_col]) if pd.notna(row[regression_target_col]) else np.nan)
                classification_targets.append(float(bool(row[classification_target_col])) if pd.notna(row[classification_target_col]) else np.nan)
                label_mask.append(bool(row[label_mask_col]) if label_mask_col in row else pd.notna(row[regression_target_col]))

    payload = {
        "images": np.stack(images, axis=0),
        "tabular": np.stack(tabular, axis=0),
        "cell_id": np.asarray(cell_ids),
        "city": np.asarray(cities),
        "regression_target": np.asarray(regression_targets, dtype=np.float32),
        "classification_target": np.asarray(classification_targets, dtype=np.float32),
        "label_mask": np.asarray(label_mask, dtype=bool),
        "feature_names": np.asarray(feature_columns),
        "channel_names": np.asarray(DEFAULT_IMAGE_CHANNELS),
        "patch_size": np.asarray([int(patch_size)], dtype=np.int32),
        "context_m": np.asarray([float(context_m)], dtype=np.float32),
    }
    np.savez_compressed(output_path, **payload)

    metadata = {
        "input_path": str(Path(input_path)),
        "study_config_path": str(Path(study_config_path)),
        "output_path": str(output_path),
        "n_rows": int(len(cell_ids)),
        "n_labeled_rows": int(np.asarray(label_mask, dtype=bool).sum()),
        "feature_columns": feature_columns,
        "channel_names": DEFAULT_IMAGE_CHANNELS,
        "patch_size": int(patch_size),
        "context_m": float(context_m),
        "regression_target_col": regression_target_col,
        "classification_target_col": classification_target_col,
        "label_mask_col": label_mask_col,
    }
    write_json(metadata, Path(metadata_path))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(f1_score(y_true, y_pred))


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(balanced_accuracy_score(y_true, y_pred))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman") if len(y_true) > 1 else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_corr": float(pearson),
        "spearman_corr": float(spearman),
    }


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Optional[float]]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "average_precision": _safe_average_precision(y_true, y_prob),
        "f1": _safe_f1(y_true, y_pred),
        "balanced_accuracy": _safe_balanced_accuracy(y_true, y_pred),
    }


def _make_protocol_splits(
    city_values: np.ndarray,
    class_target: np.ndarray,
    label_mask: np.ndarray,
    random_state: int,
    strategy: str = "auto",
) -> list[dict[str, Any]]:
    labeled_indices = np.where(label_mask)[0]
    labeled_cities = city_values[labeled_indices].astype(str)
    labeled_classes = class_target[labeled_indices].astype(int)
    stratify_labels = np.asarray(
        ["{city}__{cls}".format(city=city, cls=cls) for city, cls in zip(labeled_cities, labeled_classes)]
    )
    unique_labels, counts = np.unique(stratify_labels, return_counts=True)
    can_stratify = bool(
        len(unique_labels) >= 2
        and counts.min() >= 2
        and int(round(len(labeled_indices) * 0.2)) >= len(unique_labels)
        and (len(labeled_indices) - int(round(len(labeled_indices) * 0.2))) >= len(unique_labels)
    )
    train_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=0.2,
        random_state=int(random_state),
        stratify=stratify_labels if can_stratify else None,
    )

    protocols = [
        {
            "protocol": "pooled_random",
            "train_idx": np.asarray(train_idx),
            "test_idx": np.asarray(test_idx),
        }
    ]
    unique_cities = sorted(set(labeled_cities.tolist()))
    if strategy == "auto":
        strategy = "pairwise" if len(unique_cities) == 2 else "leave_one_city_out"

    if strategy == "pairwise" and len(unique_cities) >= 2:
        protocols.append(
            {
                "protocol": "{source}_to_{target}".format(
                    source=unique_cities[0].lower().replace(" ", "_"),
                    target=unique_cities[1].lower().replace(" ", "_"),
                ),
                "train_idx": labeled_indices[labeled_cities == unique_cities[0]],
                "test_idx": labeled_indices[labeled_cities == unique_cities[1]],
            }
        )
        if len(unique_cities) == 2:
            protocols.append(
                {
                    "protocol": "{source}_to_{target}".format(
                        source=unique_cities[1].lower().replace(" ", "_"),
                        target=unique_cities[0].lower().replace(" ", "_"),
                    ),
                    "train_idx": labeled_indices[labeled_cities == unique_cities[1]],
                    "test_idx": labeled_indices[labeled_cities == unique_cities[0]],
                }
            )
    elif strategy == "leave_one_city_out" and len(unique_cities) >= 2:
        for city in unique_cities:
            holdout_mask = labeled_cities == city
            protocols.append(
                {
                    "protocol": "holdout_{city}".format(city=city.lower().replace(" ", "_")),
                    "train_idx": labeled_indices[~holdout_mask],
                    "test_idx": labeled_indices[holdout_mask],
                }
            )
    elif strategy == "all_pairs" and len(unique_cities) >= 2:
        for source in unique_cities:
            for target in unique_cities:
                if source == target:
                    continue
                protocols.append(
                    {
                        "protocol": "{source}_to_{target}".format(
                            source=source.lower().replace(" ", "_"),
                            target=target.lower().replace(" ", "_"),
                        ),
                        "train_idx": labeled_indices[labeled_cities == source],
                        "test_idx": labeled_indices[labeled_cities == target],
                    }
                )
    else:
        raise ValueError("Unsupported protocol strategy: {strategy}".format(strategy=strategy))
    return protocols


def _split_train_validation(indices: np.ndarray, class_target: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 10:
        return indices, indices
    stratify = class_target[indices].astype(int)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.15,
        random_state=int(random_state),
        stratify=stratify,
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def _standardize_tabular(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    means = np.nanmean(train, axis=0)
    stds = np.nanstd(train, axis=0)
    stds = np.where(stds <= 1e-6, 1.0, stds)
    filled = np.where(np.isnan(values), means, values)
    return ((filled - means) / stds).astype(np.float32)


def _standardize_images(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    means = train.mean(axis=(0, 2, 3), keepdims=True)
    stds = train.std(axis=(0, 2, 3), keepdims=True)
    stds = np.where(stds <= 1e-6, 1.0, stds)
    return ((values - means) / stds).astype(np.float32)


def _run_atlas_baseline(
    table: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    score_col: str,
    regression_target_col: str,
    classification_target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg_model = LinearRegression()
    cls_model = LogisticRegression(max_iter=500)

    x_train = table.loc[train_idx, [score_col]].to_numpy(dtype=float)
    y_reg_train = table.loc[train_idx, regression_target_col].to_numpy(dtype=float)
    y_cls_train = table.loc[train_idx, classification_target_col].to_numpy(dtype=int)
    x_test = table.loc[test_idx, [score_col]].to_numpy(dtype=float)
    y_reg_test = table.loc[test_idx, regression_target_col].to_numpy(dtype=float)
    y_cls_test = table.loc[test_idx, classification_target_col].to_numpy(dtype=int)

    reg_model.fit(-x_train, y_reg_train)
    cls_model.fit(-x_train, y_cls_train)
    reg_pred = reg_model.predict(-x_test)
    cls_prob = cls_model.predict_proba(-x_test)[:, 1]

    metrics = {
        **_regression_metrics(y_reg_test, reg_pred),
        **_classification_metrics(y_cls_test, cls_prob),
    }
    metrics_frame = pd.DataFrame(
        [
            {
                "model": "atlas_linear_baseline",
                "metric": key,
                "value": value,
            }
            for key, value in metrics.items()
        ]
    )
    predictions = pd.DataFrame(
        {
            "model": "atlas_linear_baseline",
            "cell_id": table.loc[test_idx, "cell_id"].to_numpy(),
            "city": table.loc[test_idx, "city"].to_numpy(),
            "rwi_true": y_reg_test,
            "rwi_pred": reg_pred,
            "low_wealth_true": y_cls_test,
            "low_wealth_prob": cls_prob,
        }
    )
    return metrics_frame, predictions


def _pretrain_autoencoder(
    images: np.ndarray,
    checkpoint_path: Path,
    metrics_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    class PatchEncoder(nn.Module):
        def __init__(self, in_channels: int, embedding_dim: int = 128):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.projection = nn.Linear(128, embedding_dim)

        def forward(self, x):
            x = self.features(x).flatten(1)
            return self.projection(x)

    class PatchAutoencoder(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.encoder = PatchEncoder(in_channels=in_channels, embedding_dim=128)
            self.decoder = nn.Sequential(
                nn.Linear(128, 8 * 8 * 128),
                nn.ReLU(),
                nn.Unflatten(1, (128, 8, 8)),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            return self.decoder(encoded)

    torch.manual_seed(int(random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.from_numpy(images.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)

    model = PatchAutoencoder(in_channels=images.shape[1]).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    history = []

    for epoch in range(int(epochs)):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            n_batches += 1
        history.append({"epoch": epoch + 1, "train_loss": running_loss / max(n_batches, 1)})

    module = model.module if hasattr(model, "module") else model
    checkpoint = {
        "encoder_state_dict": module.encoder.state_dict(),
        "history": history,
        "in_channels": int(images.shape[1]),
    }
    ensure_parent_dir(Path(checkpoint_path))
    torch.save(checkpoint, Path(checkpoint_path))
    write_json({"history": history, "checkpoint_path": str(Path(checkpoint_path))}, Path(metrics_path))


def pretrain_patch_autoencoder(
    dataset_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    epochs: int = 12,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> None:
    payload = _load_numpy_dataset(Path(dataset_path))
    _pretrain_autoencoder(
        images=payload["images"],
        checkpoint_path=Path(checkpoint_path),
        metrics_path=Path(metrics_path),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        random_state=int(random_state),
    )


def _run_xgboost_baseline(
    table: pd.DataFrame,
    feature_columns: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    regression_target_col: str,
    classification_target_col: str,
    use_gpu: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import xgboost as xgb

    x_train_raw = table.loc[train_idx, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_test_raw = table.loc[test_idx, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_train = _standardize_tabular(x_train_raw, x_train_raw)
    x_test = _standardize_tabular(x_train_raw, x_test_raw)

    y_reg_train = table.loc[train_idx, regression_target_col].to_numpy(dtype=float)
    y_reg_test = table.loc[test_idx, regression_target_col].to_numpy(dtype=float)
    y_cls_train = table.loc[train_idx, classification_target_col].to_numpy(dtype=int)
    y_cls_test = table.loc[test_idx, classification_target_col].to_numpy(dtype=int)

    regressor = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        device="cuda" if use_gpu else "cpu",
    )
    classifier = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        random_state=42,
        device="cuda" if use_gpu else "cpu",
        eval_metric="logloss",
        scale_pos_weight=float((y_cls_train == 0).sum() / max((y_cls_train == 1).sum(), 1)),
    )
    regressor.fit(x_train, y_reg_train)
    classifier.fit(x_train, y_cls_train)

    reg_pred = regressor.predict(x_test)
    cls_prob = classifier.predict_proba(x_test)[:, 1]

    metrics = {
        **_regression_metrics(y_reg_test, reg_pred),
        **_classification_metrics(y_cls_test, cls_prob),
    }
    metrics_frame = pd.DataFrame(
        [{"model": "xgboost_tabular", "metric": key, "value": value} for key, value in metrics.items()]
    )
    predictions = pd.DataFrame(
        {
            "model": "xgboost_tabular",
            "cell_id": table.loc[test_idx, "cell_id"].to_numpy(),
            "city": table.loc[test_idx, "city"].to_numpy(),
            "rwi_true": y_reg_test,
            "rwi_pred": reg_pred,
            "low_wealth_true": y_cls_test,
            "low_wealth_prob": cls_prob,
        }
    )
    return metrics_frame, predictions


def _make_torch_models(in_channels: int, tabular_dim: int):
    import torch
    from torch import nn

    class PatchEncoder(nn.Module):
        def __init__(self, in_channels: int, embedding_dim: int = 128):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.projection = nn.Linear(128, embedding_dim)

        def forward(self, x):
            return self.projection(self.features(x).flatten(1))

    class ImageMultiTask(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.encoder = PatchEncoder(in_channels=in_channels, embedding_dim=128)
            self.reg_head = nn.Linear(128, 1)
            self.cls_head = nn.Linear(128, 1)

        def forward(self, image, tabular=None):
            embedding = self.encoder(image)
            return self.reg_head(embedding).squeeze(-1), self.cls_head(embedding).squeeze(-1)

    class FusionMultiTask(nn.Module):
        def __init__(self, in_channels: int, tabular_dim: int):
            super().__init__()
            self.encoder = PatchEncoder(in_channels=in_channels, embedding_dim=128)
            self.tabular_net = nn.Sequential(
                nn.Linear(tabular_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.shared = nn.Sequential(
                nn.Linear(192, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.reg_head = nn.Linear(128, 1)
            self.cls_head = nn.Linear(128, 1)

        def forward(self, image, tabular):
            image_embedding = self.encoder(image)
            tabular_embedding = self.tabular_net(tabular)
            fused = self.shared(torch.cat([image_embedding, tabular_embedding], dim=1))
            return self.reg_head(fused).squeeze(-1), self.cls_head(fused).squeeze(-1)

    return torch, nn, PatchEncoder, ImageMultiTask, FusionMultiTask


def _build_pretrained_resnet_backbone(in_channels: int):
    import torch
    from torch import nn

    try:
        from torchvision.models import ResNet18_Weights
        from torchvision.models import resnet18
    except Exception as exc:  # pragma: no cover - fallback when torchvision is unavailable
        raise ImportError("torchvision is required for the pretrained ResNet benchmark.") from exc

    weight_source = "random_init"
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        weight_source = str(ResNet18_Weights.DEFAULT)
    except Exception:
        model = resnet18(weights=None)

    original_weight = model.conv1.weight.detach().clone()
    original_channels = int(original_weight.shape[1])
    model.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )
    with torch.no_grad():
        if in_channels == original_channels:
            expanded_weight = original_weight
        elif in_channels > original_channels:
            mean_channel = original_weight.mean(dim=1, keepdim=True)
            extra = mean_channel.repeat(1, in_channels - original_channels, 1, 1)
            expanded_weight = torch.cat([original_weight, extra], dim=1)
        else:
            expanded_weight = original_weight[:, :in_channels, :, :]
        expanded_weight = expanded_weight * (original_channels / float(max(in_channels, 1)))
        model.conv1.weight.copy_(expanded_weight)
    model.fc = nn.Identity()
    return model, weight_source


def _encode_images_with_patch_encoder(
    images: np.ndarray,
    pretrained_encoder_path: Optional[Path],
    batch_size: int,
    random_state: int,
) -> np.ndarray:
    torch, nn, PatchEncoder, _ImageMultiTask, _FusionMultiTask = _make_torch_models(
        in_channels=images.shape[1],
        tabular_dim=1,
    )
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    torch.manual_seed(int(random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PatchEncoder(in_channels=images.shape[1], embedding_dim=128)

    if pretrained_encoder_path and Path(pretrained_encoder_path).exists():
        checkpoint = torch.load(Path(pretrained_encoder_path), map_location="cpu")
        encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)

    encoder = encoder.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)

    dataset = TensorDataset(torch.from_numpy(images.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, drop_last=False)

    encoder.eval()
    outputs = []
    with torch.no_grad():
        for (image_batch,) in loader:
            embeddings = encoder(image_batch.to(device))
            outputs.append(embeddings.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _graph_coordinates(frame: pd.DataFrame) -> np.ndarray:
    if isinstance(frame, gpd.GeoDataFrame) and "geometry" in frame.columns:
        points = frame.copy()
        points["geometry"] = points.geometry.representative_point()
        if points.crs is not None:
            try:
                points = points.to_crs(_utm_crs_for_gdf(points))
            except Exception:
                pass
        coords = np.column_stack([points.geometry.x.to_numpy(dtype=float), points.geometry.y.to_numpy(dtype=float)])
    elif {"lon", "lat"}.issubset(frame.columns):
        coords = (
            frame.loc[:, ["lon", "lat"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
    else:
        raise ValueError("Graph benchmark requires geometry or lon/lat coordinates.")

    if np.isnan(coords).any():
        col_medians = np.nanmedian(coords, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        coords = np.where(np.isnan(coords), col_medians[None, :], coords)
    return coords.astype(np.float32)


def _build_graph_edge_index(
    frame: pd.DataFrame,
    city_col: str = "city",
    k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frame) == 0:
        raise ValueError("Cannot build a graph for an empty frame.")

    frame = frame.reset_index(drop=True)
    positions = np.arange(len(frame), dtype=np.int64)
    city_values = frame[city_col].astype(str).to_numpy()
    edge_pairs: set[tuple[int, int]] = {(int(i), int(i)) for i in positions}

    for city in pd.unique(city_values):
        city_positions = positions[city_values == city]
        if len(city_positions) <= 1:
            continue
        city_frame = frame.iloc[city_positions].copy()
        coords = _graph_coordinates(city_frame)
        neighbors = min(int(k) + 1, len(city_positions))
        model = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
        model.fit(coords)
        _distances, indices = model.kneighbors(coords)
        for local_i, neighbor_indices in enumerate(indices):
            source = int(city_positions[local_i])
            for local_j in neighbor_indices[1:]:
                target = int(city_positions[int(local_j)])
                edge_pairs.add((source, target))
                edge_pairs.add((target, source))

    ordered_edges = sorted(edge_pairs)
    rows = np.asarray([row for row, _ in ordered_edges], dtype=np.int64)
    cols = np.asarray([col for _, col in ordered_edges], dtype=np.int64)
    degrees = np.bincount(rows, minlength=len(frame)).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(degrees[rows] * degrees[cols], 1.0))
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    return edge_index, weights.astype(np.float32)


def _fit_torch_model(
    model_name: str,
    images: np.ndarray,
    tabular: np.ndarray,
    regression_target: np.ndarray,
    classification_target: np.ndarray,
    cell_ids: np.ndarray,
    cities: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pretrained_encoder_path: Optional[Path],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    torch, nn, _PatchEncoder, ImageMultiTask, FusionMultiTask = _make_torch_models(
        in_channels=images.shape[1],
        tabular_dim=tabular.shape[1],
    )
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    torch.manual_seed(int(random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx, val_idx = _split_train_validation(
        indices=np.asarray(train_idx),
        class_target=classification_target,
        random_state=int(random_state),
    )
    train_images = _standardize_images(images[train_idx], images[train_idx])
    val_images = _standardize_images(images[train_idx], images[val_idx])
    test_images = _standardize_images(images[train_idx], images[test_idx])

    train_tabular = _standardize_tabular(tabular[train_idx], tabular[train_idx])
    val_tabular = _standardize_tabular(tabular[train_idx], tabular[val_idx])
    test_tabular = _standardize_tabular(tabular[train_idx], tabular[test_idx])

    if model_name == "cnn_image":
        model = ImageMultiTask(in_channels=images.shape[1])
    elif model_name == "cnn_fusion":
        model = FusionMultiTask(in_channels=images.shape[1], tabular_dim=tabular.shape[1])
    else:
        raise ValueError("Unsupported model_name: {name}".format(name=model_name))

    if pretrained_encoder_path and Path(pretrained_encoder_path).exists():
        checkpoint = torch.load(Path(pretrained_encoder_path), map_location="cpu")
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)

    model = model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_dataset = TensorDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(train_tabular),
        torch.from_numpy(regression_target[train_idx].astype(np.float32)),
        torch.from_numpy(classification_target[train_idx].astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_images),
        torch.from_numpy(val_tabular),
        torch.from_numpy(regression_target[val_idx].astype(np.float32)),
        torch.from_numpy(classification_target[val_idx].astype(np.float32)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_images),
        torch.from_numpy(test_tabular),
        torch.from_numpy(regression_target[test_idx].astype(np.float32)),
        torch.from_numpy(classification_target[test_idx].astype(np.float32)),
    )

    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    mse_loss = nn.MSELoss()
    positives = max(int((classification_target[train_idx] == 1).sum()), 1)
    negatives = max(int((classification_target[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_state = None
    best_val_score = float("inf")
    patience = 4
    bad_epochs = 0

    for _epoch in range(int(epochs)):
        model.train()
        for image_batch, tabular_batch, reg_batch, cls_batch in train_loader:
            image_batch = image_batch.to(device)
            tabular_batch = tabular_batch.to(device)
            reg_batch = reg_batch.to(device)
            cls_batch = cls_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                reg_pred, cls_logits = model(image_batch, tabular_batch)
                loss = mse_loss(reg_pred, reg_batch) + bce_loss(cls_logits, cls_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_reg_true = []
        val_reg_pred = []
        with torch.no_grad():
            for image_batch, tabular_batch, reg_batch, _cls_batch in val_loader:
                image_batch = image_batch.to(device)
                tabular_batch = tabular_batch.to(device)
                reg_pred, _cls_logits = model(image_batch, tabular_batch)
                val_reg_true.append(reg_batch.numpy())
                val_reg_pred.append(reg_pred.detach().cpu().numpy())
        val_reg_true = np.concatenate(val_reg_true)
        val_reg_pred = np.concatenate(val_reg_pred)
        val_rmse = math.sqrt(mean_squared_error(val_reg_true, val_reg_pred))
        if val_rmse < best_val_score:
            best_val_score = val_rmse
            bad_epochs = 0
            module = model.module if hasattr(model, "module") else model
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    module = model.module if hasattr(model, "module") else model
    if best_state is not None:
        module.load_state_dict(best_state)

    model.eval()
    reg_true = []
    reg_pred = []
    cls_true = []
    cls_prob = []
    with torch.no_grad():
        for image_batch, tabular_batch, reg_batch, cls_batch in test_loader:
            image_batch = image_batch.to(device)
            tabular_batch = tabular_batch.to(device)
            pred_reg, pred_cls = model(image_batch, tabular_batch)
            reg_true.append(reg_batch.numpy())
            reg_pred.append(pred_reg.detach().cpu().numpy())
            cls_true.append(cls_batch.numpy())
            cls_prob.append(torch.sigmoid(pred_cls).detach().cpu().numpy())

    reg_true = np.concatenate(reg_true)
    reg_pred = np.concatenate(reg_pred)
    cls_true = np.concatenate(cls_true).astype(int)
    cls_prob = np.concatenate(cls_prob)

    metrics = {
        **_regression_metrics(reg_true, reg_pred),
        **_classification_metrics(cls_true, cls_prob),
    }
    metrics_frame = pd.DataFrame(
        [{"model": model_name, "metric": key, "value": value} for key, value in metrics.items()]
    )
    predictions = pd.DataFrame(
        {
            "model": model_name,
            "cell_id": cell_ids[test_idx],
            "city": cities[test_idx],
            "rwi_true": reg_true,
            "rwi_pred": reg_pred,
            "low_wealth_true": cls_true,
            "low_wealth_prob": cls_prob,
        }
    )
    return metrics_frame, predictions


def _fit_resnet_fusion_model(
    images: np.ndarray,
    tabular: np.ndarray,
    regression_target: np.ndarray,
    classification_target: np.ndarray,
    cell_ids: np.ndarray,
    cities: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    class ResNetFusionMultiTask(nn.Module):
        def __init__(self, in_channels: int, tabular_dim: int):
            super().__init__()
            self.encoder, self.backbone_source = _build_pretrained_resnet_backbone(in_channels=in_channels)
            self.tabular_net = nn.Sequential(
                nn.Linear(tabular_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.shared = nn.Sequential(
                nn.Linear(512 + 128, 256),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            self.reg_head = nn.Linear(128, 1)
            self.cls_head = nn.Linear(128, 1)

        def forward(self, image, tabular):
            image_embedding = self.encoder(image)
            tabular_embedding = self.tabular_net(tabular)
            fused = self.shared(torch.cat([image_embedding, tabular_embedding], dim=1))
            return self.reg_head(fused).squeeze(-1), self.cls_head(fused).squeeze(-1)

    torch.manual_seed(int(random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx, val_idx = _split_train_validation(
        indices=np.asarray(train_idx),
        class_target=classification_target,
        random_state=int(random_state),
    )
    train_images = _standardize_images(images[train_idx], images[train_idx])
    val_images = _standardize_images(images[train_idx], images[val_idx])
    test_images = _standardize_images(images[train_idx], images[test_idx])

    train_tabular = _standardize_tabular(tabular[train_idx], tabular[train_idx])
    val_tabular = _standardize_tabular(tabular[train_idx], tabular[val_idx])
    test_tabular = _standardize_tabular(tabular[train_idx], tabular[test_idx])

    model = ResNetFusionMultiTask(in_channels=images.shape[1], tabular_dim=tabular.shape[1]).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_dataset = TensorDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(train_tabular),
        torch.from_numpy(regression_target[train_idx].astype(np.float32)),
        torch.from_numpy(classification_target[train_idx].astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_images),
        torch.from_numpy(val_tabular),
        torch.from_numpy(regression_target[val_idx].astype(np.float32)),
        torch.from_numpy(classification_target[val_idx].astype(np.float32)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_images),
        torch.from_numpy(test_tabular),
        torch.from_numpy(regression_target[test_idx].astype(np.float32)),
        torch.from_numpy(classification_target[test_idx].astype(np.float32)),
    )

    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    reg_loss = nn.SmoothL1Loss(beta=0.25)
    positives = max(int((classification_target[train_idx] == 1).sum()), 1)
    negatives = max(int((classification_target[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_state = None
    best_val_score = float("inf")
    patience = 4
    bad_epochs = 0

    for _epoch in range(int(epochs)):
        model.train()
        for image_batch, tabular_batch, reg_batch, cls_batch in train_loader:
            image_batch = image_batch.to(device)
            tabular_batch = tabular_batch.to(device)
            reg_batch = reg_batch.to(device)
            cls_batch = cls_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                reg_pred, cls_logits = model(image_batch, tabular_batch)
                loss = reg_loss(reg_pred, reg_batch) + cls_loss(cls_logits, cls_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_reg_true = []
        val_reg_pred = []
        with torch.no_grad():
            for image_batch, tabular_batch, reg_batch, _cls_batch in val_loader:
                image_batch = image_batch.to(device)
                tabular_batch = tabular_batch.to(device)
                reg_pred, _cls_logits = model(image_batch, tabular_batch)
                val_reg_true.append(reg_batch.numpy())
                val_reg_pred.append(reg_pred.detach().cpu().numpy())
        val_reg_true = np.concatenate(val_reg_true)
        val_reg_pred = np.concatenate(val_reg_pred)
        val_rmse = math.sqrt(mean_squared_error(val_reg_true, val_reg_pred))
        if val_rmse < best_val_score:
            best_val_score = val_rmse
            bad_epochs = 0
            module = model.module if hasattr(model, "module") else model
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    module = model.module if hasattr(model, "module") else model
    if best_state is not None:
        module.load_state_dict(best_state)

    model.eval()
    reg_true = []
    reg_pred = []
    cls_true = []
    cls_prob = []
    with torch.no_grad():
        for image_batch, tabular_batch, reg_batch, cls_batch in test_loader:
            image_batch = image_batch.to(device)
            tabular_batch = tabular_batch.to(device)
            pred_reg, pred_cls = model(image_batch, tabular_batch)
            reg_true.append(reg_batch.numpy())
            reg_pred.append(pred_reg.detach().cpu().numpy())
            cls_true.append(cls_batch.numpy())
            cls_prob.append(torch.sigmoid(pred_cls).detach().cpu().numpy())

    reg_true = np.concatenate(reg_true)
    reg_pred = np.concatenate(reg_pred)
    cls_true = np.concatenate(cls_true).astype(int)
    cls_prob = np.concatenate(cls_prob)

    metrics = {
        **_regression_metrics(reg_true, reg_pred),
        **_classification_metrics(cls_true, cls_prob),
    }
    metrics_frame = pd.DataFrame(
        [{"model": "resnet_fusion_pretrained", "metric": key, "value": value} for key, value in metrics.items()]
    )
    predictions = pd.DataFrame(
        {
            "model": "resnet_fusion_pretrained",
            "cell_id": cell_ids[test_idx],
            "city": cities[test_idx],
            "rwi_true": reg_true,
            "rwi_pred": reg_pred,
            "low_wealth_true": cls_true,
            "low_wealth_prob": cls_prob,
        }
    )
    return metrics_frame, predictions


def _fit_graph_fusion_model(
    table: pd.DataFrame,
    images: np.ndarray,
    tabular: np.ndarray,
    regression_target: np.ndarray,
    classification_target: np.ndarray,
    cell_ids: np.ndarray,
    cities: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pretrained_encoder_path: Optional[Path],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_state: int,
    graph_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch
    from torch import nn

    class GraphConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
            super().__init__()
            self.self_linear = nn.Linear(in_dim, out_dim)
            self.neighbor_linear = nn.Linear(in_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, adjacency):
            neighbor_features = torch.sparse.mm(adjacency, x)
            output = self.self_linear(x) + self.neighbor_linear(neighbor_features)
            output = self.norm(output)
            output = torch.relu(output)
            return self.dropout(output)

    class GraphFusionMultiTask(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, 192),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(192, 128),
                nn.ReLU(),
            )
            self.graph_layer1 = GraphConv(128, 128, dropout=0.1)
            self.graph_layer2 = GraphConv(128, 128, dropout=0.1)
            self.reg_head = nn.Linear(128, 1)
            self.cls_head = nn.Linear(128, 1)

        def forward(self, x, adjacency):
            base = self.input_proj(x)
            hidden = self.graph_layer1(base, adjacency)
            hidden = self.graph_layer2(hidden, adjacency) + base
            return self.reg_head(hidden).squeeze(-1), self.cls_head(hidden).squeeze(-1)

    torch.manual_seed(int(random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx, val_idx = _split_train_validation(
        indices=np.asarray(train_idx),
        class_target=classification_target,
        random_state=int(random_state),
    )
    standardized_images = _standardize_images(images[train_idx], images)
    image_embeddings = _encode_images_with_patch_encoder(
        images=standardized_images,
        pretrained_encoder_path=pretrained_encoder_path,
        batch_size=min(int(batch_size), 512),
        random_state=int(random_state),
    )
    standardized_tabular = _standardize_tabular(tabular[train_idx], tabular)
    node_features = np.concatenate([image_embeddings, standardized_tabular], axis=1).astype(np.float32)

    edge_index, edge_weight = _build_graph_edge_index(
        frame=table.reset_index(drop=True),
        city_col="city",
        k=int(graph_k),
    )
    if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
        torch.sparse.check_sparse_tensor_invariants.disable()
    adjacency = torch.sparse_coo_tensor(
        indices=torch.from_numpy(edge_index),
        values=torch.from_numpy(edge_weight),
        size=(len(table), len(table)),
        dtype=torch.float32,
        check_invariants=False,
    ).coalesce().to(device)

    features_tensor = torch.from_numpy(node_features).to(device)
    regression_tensor = torch.from_numpy(regression_target.astype(np.float32)).to(device)
    classification_tensor = torch.from_numpy(classification_target.astype(np.float32)).to(device)
    train_mask = torch.zeros(len(table), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(table), dtype=torch.bool, device=device)
    test_mask = torch.zeros(len(table), dtype=torch.bool, device=device)
    train_mask[np.asarray(train_idx, dtype=np.int64)] = True
    val_mask[np.asarray(val_idx, dtype=np.int64)] = True
    test_mask[np.asarray(test_idx, dtype=np.int64)] = True

    model = GraphFusionMultiTask(input_dim=node_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    reg_loss = nn.SmoothL1Loss(beta=0.25)
    positives = max(int((classification_target[train_idx] == 1).sum()), 1)
    negatives = max(int((classification_target[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_score = float("inf")
    patience = 6
    bad_epochs = 0

    for _epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reg_pred, cls_logits = model(features_tensor, adjacency)
        loss = reg_loss(reg_pred[train_mask], regression_tensor[train_mask]) + cls_loss(
            cls_logits[train_mask],
            classification_tensor[train_mask],
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_reg_pred, _val_cls_logits = model(features_tensor, adjacency)
            val_rmse = math.sqrt(
                mean_squared_error(
                    regression_tensor[val_mask].detach().cpu().numpy(),
                    val_reg_pred[val_mask].detach().cpu().numpy(),
                )
            )
        if val_rmse < best_val_score:
            best_val_score = val_rmse
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        reg_pred, cls_logits = model(features_tensor, adjacency)

    reg_true = regression_tensor[test_mask].detach().cpu().numpy()
    reg_pred_np = reg_pred[test_mask].detach().cpu().numpy()
    cls_true = classification_tensor[test_mask].detach().cpu().numpy().astype(int)
    cls_prob = torch.sigmoid(cls_logits[test_mask]).detach().cpu().numpy()

    metrics = {
        **_regression_metrics(reg_true, reg_pred_np),
        **_classification_metrics(cls_true, cls_prob),
    }
    metrics_frame = pd.DataFrame(
        [{"model": "graph_fusion", "metric": key, "value": value} for key, value in metrics.items()]
    )
    predictions = pd.DataFrame(
        {
            "model": "graph_fusion",
            "cell_id": cell_ids[test_idx],
            "city": cities[test_idx],
            "rwi_true": reg_true,
            "rwi_pred": reg_pred_np,
            "low_wealth_true": cls_true,
            "low_wealth_prob": cls_prob,
        }
    )
    return metrics_frame, predictions


def run_multimodal_rwi_benchmark(
    input_path: Path,
    dataset_path: Path,
    metrics_output_path: Path,
    predictions_output_path: Path,
    metadata_path: Path,
    pretrained_encoder_path: Optional[Path] = None,
    feature_columns: Optional[Iterable[str]] = None,
    model_names: Optional[Iterable[str]] = None,
    score_col: str = "deprivation_index_0_100",
    regression_target_col: str = "rwi_mean",
    classification_target_col: str = "rwi_bottom_quantile_flag",
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    protocol_strategy: str = "auto",
    graph_k: int = 8,
) -> None:
    table = read_table(Path(input_path))
    feature_columns = list(feature_columns or DEFAULT_TABULAR_FEATURE_COLUMNS)
    model_names = list(model_names or DEFAULT_BENCHMARK_MODELS)
    payload = _load_numpy_dataset(Path(dataset_path))

    table_geo = None
    if isinstance(table, gpd.GeoDataFrame):
        table_geo = table.set_index("cell_id").loc[payload["cell_id"]].reset_index()
        table = pd.DataFrame(table_geo.drop(columns="geometry"))
    else:
        table = table.copy().set_index("cell_id").loc[payload["cell_id"]].reset_index()
    label_mask = payload["label_mask"].astype(bool)
    city_values = payload["city"].astype(str)
    regression_target = payload["regression_target"].astype(np.float32)
    classification_target = payload["classification_target"].astype(np.float32)
    images = payload["images"].astype(np.float32)
    tabular = payload["tabular"].astype(np.float32)
    cell_ids = payload["cell_id"].astype(str)

    protocols = _make_protocol_splits(
        city_values=city_values,
        class_target=classification_target,
        label_mask=label_mask,
        random_state=int(random_state),
        strategy=str(protocol_strategy),
    )
    metrics_frames = []
    prediction_frames = []

    for protocol in protocols:
        train_idx = protocol["train_idx"]
        test_idx = protocol["test_idx"]

        protocol_metrics = []
        protocol_predictions = []

        if "atlas_linear_baseline" in model_names:
            atlas_metrics, atlas_predictions = _run_atlas_baseline(
                table=table,
                train_idx=train_idx,
                test_idx=test_idx,
                score_col=score_col,
                regression_target_col=regression_target_col,
                classification_target_col=classification_target_col,
            )
            protocol_metrics.append(atlas_metrics)
            protocol_predictions.append(atlas_predictions)

        if "xgboost_tabular" in model_names:
            xgb_metrics, xgb_predictions = _run_xgboost_baseline(
                table=table,
                feature_columns=feature_columns,
                train_idx=train_idx,
                test_idx=test_idx,
                regression_target_col=regression_target_col,
                classification_target_col=classification_target_col,
                use_gpu=bool(payload["images"].shape[0] > 0),
            )
            protocol_metrics.append(xgb_metrics)
            protocol_predictions.append(xgb_predictions)

        if "cnn_image" in model_names:
            image_metrics, image_predictions = _fit_torch_model(
                model_name="cnn_image",
                images=images,
                tabular=tabular,
                regression_target=regression_target,
                classification_target=classification_target,
                cell_ids=cell_ids,
                cities=city_values,
                train_idx=train_idx,
                test_idx=test_idx,
                pretrained_encoder_path=pretrained_encoder_path,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                random_state=int(random_state),
            )
            protocol_metrics.append(image_metrics)
            protocol_predictions.append(image_predictions)

        if "cnn_fusion" in model_names:
            fusion_metrics, fusion_predictions = _fit_torch_model(
                model_name="cnn_fusion",
                images=images,
                tabular=tabular,
                regression_target=regression_target,
                classification_target=classification_target,
                cell_ids=cell_ids,
                cities=city_values,
                train_idx=train_idx,
                test_idx=test_idx,
                pretrained_encoder_path=pretrained_encoder_path,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                random_state=int(random_state),
            )
            protocol_metrics.append(fusion_metrics)
            protocol_predictions.append(fusion_predictions)

        if "resnet_fusion_pretrained" in model_names:
            resnet_metrics, resnet_predictions = _fit_resnet_fusion_model(
                images=images,
                tabular=tabular,
                regression_target=regression_target,
                classification_target=classification_target,
                cell_ids=cell_ids,
                cities=city_values,
                train_idx=train_idx,
                test_idx=test_idx,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                random_state=int(random_state),
            )
            protocol_metrics.append(resnet_metrics)
            protocol_predictions.append(resnet_predictions)

        if "graph_fusion" in model_names:
            if table_geo is None:
                raise ValueError("graph_fusion requires a geospatial benchmark input with geometry.")
            graph_metrics, graph_predictions = _fit_graph_fusion_model(
                table=table_geo,
                images=images,
                tabular=tabular,
                regression_target=regression_target,
                classification_target=classification_target,
                cell_ids=cell_ids,
                cities=city_values,
                train_idx=train_idx,
                test_idx=test_idx,
                pretrained_encoder_path=pretrained_encoder_path,
                epochs=int(max(epochs, 12)),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                random_state=int(random_state),
                graph_k=int(graph_k),
            )
            protocol_metrics.append(graph_metrics)
            protocol_predictions.append(graph_predictions)

        for frame in protocol_metrics:
            frame.insert(0, "protocol", protocol["protocol"])
        for frame in protocol_predictions:
            frame.insert(0, "protocol", protocol["protocol"])

        metrics_frames.extend(protocol_metrics)
        prediction_frames.extend(protocol_predictions)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ensure_parent_dir(Path(metrics_output_path))
    metrics.to_csv(Path(metrics_output_path), index=False)
    predictions.to_csv(Path(predictions_output_path), index=False)
    metadata = {
        "input_path": str(Path(input_path)),
        "dataset_path": str(Path(dataset_path)),
        "metrics_output_path": str(Path(metrics_output_path)),
        "predictions_output_path": str(Path(predictions_output_path)),
        "pretrained_encoder_path": str(Path(pretrained_encoder_path)) if pretrained_encoder_path else None,
        "feature_columns": feature_columns,
        "model_names": model_names,
        "protocols": [protocol["protocol"] for protocol in protocols],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "random_state": int(random_state),
        "protocol_strategy": str(protocol_strategy),
        "graph_k": int(graph_k),
    }
    write_json(metadata, Path(metadata_path))


def summarize_multimodal_benchmark(
    metrics_input_path: Path,
    output_path: Path,
) -> None:
    metrics = pd.read_csv(Path(metrics_input_path))
    records = []
    rank_records = []
    benchmark_summary: dict[str, Any] = {"protocols": {}}
    metric_directions = [
        ("rmse", "min"),
        ("mae", "min"),
        ("spearman_corr", "max"),
        ("roc_auc", "max"),
        ("average_precision", "max"),
        ("balanced_accuracy", "max"),
    ]

    for protocol, subset in metrics.groupby("protocol"):
        protocol_summary: dict[str, Any] = {}
        for metric, direction in metric_directions:
            metric_subset = subset.loc[subset["metric"] == metric].dropna(subset=["value"]).copy()
            if metric_subset.empty:
                continue
            if direction == "min":
                ordered = metric_subset.sort_values("value", ascending=True).reset_index(drop=True)
            else:
                ordered = metric_subset.sort_values("value", ascending=False).reset_index(drop=True)
            winner = ordered.iloc[0]
            protocol_summary[metric] = {
                "winner_model": str(winner["model"]),
                "value": float(winner["value"]),
                "direction": direction,
            }
            records.append(
                {
                    "protocol": str(protocol),
                    "metric": metric,
                    "winner_model": str(winner["model"]),
                    "value": float(winner["value"]),
                    "direction": direction,
                }
            )
            for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
                rank_records.append(
                    {
                        "protocol": str(protocol),
                        "metric": metric,
                        "model": str(row["model"]),
                        "rank": rank,
                    }
                )
        benchmark_summary["protocols"][str(protocol)] = protocol_summary

    average_ranks: dict[str, Any] = {}
    if rank_records:
        rank_frame = pd.DataFrame(rank_records)
        scope_masks = {
            "all_protocols": rank_frame.index == rank_frame.index,
            "transfer_only": ~rank_frame["protocol"].astype(str).eq("pooled_random"),
            "pooled_only": rank_frame["protocol"].astype(str).eq("pooled_random"),
        }
        for scope_name, scope_mask in scope_masks.items():
            scoped = rank_frame.loc[scope_mask].copy()
            if scoped.empty:
                continue
            average_ranks[scope_name] = {}
            for metric, metric_subset in scoped.groupby("metric"):
                ordered = (
                    metric_subset.groupby("model", as_index=False)["rank"]
                    .mean()
                    .rename(columns={"rank": "mean_rank"})
                    .sort_values("mean_rank", ascending=True)
                )
                average_ranks[scope_name][str(metric)] = [
                    {
                        "model": str(row["model"]),
                        "mean_rank": float(row["mean_rank"]),
                    }
                    for _, row in ordered.iterrows()
                ]

    write_json(
        {
            "metrics_input_path": str(Path(metrics_input_path)),
            "leaders": records,
            "by_protocol": benchmark_summary["protocols"],
            "average_ranks": average_ranks,
        },
        Path(output_path),
    )


def build_benchmark_findings_artifact(
    metrics_input_path: Path,
    summary_input_path: Path,
    output_path: Path,
) -> None:
    metrics = pd.read_csv(Path(metrics_input_path))
    summary = json.loads(Path(summary_input_path).read_text())
    protocol_summaries = summary.get("by_protocol", {})
    pooled_summary = protocol_summaries.get("pooled_random", {})
    holdout_protocols = [protocol for protocol in protocol_summaries.keys() if protocol != "pooled_random"]

    hardest_holdout_rmse = None
    hardest_holdout_ap = None
    if holdout_protocols:
        rmse_records = [
            {
                "protocol": protocol,
                "best_rmse": float(protocol_summaries[protocol]["rmse"]["value"]),
                "winner_model": str(protocol_summaries[protocol]["rmse"]["winner_model"]),
            }
            for protocol in holdout_protocols
            if "rmse" in protocol_summaries[protocol]
        ]
        ap_records = [
            {
                "protocol": protocol,
                "best_average_precision": float(protocol_summaries[protocol]["average_precision"]["value"]),
                "winner_model": str(protocol_summaries[protocol]["average_precision"]["winner_model"]),
            }
            for protocol in holdout_protocols
            if "average_precision" in protocol_summaries[protocol]
        ]
        if rmse_records:
            hardest_holdout_rmse = max(rmse_records, key=lambda item: item["best_rmse"])
        if ap_records:
            hardest_holdout_ap = min(ap_records, key=lambda item: item["best_average_precision"])

    atlas_win_records = [
        record for record in summary.get("leaders", []) if str(record.get("winner_model")) == "atlas_linear_baseline"
    ]

    recommendations = {}
    for protocol in holdout_protocols:
        protocol_summary = protocol_summaries.get(protocol, {})
        recommendations[str(protocol)] = {
            "regression_recommendation": protocol_summary.get("rmse"),
            "ranking_recommendation": protocol_summary.get("spearman_corr"),
            "screening_recommendation": protocol_summary.get("average_precision"),
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_input_path": str(Path(metrics_input_path)),
        "summary_input_path": str(Path(summary_input_path)),
        "n_models": int(metrics["model"].nunique()),
        "n_protocols": int(metrics["protocol"].nunique()),
        "protocols": sorted(metrics["protocol"].astype(str).unique().tolist()),
        "pooled_random": pooled_summary,
        "holdout_protocols": {protocol: protocol_summaries.get(protocol, {}) for protocol in holdout_protocols},
        "transfer_only_average_ranks": summary.get("average_ranks", {}).get("transfer_only", {}),
        "all_protocol_average_ranks": summary.get("average_ranks", {}).get("all_protocols", {}),
        "hardest_holdouts": {
            "rmse": hardest_holdout_rmse,
            "average_precision": hardest_holdout_ap,
        },
        "atlas_retained_value": {
            "n_metric_wins": int(len(atlas_win_records)),
            "wins": atlas_win_records,
        },
        "per_holdout_recommendations": recommendations,
    }
    write_json(payload, Path(output_path))
