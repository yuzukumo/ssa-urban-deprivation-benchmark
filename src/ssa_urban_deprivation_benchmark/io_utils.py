import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


GEO_EXTENSIONS = {".geojson", ".gpkg", ".shp"}


def ensure_parent_dir(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(payload: Any, path: Path) -> None:
    ensure_parent_dir(Path(path))
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in GEO_EXTENSIONS:
        geopandas = require_optional_dependency("geopandas", "geo")
        return geopandas.read_file(path)

    raise ValueError(
        "Unsupported input format for {path}. Supported formats: .csv, .tsv, .json, "
        ".geojson, .gpkg, .shp".format(path=path)
    )


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    ensure_parent_dir(path)

    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(frame.to_dict(orient="records"), handle, indent=2, ensure_ascii=False)
        return
    if suffix in GEO_EXTENSIONS:
        geopandas = require_optional_dependency("geopandas", "geo")
        if not isinstance(frame, geopandas.GeoDataFrame):
            raise ValueError("Geospatial outputs require a GeoDataFrame.")
        driver = {
            ".geojson": "GeoJSON",
            ".gpkg": "GPKG",
            ".shp": None,
        }.get(suffix)
        if driver:
            frame.to_file(path, driver=driver, engine="pyogrio")
        else:
            frame.to_file(path)
        return

    raise ValueError("Unsupported output format for {path}.".format(path=path))


def require_optional_dependency(package_name: str, extra_name: str):
    try:
        return __import__(package_name)
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency '{package_name}'. Install the '{extra_name}' "
            "extra from pyproject.toml before running this command.".format(
                package_name=package_name,
                extra_name=extra_name,
            )
        ) from exc
