import re
from math import floor
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import geopandas as gpd
from ssa_urban_deprivation_benchmark.io_utils import read_yaml


def load_study_config(path: Path) -> Dict[str, Any]:
    return read_yaml(Path(path))


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def _worldcover_tile_code(south_edge: int, west_edge: int) -> str:
    ns = ("N" if south_edge >= 0 else "S") + "{value:02d}".format(value=abs(int(south_edge)))
    ew = ("E" if west_edge >= 0 else "W") + "{value:03d}".format(value=abs(int(west_edge)))
    return ns + ew


def worldcover_tiles_for_bounds(bounds: Iterable[float]) -> List[str]:
    minx, miny, maxx, maxy = [float(value) for value in bounds]
    west_edges = range(int(floor(minx / 3.0) * 3), int(floor(maxx / 3.0) * 3) + 1, 3)
    south_edges = range(int(floor(miny / 3.0) * 3), int(floor(maxy / 3.0) * 3) + 1, 3)
    return [
        _worldcover_tile_code(south_edge=south_edge, west_edge=west_edge)
        for south_edge in south_edges
        for west_edge in west_edges
    ]


def resolve_city_worldcover_tiles(
    city: Dict[str, Any],
    boundary: Optional[gpd.GeoDataFrame] = None,
) -> List[str]:
    explicit_tiles = city.get("worldcover_tiles")
    if explicit_tiles:
        return sorted({str(tile) for tile in explicit_tiles})

    explicit_tile = city.get("worldcover_tile")
    if explicit_tile:
        return [str(explicit_tile)]

    if boundary is None or boundary.empty:
        raise ValueError(
            "City '{city}' is missing worldcover_tile(s) and no boundary was provided for inference.".format(
                city=city.get("name", "unknown")
            )
        )

    boundary_wgs84 = boundary.to_crs("EPSG:4326")
    return worldcover_tiles_for_bounds(boundary_wgs84.total_bounds)
