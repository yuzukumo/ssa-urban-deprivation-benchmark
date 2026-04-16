from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

from ssa_urban_deprivation_benchmark.io_utils import read_yaml


DISPLAY_FIELDS = [
    ("id", "id"),
    ("route", "route"),
    ("status", "status"),
    ("access", "access"),
    ("approval_required", "approval"),
    ("recommended_role", "role"),
    ("local_target", "target"),
]


def load_catalog(path: Path) -> List[Dict]:
    payload = read_yaml(Path(path))
    datasets = payload.get("datasets", [])
    return datasets


def filter_catalog(
    datasets: Iterable[Dict],
    route: Optional[str] = None,
    status: Optional[str] = None,
    approval_required: Optional[bool] = None,
) -> List[Dict]:
    filtered = []
    for dataset in datasets:
        if route and dataset.get("route") != route:
            continue
        if status and dataset.get("status") != status:
            continue
        if approval_required is not None and dataset.get("approval_required") != approval_required:
            continue
        filtered.append(dataset)
    return filtered


def format_catalog_table(datasets: Iterable[Dict]) -> str:
    rows = []
    headers = [label for _, label in DISPLAY_FIELDS]

    for dataset in datasets:
        rows.append(
            [
                str(dataset.get(field, ""))
                for field, _ in DISPLAY_FIELDS
            ]
        )

    if not rows:
        return "No dataset entries matched the current filter."

    widths = []
    for index, header in enumerate(headers):
        content_width = max(len(row[index]) for row in rows)
        widths.append(max(len(header), content_width))

    def render_row(values: List[str]) -> str:
        padded = [
            value.ljust(widths[index])
            for index, value in enumerate(values)
        ]
        return " | ".join(padded)

    header_line = render_row(headers)
    divider = "-+-".join("-" * width for width in widths)
    body = "\n".join(render_row(row) for row in rows)
    return "\n".join([header_line, divider, body])
