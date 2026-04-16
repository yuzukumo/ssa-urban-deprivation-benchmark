from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from ssa_urban_deprivation_benchmark.io_utils import read_table


def _pretty_label(value: str) -> str:
    return str(value).replace("_", " ")


def _save_histogram(frame: pd.DataFrame, score_col: str, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(frame[score_col], bins=12, color="#c55a11", edgecolor="white")
    plt.xlabel(_pretty_label(score_col))
    plt.ylabel("Count")
    plt.title("Distribution of deprivation scores")
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=180)
    plt.close()


def _save_city_boxplot(frame: pd.DataFrame, score_col: str, group_col: str, output_dir: Path) -> None:
    groups = []
    labels = []
    for label, subset in frame.groupby(group_col):
        labels.append(str(label))
        groups.append(subset[score_col].values)

    if not groups:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(groups, labels=labels, patch_artist=True)
    plt.ylabel(_pretty_label(score_col))
    plt.title("Score distribution by group")
    plt.tight_layout()
    plt.savefig(output_dir / "score_by_group.png", dpi=180)
    plt.close()


def _save_top_bottom_chart(
    frame: pd.DataFrame,
    score_col: str,
    id_col: str,
    output_dir: Path,
    top_n: int = 8,
) -> None:
    ordered = frame.sort_values(score_col)
    lowest = ordered.head(top_n)
    highest = ordered.tail(top_n).sort_values(score_col)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(lowest[id_col].astype(str), lowest[score_col], color="#4c78a8")
    axes[0].set_title("Lowest deprivation cells")
    axes[0].set_xlabel(_pretty_label(score_col))

    axes[1].barh(highest[id_col].astype(str), highest[score_col], color="#b22222")
    axes[1].set_title("Highest deprivation cells")
    axes[1].set_xlabel(_pretty_label(score_col))

    plt.tight_layout()
    plt.savefig(output_dir / "top_bottom_cells.png", dpi=180)
    plt.close()


def _save_point_map(
    frame: pd.DataFrame,
    score_col: str,
    lon_col: str,
    lat_col: str,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(7.5, 7))
    scatter = plt.scatter(
        frame[lon_col],
        frame[lat_col],
        c=frame[score_col],
        cmap="YlOrRd",
        s=80,
        edgecolors="black",
        linewidths=0.3,
    )
    plt.colorbar(scatter, label=_pretty_label(score_col))
    plt.xlabel(_pretty_label(lon_col))
    plt.ylabel(_pretty_label(lat_col))
    plt.title("Quicklook point map")
    plt.tight_layout()
    plt.savefig(output_dir / "quicklook_point_map.png", dpi=180)
    plt.close()


def create_quicklook_outputs(
    input_path: Path,
    score_col: str,
    output_dir: Path,
    id_col: Optional[str] = None,
    group_col: Optional[str] = None,
    lon_col: Optional[str] = None,
    lat_col: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if score_col not in frame.columns:
        raise ValueError("Column '{column}' not found.".format(column=score_col))

    _save_histogram(frame, score_col, output_dir)

    if group_col and group_col in frame.columns:
        _save_city_boxplot(frame, score_col, group_col, output_dir)

    if id_col and id_col in frame.columns:
        _save_top_bottom_chart(frame, score_col, id_col, output_dir)

    if lon_col and lat_col and lon_col in frame.columns and lat_col in frame.columns:
        _save_point_map(frame, score_col, lon_col, lat_col, output_dir)


def _ensure_geodataframe(input_path: Path):
    frame = read_table(Path(input_path))
    if not hasattr(frame, "geometry"):
        raise ValueError("Map plotting requires a geospatial input file.")
    return frame


def _subplot_layout(count: int):
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    return rows, columns


def _axes_list(fig, rows: int, columns: int):
    axes = fig.subplots(rows, columns)
    if isinstance(axes, np.ndarray):
        return axes.reshape(-1)
    return [axes]


def plot_score_map(
    input_path: Path,
    score_col: str,
    output_path: Path,
    group_col: Optional[str] = None,
    cmap: str = "YlOrRd",
) -> None:
    gdf = _ensure_geodataframe(Path(input_path))
    if score_col not in gdf.columns:
        raise ValueError("Column '{column}' not found.".format(column=score_col))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vmin = float(gdf[score_col].quantile(0.01))
    vmax = float(gdf[score_col].quantile(0.99))
    if vmin == vmax:
        vmin = float(gdf[score_col].min())
        vmax = float(gdf[score_col].max())

    if group_col and group_col in gdf.columns and gdf[group_col].nunique() > 1:
        groups = list(gdf.groupby(group_col))
        rows, columns = _subplot_layout(len(groups))
        fig = plt.figure(figsize=(6.5 * columns, 6 * rows))
        axes = _axes_list(fig, rows, columns)

        for index, (label, subset) in enumerate(groups):
            ax = axes[index]
            subset.plot(
                column=score_col,
                cmap=cmap,
                ax=ax,
                linewidth=0,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(str(label))
            ax.set_axis_off()

        for ax in axes[len(groups):]:
            ax.set_axis_off()

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.subplots_adjust(left=0.02, right=0.9, bottom=0.02, top=0.92, wspace=0.02, hspace=0.08)
        cax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
        fig.colorbar(sm, cax=cax, label=_pretty_label(score_col))
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.5, 7))
    gdf.plot(column=score_col, cmap=cmap, ax=ax, linewidth=0, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(_pretty_label(score_col))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label=_pretty_label(score_col))
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_category_map(
    input_path: Path,
    category_col: str,
    output_path: Path,
    group_col: Optional[str] = None,
    palette_name: str = "tab10",
) -> None:
    gdf = _ensure_geodataframe(Path(input_path))
    if category_col not in gdf.columns:
        raise ValueError("Column '{column}' not found.".format(column=category_col))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [str(value) for value in pd.Series(gdf[category_col]).fillna("missing").unique()]
    categories = sorted(categories)
    cmap = mpl.cm.get_cmap(palette_name, max(len(categories), 3))
    color_lookup = {category: cmap(index) for index, category in enumerate(categories)}

    def draw(ax, subset, title):
        subset = subset.copy()
        subset["_plot_category"] = subset[category_col].fillna("missing").astype(str)
        for category in categories:
            part = subset.loc[subset["_plot_category"] == category]
            if part.empty:
                continue
            part.plot(ax=ax, color=color_lookup[category], linewidth=0)
        ax.set_title(title)
        ax.set_axis_off()

    if group_col and group_col in gdf.columns and gdf[group_col].nunique() > 1:
        groups = list(gdf.groupby(group_col))
        rows, columns = _subplot_layout(len(groups))
        fig = plt.figure(figsize=(6.5 * columns, 6 * rows))
        axes = _axes_list(fig, rows, columns)
        for index, (label, subset) in enumerate(groups):
            draw(axes[index], subset, str(label))
        for ax in axes[len(groups):]:
            ax.set_axis_off()
        handles = [
            mpl.patches.Patch(color=color_lookup[category], label=_pretty_label(category))
            for category in categories
        ]
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)))
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.5, 7))
    draw(ax, gdf, category_col)
    handles = [
        mpl.patches.Patch(color=color_lookup[category], label=_pretty_label(category))
        for category in categories
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)))
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hotspot_map(
    input_path: Path,
    hotspot_col: str,
    output_path: Path,
    group_col: Optional[str] = None,
) -> None:
    gdf = _ensure_geodataframe(Path(input_path))
    if hotspot_col not in gdf.columns:
        raise ValueError("Column '{column}' not found.".format(column=hotspot_col))

    hotspot_order = [
        "high_high",
        "low_low",
        "high_low",
        "low_high",
        "not_significant",
        "unknown",
        "missing",
    ]
    color_lookup = {
        "high_high": "#b2182b",
        "low_low": "#2166ac",
        "high_low": "#ef8a62",
        "low_high": "#67a9cf",
        "not_significant": "#d9d9d9",
        "unknown": "#7f7f7f",
        "missing": "#f0f0f0",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def draw(ax, subset, title):
        subset = subset.copy()
        subset["_hotspot"] = subset[hotspot_col].fillna("missing").astype(str)
        for category in hotspot_order:
            part = subset.loc[subset["_hotspot"] == category]
            if part.empty:
                continue
            part.plot(ax=ax, color=color_lookup[category], linewidth=0)
        ax.set_title(title)
        ax.set_axis_off()

    if group_col and group_col in gdf.columns and gdf[group_col].nunique() > 1:
        groups = list(gdf.groupby(group_col))
        rows, columns = _subplot_layout(len(groups))
        fig = plt.figure(figsize=(6.5 * columns, 6 * rows))
        axes = _axes_list(fig, rows, columns)
        for index, (label, subset) in enumerate(groups):
            draw(axes[index], subset, str(label))
        for ax in axes[len(groups):]:
            ax.set_axis_off()
        handles = [
            mpl.patches.Patch(color=color_lookup[category], label=_pretty_label(category))
            for category in hotspot_order
            if category in set(gdf[hotspot_col].fillna("missing").astype(str))
        ]
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)))
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.5, 7))
    draw(ax, gdf, hotspot_col)
    handles = [
        mpl.patches.Patch(color=color_lookup[category], label=_pretty_label(category))
        for category in hotspot_order
        if category in set(gdf[hotspot_col].fillna("missing").astype(str))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)))
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_by_group(
    input_path: Path,
    x_col: str,
    y_col: str,
    output_path: Path,
    group_col: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    for column in [x_col, y_col]:
        if column not in frame.columns:
            raise ValueError("Column '{column}' not found.".format(column=column))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if group_col and group_col in frame.columns and frame[group_col].nunique() > 1:
        groups = list(frame.groupby(group_col))
        rows, columns = _subplot_layout(len(groups))
        fig = plt.figure(figsize=(6.5 * columns, 5.5 * rows))
        axes = _axes_list(fig, rows, columns)

        global_min = min(float(frame[x_col].min()), float(frame[y_col].min()))
        global_max = max(float(frame[x_col].max()), float(frame[y_col].max()))

        for index, (label, subset) in enumerate(groups):
            ax = axes[index]
            ax.scatter(subset[x_col], subset[y_col], s=8, alpha=0.35, color="#c55a11")
            ax.plot([global_min, global_max], [global_min, global_max], linestyle="--", color="black", linewidth=1)
            ax.set_title(str(label))
            ax.set_xlabel(_pretty_label(x_col))
            ax.set_ylabel(_pretty_label(y_col))

        for ax in axes[len(groups):]:
            ax.set_axis_off()

        fig.tight_layout()
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(frame[x_col], frame[y_col], s=8, alpha=0.35, color="#c55a11")
    global_min = min(float(frame[x_col].min()), float(frame[y_col].min()))
    global_max = max(float(frame[x_col].max()), float(frame[y_col].max()))
    ax.plot([global_min, global_max], [global_min, global_max], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel(_pretty_label(x_col))
    ax.set_ylabel(_pretty_label(y_col))
    ax.set_title("{x} vs {y}".format(x=_pretty_label(x_col), y=_pretty_label(y_col)))
    fig.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(
    input_path: Path,
    x_col: str,
    y_col: str,
    output_path: Path,
    hue_col: Optional[str] = None,
) -> None:
    frame = read_table(Path(input_path))
    for column in [x_col, y_col]:
        if column not in frame.columns:
            raise ValueError("Column '{column}' not found.".format(column=column))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5.5))

    if hue_col and hue_col in frame.columns:
        x_values = [str(value) for value in frame[x_col].dropna().unique()]
        hue_values = [str(value) for value in frame[hue_col].dropna().unique()]
        x_values = sorted(x_values)
        hue_values = sorted(hue_values)
        x_positions = np.arange(len(x_values))
        width = 0.8 / max(len(hue_values), 1)

        for index, hue_value in enumerate(hue_values):
            subset = frame.loc[frame[hue_col].astype(str) == hue_value].copy()
            subset = subset.set_index(subset[x_col].astype(str))
            heights = [float(subset.loc[value, y_col]) if value in subset.index else np.nan for value in x_values]
            offset = (index - (len(hue_values) - 1) / 2.0) * width
            plt.bar(x_positions + offset, heights, width=width, label=_pretty_label(hue_value))

        plt.xticks(x_positions, x_values)
        plt.legend(title=_pretty_label(hue_col))
    else:
        labels = frame[x_col].astype(str).tolist()
        heights = pd.to_numeric(frame[y_col], errors="coerce").tolist()
        plt.bar(labels, heights, color="#c55a11")

    plt.xlabel(_pretty_label(x_col))
    plt.ylabel(_pretty_label(y_col))
    plt.title("{y} by {x}".format(y=_pretty_label(y_col), x=_pretty_label(x_col)).title())
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_contrast_heatmap(
    input_path: Path,
    feature_col: str,
    group_col: str,
    value_col: str,
    output_path: Path,
    sort_by_abs_mean: bool = True,
    cmap: str = "RdBu_r",
) -> None:
    frame = read_table(Path(input_path))
    required = [feature_col, group_col, value_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    working = frame[[feature_col, group_col, value_col]].copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna()
    if working.empty:
        raise ValueError("No plottable rows found for the requested heatmap.")

    pivot = working.pivot(index=feature_col, columns=group_col, values=value_col)
    if sort_by_abs_mean:
        order = pivot.abs().mean(axis=1).sort_values(ascending=False).index
        pivot = pivot.loc[order]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = pivot.to_numpy(dtype=float)
    vlim = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 1.0

    fig_width = max(5.5, 1.8 * len(pivot.columns) + 2.5)
    fig_height = max(4.0, 0.5 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap=cmap, aspect="auto", vmin=-vlim, vmax=vlim)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(column) for column in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_pretty_label(label) for label in pivot.index])
    ax.set_xlabel(_pretty_label(group_col))
    ax.set_ylabel(_pretty_label(feature_col))
    ax.set_title("{value} by {feature} and {group}".format(
        value=_pretty_label(value_col),
        feature=_pretty_label(feature_col),
        group=_pretty_label(group_col),
    ).title())

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text_color = "white" if abs(value) >= 0.55 * vlim else "black"
            ax.text(
                column_index,
                row_index,
                "{value:.2f}".format(value=value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=_pretty_label(value_col))
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_faceted_heatmap(
    input_path: Path,
    facet_col: str,
    x_col: str,
    y_col: str,
    value_col: str,
    output_path: Path,
    cmap: str = "RdBu_r",
) -> None:
    frame = read_table(Path(input_path))
    required = [facet_col, x_col, y_col, value_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: {columns}".format(columns=", ".join(missing)))

    working = frame[required].copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna()
    if working.empty:
        raise ValueError("No plottable rows found for the requested faceted heatmap.")

    facets = sorted(working[facet_col].astype(str).unique())
    x_values = sorted(working[x_col].astype(str).unique())
    y_order = (
        working.assign(_y=working[y_col].astype(str))
        .groupby("_y")[value_col]
        .apply(lambda series: float(series.abs().mean()))
        .sort_values(ascending=False)
        .index.tolist()
    )

    values = working[value_col].to_numpy(dtype=float)
    vlim = float(np.nanmax(np.abs(values)))
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 1.0

    fig_width = max(6.0, 3.2 * len(facets) + 1.5)
    fig_height = max(4.5, 0.5 * len(y_order) + 1.5)
    fig, axes = plt.subplots(
        1,
        len(facets),
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.reshape(-1)

    for index, facet_value in enumerate(facets):
        ax = axes[index]
        subset = working.loc[working[facet_col].astype(str) == facet_value].copy()
        pivot = (
            subset.assign(_x=subset[x_col].astype(str), _y=subset[y_col].astype(str))
            .pivot(index="_y", columns="_x", values=value_col)
            .reindex(index=y_order, columns=x_values)
        )
        image = ax.imshow(pivot.to_numpy(dtype=float), cmap=cmap, aspect="auto", vmin=-vlim, vmax=vlim)
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([_pretty_label(value) for value in x_values], rotation=0)
        ax.set_yticks(np.arange(len(y_order)))
        ax.set_yticklabels([_pretty_label(value) for value in y_order] if index == 0 else [])
        ax.set_title(str(facet_value))
        ax.set_xlabel(_pretty_label(x_col))

        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.iloc[row_index, column_index]
                if pd.isna(value):
                    continue
                text_color = "white" if abs(float(value)) >= 0.55 * vlim else "black"
                ax.text(
                    column_index,
                    row_index,
                    "{value:.2f}".format(value=float(value)),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    axes[0].set_ylabel(_pretty_label(y_col))
    fig.suptitle(
        "{value} by {y} and {x}".format(
            value=_pretty_label(value_col),
            y=_pretty_label(y_col),
            x=_pretty_label(x_col),
        ).title(),
        y=0.98,
    )
    fig.colorbar(image, ax=axes.tolist(), fraction=0.03, pad=0.02, label=_pretty_label(value_col))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
