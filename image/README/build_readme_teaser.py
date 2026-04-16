from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL import ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "image" / "README"

ASSETS = {
    "teaser_map": ROOT / "outputs" / "figures" / "paper_core" / "figure_01_city_score_map.png",
    "teaser_rmse": ROOT / "outputs" / "figures" / "ml" / "core6_rwi_benchmark_rmse.png",
    "teaser_districts": ROOT / "outputs" / "figures" / "paper_core" / "figure_09_admin_priority_class.png",
    "teaser_contrast": ROOT / "outputs" / "figures" / "paper_core" / "figure_05_hotspot_dimension_contrast.png",
    "absolute_relative": ROOT / "outputs" / "figures" / "paper_core" / "figure_04_absolute_relative_scatter.png",
}

WHITE = (255, 255, 255)
BORDER = "#d7d7d7"


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = max((resized.width - target_w) // 2, 0)
    top = max((resized.height - target_h) // 2, 0)
    return resized.crop((left, top, left + target_w, top + target_h))


def _contain(image: Image.Image, size: tuple[int, int], bg: tuple[int, int, int] = WHITE) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    resized = image.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    x = (target_w - resized.width) // 2
    y = (target_h - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _save_downscaled(src: Path, dst: Path, max_width: int) -> None:
    image = Image.open(src).convert("RGB")
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)
    image.save(dst, quality=95)


def _draw_panel(base: Image.Image, image: Image.Image, box: tuple[int, int, int, int], mode: str = "cover") -> None:
    x0, y0, x1, y1 = box
    panel_w = x1 - x0
    panel_h = y1 - y0
    if mode == "contain":
        panel = _contain(image, (panel_w, panel_h))
    else:
        panel = _cover(image, (panel_w, panel_h))
    base.paste(panel, (x0, y0))
    draw = ImageDraw.Draw(base)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=BORDER, width=2)


def build_teaser() -> None:
    width = 1800
    height = 1060
    margin = 42
    gap = 28

    canvas = Image.new("RGB", (width, height), WHITE)
    inner_w = width - margin * 2
    top_h = 560
    bottom_h = height - margin * 2 - top_h - gap
    small_w = (inner_w - gap * 2) // 3

    map_image = Image.open(ASSETS["teaser_map"]).convert("RGB")
    rmse_image = Image.open(ASSETS["teaser_rmse"]).convert("RGB")
    district_image = Image.open(ASSETS["teaser_districts"]).convert("RGB")
    contrast_image = Image.open(ASSETS["teaser_contrast"]).convert("RGB")

    _draw_panel(
        canvas,
        map_image,
        (margin, margin, width - margin, margin + top_h),
        mode="contain",
    )
    bottom_y0 = margin + top_h + gap
    _draw_panel(
        canvas,
        rmse_image,
        (margin, bottom_y0, margin + small_w, bottom_y0 + bottom_h),
        mode="contain",
    )
    _draw_panel(
        canvas,
        district_image,
        (margin + small_w + gap, bottom_y0, margin + small_w * 2 + gap, bottom_y0 + bottom_h),
        mode="contain",
    )
    _draw_panel(
        canvas,
        contrast_image,
        (margin + small_w * 2 + gap * 2, bottom_y0, width - margin, bottom_y0 + bottom_h),
        mode="contain",
    )

    canvas.save(OUT_DIR / "readme_teaser.png", quality=95)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_teaser()
    _save_downscaled(ASSETS["absolute_relative"], OUT_DIR / "readme_absolute_relative.png", max_width=1400)
    _save_downscaled(ASSETS["teaser_rmse"], OUT_DIR / "readme_benchmark_rmse.png", max_width=1400)


if __name__ == "__main__":
    main()
