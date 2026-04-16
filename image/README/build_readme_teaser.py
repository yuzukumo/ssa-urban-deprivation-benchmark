from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "image" / "README" / "readme_teaser.png"
ASSETS = {
    "map": ROOT / "outputs" / "figures" / "paper_core" / "figure_01_city_score_map.png",
    "rmse": ROOT / "outputs" / "figures" / "ml" / "core6_rwi_benchmark_rmse.png",
    "districts": ROOT / "outputs" / "figures" / "paper_core" / "figure_09_admin_priority_class.png",
    "contrast": ROOT / "outputs" / "figures" / "paper_core" / "figure_05_hotspot_dimension_contrast.png",
}

WIDTH = 1800
HEIGHT = 1080
MARGIN = 72
PANEL_BG = "#f8f3eb"
TEXT_MAIN = "#1b1d1f"
TEXT_MUTED = "#5b5f63"
ACCENT = "#8c2d19"
ACCENT_2 = "#0b3d91"
ACCENT_3 = "#2f6b3b"
ACCENT_4 = "#5a3e9b"
CARD_BORDER = "#d8cdc0"
CANVAS = "#f6f1e8"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = max((resized.width - target_w) // 2, 0)
    top = max((resized.height - target_h) // 2, 0)
    return resized.crop((left, top, left + target_w, top + target_h))


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


def _paste_panel(base: Image.Image, image_path: Path, box: tuple[int, int, int, int], label: str) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    radius = 30
    shadow = Image.new("RGBA", (w + 18, h + 18), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((8, 8, w + 8, h + 8), radius=radius, fill=(0, 0, 0, 72))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    base.alpha_composite(shadow, (x0 - 8, y0 - 2))

    panel = Image.new("RGBA", (w, h), ImageColor.getrgb(PANEL_BG) + (255,))
    image = _cover(Image.open(image_path).convert("RGB"), (w, h - 42)).convert("RGBA")
    panel.alpha_composite(image, (0, 42))

    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, outline=CARD_BORDER, width=2)
    draw.rounded_rectangle((18, 10, 18 + len(label) * 14 + 22, 10 + 24), radius=12, fill=(255, 255, 255, 230))
    draw.text((30, 13), label, font=_load_font(18, bold=True), fill=TEXT_MAIN)

    mask = _rounded_mask((w, h), radius)
    panel.putalpha(mask)
    base.alpha_composite(panel, (x0, y0))


def _draw_chip(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: str) -> int:
    font = _load_font(22, bold=True)
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0] + 34
    height = 40
    draw.rounded_rectangle((x, y, x + width, y + height), radius=20, fill=fill)
    draw.text((x + 17, y + 8), text, font=font, fill="white")
    return width


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    base = Image.new("RGBA", (WIDTH, HEIGHT), CANVAS)
    draw = ImageDraw.Draw(base)

    draw.rounded_rectangle((0, 0, WIDTH, 248), radius=0, fill="#efe6d8")
    draw.rectangle((MARGIN, 70, MARGIN + 14, 172), fill=ACCENT)

    title_font = _load_font(58, bold=True)
    subtitle_font = _load_font(28, bold=False)
    body_font = _load_font(24, bold=False)

    draw.text(
        (MARGIN + 36, 58),
        "Open Multimodal Benchmark for\nUrban Deprivation in Sub-Saharan Africa",
        font=title_font,
        fill=TEXT_MAIN,
        spacing=8,
    )
    draw.text(
        (MARGIN + 36, 184),
        "Public spatial data, interpretable atlas baseline, six-city transfer benchmark, and external validation.",
        font=subtitle_font,
        fill=TEXT_MUTED,
    )

    chips = [
        ("6 cities", ACCENT),
        ("11,895 labeled cells", ACCENT_2),
        ("500 m urban grids", ACCENT_3),
        ("RWI weak supervision", ACCENT_4),
    ]
    chip_x = MARGIN + 36
    chip_y = 232
    for text, fill in chips:
        chip_x += _draw_chip(draw, chip_x, chip_y, text, fill) + 14

    note_x = MARGIN + 36
    note_y = 302
    note = (
        "Current result layers: deprivation atlas, leave-one-city-out multimodal benchmark, "
        "district prioritization, GHSL validation, and VIIRS validation."
    )
    draw.text((note_x, note_y), note, font=body_font, fill=TEXT_MUTED)

    left_x = MARGIN
    right_x = WIDTH - MARGIN
    top_y = 372
    gap = 28
    map_w = 980
    right_w = right_x - left_x - map_w - gap
    row_h = 280
    bottom_h = 340

    _paste_panel(
        base,
        ASSETS["map"],
        (left_x, top_y, left_x + map_w, HEIGHT - MARGIN),
        "City-level deprivation scores",
    )
    _paste_panel(
        base,
        ASSETS["rmse"],
        (left_x + map_w + gap, top_y, right_x, top_y + row_h),
        "Six-city benchmark RMSE",
    )
    _paste_panel(
        base,
        ASSETS["districts"],
        (left_x + map_w + gap, top_y + row_h + gap, left_x + map_w + gap + right_w // 2 - gap // 2, HEIGHT - MARGIN),
        "District priority classes",
    )
    _paste_panel(
        base,
        ASSETS["contrast"],
        (left_x + map_w + gap + right_w // 2 + gap // 2, top_y + row_h + gap, right_x, HEIGHT - MARGIN),
        "Hotspot dimension contrast",
    )

    base.convert("RGB").save(OUTPUT, quality=95)


if __name__ == "__main__":
    main()
