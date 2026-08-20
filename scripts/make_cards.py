#!/usr/bin/env python3
"""
make_cards.py - render title / ranking cards as clips for MoneyPrinterTurbo.

Why this exists
---------------
MoneyPrinterTurbo draws text only for subtitles, so a countdown or explainer
video has no way to name the thing it is talking about on screen. Viewers of a
"top 5" list expect to see the rank and the name, not just hear them.

The cards are rendered as ordinary clips, so nothing in the render pipeline has
to change: a card takes the place of one clip in --video-materials. Point a
card at the beat where the narrator announces the item and the timeline keeps
exactly the duration it had, which is what stops the visuals drifting out of
sync with the narration.

Each card is drawn once with Pillow at twice the output size, then zoomed
slowly by ffmpeg, so the text stays sharp instead of being scaled up from a
frame-sized bitmap.

Spec file
---------
  {
    "aspect": "16:9",
    "font": "BeVietnamPro-Bold.ttf",
    "cards": [
      {"name": "sb_title",
       "eyebrow": "NUMBER FIVE",
       "title": "SPRINGBANK 15",
       "subtitle": "Campbeltown",
       "meta": ["46% ABV", "Sherry casks", "Non-chill-filtered"],
       "duration": 6.04,
       "accent": "#C8892B",
       "background": "storage/clips/scsb_place1.mp4"}
    ]
  }

``background`` is optional and may be an image or a video; a video contributes
one frame from its middle. It is blurred and darkened so the text stays
readable. Without one the card uses a vertical gradient tinted by ``accent``.

Examples
--------
  uv run python scripts/make_cards.py --spec cards.json
  uv run python scripts/make_cards.py --spec cards.json --preview
  uv run python scripts/make_cards.py --spec cards.json --fade 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(REPO_ROOT, "resource", "fonts")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "storage", "cards")

ASPECTS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}

# Cards are drawn at twice the output size: ffmpeg's zoom samples from this
# bitmap, and starting at output size makes the text soften as it zooms.
SUPERSAMPLE = 2

# Default frame rate matches the clips fetch_clips.py produces, so a card can
# be concatenated with them without a rate conversion.
DEFAULT_FPS = "30000/1001"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"error: {name} is required but was not found on PATH")


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"colour must be #RRGGBB, got {value!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = name if os.path.isabs(name) else os.path.join(FONT_DIR, name)
    if not os.path.isfile(path):
        sys.exit(f"error: font not found: {path}")
    return ImageFont.truetype(path, size)


def text_size(draw: ImageDraw.ImageDraw, text: str, font, tracking: int = 0):
    """Measure text as drawn from a top-left origin.

    The height is the bounding box's *bottom* edge rather than its height:
    Pillow places a glyph's ink below the origin by its top bearing, so
    subtracting that bearing makes tall lines measure short and the next block
    in the stack overlaps them.
    """
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + tracking * max(0, len(text) - 1)
    return width, box[3]


def draw_tracked(draw, xy, text: str, font, fill, tracking: int = 0) -> None:
    """Pillow has no letter-spacing, and cramped capitals read as a single
    word, so wide-set labels are drawn one glyph at a time."""
    if tracking <= 0:
        draw.text(xy, text, font=font, fill=fill)
        return
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        x += draw.textlength(char, font=font) + tracking


def fit_font(draw, text: str, font_name: str, start: int, max_width: int, minimum: int = 24):
    """Shrink a font until the line fits, so a long name cannot run off frame."""
    size = start
    while size > minimum:
        font = load_font(font_name, size)
        if text_size(draw, text, font)[0] <= max_width:
            return font
        size -= 2
    return load_font(font_name, minimum)


def background_image(spec: dict, size: tuple[int, int], accent: tuple[int, int, int]) -> Image.Image:
    """A blurred still behind the text, or a tinted gradient when none is given."""
    width, height = size
    source = spec.get("background")
    if source:
        path = source if os.path.isabs(source) else os.path.join(REPO_ROOT, source)
        if not os.path.isfile(path):
            sys.exit(f"error: background not found: {path}")
        frame = path
        tmp = None
        if os.path.splitext(path)[1].lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            probe = run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path]
            )
            try:
                middle = max(0.0, float(probe.stdout.strip()) / 2)
            except ValueError:
                middle = 0.0
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            proc = run(["ffmpeg", "-v", "error", "-ss", f"{middle:.2f}", "-i", path,
                        "-frames:v", "1", tmp, "-y"])
            if proc.returncode != 0 or not os.path.isfile(tmp):
                sys.exit(f"error: could not read a frame from {path}")
            frame = tmp
        image = Image.open(frame).convert("RGB")
        if tmp:
            os.unlink(tmp)
        # Cover the frame without distorting it.
        scale = max(width / image.width, height / image.height)
        image = image.resize((int(image.width * scale) + 1, int(image.height * scale) + 1))
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        image = image.filter(ImageFilter.GaussianBlur(radius=max(8, width // 90)))
        return Image.blend(image, Image.new("RGB", size, (0, 0, 0)), 0.62)

    top_rgb = tuple(int(c * 0.30) for c in accent)
    gradient = Image.new("RGB", (1, height))
    pixels = gradient.load()
    for y in range(height):
        ratio = y / max(1, height - 1)
        pixels[0, y] = tuple(int(top_rgb[i] * (1 - ratio) + 8 * ratio) for i in range(3))
    return gradient.resize(size)


def render_card(spec: dict, size: tuple[int, int], font_name: str) -> Image.Image:
    width, height = (dim * SUPERSAMPLE for dim in size)
    accent = hex_to_rgb(spec.get("accent", "#C8892B"))
    canvas = background_image(spec, (width, height), accent)
    draw = ImageDraw.Draw(canvas)

    margin = int(width * 0.10)
    max_width = width - margin * 2
    unit = height / 1080 / SUPERSAMPLE * SUPERSAMPLE  # scales with output height

    eyebrow = spec.get("eyebrow", "")
    title = spec.get("title", "")
    subtitle = spec.get("subtitle", "")
    meta = spec.get("meta", []) or []

    f_eyebrow = load_font(font_name, int(40 * unit / 1))
    f_title = fit_font(draw, title, font_name, int(130 * unit), max_width)
    f_sub = load_font(font_name, int(52 * unit))
    f_meta = load_font(font_name, int(32 * unit))
    tracking = int(10 * unit)

    meta_line = "   ·   ".join(str(m) for m in meta)
    gap = int(28 * unit)
    rule_h = max(2, int(4 * unit))

    blocks = []
    if eyebrow:
        blocks.append(("eyebrow", text_size(draw, eyebrow, f_eyebrow, tracking)[1], f_eyebrow))
    if title:
        blocks.append(("title", text_size(draw, title, f_title)[1], f_title))
    if subtitle or meta_line:
        blocks.append(("rule", rule_h, None))
    if subtitle:
        blocks.append(("subtitle", text_size(draw, subtitle, f_sub)[1], f_sub))
    if meta_line:
        blocks.append(("meta", text_size(draw, meta_line, f_meta, tracking)[1], f_meta))

    total = sum(b[1] for b in blocks) + gap * max(0, len(blocks) - 1)
    y = (height - total) // 2

    for kind, block_h, font in blocks:
        if kind == "eyebrow":
            w = text_size(draw, eyebrow, font, tracking)[0]
            draw_tracked(draw, ((width - w) // 2, y), eyebrow, font, accent, tracking)
        elif kind == "title":
            w = text_size(draw, title, font)[0]
            # A soft shadow keeps white text legible over a bright background.
            draw.text(((width - w) // 2 + int(3 * unit), y + int(3 * unit)),
                      title, font=font, fill=(0, 0, 0))
            draw.text(((width - w) // 2, y), title, font=font, fill=(255, 255, 255))
        elif kind == "rule":
            rule_w = int(width * 0.14)
            x0 = (width - rule_w) // 2
            draw.rectangle([x0, y, x0 + rule_w, y + rule_h], fill=accent)
        elif kind == "subtitle":
            w = text_size(draw, subtitle, font)[0]
            draw.text(((width - w) // 2, y), subtitle, font=font, fill=(232, 232, 232))
        elif kind == "meta":
            w = text_size(draw, meta_line, font, tracking)[0]
            draw_tracked(draw, ((width - w) // 2, y), meta_line, font,
                         (196, 196, 196), tracking)
        y += block_h + gap

    return canvas


def card_to_clip(png: str, out: str, duration: float, size, fps: str,
                 fade: float, zoom: float) -> bool:
    width, height = size
    frames = max(1, int(round(duration * 30)))
    filters = [
        f"zoompan=z='min(zoom+{(zoom - 1) / max(1, frames):.6f},{zoom})'"
        f":d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={width}x{height}:fps={fps}"
    ]
    if fade > 0:
        filters.append(f"fade=t=in:st=0:d={fade}")
        filters.append(f"fade=t=out:st={max(0, duration - fade):.3f}:d={fade}")
    filters.append("format=yuv420p")
    proc = run([
        "ffmpeg", "-v", "error", "-loop", "1", "-i", png,
        "-t", f"{duration:.3f}", "-vf", ",".join(filters),
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-r", fps, out, "-y",
    ])
    if proc.returncode != 0:
        print(f"  [fail] {os.path.basename(out)}: {proc.stderr.strip()[-200:]}")
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render title / ranking cards as clips for --video-materials.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spec", required=True, help="JSON card specification")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="output directory")
    parser.add_argument("--aspect", choices=sorted(ASPECTS), help="override the spec aspect")
    parser.add_argument("--font", help="override the spec font")
    parser.add_argument("--fps", default=DEFAULT_FPS, help="output frame rate")
    parser.add_argument("--fade", type=float, default=0.0,
                        help="seconds of fade at each end; 0 cuts hard")
    parser.add_argument("--zoom", type=float, default=1.06,
                        help="how far the slow push-in travels; 1.0 disables it")
    parser.add_argument("--preview", action="store_true",
                        help="also write the still PNG next to each clip")
    args = parser.parse_args(argv)

    require_tool("ffmpeg")
    require_tool("ffprobe")

    with open(args.spec) as fh:
        spec = json.load(fh)

    aspect = args.aspect or spec.get("aspect", "16:9")
    if aspect not in ASPECTS:
        sys.exit(f"error: aspect must be one of {', '.join(sorted(ASPECTS))}")
    size = ASPECTS[aspect]
    font_name = args.font or spec.get("font", "BeVietnamPro-Bold.ttf")

    cards = spec.get("cards") or []
    if not cards:
        sys.exit("error: the spec contains no cards")

    os.makedirs(args.outdir, exist_ok=True)
    produced = []
    for card in cards:
        name = card.get("name")
        if not name:
            sys.exit("error: every card needs a name")
        duration = float(card.get("duration", 5.0))
        if duration <= 0:
            sys.exit(f"error: card {name} needs a positive duration")

        image = render_card(card, size, font_name)
        image = image.resize(size, Image.LANCZOS) if args.zoom <= 1.0 else image
        png = os.path.join(args.outdir, f"{name}.png")
        image.save(png)

        out = os.path.join(args.outdir, f"{name}.mp4")
        if card_to_clip(png, out, duration, size, args.fps, args.fade, args.zoom):
            produced.append(out)
            print(f"  [ok] {os.path.basename(out)}  {size[0]}x{size[1]}  {duration:.2f}s")
        if not args.preview:
            os.unlink(png)

    print(f"\n{len(produced)} card(s) in {args.outdir}")
    if produced:
        print("\nUse with MoneyPrinterTurbo by putting a card in place of the clip\n"
              "that covers its narration, keeping --video-materials in order.")
    return 0 if len(produced) == len(cards) else 1


if __name__ == "__main__":
    raise SystemExit(main())
