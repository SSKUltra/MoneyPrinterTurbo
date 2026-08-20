#!/usr/bin/env python3
"""
fetch_clips.py - source pre-cut B-roll clips for MoneyPrinterTurbo local materials.

Why this exists
---------------
`--video-source local` gives deterministic clip ordering (clip N plays over
script paragraph N), which stock-footage search cannot guarantee. But it needs
short, pre-cut files sitting in storage/local_videos.

This tool produces exactly that, in two passes:

  Pass 1 (analysis, cheap)
      Download a tiny 144p proxy, sample one frame every N seconds, burn the
      timestamp into each frame, tile them into contact sheets, and ask Gemini
      *once per source* which timestamps match your wanted shots.
      Images cost ~258 tokens per 768px tile, versus a few hundred tokens per
      SECOND for native video input - roughly a 50-100x saving.

  Pass 2 (fetch, exact)
      yt-dlp downloads only the chosen time ranges at full quality using
      --download-sections, so a 10 minute source never lands on disk.

Use --manual to skip Gemini entirely (zero tokens).

Beat-level selection (--plan)
-----------------------------
One narration beat usually has several candidate source videos, and asking one
model call per source cannot compare them: it commits to source A before it has
ever seen source C. --plan reads a JSON file describing each beat - its subject,
its wanted shots, its measured duration and its candidate sources - and runs
four stages per beat:

  1. Rank    every candidate source's sheets go into ONE call, labelled by
             source, and the model returns a ranked pool of {source, index,
             want_number, score}. Same images, same token cost, one call per
             beat instead of one per source.
  2. Fill    code alone decides what ships: real cut-free seconds per shot, a
             per-shot and per-source cap, a minimum score, deduplication, and
             an honest shortfall when the budget cannot be met.
  3. Verify   each selected clip is shown to the model cold - only the crop
             that will be published, no sheet, no index, no hint that anything
             chose it - and rejected if it has burned-in overlays, no sign of
             the beat's subject, or a subject that is not the main focus.
  4. Replace a rejected clip is swapped for the next-best candidate and
             re-verified, for a bounded number of rounds.

--cut-plan then cuts exactly the spans that passed and runs --preflight on them.
The model never emits a timestamp in any stage: it names cells and sources, and
this side owns every conversion to seconds.

Picking shots yourself (zero tokens, and more accurate)
------------------------------------------------------
Vision models hallucinate shot contents confidently, so the reliable workflow is
a three-stage funnel where a human or agent looks at the frames:

  1. --sheets-only   scene-detect the source and write one labelled frame per
                     shot. Review the sheets and note the shot indices worth
                     keeping.
  2. --refine "..."  re-sample only those shots at --refine-fps (default 2),
                     labelling every cell with its absolute timestamp. Scene
                     detection finds *where a shot is*; this finds *which part
                     of it is usable*, which is a different question - a 4s shot
                     may hold the subject well framed for only 1.5s. Entries may
                     also be absolute "START-END" ranges, for single-take
                     sources whose shots are minutes long.
  3. --pick-window   cut the exact sub-second range read off a refine sheet.
                     --pick still exists for whole-shot cuts.

The script owns every index -> timestamp conversion. Nothing ever reads a
timestamp out of a model's answer.

Copyright note
--------------
Most YouTube content is copyrighted and downloading it generally violates
YouTube's Terms of Service. Prefer Creative Commons results, your own uploads,
or public-domain archives, especially since MoneyPrinterTurbo can auto-publish
to TikTok/Instagram/YouTube.

Examples
--------
  # AI picks the shots (1 Gemini call per source)
  uv run python scripts/fetch_clips.py \
      --source "https://www.youtube.com/watch?v=XXXX" \
      --want "red sports car driving fast" \
      --want "close up of car wheel spinning"

  # You pick the shots (no API call at all)
  uv run python scripts/fetch_clips.py \
      --manual "https://www.youtube.com/watch?v=XXXX@01:12-01:20" \
      --manual "/path/to/local.mp4@00:05-00:13"

  # Review, refine, then cut an exact window (no API call at all)
  uv run python scripts/fetch_clips.py --sheets-only --min-shot 1.0 \
      --source "https://www.youtube.com/watch?v=XXXX"
  uv run python scripts/fetch_clips.py --refine "4,12,20" \
      --source "https://www.youtube.com/watch?v=XXXX"
  uv run python scripts/fetch_clips.py --pick-window "248.9-252.4" \
      --aspect 9:16 --source "https://www.youtube.com/watch?v=XXXX"

  # Beat-level: rank across each beat's sources, verify, substitute, then cut
  uv run python scripts/fetch_clips.py --plan beats.json
  uv run python scripts/fetch_clips.py --cut-plan beats.selection.json \
      --outdir storage/auto_videos

  # beats.json
  # {"aspect": "9:16", "model": "gemini-3.1-pro-preview",
  #  "verify_model": "gemini-3.7-flash", "blocklist": ["badSourceId"],
  #  "beats": [{"tag": "beat1", "subject": "Koenigsegg Agera RS",
  #             "need": 12.09, "wants": ["the Koenigsegg Agera RS driving at speed"],
  #             "sources": ["https://www.youtube.com/watch?v=XXXX"]}]}
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

# Allow running the script directly from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "storage", "local_videos")
FONT_PATH = os.path.join(REPO_ROOT, "resource", "fonts", "BeVietnamPro-Bold.ttf")

# video.py rejects material whose smaller dimension is under 470px.
MIN_DIMENSION = 470

# --pick trims this much off a shot in total (0.1s at each boundary) so a clip
# never includes the neighbouring shot's first frame after keyframe rounding.
CLIP_EDGE_TRIM = 0.2

# Sheets are built with a coarse scene threshold, which merges real shots. Clips
# are trimmed against this sensitive re-detect so one clip is really one shot.
# Measured on the benchmark picks, this keeps 79% of the raw seconds while
# splitting 20 of 47 merged shots; over-trimming is the safe direction, because
# a short coherent clip beats a long one that jumps between setups.
FINE_SCENE_THRESHOLD = 0.10

# A beat needs a pool deeper than the beat itself: substitution spends
# candidates, and asking for exactly the beat's duration returns barely enough
# to fill it. Measured over the five benchmark beats, asking each wanted shot to
# cover twice the beat's need took the pool from 8/4/9/6/12 candidates to
# 16/5/10/15/14 and the beats filled from 1/5 to 3/5, at identical token cost.
POOL_DEPTH = 2.0

# Shots to ask for per wanted shot, and the pool size below which the ranking
# call is worth repeating. Identical inputs have returned 5 candidates on one
# run and 16 on the next, and the 5-candidate run starved a beat down to a
# single clip: the pool, not the verifier, is what decides whether substitution
# has anywhere to go.
MIN_PER_WANT = 6
MIN_POOL_FACTOR = 3

# video.py loops material from the start when the clips are shorter than the
# narration, which desynchronises every later subtitle. Same margin as
# video.py's _VIDEO_DURATION_SAFETY_MARGIN.
DURATION_SAFETY_MARGIN = 0.1

# Contact sheet layout. Cells are kept large and sheets small so the burned-in
# index stays legible after Gemini downscales the image - unreadable labels make
# the model guess timestamps instead of reading them.
SHEET_COLS = 5
SHEET_ROWS = 4
FRAMES_PER_SHEET = SHEET_COLS * SHEET_ROWS
CELL_W = 240
CELL_H = 320
LABEL_H = 34


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"error: '{name}' is not installed or not on PATH")


def parse_timecode(value: str) -> float:
    """Accept SS, MM:SS or HH:MM:SS and return seconds."""
    value = value.strip()
    if not value:
        raise ValueError("empty timecode")
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid timecode: {value}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def format_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timecode_ms(seconds: float) -> str:
    """HH:MM:SS.mmm - whole-second timecodes would quantise refined windows,
    and a sub-second window is the entire point of --pick-window."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600.0)
    minutes, secs = divmod(rem, 60.0)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def source_slug(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", source)[-60:]


def load_shots(analysis_dir: str, source: str) -> dict[int, dict]:
    """Read the shot table written by --sheets-only."""
    table_path = os.path.join(analysis_dir, source_slug(source), "shots.json")
    if not os.path.isfile(table_path):
        sys.exit(
            f"error: no shots.json for this source; run --sheets-only first\n  {table_path}"
        )
    with open(table_path) as fh:
        return {s["index"]: s for s in json.load(fh)["shots"]}


def parse_window(spec: str, flag: str) -> tuple[float, float]:
    """'248.9-252.4' -> (248.9, 252.4). Absolute seconds, read off a refine sheet."""
    if "-" not in spec:
        sys.exit(f"error: {flag} needs START-END, got: {spec}")
    start_s, end_s = spec.rsplit("-", 1)
    try:
        start, end = float(start_s), float(end_s)
    except ValueError:
        sys.exit(f"error: {flag} needs plain seconds, got: {spec}")
    if end <= start:
        sys.exit(f"error: {flag} end must follow start: {spec}")
    return start, end


def probe_duration(path: str) -> float:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            path,
        ]
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def probe_dimensions(path: str) -> tuple[int, int]:
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            path,
        ]
    )
    try:
        width, height = proc.stdout.strip().split("x")[:2]
        return int(width), int(height)
    except ValueError:
        return 0, 0


# Smallest first: analysis only needs recognisable shapes, and 144p keeps the
# proxy at a few MB. Multiple selectors because YouTube intermittently returns
# 403 for one pre-signed format URL while others on the same video succeed.
# The proxy download reuses the same player-client rotation as the clip cutter:
# yt-dlp's default rotation can land on a client whose media URLs are answered
# with HTTP 403, which used to abort analysis for an otherwise usable source.
PROXY_FORMATS = [
    "160",
    "worst[height<=240][ext=mp4]",
    "18",
    "worst[ext=mp4]",
    "worst",
]


def download_proxy(source: str, workdir: str) -> str:
    """Fetch a tiny low-resolution copy used only for frame analysis."""
    if not is_url(source):
        if not os.path.isfile(source):
            sys.exit(f"error: local source not found: {source}")
        return source

    target = os.path.join(workdir, "proxy.mp4")
    print("  [proxy] downloading low-res copy for analysis ...")
    last_err = ""
    for fmt in PROXY_FORMATS:
        for client in _client_order():
            proc = run(
                [
                    "yt-dlp",
                    source,
                    "-f",
                    fmt,
                    "--no-playlist",
                    "--no-warnings",
                    *_client_args(client),
                    "-o",
                    target,
                ]
            )
            if proc.returncode == 0 and os.path.isfile(target):
                _remember_client(client)
                size_mb = os.path.getsize(target) / 1024 / 1024
                print(
                    f"  [proxy] format {fmt}: {size_mb:.1f} MB, {probe_duration(target):.0f}s"
                )
                return target
            last_err = proc.stderr[-300:]
            if os.path.isfile(target):
                os.remove(target)
    sys.exit(f"error: proxy download failed for {source}\n{last_err}")


def detect_shots(
    video_path: str,
    threshold: float,
    min_duration: float,
    skip_head: int,
    skip_tail: int,
) -> list[dict]:
    """
    Find real cut boundaries with ffmpeg's scene filter.

    Time-based sampling cannot answer "where does this shot start and end" - it
    only proves a subject was on screen at one instant. Scene detection returns
    exact boundaries, so a clip can be cut to contain exactly one shot instead of
    spanning cuts. Shots shorter than min_duration are dropped: on montage
    sources most cuts are well under two seconds and are useless as B-roll.
    """
    meta = os.path.join(os.path.dirname(video_path), "scenes.txt")
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-vf",
            f"select='gt(scene,{threshold})',metadata=print:file={meta}",
            "-f",
            "null",
            "-",
        ]
    )
    duration = probe_duration(video_path)
    cuts: list[float] = []
    if os.path.isfile(meta):
        with open(meta) as fh:
            cuts = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", fh.read())]

    bounds = [0.0] + cuts + [duration]
    lo, hi = float(skip_head), duration - float(skip_tail)
    shots: list[dict] = []
    for i in range(len(bounds) - 1):
        # Clamp to the considered range rather than discarding. Sources with few
        # hard cuts (single-take walkarounds, slow reviews) otherwise lose their
        # entire body because one long shot straddles the skip-head boundary.
        start = max(bounds[i], lo)
        end = min(bounds[i + 1], hi)
        if end - start < min_duration:
            continue
        shots.append({"start": start, "end": end, "duration": end - start})
    for n, shot in enumerate(shots):
        shot["index"] = n
    print(
        f"  [shots] {len(cuts)} cuts -> {len(shots)} usable shots "
        f"(>= {min_duration:g}s, {sum(s['duration'] for s in shots):.0f}s total)"
    )
    return shots


def safe_area(
    width: int, height: int, aspect: str, box: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int]:
    """The rectangle of a source frame that actually survives to the output.

    The renderer scales a clip to *cover* the target and centre-crops it, so for
    a 16:9 source going to 9:16 only the middle ~32% of the width is ever seen.
    A shot with the car parked at the left edge looks perfect on a contact sheet
    and ships as an empty stretch of tarmac. Returning the real keep-rect lets
    the sheet show that, instead of leaving the model to guess.
    """
    if aspect == "none" or aspect not in ASPECT_SIZES:
        return 0, 0, width, height
    # Letterboxed sources have their bars stripped before the aspect crop, so
    # the crop is centred on the picture, not on the padded frame.
    px, py, pw, ph = 0, 0, width, height
    if box:
        pw, ph, px, py = box
    target_w, target_h = ASPECT_SIZES[aspect]
    target_ar = target_w / target_h
    src_ar = pw / ph if ph else target_ar
    if src_ar > target_ar:
        keep_w = int(round(ph * target_ar))
        keep_h = ph
    else:
        keep_w = pw
        keep_h = int(round(pw / target_ar))
    x0 = px + (pw - keep_w) // 2
    y0 = py + (ph - keep_h) // 2
    return x0, y0, x0 + keep_w, y0 + keep_h


def _paste_cell(
    sheet,
    draw,
    frame_path: str,
    x0: int,
    y0: int,
    keep: tuple[float, float, float, float] | None,
):
    """Paste one thumbnail, dimming whatever the output crop will discard."""
    from PIL import Image, ImageEnhance

    thumb = Image.open(frame_path).convert("RGB")
    src_w, src_h = thumb.size
    thumb.thumbnail((CELL_W, CELL_H))
    if keep and src_w and src_h:
        scale_x = thumb.width / src_w
        scale_y = thumb.height / src_h
        kx0 = max(0, min(thumb.width, int(keep[0] * scale_x)))
        ky0 = max(0, min(thumb.height, int(keep[1] * scale_y)))
        kx1 = max(0, min(thumb.width, int(keep[2] * scale_x)))
        ky1 = max(0, min(thumb.height, int(keep[3] * scale_y)))
        if kx1 - kx0 > 1 and ky1 - ky0 > 1:
            dimmed = ImageEnhance.Brightness(thumb).enhance(0.32)
            dimmed.paste(thumb.crop((kx0, ky0, kx1, ky1)), (kx0, ky0))
            thumb = dimmed
    ox = x0 + (CELL_W - thumb.width) // 2
    oy = y0 + LABEL_H + (CELL_H - thumb.height) // 2
    sheet.paste(thumb, (ox, oy))
    if keep and src_w and src_h and kx1 - kx0 > 1:
        draw.rectangle(
            [ox + kx0, oy + ky0, ox + kx1 - 1, oy + ky1 - 1],
            outline=(255, 60, 60),
            width=2,
        )
    return thumb


def build_shot_sheets(
    video_path: str,
    workdir: str,
    shots: list[dict],
    aspect: str = "none",
) -> list[str]:
    """One representative frame per shot, taken from the shot's midpoint."""
    from PIL import Image, ImageDraw, ImageFont

    frames_dir = os.path.join(workdir, "shotframes")
    os.makedirs(frames_dir, exist_ok=True)
    for shot in shots:
        mid = shot["start"] + shot["duration"] / 2
        out = os.path.join(frames_dir, f"s_{shot['index']:04d}.jpg")
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{mid:.2f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-q:v",
                "3",
                out,
                "-y",
            ]
        )
        shot["frame"] = out if os.path.isfile(out) else None

    usable = [s for s in shots if s.get("frame")]
    keep = None
    if aspect != "none":
        width, height = probe_dimensions(video_path)
        keep = safe_area(width, height, aspect, detect_letterbox(video_path))
        pct = (keep[2] - keep[0]) * (keep[3] - keep[1]) / max(1, width * height)
        print(
            f"  [safe] {aspect} output keeps x={keep[0]}..{keep[2]} of {width}px "
            f"({pct * 100:.0f}% of frame area); rest is dimmed on the sheets"
        )

    try:
        font = ImageFont.truetype(FONT_PATH, 22)
    except Exception:
        font = ImageFont.load_default()

    sheets: list[str] = []
    for page_start in range(0, len(usable), FRAMES_PER_SHEET):
        page = usable[page_start : page_start + FRAMES_PER_SHEET]
        cols = min(SHEET_COLS, len(page))
        rows = (len(page) + cols - 1) // cols
        sheet = Image.new(
            "RGB", (cols * CELL_W, rows * (CELL_H + LABEL_H)), (20, 20, 20)
        )
        draw = ImageDraw.Draw(sheet)
        for slot, shot in enumerate(page):
            col, row = slot % cols, slot // cols
            x0, y0 = col * CELL_W, row * (CELL_H + LABEL_H)
            _paste_cell(sheet, draw, shot["frame"], x0, y0, keep)
            draw.rectangle([x0, y0, x0 + CELL_W, y0 + LABEL_H], fill=(0, 0, 0))
            draw.text(
                (x0 + 6, y0 + 5),
                f"#{shot['index']}  {shot['duration']:.1f}s",
                fill=(255, 220, 0),
                font=font,
            )
        path = os.path.join(workdir, f"shots_{len(sheets) + 1:02d}.jpg")
        sheet.save(path, quality=85)
        sheets.append(path)
    return sheets


def build_refine_sheets(
    video_path: str,
    workdir: str,
    segments: list[dict],
    fps: float,
    aspect: str = "none",
) -> tuple[list[str], list[dict]]:
    """
    Second-tier review: sample `fps` frames per second *inside* shortlisted
    segments only.

    Scene detection answers "where is the shot"; it cannot answer "which part of
    the shot is usable". A four second shot may only hold the subject centred
    and sharp for one and a half of those seconds. Sampling the whole source at
    2 fps would produce hundreds of unreviewable frames and would not improve
    cut points, because the shot boundaries are already frame exact. Sampling
    only the shortlist keeps the frame count reviewable.

    Each cell carries its absolute timestamp so --pick-window can be given a
    real range, and this function owns the frame-index -> timestamp arithmetic.
    """
    from PIL import Image, ImageDraw, ImageFont

    frames_dir = os.path.join(workdir, "refineframes")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)

    cells: list[tuple[str, float, str]] = []
    for n, seg in enumerate(segments):
        stem = f"r_{n:03d}_"
        pattern = os.path.join(frames_dir, stem + "%04d.jpg")
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                format_timecode_ms(seg["start"]),
                "-i",
                video_path,
                "-t",
                f"{seg['duration']:.3f}",
                "-vf",
                f"fps={fps}",
                "-fps_mode",
                "vfr",
                "-q:v",
                "3",
                pattern,
                "-y",
            ]
        )
        produced = sorted(
            f
            for f in os.listdir(frames_dir)
            if f.startswith(stem) and f.endswith(".jpg")
        )
        for k, name in enumerate(produced):
            stamp = seg["start"] + k / fps
            cells.append((seg["label"], stamp, os.path.join(frames_dir, name)))
        print(
            f"  [refine] {seg['label']} "
            f"{seg['start']:.1f}-{seg['end']:.1f}s -> {len(produced)} frames"
        )

    if not cells:
        sys.exit("error: --refine produced no frames")

    keep = None
    if aspect != "none":
        width, height = probe_dimensions(video_path)
        keep = safe_area(width, height, aspect, detect_letterbox(video_path))

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except Exception:
        font = ImageFont.load_default()

    sheets: list[str] = []
    for page_start in range(0, len(cells), FRAMES_PER_SHEET):
        page = cells[page_start : page_start + FRAMES_PER_SHEET]
        cols = min(SHEET_COLS, len(page))
        rows = (len(page) + cols - 1) // cols
        sheet = Image.new(
            "RGB", (cols * CELL_W, rows * (CELL_H + LABEL_H)), (20, 20, 20)
        )
        draw = ImageDraw.Draw(sheet)
        for slot, (label, stamp, frame_path) in enumerate(page):
            number = page_start + slot
            col, row = slot % cols, slot // cols
            x0, y0 = col * CELL_W, row * (CELL_H + LABEL_H)
            _paste_cell(sheet, draw, frame_path, x0, y0, keep)
            draw.rectangle([x0, y0, x0 + CELL_W, y0 + LABEL_H], fill=(0, 0, 0))
            # The [N] prefix gives every cell one flat address across all
            # sheets, so --auto-window can name a range without ever emitting
            # a timestamp of its own.
            draw.text(
                (x0 + 6, y0 + 6),
                f"[{number}] {label} t={stamp:.1f}s",
                fill=(0, 255, 180),
                font=font,
            )
        path = os.path.join(workdir, f"refine_{len(sheets) + 1:02d}.jpg")
        sheet.save(path, quality=85)
        sheets.append(path)

    table = [
        {"cell": i, "label": label, "time": stamp}
        for i, (label, stamp, _) in enumerate(cells)
    ]
    print(f"  [refine] {len(cells)} frames at {fps:g} fps -> {len(sheets)} sheet(s)")
    return sheets, table


def build_contact_sheets(
    video_path: str,
    workdir: str,
    sample_every: int,
    skip_head: int = 0,
    skip_tail: int = 0,
) -> tuple[list[str], set[int]]:
    """
    Sample one frame every `sample_every` seconds and tile them into sheets,
    labelling each cell with a sequential index.

    The model reads a small integer instead of a timestamp, and this function
    owns the index -> timestamp mapping (index * sample_every). That keeps the
    arithmetic on our side, where it can't be hallucinated.

    skip_head/skip_tail drop frames from the start/end of the source. Intros and
    especially end credits are a reliable failure mode: they are visually busy,
    so the model happily describes a "hero shot" that is really a title card.
    Excluded frames never reach the model, and indices stay tied to real
    timestamps so the mapping is unaffected.
    """
    from PIL import Image, ImageDraw, ImageFont

    frames_dir = os.path.join(workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    proc = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            video_path,
            "-vf",
            f"fps=1/{sample_every}",
            "-fps_mode",
            "vfr",
            "-q:v",
            "3",
            os.path.join(frames_dir, "f_%04d.jpg"),
            "-y",
        ]
    )
    frames = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".jpg")
    )
    if not frames:
        sys.exit(f"error: could not sample frames\n{proc.stderr[-800:]}")

    total = len(frames)
    first = skip_head // sample_every
    last = total - 1 - (skip_tail // sample_every)
    indexed = [(i, path) for i, path in enumerate(frames) if first <= i <= last]
    if not indexed:
        sys.exit("error: skip-head/skip-tail excluded every sampled frame")
    if len(indexed) < total:
        print(
            f"  [range] considering #{indexed[0][0]}..#{indexed[-1][0]} "
            f"of #0..#{total - 1} (skipped {total - len(indexed)} intro/outro frames)"
        )

    try:
        font = ImageFont.truetype(FONT_PATH, 26)
    except Exception:
        font = ImageFont.load_default()

    sheets: list[str] = []
    for page_start in range(0, len(indexed), FRAMES_PER_SHEET):
        page = indexed[page_start : page_start + FRAMES_PER_SHEET]
        cols = min(SHEET_COLS, len(page))
        rows = (len(page) + cols - 1) // cols
        sheet = Image.new(
            "RGB", (cols * CELL_W, rows * (CELL_H + LABEL_H)), (20, 20, 20)
        )
        draw = ImageDraw.Draw(sheet)

        for slot, (index, frame_path) in enumerate(page):
            col, row = slot % cols, slot // cols
            x0, y0 = col * CELL_W, row * (CELL_H + LABEL_H)

            thumb = Image.open(frame_path).convert("RGB")
            thumb.thumbnail((CELL_W, CELL_H))
            sheet.paste(
                thumb,
                (
                    x0 + (CELL_W - thumb.width) // 2,
                    y0 + LABEL_H + (CELL_H - thumb.height) // 2,
                ),
            )
            draw.rectangle([x0, y0, x0 + CELL_W, y0 + LABEL_H], fill=(0, 0, 0))
            draw.text((x0 + 8, y0 + 4), f"#{index}", fill=(255, 220, 0), font=font)

        path = os.path.join(workdir, f"sheet_{len(sheets) + 1:02d}.jpg")
        sheet.save(path, quality=85)
        sheets.append(path)

    total_mb = sum(os.path.getsize(s) for s in sheets) / 1024 / 1024
    print(
        f"  [sheets] {len(indexed)} frames -> {len(sheets)} sheet(s), {total_mb:.1f} MB "
        f"(#{indexed[0][0]}..#{indexed[-1][0]}, {sample_every}s apart)"
    )
    return sheets, {i for i, _ in indexed}


_TOKEN_TOTALS = {"in": 0, "out": 0, "thoughts": 0, "calls": 0}

# Reasoning tiers bill thinking against the output allowance, so an uncapped
# request can spend the whole budget thinking and return a truncated JSON body.
_MAX_OUTPUT_TOKENS = 16384


def _thinking_config(model: str, level: str | None):
    """Thinking knob for the model family, or None to keep the model default.

    Gemini 3 exposes ``thinking_level``; 2.5 exposes a numeric ``thinking_budget``.
    Passing the wrong one is a hard API error, so map it per family.
    """
    from google.genai import types

    if not level:
        return None
    name = model.lower()
    if "gemini-3" in name or "gemini-4" in name:
        return types.ThinkingConfig(thinking_level=level)
    if "gemini-2.5" in name:
        budget = {"low": 512, "medium": 2048, "high": 8192}.get(level)
        return types.ThinkingConfig(thinking_budget=budget) if budget else None
    return None


class GeminiJSONError(RuntimeError):
    """The model returned nothing usable for one source.

    Raised instead of exiting so a batch run loses one source rather than all
    the work done for the sources after it.
    """


def _loads_tolerant(text: str) -> list[dict]:
    """Parse a JSON array, salvaging the complete objects from a truncated one.

    Observed in practice: a model ends its turn (finish_reason STOP, well under
    the token cap) having emitted a valid prefix of the array but no closing
    bracket. Every complete object in that prefix is still good data, and
    throwing it away costs a whole source.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(data, list):
            return data
        raise GeminiJSONError(f"expected a JSON array, got: {text[:200]}")

    start = text.find("[")
    if start < 0:
        raise GeminiJSONError(f"no JSON array in reply: {text[:200]}")
    decoder = json.JSONDecoder()
    salvaged: list[dict] = []
    pos = start + 1
    while pos < len(text):
        while pos < len(text) and text[pos] in ", \n\r\t":
            pos += 1
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            salvaged.append(obj)
    if not salvaged:
        raise GeminiJSONError(f"unparsable JSON: {text[:200]}")
    print(f"  [warn] reply was truncated; salvaged {len(salvaged)} complete entr(ies)")
    return salvaged


_CLIENT = None
_TOKEN_LOCK = threading.Lock()


def _gemini_client():
    """One shared client: verification fans out over threads."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai

        from app.config import config

        api_key = config.app.get("gemini_api_key", "")
        if not api_key:
            sys.exit("error: gemini_api_key is not set in config.toml")
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _image_part(path: str):
    from google.genai import types

    with open(path, "rb") as fh:
        return types.Part.from_bytes(data=fh.read(), mime_type="image/jpeg")


def _gemini_json(
    prompt: str, sheets: list[str], model: str, thinking: str | None = None
) -> list[dict]:
    """One multimodal request: prompt first, then every sheet."""
    from google.genai import types

    parts = [types.Part(text=prompt)] + [_image_part(s) for s in sheets]
    return _gemini_parts_json(parts, model, thinking)


def _gemini_parts_json(
    parts: list,
    model: str,
    thinking: str | None = None,
    as_object: bool = False,
    temperature: float = 0.2,
    quiet: bool = False,
) -> list | dict:
    """One multimodal request from pre-built parts. Returns the parsed JSON.

    Taking parts rather than a prompt plus a flat image list lets a caller
    interleave text between image groups, which is what labels several sources
    inside a single call. Token usage is accumulated across calls so a batch run
    can report what the unattended path actually cost.
    """
    from google.genai import types

    client = _gemini_client()
    resp = client.models.generate_content(
        model=model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            thinking_config=_thinking_config(model, thinking),
        ),
    )

    usage = getattr(resp, "usage_metadata", None)
    if usage:
        thoughts = getattr(usage, "thoughts_token_count", None) or 0
        with _TOKEN_LOCK:
            _TOKEN_TOTALS["in"] += usage.prompt_token_count or 0
            _TOKEN_TOTALS["out"] += usage.candidates_token_count or 0
            _TOKEN_TOTALS["thoughts"] += thoughts
            _TOKEN_TOTALS["calls"] += 1
        if not quiet:
            print(
                f"  [gemini] {model} tokens in={usage.prompt_token_count} "
                f"out={usage.candidates_token_count} thoughts={thoughts} "
                f"total={usage.total_token_count}"
            )

    finish = ""
    for cand in getattr(resp, "candidates", None) or []:
        finish = str(getattr(cand, "finish_reason", "") or "")
        break

    text = (resp.text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    if as_object:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise GeminiJSONError(
                f"unparsable JSON object: {text[:200]} (finish_reason={finish})"
            ) from None
        if not isinstance(data, dict):
            raise GeminiJSONError(f"expected a JSON object, got: {text[:200]}")
        return data
    try:
        return _loads_tolerant(text)
    except GeminiJSONError as exc:
        raise GeminiJSONError(f"{exc} (finish_reason={finish})") from None


_SHOT_RULES = """Rules:
- "index" MUST be the integer from the yellow #N label of the cell you chose.
  Read the label; never estimate or interpolate it.
- The subject must be the specific vehicle named in the wanted shot. A different
  make or model is a failure, not a near miss.
- Reject title cards, end credits, channel intros, logos, watermarks,
  "subscribe" banners, and any frame whose main content is text.
- Reject frames that are a screen recording of a video player (progress bar,
  playback controls or burned-in captions along the bottom).
- Reject video-game, simulator and toy/scale-model footage; they look plausible
  but are not the real car.
- Reject frames whose main subject is a person rather than the vehicle.
- If no cell reasonably matches a wanted shot, set "index" to null. Do not guess."""


_CROP_RULE = """
OUTPUT FRAMING - this decides whether a shot is usable at all:
The finished video is {w}x{h} ({aspect}). Each cell shows the whole source frame,
but only the area inside the RED RECTANGLE survives; everything DIMMED around it
is cropped away and will never be seen.
- Judge every shot on the bright area alone.
- If the vehicle is mostly in the dimmed region, that shot is UNUSABLE even
  though the car is clearly visible in the cell. Reject it and choose another.
- Prefer shots where the vehicle sits inside, or crosses the middle of, the red
  rectangle."""


def _shot_rules(aspect: str) -> str:
    if aspect in ASPECT_SIZES:
        w, h = ASPECT_SIZES[aspect]
        return _SHOT_RULES + _CROP_RULE.format(w=w, h=h, aspect=aspect)
    return _SHOT_RULES


def fine_cuts(proxy: str, threshold: float | None = None) -> list[float]:
    """Cut times from a sensitive re-detect, cached next to the proxy.

    The sheet-building pass runs a deliberately coarse threshold, which misses
    ordinary hard cuts and merges several real shots into one shots.json entry.
    Measured on one benchmark run, 42% of picked shots contained an undetected
    cut and those shots held 79% of the picked seconds, so cutting on shot
    boundaries alone ships clips that jump between unrelated setups.

    The cache filename carries the threshold: a different threshold is a
    different answer, and silently reusing the old one hides that.
    """
    threshold = FINE_SCENE_THRESHOLD if threshold is None else threshold
    meta = os.path.join(os.path.dirname(proxy), f"scenes_fine_{threshold:g}.txt")
    if not os.path.isfile(meta):
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                proxy,
                "-vf",
                f"select='gt(scene,{threshold})',metadata=print:file={meta}",
                "-f",
                "null",
                "-",
            ]
        )
        if not os.path.isfile(meta):
            # No cuts at all: record that, so the pass is not repeated.
            with open(meta, "w"):
                pass
    with open(meta) as fh:
        return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", fh.read())]


def coherent_span(
    proxy: str, start: float, end: float, threshold: float | None = None
) -> tuple[float, float]:
    """Largest sub-range of a shot that contains no detected cut.

    Returns the shot unchanged when the proxy is unavailable, so a missing
    analysis directory degrades to the old behaviour instead of failing.
    """
    if not os.path.isfile(proxy):
        return start, end
    inside = [c for c in fine_cuts(proxy, threshold) if start + 0.05 < c < end - 0.05]
    if not inside:
        return start, end
    bounds = [start] + sorted(inside) + [end]
    return max(zip(bounds, bounds[1:]), key=lambda pair: pair[1] - pair[0])


def coherent_seconds(proxy: str, shot: dict, threshold: float | None = None) -> float:
    """Seconds of single-shot footage a shot actually yields once trimmed."""
    lo, hi = coherent_span(proxy, shot["start"], shot["end"], threshold)
    return usable_seconds(hi - lo)


def _budget_rule(need_seconds: float) -> str:
    """Tell the model that duration is part of the objective, not decoration.

    The sheets have always shown each shot's duration, but nothing in the prompt
    said it mattered, so models happily returned three 1.5s cutaways for a 14s
    beat. Under-filled beats make video.py loop the footage from the start,
    which desynchronises every later subtitle.
    """
    return f"""
DURATION BUDGET - this decides whether the video can be built at all:
Each wanted shot has to be filled with about {need_seconds:.0f}s of footage taken
from the shots you list. The second number in a cell's label is that shot's
duration in seconds.
- A 1.5s shot contributes almost nothing; a 12s shot can fill a slot by itself.
- Put longer, clearly-matching shots first, and keep listing shots until their
  durations add up to at least {need_seconds:.0f}s.
- Do NOT pad the list with shots that fail the rules above. A short honest list
  is the correct answer when the video genuinely has little usable footage."""


def ask_gemini_shortlist(
    sheets: list[str],
    wants: list[str],
    shots: dict[int, dict],
    model: str,
    aspect: str = "none",
    need_seconds: float = 12.0,
    thinking: str | None = None,
) -> list[dict]:
    """Rank every usable shot per wanted shot so the caller can fill a budget.

    One-pick-per-want cannot express "this beat still needs 9 more seconds", so
    the model only ranks here and all duration arithmetic stays on this side.
    That keeps the original property that a model can never emit a timestamp.
    """
    wanted = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(wants))
    valid = sorted(shots)
    prompt = f"""These images are contact sheets. Each cell is one representative frame
of a distinct SHOT that was detected in a single video, in chronological order.
Every cell has a yellow label like "#12  3.4s": #12 is the shot index and 3.4s is
that shot's duration in seconds. Valid shot indices are #{valid[0]} to #{valid[-1]}.

List EVERY shot that genuinely matches each wanted shot below, best first:
{wanted}

{_shot_rules(aspect)}
{_budget_rule(need_seconds)}
- "want_number" is the number of the wanted shot from the list above.
- "score" is 1-5: how certain you are that this is the named vehicle, judged
  only on the bright area that survives the crop.
- Never list the same shot index twice for the same wanted shot.
- If nothing matches a wanted shot, list nothing for it. Do not guess.

Return JSON: [{{"want_number": <int>, "index": <int>, "score": <1-5>, "why": "<what you see in that cell>"}}]"""
    return _gemini_json(prompt, sheets, model, thinking)


def ask_gemini_beat_shortlist(
    groups: list[dict],
    wants: list[str],
    model: str,
    aspect: str = "none",
    need_seconds: float = 12.0,
    thinking: str | None = None,
) -> list[dict]:
    """Rank shots across ALL of a beat's sources in ONE call.

    ``groups`` is one entry per source: ``{"label", "source", "sheets", "shots"}``.

    One call per source cannot compare source A's mediocre shot against source
    C's good one - it has to commit to A before it has ever seen C. Merging the
    same images into a single call costs the same input tokens (they are the
    same sheets) while turning 19 calls into 5 and making the comparison
    possible. Each source keeps its own #N numbering, so an answer has to name
    both a source label and an index; a label/index pair that does not exist is
    the same hallucination signal an out-of-range index has always been.
    """
    from google.genai import types

    wanted = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(wants))
    catalogue = []
    parts: list = []
    for group in groups:
        valid = sorted(group["shots"])
        catalogue.append(
            f"- SOURCE {group['label']}: {len(group['sheets'])} sheet(s), "
            f"shot indices #{valid[0]} to #{valid[-1]}"
        )
    header = f"""The images below are contact sheets from {len(groups)} DIFFERENT videos of the
same subject. Each cell is one representative frame of a distinct SHOT, in
chronological order within its own video. Every cell has a yellow label like
"#12  3.4s": #12 is the shot index and 3.4s is that shot's duration in seconds.

Each video numbers its shots independently, so "#12" means nothing without the
source it came from. A text line naming the source ("SOURCE A:") precedes that
source's sheets:
{chr(10).join(catalogue)}

List EVERY shot, from ANY of these sources, that genuinely matches each wanted
shot below, best first. Compare the sources against each other: prefer the best
footage wherever it comes from, and do not favour the first source.
{wanted}

{_shot_rules(aspect)}
{_budget_rule(need_seconds)}
- Aim for at least {MIN_PER_WANT} shots per wanted shot where that many
  genuinely match. Later entries are used only when an earlier one turns out to
  be unusable, so a thin list leaves the video with nothing to fall back on. A
  short honest list is still the right answer when the footage is not there.
- "source" MUST be the single letter of the SOURCE heading that immediately
  preceded the sheet you read the cell from. Getting this wrong attaches the
  wrong video to the shot, which is worse than not listing it.
- "want_number" is the number of the wanted shot from the list above.
- "score" is 1-5: how certain you are that this is the named vehicle, judged
  only on the bright area that survives the crop.
- Never list the same source+index pair twice for the same wanted shot.
- If nothing matches a wanted shot, list nothing for it. Do not guess.

Return JSON: [{{"source": "<letter>", "index": <int>, "want_number": <int>, "score": <1-5>, "why": "<what you see in that cell>"}}]"""
    parts.append(types.Part(text=header))
    for group in groups:
        parts.append(
            types.Part(
                text=f"SOURCE {group['label']}: the next "
                f"{len(group['sheets'])} image(s) are this source's sheets."
            )
        )
        parts.extend(_image_part(sheet) for sheet in group["sheets"])
    return _gemini_parts_json(parts, model, thinking)


def fill_budget(
    candidates: list[dict],
    shots: dict[int, dict],
    need_seconds: float,
    seconds_of=None,
) -> tuple[list[int], float]:
    """Take ranked candidates until their durations cover need_seconds.

    Highest score first, model order as the tie-break, so the model's own
    ranking survives. ``seconds_of`` supplies the usable length of a shot and
    defaults to the trimmed shot length; callers pass a cut-aware version so the
    budget is not filled with shots that secretly span several scenes.
    """
    if seconds_of is None:

        def seconds_of(shot):
            return usable_seconds(shot["duration"])

    ordered = sorted(
        enumerate(candidates), key=lambda pair: (-pair[1]["score"], pair[0])
    )
    chosen: list[int] = []
    total = 0.0
    for _, cand in ordered:
        if total >= need_seconds:
            break
        index = cand["index"]
        if index in chosen:
            continue
        chosen.append(index)
        total += seconds_of(shots[index])
    return chosen, total


def usable_seconds(duration: float) -> float:
    """Seconds a shot actually contributes once --pick trims the boundaries."""
    return max(1.0, duration - CLIP_EDGE_TRIM)


def fill_beat_budget(
    candidates: list[dict],
    need_seconds: float,
    per_source_seconds: float,
    min_score: int,
    exclude: set | None = None,
    want_coverage: bool = True,
) -> tuple[list[dict], float]:
    """Choose clips for one beat from a pool ranked across several sources.

    Every candidate carries the real seconds it yields (``seconds``), already
    trimmed to one cut-free shot and capped, so this is pure arithmetic: the
    model ranked, code decides. Rules, in order of importance:

    - never take a candidate the verifier has rejected (``exclude``)
    - never take the same (source, shot) twice
    - drop anything under ``min_score``
    - cap the seconds any one source may contribute, so a long trap source
      cannot own a beat merely by being long
    - with ``want_coverage``, seed the beat with the best candidate for each
      wanted shot before filling by score, so a beat is not five near-identical
      drive-bys

    Returns the chosen candidates and their total seconds; a total under
    ``need_seconds`` is an honest shortfall, never something to paper over.
    """
    exclude = exclude or set()
    pool = [
        c for c in candidates if c["score"] >= min_score and c["key"] not in exclude
    ]
    ordered = sorted(pool, key=lambda c: (-c["score"], c["order"]))

    chosen: list[dict] = []
    taken: set = set()
    per_source: dict[str, float] = {}
    total = 0.0

    def take(cand) -> bool:
        nonlocal total
        if cand["key"] in taken:
            return False
        used = per_source.get(cand["src_id"], 0.0)
        if used + cand["seconds"] > per_source_seconds + 0.001:
            return False
        taken.add(cand["key"])
        per_source[cand["src_id"]] = used + cand["seconds"]
        chosen.append(cand)
        total += cand["seconds"]
        return True

    if want_coverage:
        for want_number in sorted({c["want_number"] for c in ordered}):
            if total >= need_seconds:
                break
            for cand in ordered:
                if cand["want_number"] == want_number and take(cand):
                    break

    for cand in ordered:
        if total >= need_seconds:
            break
        take(cand)

    chosen.sort(key=lambda c: (c["want_number"], -c["score"], c["order"]))
    return chosen, total


def trim_clips_to_target(clips: list[dict], target: float, floor: float) -> float:
    """Shrink a beat's clips so their durations sum to ``target``.

    video.py has no idea which clips belong to which narration beat: it
    concatenates them in order until the audio is covered, so a beat's visuals
    start exactly as late as every earlier beat's surplus. Over-filling is
    therefore not the safe direction it looks like - shipping 15.04s for a
    12.09s beat pushes every later beat 3s out of sync with its narration.

    Surplus is taken proportionally, so no clip collapses while another keeps
    its full length, and each clip is shrunk **towards its own centre**. That
    matters: the span was verified as a filmstrip of itself, so shrinking can
    only ever show frames that were already checked, while extending a span
    would put unverified footage on screen.
    """
    total = sum(c["duration"] for c in clips)
    surplus = total - target
    if not clips or surplus <= 0.01:
        return total
    while surplus > 0.01:
        flexible = [c for c in clips if c["duration"] > floor + 0.01]
        if not flexible:
            break
        headroom = sum(c["duration"] - floor for c in flexible)
        take = min(surplus, headroom)
        for clip in flexible:
            share = (clip["duration"] - floor) / headroom * take
            new_duration = clip["duration"] - share
            clip["start"] = round(
                clip["start"] + (clip["duration"] - new_duration) / 2, 2
            )
            clip["duration"] = round(new_duration, 2)
            clip["seconds"] = clip["duration"]
        new_surplus = sum(c["duration"] for c in clips) - target
        # Durations are rounded to 1/100s, so a surplus spread thinly enough
        # across a beat's clips can round away to nothing on every pass. Without
        # this guard the loop spins forever instead of accepting a sub-frame
        # residue it can no longer remove.
        if new_surplus >= surplus - 1e-9:
            break
        surplus = new_surplus
    return sum(c["duration"] for c in clips)


_VERIFY_PROMPT = """This image is a single frame from a short video clip, shown in the exact
framing in which it will be published. Nothing has been cropped away since.

Answer about this frame alone. Judge only what is visible.

1. Could this frame plausibly be used as B-roll illustrating the topic "{car}"?
   Answer true when the scene is thematically consistent with the topic - the
   right kind of place, object, material or process - even if you cannot
   confirm an exact named location, brand or vintage.
2. Is what is visible specifically part of "{car}"? Answer false if you cannot
   confirm the exact identity.
3. Is that scene the clear focus of the frame? Answer false only if unrelated
   people, unrelated text or clutter dominate it. A wide landscape or an
   establishing shot counts as its own focus.
4. Does the frame match this description: "{want}"?
5. Does the frame contain burned-in overlay graphics? That means video-player
   controls or progress bars, channel watermarks, "subscribe" banners,
   reaction-video furniture, subtitles or captions, or a split-screen /
   picture-in-picture inset. Lettering that is part of the scene itself - a
   sign, a label or a logo on an object - is NOT an overlay.

Return JSON:
{{"subject_visible": true|false, "is_named_subject": true|false,
  "subject_is_main": true|false, "matches_description": true|false,
  "has_overlay": true|false, "confidence": 1-5,
  "what_you_see": "<one short sentence>"}}"""

_VERIFY_PROMPT_STRIP = """These {n} images are frames taken from the start, middle and end of ONE
short video clip, in that order, shown in the exact framing in which the
clip will be published. Nothing has been cropped away since.

Answer about the clip as a whole, judging only what is visible in these frames.

1. Could these frames plausibly be used as B-roll illustrating the topic
   "{car}"? Answer true when the scene is thematically consistent with the
   topic - the right kind of place, object, material or process - even if you
   cannot confirm an exact named location, brand or vintage.
2. Is what is visible specifically part of "{car}"? Answer false if you cannot
   confirm the exact identity.
3. Is that scene the clear focus? Answer false only if unrelated people,
   unrelated text or clutter dominate the frames. A wide landscape or an
   establishing shot counts as its own focus.
4. Do the frames match this description: "{want}"?
5. Do ANY of the frames contain burned-in overlay graphics? That means
   video-player controls or progress bars, channel watermarks, "subscribe"
   banners, reaction-video furniture, subtitles or captions, or a split-screen /
   picture-in-picture inset. Lettering that is part of the scene itself - a
   sign, a label or a logo on an object - is NOT an overlay. Answer true if any
   single frame has one.

Return JSON:
{{"subject_visible": true|false, "is_named_subject": true|false,
  "subject_is_main": true|false, "matches_description": true|false,
  "has_overlay": true|false, "confidence": 1-5,
  "what_you_see": "<one short sentence>"}}"""


def verdict_accepts(verdict: dict) -> bool:
    """The measured accept rule.

    Rejecting on overlay, a missing subject, or the subject not being the main
    focus removed 8 of 12 bad frames while keeping 29 of 29 good ones. Adding
    is_named_subject or matches_description catches one or two more bad frames
    and throws away a third to a half of the good footage, which beats this
    short cannot afford.
    """
    if verdict.get("error"):
        return True  # a failed call is not evidence against the clip
    return bool(
        verdict.get("subject_visible")
        and verdict.get("subject_is_main")
        and not verdict.get("has_overlay")
    )


def shipped_frames(
    proxy: str,
    keep: tuple[int, int, int, int],
    start: float,
    end: float,
    count: int,
    outdir: str,
    tag: str,
) -> list[str]:
    """Extract the frames that actually ship: the aspect crop, nothing else.

    A contact-sheet cell shows the whole source frame with the discarded area
    dimmed; the verifier must see only what survives, with no sheet furniture,
    no index and no hint that anything selected it.
    """
    os.makedirs(outdir, exist_ok=True)
    span = max(0.0, end - start)
    if count <= 1:
        times = [start + span / 2]
    else:
        inset = min(0.25, span * 0.1)
        first, last = start + inset, end - inset
        times = [first + (last - first) * i / (count - 1) for i in range(count)]
    x0, y0, x1, y1 = keep
    out: list[str] = []
    for i, when in enumerate(times):
        path = os.path.join(outdir, f"{tag}_{i}_{when:.2f}.jpg")
        if not os.path.isfile(path):
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{when:.2f}",
                    "-i",
                    proxy,
                    "-frames:v",
                    "1",
                    "-vf",
                    f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}",
                    "-q:v",
                    "3",
                    path,
                    "-y",
                ]
            )
        if os.path.isfile(path):
            out.append(path)
    return out


def verify_shipped(frames: list[str], subject: str, want: str, model: str) -> dict:
    """Ask the model, cold, what is in the frames that will be published."""
    from google.genai import types

    if not frames:
        return {"error": "no frames extracted"}
    if len(frames) == 1:
        prompt = _VERIFY_PROMPT.format(car=subject, want=want)
    else:
        prompt = _VERIFY_PROMPT_STRIP.format(n=len(frames), car=subject, want=want)
    parts = [types.Part(text=prompt)] + [_image_part(f) for f in frames]
    for attempt in range(3):
        try:
            return _gemini_parts_json(
                parts, model, as_object=True, temperature=0.0, quiet=True
            )
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return {"error": str(exc)[:160]}
    return {"error": "unreachable"}


def ask_gemini_shots(
    sheets: list[str],
    wants: list[str],
    shots: dict[int, dict],
    model: str,
    aspect: str = "none",
    thinking: str | None = None,
) -> list[dict]:
    """Pick whole scene-detected shots. One request per source.

    This is the model-driven equivalent of reviewing the --sheets-only output
    by eye: same sheets, same shot indices, so the two can be compared directly.
    """
    wanted = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(wants))
    valid = sorted(shots)
    prompt = f"""These images are contact sheets. Each cell is one representative frame
of a distinct SHOT that was detected in a single video, in chronological order.
Every cell has a yellow label like "#12  3.4s": #12 is the shot index and 3.4s is
that shot's duration. Valid shot indices are #{valid[0]} to #{valid[-1]}.

Pick the single best matching shot for each wanted shot below:
{wanted}

{_shot_rules(aspect)}
- Prefer a shot where the vehicle is large in frame, sharp and well lit.
- Return exactly one entry per wanted shot, in the same order as listed.

Return JSON: [{{"want": "<the wanted shot text>", "index": <int or null>, "why": "<what you see in that cell>"}}]"""
    return _gemini_json(prompt, sheets, model, thinking)


def ask_gemini_window(
    sheets: list[str],
    cells: list[dict],
    want: str,
    model: str,
    aspect: str = "none",
    thinking: str | None = None,
) -> dict:
    """Pick the best sub-range inside already-shortlisted shots.

    The model returns two cell numbers; this side converts them to timestamps,
    so a hallucinated time is structurally impossible.
    """
    prompt = f"""These images are contact sheets of frames sampled twice per second from
inside a few candidate shots of one video. Every cell has a green label like
"[7] #12 t=248.9s": [7] is the cell number you must use. Valid cell numbers are
[0] to [{len(cells) - 1}].

Wanted shot: {want}

Choose the longest run of CONSECUTIVE cells that all clearly show the wanted
subject, then report its first and last cell number.

{_shot_rules(aspect)}
- "first" and "last" MUST be cell numbers read from the [N] labels.
- The run must be at least 3 cells long. If nothing qualifies, set both to null.

Return JSON: [{{"first": <int or null>, "last": <int or null>, "why": "<what you see>"}}]"""
    data = _gemini_json(prompt, sheets, model, thinking)
    return data[0] if data else {}


def ask_gemini(
    sheets: list[str],
    wants: list[str],
    sample_every: int,
    clip_seconds: int,
    valid_indices: set[int],
    model: str,
) -> list[dict]:
    """One request per source video. Returns [{want, index, why}]."""
    wanted = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(wants))
    prompt = f"""These images are contact sheets of frames sampled from one video, {sample_every} seconds apart.
Every cell has a yellow index label like #0, #1, #2 in the black bar above it.
Valid indices are #{min(valid_indices)} to #{max(valid_indices)} across all sheets, in order.

Pick the single best matching cell for each wanted shot below:
{wanted}

{_SHOT_RULES}
- Prefer clear, well-lit, non-blurry frames that actually contain the subject.
- Return exactly one entry per wanted shot, in the same order as listed.

Return JSON: [{{"want": "<the wanted shot text>", "index": <int or null>, "why": "<what you see in that cell>"}}]"""
    return _gemini_json(prompt, sheets, model)


# YouTube sometimes returns 403 for one pre-signed format URL while other
# formats on the same video work fine. Try progressively lower quality instead
# of failing the clip. Progressive formats are last: they are usually 360p,
# which is below MIN_DIMENSION.
SECTION_FORMATS = [
    "bv*[height>=1080][ext=mp4]",
    "137",
    "bv*[height>=720][ext=mp4]",
    "136",
    "bv*[ext=mp4]",
    "best[ext=mp4]",
]

# Sources that fell back to a full download, cached so each source is fetched
# at most once no matter how many clips are cut from it.
_FULL_CACHE: dict[str, str] = {}

# YouTube hands different player clients different pre-signed URLs, and whole
# clients go bad for weeks at a time: the default choice currently yields
# formats whose URLs answer 403 to ffmpeg's ranged request, while web_safari
# serves the same 1080p video fine. Ordering clients and remembering the one
# that worked turns a dead run into one extra attempt on the first clip.
PLAYER_CLIENTS = ["web_safari", "web_embedded", "default", "tv", "mweb"]
_GOOD_CLIENT: list[str] = []


def _client_args(client: str) -> list[str]:
    if client == "default":
        return []
    return ["--extractor-args", f"youtube:player_client={client}"]


def _client_order() -> list[str]:
    if _GOOD_CLIENT:
        first = _GOOD_CLIENT[0]
        return [first] + [c for c in PLAYER_CLIENTS if c != first]
    return list(PLAYER_CLIENTS)


def _remember_client(client: str) -> None:
    if not _GOOD_CLIENT or _GOOD_CLIENT[0] != client:
        _GOOD_CLIENT[:] = [client]


def _cut_local(source: str, start: float, duration: float, outfile: str) -> bool:
    proc = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            format_timecode_ms(start),
            "-i",
            source,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            outfile,
            "-y",
        ]
    )
    if proc.returncode != 0 or not os.path.isfile(outfile):
        print(f"  [warn] local cut failed: {proc.stderr[-300:]}")
        return False
    return True


def _ranged_download(
    source: str, start: float, duration: float, outfile: str, fmt: str, client: str
) -> bool:
    end = start + duration
    proc = run(
        [
            "yt-dlp",
            source,
            "-f",
            fmt,
            "--download-sections",
            f"*{format_timecode_ms(start)}-{format_timecode_ms(end)}",
            "--force-keyframes-at-cuts",
            "--no-playlist",
            "--no-warnings",
            *_client_args(client),
            "-o",
            outfile,
        ]
    )
    return proc.returncode == 0 and os.path.isfile(outfile)


def _ensure_full_download(source: str, workdir: str, quality: str) -> str | None:
    """Download a whole source once, so clips can be cut locally when ranged
    fetching is refused."""
    if source in _FULL_CACHE:
        return _FULL_CACHE[source]
    target = os.path.join(workdir, "full.mp4")
    print("  [fallback] ranged download unavailable; fetching full source once ...")
    for client in _client_order():
        proc = run(
            [
                "yt-dlp",
                source,
                "-f",
                f"{quality}/bv*[ext=mp4]/best[ext=mp4]/best",
                "--no-playlist",
                "--no-warnings",
                *_client_args(client),
                "-o",
                target,
            ]
        )
        if proc.returncode != 0 or not os.path.isfile(target):
            print(f"  [warn] {client} full download failed: {proc.stderr[-200:]}")
            continue
        width, height = content_dimensions(target)
        if min(width, height) < MIN_DIMENSION:
            # A 403 cascade can end at a 360p/480p progressive format. Cutting
            # from that produces material video.py will drop, or a heavy upscale.
            print(
                f"  [warn] {client} served only {width}x{height} of real picture; "
                f"refusing it"
            )
            os.remove(target)
            continue
        _remember_client(client)
        print(
            f"  [fallback] cached {os.path.getsize(target) / 1024 / 1024:.0f} MB source"
        )
        _FULL_CACHE[source] = target
        return target
    print("  [warn] no client served a usable full source; 403s are often transient")
    return None


def download_section(
    source: str, start: float, duration: float, outfile: str, quality: str, workdir: str
) -> bool:
    if not is_url(source):
        return _cut_local(source, start, duration, outfile)

    if source in _FULL_CACHE:
        return _cut_local(_FULL_CACHE[source], start, duration, outfile)

    selectors = [quality] + [f for f in SECTION_FORMATS if f != quality]
    # Client outside, format inside: a bad client is the usual cause of both a
    # 403 and a 360p-only format list, so exhausting the formats of a good
    # client beats trying every client on a format that client cannot serve.
    for client in _client_order():
        for fmt in selectors:
            if _ranged_download(source, start, duration, outfile, fmt, client):
                width, height = content_dimensions(outfile)
                if min(width, height) >= MIN_DIMENSION:
                    _remember_client(client)
                    return True
                print(
                    f"  [retry] {client} format '{fmt}' gave {width}x{height} of "
                    f"real picture (<{MIN_DIMENSION}px)"
                )
            if os.path.isfile(outfile):
                os.remove(outfile)

    full = _ensure_full_download(source, workdir, quality)
    if not full:
        return False
    return _cut_local(full, start, duration, outfile)


# video.py never crops: a clip whose aspect differs from the target is scaled to
# fit and centred on a black background. A landscape YouTube source therefore
# renders as a thin band inside a 9:16 frame. Cropping here, at fetch time, is
# the only place that knows the intended output aspect.
ASPECT_SIZES = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}


def detect_letterbox(path: str) -> tuple[int, int, int, int] | None:
    """
    Find the real picture inside baked-in black bars.

    Cinematic YouTube uploads are often 2.35:1 letterboxed inside a 16:9 frame.
    Scaling that to 9:16 keeps the bars, so the finished short shows black
    stripes across the middle of the screen. cropdetect reports one box per
    frame; taking the union of every box is deliberate - a dark frame reports a
    box smaller than the true picture, and the union can therefore only ever be
    too generous, never too tight.
    """
    width, height = probe_dimensions(path)
    if not width or not height:
        return None
    proc = run(["ffmpeg", "-i", path, "-vf", "cropdetect=24:2:0", "-f", "null", "-"])
    boxes = [
        tuple(int(v) for v in m)
        for m in re.findall(r"crop=(\d+):(\d+):(-?\d+):(-?\d+)", proc.stderr)
    ]
    boxes = [b for b in boxes if b[0] > 0 and b[1] > 0 and b[2] >= 0 and b[3] >= 0]
    if not boxes:
        return None

    left = min(b[2] for b in boxes)
    top = min(b[3] for b in boxes)
    right = max(b[2] + b[0] for b in boxes)
    bottom = max(b[3] + b[1] for b in boxes)
    crop_w = min(right, width) - left
    crop_h = min(bottom, height) - top
    if crop_w <= 0 or crop_h <= 0:
        return None
    # Only act on a clear letterbox. Small deltas are encoder noise, and a
    # drastic crop means cropdetect was fooled by a genuinely dark shot.
    if crop_w * crop_h > 0.97 * width * height:
        return None
    if crop_w < 0.5 * width or crop_h < 0.4 * height:
        return None
    return crop_w - crop_w % 2, crop_h - crop_h % 2, left, top


def content_dimensions(path: str) -> tuple[int, int]:
    """Dimensions of the real picture, ignoring baked-in black bars.

    The raw frame size lies about quality when a source is letterboxed: an
    854x480 upload whose picture is only 364px tall gets upscaled 5x into a
    1080x1920 short and looks it.
    """
    box = detect_letterbox(path)
    if box:
        return box[0], box[1]
    return probe_dimensions(path)


def normalize_aspect(path: str, aspect: str, max_seconds: float = 0.0) -> bool:
    """Strip baked-in letterboxing, then centre-crop/scale to the target
    aspect, in place.

    ``max_seconds`` also pins the output length. A ranged fetch rounds out to
    the enclosing keyframes, so a 4.80s request lands as a 4.86s file; that
    overshoot accumulates across a timeline video.py concatenates by duration
    alone. Trimming here is free because the file is being re-encoded anyway.
    """
    if aspect == "none" and not max_seconds:
        return True
    target_w, target_h = ASPECT_SIZES.get(aspect, (0, 0))
    width, height = probe_dimensions(path)

    box = detect_letterbox(path) if aspect != "none" else None
    prefix = ""
    if box:
        crop_w, crop_h, x, y = box
        prefix = f"crop={crop_w}:{crop_h}:{x}:{y},"
        print(f"  [letterbox] {width}x{height} -> {crop_w}x{crop_h} picture area")
    elif (width, height) == (target_w, target_h) and (
        not max_seconds or probe_duration(path) <= max_seconds + 0.02
    ):
        return True

    tmp = path + ".crop.mp4"
    filters = []
    if aspect != "none":
        filters.append(
            f"{prefix}scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h}"
        )
    elif prefix:
        filters.append(prefix.rstrip(","))
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", tmp, "-y"]
    proc = run(cmd)
    if proc.returncode != 0 or not os.path.isfile(tmp):
        print(f"  [warn] aspect crop failed: {proc.stderr[-300:]}")
        if os.path.isfile(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, path)
    trimmed = f", capped at {max_seconds:.2f}s" if max_seconds else ""
    print(f"  [crop] {width}x{height} -> {target_w}x{target_h} ({aspect}){trimmed}")
    return True


def parse_beat(spec: str) -> tuple[float, list[str]]:
    """'12.09:a.mp4,b.mp4' -> (12.09, ['a.mp4', 'b.mp4'])."""
    if ":" not in spec:
        sys.exit(f"error: --beat needs SECONDS:file1,file2, got: {spec}")
    need_s, files_s = spec.split(":", 1)
    try:
        need = float(need_s)
    except ValueError:
        sys.exit(f"error: --beat needs a number before ':', got: {need_s!r}")
    files = [f.strip() for f in files_s.split(",") if f.strip()]
    if not files:
        sys.exit(f"error: --beat has no files: {spec}")
    return need, files


def black_spans(path: str, min_seconds: float = 0.15) -> list[tuple[float, float]]:
    """Stretches of near-black inside a clip.

    A fade to black inside a shot survives every check made so far: the span is
    one cut-free shot, and the verifier only ever sees three sampled frames, so
    a half-second dip between them is invisible until someone watches the
    render. Cheap to detect here, and reported rather than fatal - it is a
    quality flaw, not something that corrupts the timeline.
    """
    proc = run(
        [
            "ffmpeg",
            "-i",
            path,
            "-vf",
            f"blackdetect=d={min_seconds}:pix_th=0.10",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    return [
        (float(a), float(b))
        for a, b in re.findall(
            r"black_start:([0-9.]+) black_end:([0-9.]+)", proc.stderr
        )
    ]


def preflight(
    beats: list[str],
    outdir: str,
    tolerance: float = 0.35,
    check_black: bool = True,
) -> int:
    """Refuse to ship material that video.py would silently corrupt.

    Three failure modes are silent at render time and only visible by watching
    the finished video. Material shorter than the narration makes video.py loop
    the clips from the start, and anything under MIN_DIMENSION is dropped
    without a word. The third is the mirror image of the first: video.py has no
    concept of a beat, so a beat that runs LONG delays every beat after it by
    that surplus, and the narration ends up over the wrong car. Only the last
    beat may run long, because nothing follows it and the total has to outlast
    the audio.
    """
    print("=== preflight ===")
    problems: list[str] = []
    warnings: list[str] = []
    grand_need = grand_have = 0.0
    longest = 0.0
    for n, spec in enumerate(beats, start=1):
        need, files = parse_beat(spec)
        have = 0.0
        print(f"\n[beat {n}] needs {need:.2f}s")
        for name in files:
            path = name if os.path.isabs(name) else os.path.join(outdir, name)
            if not os.path.isfile(path):
                problems.append(f"beat {n}: missing file {name}")
                print(f"  [missing] {name}")
                continue
            width, height = probe_dimensions(path)
            seconds = probe_duration(path)
            if min(width, height) < MIN_DIMENSION:
                problems.append(
                    f"beat {n}: {name} is {width}x{height}, under {MIN_DIMENSION}px "
                    f"- video.py will drop it"
                )
                print(f"  [small  ] {name}  {width}x{height}  {seconds:.1f}s")
                continue
            if seconds <= 0:
                problems.append(f"beat {n}: {name} has no readable duration")
                print(f"  [bad    ] {name}")
                continue
            have += seconds
            longest = max(longest, seconds)
            print(f"  [ok     ] {name}  {width}x{height}  {seconds:.1f}s")
            for lo, hi in black_spans(path) if check_black else []:
                warnings.append(
                    f"beat {n}: {name} fades to black {lo:.2f}-{hi:.2f}s "
                    f"({hi - lo:.2f}s) - three sampled frames cannot see this"
                )
                print(f"  [black  ] {name}  {lo:.2f}-{hi:.2f}s")
        short = need - have
        grand_need += need
        grand_have += have
        if short > DURATION_SAFETY_MARGIN:
            problems.append(
                f"beat {n}: {have:.2f}s of material for {need:.2f}s of narration (short {short:.2f}s)"
            )
            print(f"  -> {have:.2f}s / {need:.2f}s  SHORT by {short:.2f}s")
        elif -short > tolerance and n < len(beats):
            problems.append(
                f"beat {n}: {have:.2f}s of material for {need:.2f}s of narration "
                f"(long {-short:.2f}s) - every later beat starts {-short:.2f}s late"
            )
            print(f"  -> {have:.2f}s / {need:.2f}s  LONG by {-short:.2f}s")
        else:
            print(f"  -> {have:.2f}s / {need:.2f}s  ok")

    print(f"\ntotal {grand_have:.2f}s of material for {grand_need:.2f}s of narration")
    if grand_have < grand_need + DURATION_SAFETY_MARGIN:
        problems.append(
            f"total material {grand_have:.2f}s does not outlast the narration "
            f"{grand_need:.2f}s by the {DURATION_SAFETY_MARGIN}s safety margin - "
            f"video.py would loop the timeline from the first clip"
        )
    if not problems:
        print("preflight PASSED")
        for w in warnings:
            print(f"  [warn] {w}")
        print(
            f"\nRender with --video-clip-duration {math.ceil(longest)} or more: "
            f"sequential concat keeps only the first --video-clip-duration "
            f"seconds of each file, and the longest clip here is {longest:.2f}s."
        )
        return 0
    print(f"\npreflight FAILED - {len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
    print(
        "\nFix by picking more or longer shots for the short beats, by trimming\n"
        "the long ones, or by falling back to another source. Rendering now\n"
        "would loop footage from the start, or drift the visuals out of sync\n"
        "with the narration, and desynchronise every later subtitle."
    )
    return 1


# ---------------------------------------------------------------------------
# Beat-level clip selection: rank across a beat's sources, satisfy the duration
# constraint in code, verify the shipped crop cold, substitute what it rejects.
# ---------------------------------------------------------------------------

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_plan(path: str) -> dict:
    """Read the beat plan. Want strings live in the file so runs stay comparable."""
    with open(path) as fh:
        plan = json.load(fh)
    if not plan.get("beats"):
        sys.exit(f"error: {path} has no beats")
    for n, beat in enumerate(plan["beats"], start=1):
        for key in ("tag", "need", "subject", "wants", "sources"):
            if key not in beat:
                sys.exit(f"error: beat {n} in {path} is missing {key!r}")
    return plan


def source_max_height(source: str, workdir: str) -> int:
    """Best available height for a source, cached; 0 when it cannot be read.

    video.py silently drops material under MIN_DIMENSION, and one of the
    benchmark sources is a 480p upload whose real picture is 364px tall, so
    every clip cut from it disappears at render time. Reading the height costs
    one metadata request and no download, which is far cheaper than discovering
    it after the beat has been selected, verified and cut.
    """
    cache = os.path.join(workdir, "max_height.txt")
    if os.path.isfile(cache):
        try:
            return int(open(cache).read().strip() or 0)
        except ValueError:
            pass
    height = 0
    if is_url(source):
        proc = run(
            ["yt-dlp", "--no-warnings", "--print", "%(height)s", "-f", "bv*", source]
        )
        for line in proc.stdout.splitlines():
            try:
                height = max(height, int(line.strip()))
            except ValueError:
                continue
    else:
        height = probe_dimensions(source)[1]
    if height:
        os.makedirs(workdir, exist_ok=True)
        with open(cache, "w") as fh:
            fh.write(str(height))
    return height


def beat_groups(beat: dict, plan: dict, args) -> list[dict]:
    """Label each of a beat's sources and gather its cached sheets and shots."""
    blocked = set(plan.get("blocklist", []))
    groups: list[dict] = []
    for source in beat["sources"]:
        slug = source_slug(source)
        if any(bad in source for bad in blocked):
            print(f"  [block] {source} - on the plan's blocklist")
            continue
        workdir = os.path.join(args.analysis_dir, slug)
        sheets = sorted(glob.glob(os.path.join(workdir, "shots_*.jpg")))
        proxy = os.path.join(workdir, "proxy.mp4")
        if not sheets or not os.path.isfile(proxy):
            print(f"  [skip ] {source} - no cached sheets; run --sheets-only first")
            continue
        shots = load_shots(args.analysis_dir, source)
        width, height = probe_dimensions(proxy)
        keep = safe_area(width, height, args.aspect, detect_letterbox(proxy))
        group = {
            "label": _LABELS[len(groups)],
            "source": source,
            "src_id": source.rsplit("=", 1)[-1],
            "slug": slug,
            "workdir": workdir,
            "proxy": proxy,
            "sheets": sheets,
            "shots": shots,
            "keep": keep,
        }
        if args.check_resolution:
            real_h = source_max_height(source, workdir)
            content = detect_letterbox(proxy) or (width, height, 0, 0)
            if real_h and height:
                # Clips are always scaled to the full 1080x1920 target, so the
                # question is not the file's dimensions but how much real
                # picture is behind them. One benchmark source is a 480p upload
                # whose picture is only 364px tall; every clip cut from it is
                # upscaled 5x and video.py drops material that small.
                picture_h = content[1] * real_h / height
                group["picture_height"] = round(picture_h)
                if picture_h < MIN_DIMENSION:
                    print(
                        f"  [small] {source} has {picture_h:.0f}px of real picture "
                        f"height, under {MIN_DIMENSION}px - skipping"
                    )
                    continue
        groups.append(group)
    return groups


def candidate_span(group: dict, shot: dict, args) -> tuple[float, float]:
    """The largest cut-free span of a shot, before any duration cap."""
    lo, hi = shot["start"], shot["end"]
    if not args.no_cut_trim:
        lo, hi = coherent_span(group["proxy"], lo, hi, args.fine_threshold)
    return lo, hi


def apply_shot_cap(cand: dict, cap: float) -> dict:
    """Set the span that would actually ship for a candidate under ``cap``.

    The cap buys variety, not quality, so it is relaxed when a beat would
    otherwise ship short. Because a longer span shows the model different
    frames, the verification key carries the duration: a re-capped clip is
    verified again rather than inheriting a verdict about a shorter cut.
    """
    lo, hi = cand["span"]
    start = lo + CLIP_EDGE_TRIM / 2
    duration = usable_seconds(hi - lo)
    if cap and duration > cap:
        # Keep the middle of the shot: the ends sit against the cuts.
        start += (duration - cap) / 2
        duration = cap
    cand["start"] = round(start, 2)
    cand["duration"] = round(duration, 2)
    cand["seconds"] = round(duration, 2)
    cand["vkey"] = (*cand["key"], round(duration, 1))
    return cand


def collect_candidates(
    groups: list[dict], raw: list[dict], args
) -> tuple[list[dict], dict]:
    """Turn one beat-level reply into candidates whose seconds are already real."""
    by_label = {g["label"]: g for g in groups}
    cands: list[dict] = []
    stats = {"raw": len(raw), "bad_label": 0, "oob": 0, "bad_field": 0, "dupe": 0}
    seen: set = set()
    for order, entry in enumerate(raw):
        label = str(entry.get("source", "")).strip().upper()[:1]
        group = by_label.get(label)
        if group is None:
            stats["bad_label"] += 1
            print(f"  [src ] no source {label!r} in this beat: {entry!r}")
            continue
        try:
            cell = int(entry.get("index"))
            want_number = int(entry.get("want_number", 0)) - 1
            score = int(entry.get("score", 3))
        except (TypeError, ValueError):
            stats["bad_field"] += 1
            print(f"  [bad ] unreadable candidate: {entry!r}")
            continue
        if cell not in group["shots"]:
            # Still the clearest hallucination signal there is.
            stats["oob"] += 1
            print(f"  [oob ] source {label} has no shot #{cell}")
            continue
        key = (group["src_id"], cell)
        if key in seen:
            stats["dupe"] += 1
            continue
        seen.add(key)
        shot = group["shots"][cell]
        cands.append(
            apply_shot_cap(
                {
                    "key": key,
                    "order": order,
                    "label": label,
                    "source": group["source"],
                    "src_id": group["src_id"],
                    "shot": cell,
                    "want_number": max(0, want_number),
                    "score": max(1, min(5, score)),
                    "why": str(entry.get("why", ""))[:300],
                    "span": candidate_span(group, shot, args),
                    "shot_duration": shot["duration"],
                },
                args.per_shot_seconds,
            )
        )
    return cands, stats


def verify_candidates(cands: list[dict], beat: dict, groups: list[dict], args) -> dict:
    """Verify each clip cold, in parallel. Returns key -> verdict."""
    import concurrent.futures as cf

    by_id = {g["src_id"]: g for g in groups}
    jobs = []
    for cand in cands:
        group = by_id[cand["src_id"]]
        frames = shipped_frames(
            group["proxy"],
            group["keep"],
            cand["start"],
            cand["start"] + cand["duration"],
            args.verify_frames,
            os.path.join(group["workdir"], "verifyframes"),
            f"s{cand['shot']:04d}",
        )
        want = beat["wants"][min(cand["want_number"], len(beat["wants"]) - 1)]
        jobs.append((cand, frames, want))

    out: dict = {}
    if not jobs:
        return out
    with cf.ThreadPoolExecutor(max_workers=args.verify_workers) as pool:
        futures = {
            pool.submit(
                verify_shipped, frames, beat["subject"], want, args.verify_model
            ): (cand, frames)
            for cand, frames, want in jobs
        }
        for future in cf.as_completed(futures):
            cand, frames = futures[future]
            verdict = future.result()
            verdict["frames"] = frames
            out[cand["vkey"]] = verdict
    return out


def run_beat(beat: dict, plan: dict, args, is_last: bool = False) -> dict:
    """Stages 1-4 for one beat."""
    need = float(beat["need"])
    print(f"\n=== beat {beat['tag']}: {beat['subject']} needs {need:.2f}s ===")
    groups = beat_groups(beat, plan, args)
    result = {
        "tag": beat["tag"],
        "subject": beat["subject"],
        "need": need,
        "wants": beat["wants"],
        "sources": [
            {"label": g["label"], "source": g["source"], "sheets": len(g["sheets"])}
            for g in groups
        ],
        "clips": [],
        "rejected": [],
        "status": "hard fail",
    }
    if not groups:
        print("  [fail] no usable sources for this beat")
        return result

    for group in groups:
        print(
            f"  [src  ] {group['label']}: {group['src_id']} "
            f"{len(group['sheets'])} sheet(s), {len(group['shots'])} shot(s)"
        )
    pool_seconds = args.pool_seconds or need * POOL_DEPTH
    raw: list[dict] = []
    cands: list[dict] = []
    stats = {"raw": 0, "bad_label": 0, "oob": 0, "bad_field": 0, "dupe": 0}
    # One re-ask when the pool is too thin to survive substitution. The reply
    # varies run to run for identical input, so asking again genuinely deepens
    # the pool; it is capped at one extra call so a barren beat cannot loop.
    for attempt in range(1, args.rank_attempts + 1):
        try:
            reply = ask_gemini_beat_shortlist(
                groups,
                beat["wants"],
                args.model,
                args.aspect,
                pool_seconds,
                args.thinking,
            )
        except GeminiJSONError as exc:
            print(f"  [fail] ranking call failed: {exc}")
            if cands:
                break
            return result
        raw += reply
        cands, stats = collect_candidates(groups, raw, args)
        offered = sum(c["seconds"] for c in cands)
        deep = (
            len(cands) >= MIN_POOL_FACTOR * len(beat["wants"]) and offered >= need * 1.5
        )
        print(
            f"  [pool {attempt}] {len(cands)} candidate(s), {offered:.1f}s offered "
            f"from {stats['raw']} entries (bad source label {stats['bad_label']}, "
            f"out-of-range {stats['oob']}, unreadable {stats['bad_field']}, "
            f"duplicate {stats['dupe']})"
        )
        if deep:
            break
        if attempt < args.rank_attempts:
            print("  [pool ] too thin to substitute from; asking once more")
    if not cands:
        print("  [fail] no usable candidates")
        return result
    result["candidates"] = cands
    result["rank_stats"] = stats
    by_src: dict[str, float] = {}
    for cand in cands:
        by_src[cand["src_id"]] = by_src.get(cand["src_id"], 0.0) + cand["seconds"]
    for src_id, seconds in by_src.items():
        print(f"          {src_id}: {seconds:.1f}s offered")

    per_source = args.per_source_seconds or max(need * 0.6, args.per_shot_seconds)
    # Both caps buy variety, not quality, so they are what gets relaxed rather
    # than shipping a short beat: a beat whose other sources were rejected has
    # nowhere else to go, and a 6s ceiling on a 7.5s cut-free shot throws away
    # seconds for nothing. Score, the verifier and the verdict rule never move.
    shot_caps = [args.per_shot_seconds]
    if args.per_shot_seconds:
        shot_caps.append(args.per_shot_seconds * 2)
    levels = [
        (source_cap, shot_cap)
        for shot_cap in shot_caps
        for source_cap in ([per_source, need] if per_source < need else [per_source])
    ]
    verdicts: dict = {}
    approved: dict = {}
    rejected: set = set()
    strikes: dict[str, int] = {}
    keeps: dict[str, int] = {}
    struck: set = set()
    chosen: list[dict] = []
    for round_n in range(1, args.verify_rounds + 1):
        for source_cap, shot_cap in levels:
            for cand in cands:
                apply_shot_cap(cand, shot_cap)
            chosen, total = fill_beat_budget(
                cands,
                need,
                source_cap,
                args.min_score,
                exclude=rejected,
                want_coverage=not args.no_want_coverage,
            )
            used = (source_cap, shot_cap)
            if total >= need - DURATION_SAFETY_MARGIN:
                break
        note = "" if total >= need - DURATION_SAFETY_MARGIN else "  SHORT"
        if used != levels[0]:
            note += (
                f"  (caps relaxed: per-source {used[0]:.1f}s, per-shot {used[1]:.1f}s)"
            )
        print(
            f"  [fill {round_n}] {len(chosen)} clip(s), {total:.1f}s / {need:.1f}s"
            + note
        )
        for cand in chosen:
            print(
                f"          {cand['src_id']} #{cand['shot']} w{cand['want_number'] + 1} "
                f"score {cand['score']} {cand['start']:.1f}s +{cand['duration']:.1f}s"
                f"  {cand['why'][:80]}"
            )
        if args.no_verify:
            break
        todo = [c for c in chosen if c["vkey"] not in verdicts]
        verdicts.update(verify_candidates(todo, beat, groups, args))
        bad = [c for c in chosen if not verdict_accepts(verdicts.get(c["vkey"], {}))]
        for cand in chosen:
            verdict = verdicts.get(cand["vkey"], {})
            mark = "keep  " if verdict_accepts(verdict) else "REJECT"
            if verdict_accepts(verdict):
                # Snapshot the clip exactly as it was verified. Later rounds
                # re-cap candidates, and a beat must ship the spans that were
                # actually checked, not a longer cut of them.
                approved[cand["key"]] = {**cand, "verdict": verdict}
            print(
                f"  [{mark}] {cand['src_id']} #{cand['shot']}: "
                f"subject={verdict.get('subject_visible')} main={verdict.get('subject_is_main')} "
                f"overlay={verdict.get('has_overlay')} named={verdict.get('is_named_subject')} "
                f"- {str(verdict.get('what_you_see', verdict.get('error', '')))[:90]}"
            )
        if not bad:
            break
        for cand in bad:
            rejected.add(cand["key"])
            strikes[cand["src_id"]] = strikes.get(cand["src_id"], 0) + 1
            result["rejected"].append(
                {
                    **{k: cand[k] for k in ("src_id", "shot", "score", "why")},
                    "verdict": verdicts.get(cand["vkey"], {}),
                }
            )
        for src_id, count in strikes.items():
            keeps[src_id] = sum(1 for c in approved.values() if c["src_id"] == src_id)
            # A trap source occasionally lands one usable shot; that is not a
            # reason to let it burn every remaining substitution round. Strike
            # it once its rejections outnumber its accepted clips two to one.
            if (
                count >= args.source_strikes
                and count >= 2 * keeps[src_id]
                and src_id not in struck
            ):
                struck.add(src_id)
                dropped = [
                    c["key"]
                    for c in cands
                    if c["src_id"] == src_id and c["key"] not in rejected
                ]
                rejected.update(dropped)
                print(
                    f"  [strike] {src_id} rejected {count}x with nothing accepted; "
                    f"dropping its {len(dropped)} remaining candidate(s)"
                )
        if round_n == args.verify_rounds:
            print(f"  [stop ] substitution capped at {args.verify_rounds} round(s)")

    # Ship the best set of clips that actually passed verification, drawn from
    # every round rather than only the last one: a round that replaced a good
    # clip with a rejected one must not cost the beat the good clip.
    if args.no_verify:
        kept = [c for c in chosen if c["key"] not in rejected]
    else:
        kept, _ = fill_beat_budget(
            [c for c in approved.values() if c["key"] not in rejected],
            need,
            used[0],
            args.min_score,
            want_coverage=not args.no_want_coverage,
        )
    total = sum(c["seconds"] for c in kept)
    # Only the final beat may run long: nothing follows it to push out of sync,
    # and video.py loops the whole timeline from clip 1 unless the material
    # outlasts the narration by its safety margin.
    target = need + (args.tail_seconds if is_last else 0.0)
    if total > target + 0.01:
        total = trim_clips_to_target(kept, target, args.min_clip_seconds)
        print(f"  [trim ] beat trimmed to {total:.2f}s for a {need:.2f}s beat")
    result["per_source_cap"] = round(used[0], 2)
    result["per_shot_cap"] = round(used[1], 2)
    result["verified"] = len(verdicts)
    result["clips"] = [
        {k: v for k, v in c.items() if k not in ("key", "vkey", "span")} for c in kept
    ]
    result["have"] = round(total, 2)
    short = need - total
    if not kept:
        result["status"] = "hard fail"
    elif short > DURATION_SAFETY_MARGIN:
        result["status"] = "short"
        result["short_by"] = round(short, 2)
    else:
        result["status"] = "filled"
    print(
        f"  [beat ] {result['status'].upper()} {total:.2f}s / {need:.2f}s "
        f"from {len({c['src_id'] for c in kept})} source(s), {len(kept)} clip(s)"
    )
    return result


def select_beats(plan: dict, args) -> dict:
    """Run the whole pipeline over every beat and write the selection."""
    beats = [
        run_beat(beat, plan, args, is_last=(n == len(plan["beats"]) - 1))
        for n, beat in enumerate(plan["beats"])
    ]
    selection = {
        "plan": os.path.abspath(args.plan),
        "model": args.model,
        "verify_model": None if args.no_verify else args.verify_model,
        "verify_frames": args.verify_frames,
        "aspect": args.aspect,
        "settings": {
            "per_shot_seconds": args.per_shot_seconds,
            "per_source_seconds": args.per_source_seconds,
            "min_score": args.min_score,
            "verify_rounds": args.verify_rounds,
            "want_coverage": not args.no_want_coverage,
            "pool_seconds": args.pool_seconds,
        },
        "beats": beats,
        "tokens": dict(_TOKEN_TOTALS),
    }
    with open(args.select_out, "w") as fh:
        json.dump(selection, fh, indent=1)

    print("\n" + "=" * 60)
    filled = sum(1 for b in beats if b["status"] == "filled")
    for beat in beats:
        line = (
            f"  {beat['tag']:<9} {beat.get('have', 0.0):5.2f}s / {beat['need']:5.2f}s "
            f"{beat['status'].upper():<6} {len(beat['clips'])} clip(s)"
        )
        if beat["status"] == "short":
            line += f"  short {beat['short_by']:.2f}s"
        print(line)
    print(f"\n{filled}/{len(beats)} beat(s) filled; selection in {args.select_out}")
    print(
        f"[gemini] {_TOKEN_TOTALS['calls']} call(s), in={_TOKEN_TOTALS['in']} "
        f"out={_TOKEN_TOTALS['out']} thoughts={_TOKEN_TOTALS['thoughts']}"
    )
    return selection


def cut_selection(selection: dict, args) -> int:
    """Cut every selected span, then gate the result with preflight."""
    os.makedirs(args.outdir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="mpt_clips_beat_")
    beat_files: list[str] = []
    produced: list[str] = []
    missing: list[str] = []
    # Idempotent: a selection produced by --plan is already trimmed, but one
    # edited by hand, or produced before this rule existed, is not, and a beat
    # that runs long desynchronises every beat after it.
    for n, beat in enumerate(selection["beats"]):
        is_last = n == len(selection["beats"]) - 1
        target = beat["need"] + (args.tail_seconds if is_last else 0.0)
        have = sum(c["duration"] for c in beat["clips"])
        if have > target + 0.01:
            have = trim_clips_to_target(beat["clips"], target, args.min_clip_seconds)
            print(
                f"[trim] beat {beat['tag']}: {beat.get('have', 0):.2f}s -> "
                f"{have:.2f}s for a {beat['need']:.2f}s beat"
            )
            beat["have"] = round(have, 2)
    try:
        for beat in selection["beats"]:
            names: list[str] = []
            for n, clip in enumerate(beat["clips"], start=1):
                name = f"{args.prefix}{beat['tag']}{n}.mp4"
                outfile = os.path.join(args.outdir, name)
                print(
                    f"\n[{beat['tag']} {n}] {clip['src_id']} #{clip['shot']} "
                    f"{format_timecode(clip['start'])} +{clip['duration']:.1f}s"
                )
                if os.path.isfile(outfile) and args.reuse_clips:
                    print("  [reuse] already cut")
                elif download_section(
                    clip["source"],
                    clip["start"],
                    clip["duration"],
                    outfile,
                    args.quality,
                    workdir,
                ):
                    normalize_aspect(outfile, args.aspect, clip["duration"])
                    if not validate_clip(outfile):
                        continue
                else:
                    continue
                names.append(name)
                produced.append(name)
            if names:
                beat_files.append(f"{beat['need']}:{','.join(names)}")
            else:
                missing.append(beat["tag"])
                print(f"  [fail] beat {beat['tag']} produced no clips")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"{len(produced)} clip(s) in {args.outdir}")
    if args.cut_plan:
        # Record exactly what was cut: trimming changes every span, and the
        # selection on disk still describes the untrimmed one.
        cut_path = f"{os.path.splitext(args.cut_plan)[0]}.cut.json"
        with open(cut_path, "w") as fh:
            json.dump(selection, fh, indent=1)
        print(f"cut spans written to {cut_path}")
    if not produced:
        return 1
    print(f'\n  --video-source local --video-materials "{",".join(produced)}"\n')
    status = preflight(beat_files, args.outdir, args.beat_tolerance, args.black_check)
    if missing:
        # A beat with no files cannot be expressed as a --beat spec, so it would
        # otherwise vanish from the gate and look like a pass.
        print(
            f"\npreflight FAILED - {len(missing)} beat(s) produced no clips at all: "
            f"{', '.join(missing)}"
        )
        return 1
    return status


def validate_clip(path: str) -> bool:
    width, height = probe_dimensions(path)
    duration = probe_duration(path)
    if min(width, height) < MIN_DIMENSION:
        print(
            f"  [warn] {os.path.basename(path)} is {width}x{height}; "
            f"MoneyPrinterTurbo drops material under {MIN_DIMENSION}px - removing"
        )
        os.remove(path)
        return False
    if duration <= 0:
        print(f"  [warn] {os.path.basename(path)} has no readable duration - removing")
        os.remove(path)
        return False
    print(f"  [ok] {os.path.basename(path)}  {width}x{height}  {duration:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pre-cut B-roll into storage/local_videos for --video-source local.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="YouTube URL or local video path (repeatable)",
    )
    parser.add_argument(
        "--want",
        action="append",
        default=[],
        help="description of a shot to find (repeatable, order is preserved)",
    )
    parser.add_argument(
        "--manual",
        action="append",
        default=[],
        help="SOURCE@START-END, e.g. 'https://...@01:12-01:20'. Skips Gemini entirely.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=int,
        default=6,
        help="length of each downloaded clip (default: 6)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="seconds between sampled frames during analysis (default: 5)",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="output directory (default: storage/local_videos)",
    )
    parser.add_argument(
        "--prefix", default="clip", help="output filename prefix (default: clip)"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model for shot selection (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--need-seconds",
        type=float,
        default=0.0,
        help=(
            "seconds of footage each --want must be filled with. When set, "
            "--auto-shots asks the model to rank every usable shot and then "
            "takes shots until the budget is covered, instead of taking exactly "
            "one shot per want (default: 0 = one per want)"
        ),
    )
    parser.add_argument(
        "--fine-threshold",
        type=float,
        default=FINE_SCENE_THRESHOLD,
        help=(
            "scene sensitivity used to trim clips to a single shot "
            f"(default: {FINE_SCENE_THRESHOLD}). Lower catches more cuts but "
            "can split fast motion; over-trimming is the safe direction"
        ),
    )
    parser.add_argument(
        "--no-cut-trim",
        action="store_true",
        help=(
            "do not trim clips to the largest cut-free span inside a shot. "
            "The sheet-building scene threshold is coarse and merges real "
            "shots, so trimming is on by default"
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check that clips can fill the narration before rendering; see --beat",
    )
    parser.add_argument(
        "--plan",
        default="",
        help="JSON beat plan: one entry per narration beat with its subject, "
        "wanted shots, candidate sources and measured duration. Runs the "
        "beat-level pipeline: rank across all of a beat's sources in one call, "
        "fill the duration budget in code, verify the shipped crop cold, and "
        "substitute what the verifier rejects. One ranking call per beat.",
    )
    parser.add_argument(
        "--select-out",
        default="",
        help="where --plan writes its selection (default: <plan>.selection.json)",
    )
    parser.add_argument(
        "--cut-plan",
        default="",
        help="cut every clip named in a selection written by --plan, then "
        "preflight the result",
    )
    parser.add_argument(
        "--reuse-clips",
        action="store_true",
        help="with --cut-plan, keep clip files that already exist in --outdir",
    )
    parser.add_argument(
        "--per-shot-seconds",
        type=float,
        default=6.0,
        help="most seconds one shot may contribute to a beat (default: 6.0); "
        "trades fill against variety. Doubled automatically when a beat would "
        "otherwise ship short, and the re-cut clip is verified again. "
        "0 disables the cap",
    )
    parser.add_argument(
        "--per-source-seconds",
        type=float,
        default=0.0,
        help="most seconds one source may contribute to a beat "
        "(default: 0 = 60%% of the beat's need), so a long trap source cannot "
        "own a beat merely by being long",
    )
    parser.add_argument(
        "--rank-attempts",
        type=int,
        default=2,
        help="how many times a beat's ranking call may be repeated while its "
        f"candidate pool is thinner than {MIN_POOL_FACTOR} shots per wanted "
        "shot (default: 2). Replies vary run to run, so a second ask usually "
        "deepens the pool substitution spends",
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=0.5,
        help="surplus allowed on the LAST beat only (default: 0.5). Earlier "
        "beats are trimmed to their exact need because video.py concatenates "
        "clips by duration alone, so one long beat delays every beat after it. "
        "The tail keeps the timeline longer than the audio, which is what stops "
        "video.py looping it from the first clip",
    )
    parser.add_argument(
        "--min-clip-seconds",
        type=float,
        default=1.0,
        help="shortest a clip may be trimmed to when a beat is trimmed to its "
        "measured need (default: 1.0)",
    )
    parser.add_argument(
        "--beat-tolerance",
        type=float,
        default=0.35,
        help="seconds a beat may exceed its narration before --preflight calls "
        "it a problem (default: 0.35)",
    )
    parser.add_argument(
        "--black-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="report clips that fade to black; a dip between the verifier's "
        "three sampled frames is otherwise invisible until the render is "
        "watched (default: on)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="drop ranked candidates scoring below this 1-5 value (default: 3)",
    )
    parser.add_argument(
        "--pool-seconds",
        type=float,
        default=0.0,
        help="seconds per wanted shot the ranking call is asked to cover "
        f"(default: 0 = {POOL_DEPTH:g}x the beat's need). Larger asks for a "
        "deeper candidate pool, which is what substitution spends",
    )
    parser.add_argument(
        "--no-want-coverage",
        action="store_true",
        help="fill a beat purely by score instead of seeding it with the best "
        "candidate for each wanted shot",
    )
    parser.add_argument(
        "--verify-model",
        default="gemini-3.7-flash",
        help="model that verifies the shipped crop cold (default: gemini-3.7-flash)",
    )
    parser.add_argument(
        "--verify-frames",
        type=int,
        default=3,
        help="frames per clip shown to the verifier: 3 is a start/middle/end "
        "filmstrip of the span that actually ships, 1 is the midpoint alone "
        "(default: 3). Measured paired on 26 clips, the filmstrip changed 3 "
        "verdicts and was right on all 3: it caught two clips whose player "
        "progress bar is only visible away from the midpoint, and kept one good "
        "clip whose midpoint happens to be black",
    )
    parser.add_argument(
        "--source-strikes",
        type=int,
        default=2,
        help="how many rejections a source may collect in one beat, with "
        "nothing of its own accepted, before the rest of its candidates are "
        "dropped (default: 2). Trap sources - reaction videos, burned-in "
        "telemetry - fail the same way repeatedly, and each retry costs a "
        "substitution round. A source that has had a clip accepted is never "
        "struck: a couple of bad shots is normal, and dropping a good source "
        "over them collapses the beat",
    )
    parser.add_argument(
        "--verify-rounds",
        type=int,
        default=3,
        help="how many times a rejected clip may be replaced before the beat is "
        "reported short (default: 3)",
    )
    parser.add_argument(
        "--verify-workers",
        type=int,
        default=6,
        help="parallel verification requests (default: 6)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip verification and substitution; selection only",
    )
    parser.add_argument(
        "--check-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read each source's real height and skip sources whose shipped "
        "crop would fall under the size video.py silently drops (default: on)",
    )
    parser.add_argument(
        "--beat",
        action="append",
        default=[],
        help=(
            "one narration beat as SECONDS:file1,file2 (repeatable). Files "
            "resolve against --outdir unless absolute. Used with --preflight"
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=["low", "medium", "high"],
        default=None,
        help=(
            "cap reasoning effort on thinking-capable models; omit to use the "
            "model default. Reasoning tiers bill thinking against the output "
            "allowance and can truncate the JSON reply"
        ),
    )
    parser.add_argument(
        "--quality",
        default="bv*[height>=1080][ext=mp4]/bv*[ext=mp4]/bv*",
        help="yt-dlp format selector for the final download",
    )
    parser.add_argument(
        "--sheets-only",
        action="store_true",
        help="detect shots, write contact sheets + shots.json, then stop "
        "(for human/agent review; makes no API call)",
    )
    parser.add_argument(
        "--pick",
        default="",
        help="comma-separated shot indices from a previous --sheets-only run, "
        "cut at exact shot boundaries. Order is preserved.",
    )
    parser.add_argument(
        "--auto-shots",
        action="store_true",
        help="let the model pick whole shots from the --sheets-only contact "
        "sheets (needs --source and --want). Writes auto_picks.json and prints "
        "a --pick string; does not download. One API call per source.",
    )
    parser.add_argument(
        "--auto-window",
        default="",
        help="comma-separated shot indices to shortlist, then let the model "
        "choose the exact sub-range inside them (needs --source and one --want). "
        "Writes auto_window.json. One API call per source.",
    )
    parser.add_argument(
        "--refine",
        default="",
        help="comma-separated shot indices to re-sample at --refine-fps inside "
        "each shot, writing timestamp-labelled sheets, then stop. Use this to "
        "find the exact usable sub-window of a shot. Makes no API call.",
    )
    parser.add_argument(
        "--refine-fps",
        type=float,
        default=2.0,
        help="frames per second sampled inside each --refine shot (default: 2.0)",
    )
    parser.add_argument(
        "--pick-window",
        action="append",
        default=[],
        help="absolute 'START-END' in seconds (float), e.g. '248.9-252.4', read "
        "off a --refine sheet. Repeatable; order is preserved.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=os.path.join(REPO_ROOT, "storage", "analysis"),
        help="where --sheets-only stores sheets and shots.json",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.3,
        help="ffmpeg scene-change sensitivity, 0-1 (default: 0.3)",
    )
    parser.add_argument(
        "--min-shot",
        type=float,
        default=2.0,
        help="ignore shots shorter than this many seconds (default: 2.0)",
    )
    parser.add_argument(
        "--skip-head",
        type=int,
        default=0,
        help="ignore the first N seconds when picking shots (channel intros)",
    )
    parser.add_argument(
        "--skip-tail",
        type=int,
        default=20,
        help="ignore the last N seconds when picking shots (end credits, default: 20)",
    )
    parser.add_argument(
        "--aspect",
        default="9:16",
        choices=["9:16", "16:9", "1:1", "none"],
        help="centre-crop clips to this aspect (default: 9:16). "
        "MoneyPrinterTurbo letterboxes mismatched material instead of cropping.",
    )
    parser.add_argument(
        "--keep-proxy",
        action="store_true",
        help="keep the analysis proxy and contact sheets for inspection",
    )
    args = parser.parse_args()

    if not (
        args.manual
        or args.sheets_only
        or args.pick
        or args.refine
        or args.pick_window
        or args.auto_shots
        or args.auto_window
        or args.preflight
        or args.plan
        or args.cut_plan
        or (args.source and args.want)
    ):
        parser.error(
            "provide --manual, --sheets-only, --refine, --pick, --pick-window, "
            "--auto-shots, --auto-window, --preflight, --plan, --cut-plan, or "
            "both --source and --want"
        )

    require_tool("ffmpeg")
    require_tool("ffprobe")

    # ---- preflight: no download, no API call, just refuse bad material -----
    if args.preflight:
        if not args.beat:
            parser.error("--preflight needs at least one --beat SECONDS:file1,file2")
        raise SystemExit(
            preflight(args.beat, args.outdir, args.beat_tolerance, args.black_check)
        )

    # ---- beat-level pipeline: rank, fill, verify, substitute ---------------
    if args.plan:
        plan = load_plan(args.plan)
        args.model = plan.get("model", args.model)
        args.verify_model = plan.get("verify_model", args.verify_model)
        args.aspect = plan.get("aspect", args.aspect)
        if not args.select_out:
            args.select_out = f"{os.path.splitext(args.plan)[0]}.selection.json"
        if args.check_resolution:
            require_tool("yt-dlp")
        select_beats(plan, args)
        return

    # ---- cut what the pipeline selected, then gate it ----------------------
    if args.cut_plan:
        with open(args.cut_plan) as fh:
            selection = json.load(fh)
        args.aspect = selection.get("aspect", args.aspect)
        require_tool("yt-dlp")
        raise SystemExit(cut_selection(selection, args))

    if any(is_url(s) for s in args.source) or any(
        is_url(m.rsplit("@", 1)[0]) for m in args.manual
    ):
        require_tool("yt-dlp")

    os.makedirs(args.outdir, exist_ok=True)
    produced: list[str] = []
    index = 0

    # ---- shot analysis: sheets for review, no API call ---------------------
    if args.sheets_only:
        if not args.source:
            parser.error("--sheets-only needs --source")
        for source in args.source:
            slug = source_slug(source)
            workdir = os.path.join(args.analysis_dir, slug)
            # Keep any cached proxy: re-analysis is common (different --min-shot
            # or --aspect) and the download is by far the slowest step.
            os.makedirs(workdir, exist_ok=True)
            for stale in glob.glob(os.path.join(workdir, "shots_*.jpg")) + glob.glob(
                os.path.join(workdir, "refine_*.jpg")
            ):
                os.remove(stale)
            shutil.rmtree(os.path.join(workdir, "shotframes"), ignore_errors=True)
            print(f"\n=== analysing {source} ===")
            cached = os.path.join(workdir, "proxy.mp4")
            if os.path.isfile(cached):
                print(
                    f"  [proxy] reusing cached {os.path.getsize(cached) / 1e6:.1f} MB"
                )
                proxy = cached
            else:
                proxy = download_proxy(source, workdir)
            shots = detect_shots(
                proxy,
                args.scene_threshold,
                args.min_shot,
                args.skip_head,
                args.skip_tail,
            )
            if not shots:
                print(
                    "  [warn] no shots met the minimum duration; lower --min-shot "
                    "or --scene-threshold. Skipping this source."
                )
                continue
            sheets = build_shot_sheets(proxy, workdir, shots, args.aspect)
            table = [
                {k: s[k] for k in ("index", "start", "end", "duration")} for s in shots
            ]
            with open(os.path.join(workdir, "shots.json"), "w") as fh:
                json.dump({"source": source, "shots": table}, fh, indent=2)
            print(f"  [sheets] {len(sheets)} sheet(s) in {workdir}")
            for sheet in sheets:
                print(f"    {sheet}")
            print(f"  [shots]  {os.path.join(workdir, 'shots.json')}")
            print("\n  Review the sheets, then cut exact shots with:")
            print(f'    --source "{source}" --pick "3,17,42"')
            print("  or refine a shortlist to sub-shot precision with:")
            print(f'    --source "{source}" --refine "3,17,42"')
        return

    # ---- model picks whole shots from the same sheets a human would review --
    if args.auto_shots:
        if not args.source or not args.want:
            parser.error("--auto-shots needs --source and --want")
        for source in args.source:
            workdir = os.path.join(args.analysis_dir, source_slug(source))
            sheets = sorted(glob.glob(os.path.join(workdir, "shots_*.jpg")))
            if not sheets:
                print(f"  [skip] no shot sheets for {source}; run --sheets-only first")
                continue
            shots = load_shots(args.analysis_dir, source)
            print(f"\n=== auto-shots {source} ({len(sheets)} sheet(s)) ===")
            if args.need_seconds > 0:
                try:
                    raw = ask_gemini_shortlist(
                        sheets,
                        args.want,
                        shots,
                        args.model,
                        args.aspect,
                        args.need_seconds,
                        args.thinking,
                    )
                except GeminiJSONError as exc:
                    print(f"  [skip] {source}: {exc}")
                    continue
                by_want: dict[int, list[dict]] = {i: [] for i in range(len(args.want))}
                for cand in raw:
                    try:
                        want_i = int(cand.get("want_number", 0)) - 1
                        cell = int(cand.get("index"))
                        score = int(cand.get("score", 3))
                    except (TypeError, ValueError):
                        print(f"  [bad ] unreadable candidate: {cand!r}")
                        continue
                    if want_i not in by_want:
                        print(f"  [bad ] want_number {want_i + 1} does not exist")
                        continue
                    if cell not in shots:
                        # Still the clearest hallucination signal there is.
                        print(f"  [oob ] shot #{cell} does not exist")
                        continue
                    by_want[want_i].append(
                        {"index": cell, "score": score, "why": cand.get("why", "")}
                    )

                chosen = []
                auto_proxy = os.path.join(workdir, "proxy.mp4")

                def shot_seconds(shot, _proxy=auto_proxy):
                    if args.no_cut_trim:
                        return usable_seconds(shot["duration"])
                    return coherent_seconds(_proxy, shot, args.fine_threshold)

                for want_i, want in enumerate(args.want):
                    cands = by_want[want_i]
                    if not cands:
                        print(f"  [none] {want}")
                        continue
                    picked, have = fill_budget(
                        cands, shots, args.need_seconds, shot_seconds
                    )
                    short = args.need_seconds - have
                    status = (
                        f"SHORT by {short:.1f}s"
                        if short > DURATION_SAFETY_MARGIN
                        else "budget met"
                    )
                    print(
                        f"  [fill] {want}\n"
                        f"         {len(cands)} candidate(s), took {len(picked)} "
                        f"-> {have:.1f}s / {args.need_seconds:.1f}s  {status}"
                    )
                    for cell in picked:
                        shot = shots[cell]
                        why = next(c["why"] for c in cands if c["index"] == cell)
                        real = shot_seconds(shot)
                        span = (
                            f" [single-shot {real:.1f}s]"
                            if real < shot["duration"] - 0.25
                            else ""
                        )
                        print(
                            f"         -> shot #{cell} {shot['start']:.1f}-"
                            f"{shot['end']:.1f}s ({shot['duration']:.1f}s)"
                            f"{span}: {why}"
                        )
                        chosen.append(cell)
                picks = raw
            else:
                try:
                    picks = ask_gemini_shots(
                        sheets, args.want, shots, args.model, args.aspect, args.thinking
                    )
                except GeminiJSONError as exc:
                    print(f"  [skip] {source}: {exc}")
                    continue
                chosen = []
                for pick in picks:
                    cell = pick.get("index")
                    want = pick.get("want", "?")
                    if cell is None:
                        print(f"  [none] {want}")
                        continue
                    try:
                        cell = int(cell)
                    except (TypeError, ValueError):
                        print(f"  [bad ] {want}: index {cell!r}")
                        continue
                    if cell not in shots:
                        # A shot index that does not exist is the clearest
                        # possible hallucination signal, so it is reported
                        # rather than clamped.
                        print(f"  [oob ] {want}: shot #{cell} does not exist")
                        continue
                    shot = shots[cell]
                    chosen.append(cell)
                    print(
                        f"  [pick] {want}\n"
                        f"         -> shot #{cell} {shot['start']:.1f}-"
                        f"{shot['end']:.1f}s "
                        f"({shot['duration']:.1f}s): {pick.get('why', '')}"
                    )
            out = os.path.join(workdir, "auto_picks.json")
            with open(out, "w") as fh:
                json.dump(
                    {"source": source, "model": args.model, "picks": picks},
                    fh,
                    indent=2,
                )
            print(f"  [saved] {out}")
            if chosen:
                print(f'  --source "{source}" --pick "{",".join(map(str, chosen))}"')
        print(
            f"\n[gemini] {_TOKEN_TOTALS['calls']} call(s), "
            f"in={_TOKEN_TOTALS['in']} out={_TOKEN_TOTALS['out']} "
            f"thoughts={_TOKEN_TOTALS['thoughts']}"
        )
        return

    # ---- model picks an exact sub-range inside shortlisted shots ------------
    if args.auto_window:
        if len(args.source) != 1 or len(args.want) != 1:
            parser.error("--auto-window needs exactly one --source and one --want")
        source = args.source[0]
        workdir = os.path.join(args.analysis_dir, source_slug(source))
        shots = load_shots(args.analysis_dir, source)
        segments = []
        for token in args.auto_window.split(","):
            token = token.strip()
            if not token:
                continue
            shot = shots.get(int(token))
            if shot is None:
                print(f"  [skip] unknown shot #{token}")
                continue
            segments.append({**shot, "label": f"#{shot['index']}"})
        if not segments:
            sys.exit("error: --auto-window selected nothing")

        proxy = os.path.join(workdir, "proxy.mp4")
        if not os.path.isfile(proxy):
            os.makedirs(workdir, exist_ok=True)
            proxy = download_proxy(source, workdir)
        print(f"\n=== auto-window {source} ===")
        sheets, cells = build_refine_sheets(
            proxy, workdir, segments, args.refine_fps, args.aspect
        )
        answer = ask_gemini_window(
            sheets, cells, args.want[0], args.model, args.aspect, args.thinking
        )
        if not answer:
            sys.exit("error: no usable window in the model reply")

        first, last = answer.get("first"), answer.get("last")
        if first is None or last is None:
            print(f"  [none] model declined: {answer.get('why', '')}")
            return
        try:
            first, last = int(first), int(last)
        except (TypeError, ValueError):
            sys.exit(f"error: non-integer cell numbers: {first!r},{last!r}")
        if not (0 <= first <= last < len(cells)):
            sys.exit(f"error: cells [{first},{last}] outside [0,{len(cells) - 1}]")

        # We own this arithmetic: the model only ever named cell numbers.
        start = cells[first]["time"]
        end = cells[last]["time"] + 1.0 / args.refine_fps
        out = os.path.join(workdir, "auto_window.json")
        with open(out, "w") as fh:
            json.dump(
                {
                    "source": source,
                    "model": args.model,
                    "want": args.want[0],
                    "answer": answer,
                    "start": start,
                    "end": end,
                },
                fh,
                indent=2,
            )
        print(
            f"  [window] cells [{first}]..[{last}] -> {start:.2f}-{end:.2f}s "
            f"({end - start:.2f}s): {answer.get('why', '')}"
        )
        print(f'  --source "{source}" --pick-window "{start:.2f}-{end:.2f}"')
        print(
            f"\n[gemini] {_TOKEN_TOTALS['calls']} call(s), "
            f"in={_TOKEN_TOTALS['in']} out={_TOKEN_TOTALS['out']} "
            f"thoughts={_TOKEN_TOTALS['thoughts']}"
        )
        return

    # ---- second-tier refinement: 2 fps inside a shortlist, no API call -----
    if args.refine:
        if len(args.source) != 1:
            parser.error("--refine needs exactly one --source")
        source = args.source[0]
        workdir = os.path.join(args.analysis_dir, source_slug(source))

        # Entries are either a shot index from shots.json, or an absolute
        # "START-END" range. Ranges matter for single-take sources, where one
        # "shot" can be minutes long and refining all of it is unreviewable.
        segments: list[dict] = []
        shots: dict[int, dict] | None = None
        for token in args.refine.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start, end = parse_window(token, "--refine")
                segments.append(
                    {
                        "label": f"t{start:.0f}",
                        "start": start,
                        "end": end,
                        "duration": end - start,
                    }
                )
                continue
            if shots is None:
                shots = load_shots(args.analysis_dir, source)
            shot = shots.get(int(token))
            if shot is None:
                print(f"  [skip] unknown shot #{token}")
                continue
            segments.append({**shot, "label": f"#{shot['index']}"})
        if not segments:
            sys.exit("error: --refine selected nothing")

        frames = sum(s["duration"] * args.refine_fps for s in segments)
        if frames > 200:
            print(
                f"  [warn] {frames:.0f} frames requested; that is "
                f"{frames / FRAMES_PER_SHEET:.0f} sheets. Narrow the shortlist "
                f"or pass absolute ranges like '120-135'."
            )

        proxy = os.path.join(workdir, "proxy.mp4")
        if not os.path.isfile(proxy):
            os.makedirs(workdir, exist_ok=True)
            proxy = download_proxy(source, workdir)
        print(f"\n=== refining {len(segments)} segment(s) of {source} ===")
        sheets, _cells = build_refine_sheets(
            proxy, workdir, segments, args.refine_fps, args.aspect
        )
        for sheet in sheets:
            print(f"    {sheet}")
        print("\n  Read the t= label of the first and last good cell, then cut:")
        print(f'    --source "{source}" --pick-window "248.9-252.4"')
        return

    # ---- cut exact absolute-second windows chosen from refine sheets -------
    if args.pick_window:
        if len(args.source) != 1:
            parser.error("--pick-window needs exactly one --source")
        source = args.source[0]
        workdir = tempfile.mkdtemp(prefix="mpt_clips_window_")
        try:
            for spec in args.pick_window:
                start, end = parse_window(spec, "--pick-window")
                duration = end - start
                index += 1
                outfile = os.path.join(args.outdir, f"{args.prefix}{index}.mp4")
                print(
                    f"\n[{index}] window {start:.2f}-{end:.2f}s ({duration:.2f}s) "
                    f"of {source}"
                )
                if download_section(
                    source, start, duration, outfile, args.quality, workdir
                ):
                    normalize_aspect(outfile, args.aspect)
                    if validate_clip(outfile):
                        produced.append(os.path.basename(outfile))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        print("\n" + "=" * 60)
        if not produced:
            sys.exit("no clips were produced")
        print(f"{len(produced)} clip(s) in {args.outdir}\n")
        print(f'  --video-source local --video-materials "{",".join(produced)}"')
        return

    # ---- cut chosen shots at exact boundaries ------------------------------
    if args.pick:
        if len(args.source) != 1:
            parser.error("--pick needs exactly one --source")
        source = args.source[0]
        shots = load_shots(args.analysis_dir, source)
        pick_proxy = os.path.join(args.analysis_dir, source_slug(source), "proxy.mp4")
        if not args.no_cut_trim and not os.path.isfile(pick_proxy):
            print(
                "  [warn] no analysis proxy for this source, so clips cannot be "
                "trimmed to a single shot; they may span cuts"
            )
        workdir = tempfile.mkdtemp(prefix="mpt_clips_pick_")
        try:
            for token in args.pick.split(","):
                token = token.strip()
                if not token:
                    continue
                shot = shots.get(int(token))
                if shot is None:
                    print(f"  [skip] unknown shot #{token}")
                    continue
                index += 1
                outfile = os.path.join(args.outdir, f"{args.prefix}{index}.mp4")
                # The sheet threshold merges real shots, so cut the largest
                # cut-free span inside the chosen shot rather than the whole
                # shot, which would jump between unrelated setups.
                lo, hi = shot["start"], shot["end"]
                if not args.no_cut_trim:
                    lo, hi = coherent_span(pick_proxy, lo, hi, args.fine_threshold)
                start = lo + CLIP_EDGE_TRIM / 2
                duration = usable_seconds(hi - lo)
                trimmed = (
                    f", trimmed to a single shot from {shot['duration']:.1f}s"
                    if hi - lo < shot["duration"] - 0.05
                    else ""
                )
                print(
                    f"\n[{index}] shot #{shot['index']} "
                    f"{format_timecode(start)} +{duration:.1f}s "
                    f"(full shot {shot['duration']:.1f}s{trimmed})"
                )
                if download_section(
                    source, start, duration, outfile, args.quality, workdir
                ):
                    normalize_aspect(outfile, args.aspect)
                    if validate_clip(outfile):
                        produced.append(os.path.basename(outfile))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        print("\n" + "=" * 60)
        if not produced:
            sys.exit("no clips were produced")
        print(f"{len(produced)} clip(s) in {args.outdir}\n")
        print(f'  --video-source local --video-materials "{",".join(produced)}"')
        return
    # Shared scratch space for full-source fallback downloads in manual mode.
    manual_workdir = tempfile.mkdtemp(prefix="mpt_clips_manual_")

    # ---- manual mode: no tokens spent -------------------------------------
    for entry in args.manual:
        if "@" not in entry:
            sys.exit(f"error: --manual needs SOURCE@START-END, got: {entry}")
        source, span = entry.rsplit("@", 1)
        if "-" not in span:
            sys.exit(f"error: --manual needs START-END, got: {span}")
        start_s, end_s = span.split("-", 1)
        start = parse_timecode(start_s)
        duration = max(1.0, parse_timecode(end_s) - start)
        index += 1
        outfile = os.path.join(args.outdir, f"{args.prefix}{index}.mp4")
        print(
            f"\n[{index}] manual {source} @ {format_timecode(start)} +{duration:.1f}s"
        )
        if download_section(
            source, start, duration, outfile, args.quality, manual_workdir
        ):
            normalize_aspect(outfile, args.aspect)
            if validate_clip(outfile):
                produced.append(os.path.basename(outfile))

    # ---- AI mode: one Gemini call per source ------------------------------
    for source in args.source:
        print(f"\n=== analysing {source} ===")
        workdir = tempfile.mkdtemp(prefix="mpt_clips_")
        try:
            proxy = download_proxy(source, workdir)
            sheets, valid_indices = build_contact_sheets(
                proxy, workdir, args.sample_every, args.skip_head, args.skip_tail
            )
            picks = ask_gemini(
                sheets,
                args.want,
                args.sample_every,
                args.clip_seconds,
                valid_indices,
                args.model,
            )
            duration = probe_duration(proxy)

            for pick in picks:
                want = pick.get("want", "?")
                cell = pick.get("index")
                if cell is None:
                    print(f"  [skip] no match for: {want}")
                    continue
                try:
                    cell = int(cell)
                except (TypeError, ValueError):
                    print(f"  [skip] bad index for {want}: {cell!r}")
                    continue
                if cell not in valid_indices:
                    print(
                        f"  [skip] index #{cell} is outside the considered range for {want}"
                    )
                    continue

                # We own this arithmetic so the model cannot hallucinate a timestamp.
                start = float(cell * args.sample_every)
                # Keep the requested window inside the source.
                if duration and start + args.clip_seconds > duration:
                    start = max(0.0, duration - args.clip_seconds)

                index += 1
                outfile = os.path.join(args.outdir, f"{args.prefix}{index}.mp4")
                print(f"\n[{index}] {want}")
                print(
                    f"      -> cell #{cell} = {format_timecode(start)} +{args.clip_seconds}s "
                    f"({pick.get('why', '')})"
                )
                if download_section(
                    source, start, args.clip_seconds, outfile, args.quality, workdir
                ):
                    normalize_aspect(outfile, args.aspect)
                    if validate_clip(outfile):
                        produced.append(os.path.basename(outfile))
        finally:
            if args.keep_proxy:
                print(f"  [kept] analysis artifacts in {workdir}")
            else:
                shutil.rmtree(workdir, ignore_errors=True)

    # ---- result ------------------------------------------------------------
    shutil.rmtree(manual_workdir, ignore_errors=True)
    print("\n" + "=" * 60)
    if not produced:
        sys.exit("no clips were produced")
    print(f"{len(produced)} clip(s) in {args.outdir}\n")
    print("Use with MoneyPrinterTurbo:\n")
    print(f'  --video-source local --video-materials "{",".join(produced)}"')
    print("\nClips are listed in request order, so pair this with")
    print("--match-materials-to-script to keep clip N on script paragraph N.")


if __name__ == "__main__":
    main()
