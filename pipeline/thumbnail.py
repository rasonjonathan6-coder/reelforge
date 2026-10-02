"""Thumbnail generation: free AI background + overlaid title.

Text is drawn with Pillow rather than FFmpeg's drawtext, which is absent from
many minimal/static FFmpeg builds.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont

from config import FFMPEG_THREADS, VIDEO_HEIGHT, VIDEO_WIDTH

WIDTH, HEIGHT = VIDEO_WIDTH, VIDEO_HEIGHT
IMAGE_URL = "https://image.pollinations.ai"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def _wrap(draw: ImageDraw.ImageDraw, title: str, font, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in title.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]


def _download_background(prompt: str, dest: Path) -> Path | None:
    seed = random.randint(1, 10**6)
    url = f"{IMAGE_URL}/prompt/{quote(prompt)}?width={WIDTH}&height={HEIGHT}&nologo=true&seed={seed}"
    try:
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as exc:  # noqa: BLE001 - fall back to a gradient background
        print(f"[thumbnail] image API unavailable ({exc}); using gradient")
        return None


def _gradient_background(dest: Path) -> Path:
    src = f"gradients=s={WIDTH}x{HEIGHT}:c0=0x1a1a2e:c1=0x533483:c2=0x0f3460:nb_colors=3"
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-threads", str(FFMPEG_THREADS),
        "-f", "lavfi", "-i", src,
        "-vf", "gblur=sigma=30,format=rgb24",
        "-frames:v", "1", str(dest),
    ])
    return dest


def generate_thumbnail(title: str, prompt: str, out_path: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    background = _download_background(prompt, work_dir / "thumb_bg.jpg")
    if not background or background.stat().st_size < 1024:
        background = _gradient_background(work_dir / "thumb_bg.png")

    image = Image.open(background).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)

    # Darken the lower half so the title stays readable on any image.
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        [0, int(HEIGHT * 0.5), WIDTH, HEIGHT], fill=(0, 0, 0, 140)
    )
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_BOLD, 92)
    lines = _wrap(draw, title, font, int(WIDTH * 0.86))
    line_height = font.size + 20
    y = int(HEIGHT * 0.60) - (line_height * len(lines)) // 2

    for line in lines:
        width = draw.textlength(line, font=font)
        draw.text(((WIDTH - width) / 2, y), line, font=font, fill="white",
                  stroke_width=8, stroke_fill="black")
        y += line_height

    image.convert("RGB").save(out_path, "JPEG", quality=90)
    return out_path
