"""Overlay pass: progress bar and optional logo/brand text.

Uses FFmpeg `drawtext`/`drawbox`, which the Debian/Docker build provides. On a
minimal static build without `drawtext` the caller should catch the failure and
skip the overlay rather than fail the whole job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from config import FFMPEG_PRESET, FFMPEG_THREADS, VIDEO_HEIGHT, VIDEO_WIDTH

WIDTH, HEIGHT = VIDEO_WIDTH, VIDEO_HEIGHT
BAR_HEIGHT = 12
BAR_COLOR = "0x00E5FF"  # matches the caption highlight (orange-yellow in BGR)


def _ffmpeg_has_drawtext() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:  # noqa: BLE001
        return False
    return " drawtext " in out or "\ndrawtext " in out


def available() -> bool:
    return bool(shutil.which("ffmpeg")) and _ffmpeg_has_drawtext()


def _escape_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "")
        .replace("%", "\\%")
    )


def add_overlay(
    video: Path,
    out_path: Path,
    duration: float,
    logo_text: str | None = None,
) -> Path:
    """Add a bottom progress bar and optional brand text to `video`."""
    filters = [
        # Progress bar: a growing box along the very bottom edge.
        f"drawbox=x=0:y=ih-{BAR_HEIGHT}:w='iw*t/{max(duration, 0.1):.3f}'"
        f":h={BAR_HEIGHT}:color={BAR_COLOR}@0.95:t=fill",
    ]
    if logo_text:
        font = os.environ.get(
            "OVERLAY_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        )
        filters.append(
            f"drawtext=fontfile={font}:text='{_escape_text(logo_text)}'"
            f":fontcolor=white@0.85:fontsize=44:x=(w-text_w)/2:y=120"
            f":box=1:boxcolor=black@0.35:boxborderw=18"
        )

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-threads", str(FFMPEG_THREADS),
            "-i", str(video),
            "-vf", ",".join(filters),
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", "21",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return out_path
