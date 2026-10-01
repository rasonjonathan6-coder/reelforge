"""Final assembly: background + audio + karaoke subtitles -> 9:16 mp4."""

from __future__ import annotations

import subprocess
from pathlib import Path

WIDTH, HEIGHT = 1080, 1920
FPS = 30


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def _ass_filter(ass_path: Path) -> str:
    escaped = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    return f"ass='{escaped}'"


def compose(
    background: Path,
    audio: Path,
    ass: Path,
    out_path: Path,
    duration: float,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Dark vignette-ish overlay keeps white captions legible on any footage.
    vf = (
        f"[0:v]scale={WIDTH}:{HEIGHT},setsar=1,"
        f"drawbox=x=0:y=0:w={WIDTH}:h={HEIGHT}:color=black@0.28:t=fill,"
        f"{_ass_filter(ass)}[v]"
    )
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(background),
        "-i", str(audio),
        "-filter_complex", vf,
        "-map", "[v]", "-map", "1:a",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-movflags", "+faststart", "-shortest",
        str(out_path),
    ])
    return out_path
