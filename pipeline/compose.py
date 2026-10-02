"""Final assembly: background + audio + karaoke subtitles -> 9:16 mp4."""

from __future__ import annotations

import subprocess
from pathlib import Path

WIDTH, HEIGHT = 1080, 1920
FPS = 30

# Characters stand above the caption band (captions sit ~420px from the bottom).
AVATAR_RATIO = 860 / 520
CAPTION_TOP = 1160


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def _ass_filter(ass_path: Path) -> str:
    escaped = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
    return f"ass='{escaped}'"


def stage_avatar_width(count: int) -> int:
    """Overlay width that fits `count` characters side by side."""
    return int(min(WIDTH * 0.46, WIDTH / max(count, 1) * 0.92))


def stage_positions(count: int) -> list[tuple[int, int]]:
    """Top-left corner of each character, spread evenly and above the captions."""
    if count <= 0:
        return []
    w = stage_avatar_width(count)
    height = int(w * AVATAR_RATIO)
    gap = (WIDTH - count * w) / (count + 1)
    y = max(40, CAPTION_TOP - height)
    return [(int(gap + index * (w + gap)), y) for index in range(count)]


def compose(
    background: Path,
    audio: Path,
    ass: Path,
    out_path: Path,
    duration: float,
    characters: list[Path] | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Dark vignette-ish overlay keeps white captions legible on any footage.
    chain = (
        f"[0:v]scale={WIDTH}:{HEIGHT},setsar=1,"
        f"drawbox=x=0:y=0:w={WIDTH}:h={HEIGHT}:color=black@0.28:t=fill"
    )
    inputs = ["-i", str(background), "-i", str(audio)]
    if characters:
        positions = stage_positions(len(characters))
        w = stage_avatar_width(len(characters))
        prev = "[base]"
        chain += "[base]"
        for index, avatar in enumerate(characters):
            inputs += ["-i", str(avatar)]
            x, y = positions[index]
            label = f"[c{index}]"
            chain += (
                f";[{index + 2}:v]scale={w}:-1[av{index}]"
                f";{prev}[av{index}]overlay=x={x}:y={y}:format=auto:shortest=0{label}"
            )
            prev = label
        chain += f";{prev}{_ass_filter(ass)}[v]"
    else:
        chain += f",{_ass_filter(ass)}[v]"

    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", chain,
        "-map", "[v]", "-map", "1:a",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-movflags", "+faststart", "-shortest",
        str(out_path),
    ])
    return out_path
