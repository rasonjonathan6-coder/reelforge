"""Background visuals for the reel.

Two sources, both free:
  * Pexels stock footage (portrait) when PEXELS_API_KEY is set.
  * A generated animated gradient when no key is available, so the pipeline
    still produces a real video with zero credentials.
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

WIDTH, HEIGHT = 1080, 1920
FPS = 30

PALETTES = [
    ("0x0f0c29", "0x302b63", "0x24243e", "0x1b1b3a"),
    ("0x1a1a2e", "0x16213e", "0x0f3460", "0x533483"),
    ("0x000428", "0x004e92", "0x0b8793", "0x1b2a4a"),
    ("0x232526", "0x414345", "0x2c3e50", "0x1c1c1c"),
    ("0x3a1c71", "0xd76d77", "0x4a2c8f", "0x1e1147"),
]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def generate_gradient(duration: float, out_path: Path, seed: int | None = None) -> Path:
    rng = random.Random(seed)
    c0, c1, c2, c3 = rng.choice(PALETTES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = (
        f"gradients=s={WIDTH}x{HEIGHT}:rate={FPS}:"
        f"c0={c0}:c1={c1}:c2={c2}:c3={c3}:nb_colors=4:speed=0.015:duration={duration:.3f}"
    )
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", src,
        "-vf", "gblur=sigma=40,format=yuv420p",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        str(out_path),
    ])
    return out_path


def _pexels_clips(query: str, count: int, work_dir: Path) -> list[Path]:
    import requests

    api_key = os.environ["PEXELS_API_KEY"]
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": api_key},
        params={"query": query, "orientation": "portrait", "per_page": 15},
        timeout=30,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    rng = random.Random(query)
    rng.shuffle(videos)

    clips: list[Path] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for index, video in enumerate(videos[:count]):
        files = sorted(
            video.get("video_files", []),
            key=lambda f: abs((f.get("height") or 0) - HEIGHT),
        )
        files = [f for f in files if (f.get("height") or 0) >= 720]
        if not files:
            continue
        url = files[0]["link"]
        raw = work_dir / f"raw_{index}.mp4"
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(raw, "wb") as handle:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
        clips.append(raw)
    return clips


def build_background(
    duration: float,
    work_dir: Path,
    query: str = "city night vertical",
    use_stock: bool = True,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "background.mp4"

    clips: list[Path] = []
    if use_stock and os.environ.get("PEXELS_API_KEY"):
        try:
            clips = _pexels_clips(query, count=4, work_dir=work_dir)
        except Exception as exc:  # noqa: BLE001 - fall back to generated visuals
            print(f"[visuals] Pexels unavailable ({exc}); using generated gradient")

    if not clips:
        return generate_gradient(duration, out_path)

    seg_dir = work_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    per_clip = duration / len(clips)
    segments: list[Path] = []
    for index, clip in enumerate(clips):
        seg = seg_dir / f"seg_{index}.mp4"
        # Cover-crop portrait then slow zoom for motion.
        vf = (
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},"
            f"zoompan=z='min(zoom+0.0006,1.15)':d={int(per_clip * FPS)}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps={FPS},"
            f"format=yuv420p"
        )
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(clip),
            "-t", f"{per_clip:.3f}",
            "-vf", vf, "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            str(seg),
        ])
        segments.append(seg)

    concat_file = work_dir / "concat.txt"
    concat_file.write_text("".join(f"file '{s.resolve()}'\n" for s in segments))
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        str(out_path),
    ])
    return out_path
