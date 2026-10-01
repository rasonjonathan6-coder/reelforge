"""Background visuals for the reel.

Three sources, all free:
  * Pexels stock footage when PEXELS_API_KEY is set.
  * Pixabay stock footage when PIXABAY_API_KEY is set.
  * Otherwise multi-scene animated gradients with a slow zoom, film grain and
    vignette, so the reel still moves and looks composed with zero credentials.
  * A caller may also pass its own `clips` (e.g. AI-generated scenes).

Segments are joined with cross-dissolves (xfade) rather than hard cuts.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
from pathlib import Path

WIDTH, HEIGHT = 1080, 1920
FPS = 30

TRANSITION = 0.6  # cross-dissolve duration, seconds
SCENE_SECONDS = 4.0  # target on-screen time per generated scene

PALETTES = [
    ("0x0f0c29", "0x302b63", "0x24243e", "0x1b1b3a"),
    ("0x1a1a2e", "0x16213e", "0x0f3460", "0x533483"),
    ("0x000428", "0x004e92", "0x0b8793", "0x1b2a4a"),
    ("0x232526", "0x414345", "0x2c3e50", "0x1c1c1c"),
    ("0x3a1c71", "0xd76d77", "0x4a2c8f", "0x1e1147"),
    ("0x42275a", "0x734b6d", "0x2c1b47", "0x14061f"),
    ("0x0b486b", "0xf56217", "0x3b1f2b", "0x1a1a2e"),
]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def _join_xfade(segments: list[Path], duration: float, out_path: Path) -> Path:
    """Cross-dissolve the segments into a single clip of `duration` seconds."""
    if len(segments) == 1:
        _run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(segments[0]),
            "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-pix_fmt", "yuv420p", str(out_path),
        ])
        return out_path

    inputs: list[str] = []
    for seg in segments:
        inputs += ["-i", str(seg)]

    per_scene = (duration + TRANSITION * (len(segments) - 1)) / len(segments)
    steps = []
    prev = "[0:v]"
    for index in range(1, len(segments)):
        offset = index * (per_scene - TRANSITION)
        label = f"[v{index}]"
        steps.append(
            f"{prev}[{index}:v]xfade=transition=fade:duration={TRANSITION:.3f}"
            f":offset={offset:.3f}{label}"
        )
        prev = label
    filter_complex = ";".join(steps)

    _run([
        "ffmpeg", "-y", "-loglevel", "error", *inputs,
        "-filter_complex", filter_complex,
        "-map", prev,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        str(out_path),
    ])
    return out_path


def _scene_filters() -> str:
    """Slow zoom + grain + vignette: reads as footage rather than a flat loop."""
    return (
        f"zoompan=z='min(zoom+0.0005,1.10)':d={int(SCENE_SECONDS * FPS) + 1}"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"noise=alls=7:allf=t+u,"
        f"vignette=PI/4.5,"
        f"format=yuv420p"
    )


def generate_scenes(duration: float, out_path: Path, seed: int | None = None) -> Path:
    """Several animated gradient scenes with distinct palettes, cross-dissolved."""
    rng = random.Random(seed)
    count = max(1, math.ceil(duration / SCENE_SECONDS))
    per_scene = (duration + TRANSITION * (count - 1)) / count
    seg_dir = out_path.parent / "scenes"
    seg_dir.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    for index in range(count):
        c0, c1, c2, c3 = rng.choice(PALETTES)
        src = (
            f"gradients=s={WIDTH}x{HEIGHT}:rate={FPS}:"
            f"c0={c0}:c1={c1}:c2={c2}:c3={c3}:nb_colors=4"
            f":speed={rng.uniform(0.010, 0.028):.4f}:duration={per_scene + 1:.3f}"
        )
        seg = seg_dir / f"scene_{index}.mp4"
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", src,
            "-vf", f"gblur=sigma=38,{_scene_filters()}",
            "-t", f"{per_scene:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            str(seg),
        ])
        segments.append(seg)

    return _join_xfade(segments, duration, out_path)


def generate_gradient(duration: float, out_path: Path, seed: int | None = None) -> Path:
    """Single-scene gradient (kept for thumbnails and simple callers)."""
    return generate_scenes(duration, out_path, seed=seed)


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


def _pixabay_clips(query: str, count: int, work_dir: Path) -> list[Path]:
    import requests

    resp = requests.get(
        "https://pixabay.com/api/videos/",
        params={
            "key": os.environ["PIXABAY_API_KEY"],
            "q": query,
            "video_type": "film",
            "safesearch": "true",
            "per_page": 20,
        },
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    rng = random.Random(query)
    rng.shuffle(hits)

    clips: list[Path] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for index, hit in enumerate(hits[:count]):
        variants = hit.get("videos", {})
        chosen = None
        for key in ("large", "medium", "small"):
            candidate = variants.get(key)
            if candidate and (candidate.get("height") or 0) >= 720:
                chosen = candidate
                break
        if not chosen:
            continue
        raw = work_dir / f"px_{index}.mp4"
        with requests.get(chosen["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(raw, "wb") as handle:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
        clips.append(raw)
    return clips


def _cover_filter(per_scene: float) -> str:
    return (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='min(zoom+0.0006,1.15)':d={int(per_scene * FPS)}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"format=yuv420p"
    )


def build_background(
    duration: float,
    work_dir: Path,
    query: str = "city night vertical",
    use_stock: bool = True,
    clips: list[Path] | None = None,
) -> Path:
    """Produce the moving background for the reel.

    `clips` lets a caller supply its own footage (e.g. AI-generated scenes);
    those take priority over every stock source.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "background.mp4"

    if not clips and use_stock:
        for name, fetch in (("PEXELS", _pexels_clips), ("PIXABAY", _pixabay_clips)):
            if not os.environ.get(f"{name}_API_KEY"):
                continue
            try:
                clips = fetch(query, count=5, work_dir=work_dir)
                if clips:
                    break
            except Exception as exc:  # noqa: BLE001 - fall back to next source
                print(f"[visuals] {name} unavailable ({exc}); trying next source")

    if not clips:
        return generate_scenes(duration, out_path)

    count = len(clips)
    per_scene = (duration + TRANSITION * (count - 1)) / count
    seg_dir = work_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    segments: list[Path] = []
    for index, clip in enumerate(clips):
        seg = seg_dir / f"seg_{index}.mp4"
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(clip),
            "-t", f"{per_scene:.3f}",
            "-vf", _cover_filter(per_scene), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            str(seg),
        ])
        segments.append(seg)

    return _join_xfade(segments, duration, out_path)
