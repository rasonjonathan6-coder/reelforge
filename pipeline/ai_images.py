"""Free AI images for the reel background — no key, no account, no GPU.

The CPU-only server cannot run a text-to-video model, so real photoreal
footage has to come from somewhere else. This module uses Pollinations, a
free text-to-image endpoint that needs no API key, then turns each image into
a short Ken Burns clip (slow zoom + grain) so the reel still moves. The result
is a genuine AI-generated background produced entirely inside the app.

Honest limit: these are moving stills, not true video. Motion is camera
motion, not subject motion. That is the price of doing it on a CPU for free;
the Colab bridge stays available for anyone who wants real motion.
"""

from __future__ import annotations

import random
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import requests

from pipeline import video_prompts

from config import FFMPEG_THREADS, VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH

WIDTH, HEIGHT = VIDEO_WIDTH, VIDEO_HEIGHT
FPS = VIDEO_FPS
ZOOM_PER_SECOND = 0.03
ZOOM_MAX = 1.12
SCENE_SECONDS = 4.0
# Slow zoom on a still reads as footage; without grain it reads as a slide.
GRAIN = "noise=alls=8:allf=t+u"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")


def build_prompt(topic: str, scene_text: str, index: int = 0) -> str:
    """An English image prompt for one scene, reusing the video prompt builder."""
    return video_prompts.prompt_for(scene_text, topic=topic, index=index)


def fetch_image(prompt: str, dest: Path, seed: int | None = None,
                base_url: str | None = None, timeout: int = 90,
                attempts: int = 6) -> Path | None:
    """Download one free AI image. Returns None when the service is unavailable.

    The free endpoint throttles (HTTP 402 when the anonymous quota is busy), so
    a single request is unreliable; we retry with a growing pause. Never raises:
    a missing image must degrade to the local fallback, not fail the job.
    """
    import os

    base = (base_url or os.environ.get("AI_IMAGE_BASE_URL")
            or "https://image.pollinations.ai").rstrip("/")
    model = os.environ.get("AI_IMAGE_MODEL", "sdxl")
    attempts = int(os.environ.get("AI_IMAGE_ATTEMPTS", attempts))
    seed = random.randint(1, 10**6) if seed is None else seed
    url = (
        f"{base}/prompt/{quote(prompt)}?width={WIDTH}&height={HEIGHT}"
        f"&nologo=true&seed={seed}&model={model}"
    )
    for attempt in range(attempts):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                return dest
            # 402 = anonymous quota busy right now; wait and try again.
            if attempt < attempts - 1:
                time.sleep(3 + 4 * attempt)
        except Exception as exc:  # noqa: BLE001 - free service, must degrade quietly
            print(f"[ai_images] tentative {attempt + 1} échouée ({exc})")
            if attempt < attempts - 1:
                time.sleep(3 + 4 * attempt)
    print(f"[ai_images] image indisponible après {attempts} tentatives")
    return None


def image_to_clip(image: Path, out_path: Path, seconds: float,
                  pan: bool = False) -> Path:
    """Turn a still into a moving clip with a slow zoom (Ken Burns)."""
    if pan:
        x_expr = "iw/2-(iw/zoom/2)+(iw-iw/zoom)/2*sin(2*PI*in_time/8)"
    else:
        x_expr = "iw/2-(iw/zoom/2)"
    vf = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='min(1+{ZOOM_PER_SECOND}*in_time,{ZOOM_MAX})':d=1:"
        f"x='{x_expr}':y='ih/2-(ih/zoom/2)':s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"{GRAIN},format=yuv420p"
    )
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-threads", str(FFMPEG_THREADS),
        "-loop", "1", "-i", str(image),
        "-t", f"{seconds:.3f}", "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", str(out_path),
    ])
    return out_path


def generate_scene_clips(topic: str, scene_texts: list[str], work_dir: Path,
                         pans: list[bool] | None = None,
                         seed: int | None = None) -> list[Path]:
    """One AI-image clip per scene, in order. Failed scenes are skipped.

    The caller checks the count: fewer clips than scenes means the montage
    still works (clips are reused), but the source report stays honest.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    pans = pans if pans and len(pans) == len(scene_texts) else [False] * len(scene_texts)
    rng = random.Random(seed)
    clips: list[Path] = []
    for index, text in enumerate(scene_texts):
        prompt = build_prompt(topic, text, index=index)
        image = work_dir / f"scene_{index}.jpg"
        if not fetch_image(prompt, image, seed=rng.randint(1, 10**6)):
            continue
        clip = work_dir / f"scene_{index}.mp4"
        try:
            image_to_clip(image, clip, SCENE_SECONDS, pan=pans[index])
        except RuntimeError as exc:
            print(f"[ai_images] encodage impossible ({exc})")
            continue
        clips.append(clip)
        # The free endpoint throttles: a short pause between scenes keeps
        # the anonymous quota from rejecting the whole batch.
        time.sleep(2)
    return clips
