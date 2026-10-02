"""Animation providers: pluggable sources of real animated scene video.

`AnimationProvider` is the interface; `LocalAnimationProvider` is the free,
CPU-only default that draws every frame itself (no Ken Burns, no still image).
`RemoteAnimationProvider` is a thin HTTP client for an external service and is
selected with `ANIMATION_PROVIDER=remote` plus `ANIMATION_REMOTE_URL`.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline import avatars

WIDTH, HEIGHT, FPS = 1080, 1920, 30


class AnimationFailed(RuntimeError):
    """Raised when a scene cannot be animated (never silently downgraded)."""

    def __init__(self, scene_id: int, provider: str, error: str, retryable: bool = True):
        super().__init__(f"ANIMATION_GENERATION_FAILED scene={scene_id} provider={provider}: {error}")
        self.scene_id = scene_id
        self.provider = provider
        self.error = error
        self.retryable = retryable

    def as_dict(self) -> dict:
        return {
            "code": "ANIMATION_GENERATION_FAILED",
            "scene_id": self.scene_id,
            "provider": self.provider,
            "error": self.error,
            "retryable": self.retryable,
        }


@dataclass
class ProviderOutput:
    """What every provider returns for one animated scene."""

    video_path: Path
    duration: float
    width: int
    height: int
    provider: str
    metadata: dict = field(default_factory=dict)


class AnimationProvider(ABC):
    """Interface every animation provider must implement."""

    name = "abstract"

    @abstractmethod
    def generate_scene(self, character, scene, spans, work_dir: Path) -> ProviderOutput:
        """Animate one scene for `character` with `spans` (word timings)."""


# ---------------------------------------------------------------------------
# Environment backdrops — drawn per frame so the decor itself moves
# ---------------------------------------------------------------------------
_PALETTES = {
    "chambre": ((38, 32, 58), (86, 72, 120), (200, 176, 150)),
    "cuisine": ((44, 54, 62), (120, 150, 150), (222, 214, 196)),
    "rue": ((30, 36, 58), (70, 86, 120), (200, 200, 210)),
    "bureau": ((34, 40, 48), (96, 110, 120), (210, 210, 200)),
    "parc": ((28, 52, 40), (70, 130, 80), (200, 220, 160)),
    "restaurant": ((52, 36, 34), (130, 90, 70), (230, 200, 160)),
    "salon": ((40, 34, 44), (110, 90, 110), (220, 200, 190)),
    "magasin": ((36, 40, 56), (100, 110, 150), (215, 215, 225)),
    "exterieur": ((40, 60, 90), (90, 130, 170), (210, 220, 200)),
}
_DEFAULT_ENV = ((36, 36, 50), (90, 96, 130), (205, 205, 215))


def _env_palette(env: str):
    return _PALETTES.get((env or "").strip().lower(), _DEFAULT_ENV)


def _backdrop(env: str, t: float, camera: str) -> Image.Image:
    """One animated environment frame; the motifs drift with `t`."""
    top, mid, accent = _env_palette(env)
    img = Image.new("RGB", (WIDTH, HEIGHT), top)
    draw = ImageDraw.Draw(img)
    for y in range(0, HEIGHT, 8):
        frac = y / HEIGHT
        color = tuple(int(top[i] + (mid[i] - top[i]) * frac) for i in range(3))
        draw.rectangle((0, y, WIDTH, y + 8), fill=color)
    drift = int(40 * math.sin(2 * math.pi * t / 6))
    if camera == "tracking":
        drift += int((t * 120) % 200) - 100
    # Floor band + a few parallax motifs that read as a room/street.
    draw.rectangle((0, 1320, WIDTH, HEIGHT), fill=tuple(int(c * 0.8) for c in mid))
    for index in range(5):
        x = (index * 260 + drift) % (WIDTH + 260) - 130
        draw.rectangle((x, 1180, x + 180, 1340), fill=tuple(int(c * 0.7) for c in accent))
    for index in range(3):
        x = 120 + index * 380 + drift // 2
        draw.rectangle((x, 560, x + 220, 1120), fill=tuple(int(c * 0.85) for c in mid))
        draw.rectangle((x + 40, 640, x + 180, 760), fill=accent)
    return img.convert("RGBA")


# ---------------------------------------------------------------------------
# Local provider — the real, free animation engine
# ---------------------------------------------------------------------------
class LocalAnimationProvider(AnimationProvider):
    """Draw every frame on CPU: animated backdrop + lip-synced character."""

    name = "local"

    def generate_scene(self, character, scene, spans, work_dir: Path) -> ProviderOutput:
        out = work_dir / f"scene_{scene.scene_id:02d}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.5, float(scene.duration))
        frames = max(1, int(round(duration * FPS)))
        scale = _camera_scale(scene.camera)
        pal = character.palette()
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
             "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
             "-pix_fmt", "yuv420p", str(out)],
            stdin=subprocess.PIPE,
        )
        assert proc.stdin is not None
        try:
            for index in range(frames):
                t = index / FPS
                frame = _backdrop(scene.environment, t, scene.camera)
                char = avatars.draw_frame(scene.speaker, spans, t, scene.character_action, pal)
                frame = _place_character(frame, char, scale, scene.camera, t)
                proc.stdin.write(frame.convert("RGB").tobytes())
        finally:
            proc.stdin.close()
            proc.wait()
        if proc.returncode != 0 or not out.exists():
            raise AnimationFailed(scene.scene_id, self.name, "ffmpeg encode failed")
        return ProviderOutput(
            video_path=out, duration=duration, width=WIDTH, height=HEIGHT,
            provider=self.name,
            metadata={"camera": scene.camera, "environment": scene.environment,
                      "frames": frames, "animated": True},
        )


def _camera_scale(camera: str) -> float:
    return {
        "wide": 0.75, "medium": 1.0, "close_up": 1.55,
        "over_the_shoulder": 1.2, "tracking": 1.0, "static": 1.0,
        "zoom_in": 1.0, "zoom_out": 1.0,
    }.get(camera, 1.0)


def _place_character(frame: Image.Image, char: Image.Image, scale: float,
                     camera: str, t: float) -> Image.Image:
    """Composite the animated character; camera moves the animated frames."""
    if camera == "zoom_in":
        scale *= 1.0 + 0.12 * t
    elif camera == "zoom_out":
        scale *= 1.25 - 0.12 * t
    width = max(1, int(char.width * scale))
    height = max(1, int(char.height * scale))
    sprite = char.resize((width, height))
    # Sit the character on the floor band, centred (or offset for tracking).
    x = (WIDTH - width) // 2
    if camera == "tracking":
        x += int(30 * math.sin(2 * math.pi * t / 3))
    y = 1330 - height
    frame.alpha_composite(sprite, (x, max(0, y)))
    return frame


# ---------------------------------------------------------------------------
# Remote provider — plug an external service without touching the pipeline
# ---------------------------------------------------------------------------
class RemoteAnimationProvider(AnimationProvider):
    """Call an external animation service over HTTP (JSON in, video out)."""

    name = "remote"

    def __init__(self, url: str = "", token: str = ""):
        self.url = url or os.environ.get("ANIMATION_REMOTE_URL", "")
        self.token = token or os.environ.get("ANIMATION_REMOTE_TOKEN", "")

    def generate_scene(self, character, scene, spans, work_dir: Path) -> ProviderOutput:
        if not self.url:
            raise AnimationFailed(scene.scene_id, self.name,
                                  "ANIMATION_REMOTE_URL is not configured", retryable=False)
        payload = json.dumps({
            "character": character.as_dict(),
            "scene": scene.as_dict(),
            "spans": [[s, a, b] for s, a, b in spans],
        }).encode()
        request = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.token}"} if self.token else {})},
        )
        out = work_dir / f"scene_{scene.scene_id:02d}.mp4"
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                out.write_bytes(response.read())
        except Exception as exc:  # noqa: BLE001 - surfaced as AnimationFailed
            raise AnimationFailed(scene.scene_id, self.name, str(exc)) from exc
        if not out.exists() or out.stat().st_size == 0:
            raise AnimationFailed(scene.scene_id, self.name, "empty response")
        return ProviderOutput(video_path=out, duration=float(scene.duration),
                              width=WIDTH, height=HEIGHT, provider=self.name,
                              metadata={"animated": True, "source": "remote"})


class AnimationProviderFactory:
    """Build a provider from an explicit name or `ANIMATION_PROVIDER`."""

    _REGISTRY = {"local": LocalAnimationProvider, "remote": RemoteAnimationProvider}

    @classmethod
    def create(cls, name: str | None = None) -> AnimationProvider:
        key = (name or os.environ.get("ANIMATION_PROVIDER", "local")).strip().lower()
        provider_cls = cls._REGISTRY.get(key)
        if provider_cls is None:
            raise ValueError(f"Unknown ANIMATION_PROVIDER: {key!r} (have {sorted(cls._REGISTRY)})")
        return provider_cls()
