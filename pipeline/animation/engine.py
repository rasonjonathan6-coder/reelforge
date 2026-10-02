"""Engine: scene breakdown -> animated scene clips -> 1080x1920 MP4.

This is the orchestration for the "Animated Character" video type. It reuses
the existing TTS/subtitle/music stages from the rest of the pipeline and only
replaces the visual stage, so nothing that already works is removed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .character import CharacterProfile, profile_for
from .providers import AnimationFailed, AnimationProviderFactory, WIDTH, HEIGHT
from .scenes import Scene, breakdown


@dataclass
class AnimationResult:
    """What the engine produced, with the facts the caller must verify."""

    path: Path
    duration: float
    scenes: list[Scene]
    provider: str
    profiles: dict[str, CharacterProfile] = field(default_factory=dict)
    scene_clips: list[Path] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "duration": self.duration,
            "provider": self.provider,
            "scenes": [s.as_dict() for s in self.scenes],
            "characters": {k: v.as_dict() for k, v in self.profiles.items()},
            "scene_clips": [str(p) for p in self.scene_clips],
        }


def _spans_for(speech, scene: Scene, offset: float) -> list[tuple[str, float, float]]:
    """Words this scene's speaker says, re-based to the scene's own clock."""
    spans: list[tuple[str, float, float]] = []
    for word in getattr(speech, "words", None) or []:
        if getattr(word, "speaker", "") != scene.speaker:
            continue
        start, end = float(word.start), float(word.end)
        if start >= offset - 0.05 and start < offset + scene.duration + 0.4:
            spans.append((word.text, max(0.0, start - offset), max(0.0, end - offset)))
    return spans


def _concat(clips: list[Path], work_dir: Path) -> Path:
    listing = work_dir / "scenes.txt"
    listing.write_text(
        "".join(f"file '{clip.resolve()}'\n" for clip in clips), encoding="utf-8")
    out = work_dir / "animation.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)],
        check=True, capture_output=True, timeout=900,
    )
    return out


def _burn_captions(video: Path, ass: Path, out: Path, fonts_dir: Path | None = None) -> Path:
    escaped = str(ass).replace("\\", "/").replace(":", r"\:")
    filt = f"subtitles='{escaped}'"
    if fonts_dir and fonts_dir.exists():
        filt += f":fontsdir='{str(fonts_dir).replace(chr(92), '/')}'"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
         "-vf", filt, "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True, timeout=900,
    )
    return out


def _mux_audio(video: Path, audio: Path, out: Path, duration: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-t", f"{duration:.3f}",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)],
        check=True, capture_output=True, timeout=900,
    )
    return out


def generate_animated_reel(
    speech,
    lines: list[tuple[str, str]],
    out_path: Path,
    work_dir: Path,
    *,
    target_duration: float | None = None,
    voice_map: dict[str, str] | None = None,
    environment: str = "",
    visual_style: str = "anime",
    provider_name: str | None = None,
    captions: Path | None = None,
    fonts_dir: Path | None = None,
    audio_path: Path | None = None,
    on_step=None,
) -> AnimationResult:
    """Produce a real animated-character reel.

    Every scene is animated by the selected provider; if a provider fails and
    `ALLOW_STATIC_FALLBACK` is not `true`, the failure is raised as
    `AnimationFailed` instead of being hidden behind a still image.
    """
    provider = AnimationProviderFactory.create(provider_name)
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = float(target_duration or getattr(speech, "duration", 0.0) or 1.0)
    scenes = breakdown(lines, duration, voices=voice_map or {}, environment=environment)
    if not scenes:
        raise AnimationFailed(0, provider.name, "no scene to animate", retryable=False)

    profiles: dict[str, CharacterProfile] = {}
    for speaker in dict.fromkeys(s for s, _ in lines):
        profiles[speaker] = profile_for(
            speaker, voice_id=(voice_map or {}).get(speaker, ""), visual_style=visual_style)

    clips: list[Path] = []
    offset = 0.0
    for scene in scenes:
        spans = _spans_for(speech, scene, offset)
        try:
            result = provider.generate_scene(profiles[scene.speaker], scene, spans, work_dir)
        except AnimationFailed:
            if os.environ.get("ALLOW_STATIC_FALLBACK", "false").strip().lower() == "true":
                result = _static_fallback(provider.name, scene, work_dir)
            else:
                raise
        clips.append(result.video_path)
        offset += scene.duration
        if on_step:
            on_step("animation", min(95, 10 + int(80 * len(clips) / len(scenes))))

    assembled = _concat(clips, work_dir)
    final = assembled
    if captions and Path(captions).exists():
        final = _burn_captions(final, Path(captions), work_dir / "captioned.mp4", fonts_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path and Path(audio_path).exists():
        final = _mux_audio(final, Path(audio_path), out_path, duration)
    else:
        shutil.move(str(final), str(out_path))
    return AnimationResult(path=out_path, duration=duration, scenes=scenes,
                           provider=provider.name, profiles=profiles, scene_clips=clips)


def _static_fallback(provider: str, scene: Scene, work_dir: Path) -> object:
    """Explicit opt-in fallback: a still frame. Off unless the user enables it."""
    from .providers import ProviderOutput

    out = work_dir / f"scene_{scene.scene_id:02d}_static.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=0x202030:s={WIDTH}x{HEIGHT}:d={max(0.5, scene.duration)}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True, timeout=300,
    )
    return ProviderOutput(out, float(scene.duration), WIDTH, HEIGHT, provider,
                          {"animated": False, "static_fallback": True})


def write_animation_info(result: AnimationResult, info_out: Path) -> None:
    """Archive the scene breakdown and character profiles next to the video."""
    info_out.mkdir(parents=True, exist_ok=True)
    (info_out / "animation_scenes.json").write_text(
        json.dumps([s.as_dict() for s in result.scenes], ensure_ascii=False, indent=2),
        encoding="utf-8")
    (info_out / "characters.json").write_text(
        json.dumps({k: v.as_dict() for k, v in result.profiles.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
