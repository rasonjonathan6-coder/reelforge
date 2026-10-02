"""Engine: scene breakdown -> animated scene clips -> 1080x1920 MP4.

This is the orchestration for the "Animated Character" video type. It reuses
the existing TTS/subtitle/music stages from the rest of the pipeline and only
replaces the visual stage, so nothing that already works is removed.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pipeline import music as music_pipeline

from .character import CharacterProfile, profile_for
from .providers import (
    AnimationFailed,
    AnimationProviderFactory,
    WIDTH,
    HEIGHT,
    encoder_args as provider_encoder_args,
)
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


def _encoder_args(preset: str = "veryfast", crf: str = "21") -> list[str]:
    """Shared encoder selection (NVENC when available, else libx264)."""
    return provider_encoder_args(preset=preset, crf=crf)


def _audio_graph(has_music: bool, has_voice: bool, duration: float) -> str:
    """Audio filtergraph with the voice split so it can both duck the music and
    still be mixed in (a plain `[voice]` label is consumed by sidechaincompress)."""
    sr = music_pipeline.SAMPLE_RATE
    fade_in = min(music_pipeline.FADE_IN, duration / 3)
    fade_out = min(music_pipeline.FADE_OUT, duration / 3)
    fade_out_start = max(0.0, duration - music_pipeline.FADE_OUT)
    if has_music and has_voice:
        return (
            f"[1:a]aformat=sample_rates={sr}:channel_layouts=mono,"
            f"volume={music_pipeline.MUSIC_GAIN},"
            f"afade=t=in:st=0:d={fade_in:.2f},"
            f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f}[mus];"
            f"[2:a]aformat=sample_rates={sr}:channel_layouts=mono,"
            f"asplit=2[voice][voice2];"
            f"[mus][voice]sidechaincompress="
            f"threshold=0.05:ratio=8:attack=15:release=350:makeup=1[duck];"
            f"[voice2][duck]amix=inputs=2:duration=first:normalize=0,"
            f"alimiter=limit=0.95[aout]"
        )
    return f"[1:a]aformat=sample_rates={sr}:channel_layouts=mono,alimiter=limit=0.95[aout]"


def _finish(
    assembled: Path, ass: Path | None, out_path: Path, duration: float, *,
    audio: Path | None, music_bed: Path | None, logo_text: str | None,
    fonts_dir: Path | None,
) -> Path:
    """One FFmpeg pass: burn captions, draw the overlay, mix voice + music.

    Everything runs in a single filtergraph so a 1080x1920 reel is encoded
    once instead of four times (which is what made long renders look stuck).
    """
    vf: list[str] = []
    if ass and Path(ass).exists():
        escaped = str(ass).replace("\\", "/").replace(":", r"\:")
        filt = f"subtitles='{escaped}'"
        if fonts_dir and Path(fonts_dir).exists():
            filt += f":fontsdir='{str(fonts_dir).replace(chr(92), '/')}'"
        vf.append(filt)
    vf.append(
        f"drawbox=x=0:y=ih-12:w='iw*t/{max(duration, 0.1):.3f}':h=12"
        f":color=0x00E5FF@0.95:t=fill")
    if logo_text:
        font = os.environ.get(
            "OVERLAY_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        safe = (logo_text.replace("\\", "\\\\").replace(":", "\\:")
                .replace("'", "").replace("%", "\\%"))
        vf.append(
            f"drawtext=fontfile={font}:text='{safe}':fontcolor=white@0.85"
            f":fontsize=44:x=(w-text_w)/2:y=120:box=1:boxcolor=black@0.35:boxborderw=18")

    has_voice = bool(audio and Path(audio).exists())
    has_music = bool(music_bed and Path(music_bed).exists())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(assembled)]
    if has_music:
        cmd += ["-i", str(music_bed)]
    if has_voice:
        cmd += ["-i", str(audio)]

    graph = [f"[0:v]{','.join(vf)}[vout]"]
    if has_music or has_voice:
        graph.append(_audio_graph(has_music, has_voice, duration))
        cmd += ["-filter_complex", ";".join(graph),
                "-map", "[vout]", "-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-filter_complex", ";".join(graph), "-map", "[vout]", "-an"]
    cmd += [*_encoder_args(), "-profile:v", "high", "-level", "4.1",
            "-movflags", "+faststart", "-t", f"{duration:.3f}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    return out_path


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
    music_bed: Path | None = None,
    logo_text: str | None = None,
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _finish(assembled, captions, out_path, duration, audio=audio_path,
            music_bed=music_bed, logo_text=logo_text, fonts_dir=fonts_dir)
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
