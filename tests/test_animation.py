"""Tests for the animated-character engine: profile, scenes, providers, E2E.

All tests run offline with real Pillow/ffmpeg. The E2E writes a real MP4 and
inspects it with ffprobe; nothing is mocked.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from pipeline import animation
from pipeline.animation import providers


class _Word:
    """Timed word, same shape as `tts.WordTiming`."""

    def __init__(self, text, start, end, speaker="Goku"):
        self.text, self.start, self.end, self.speaker = text, start, end, speaker


class _Speech:
    """Minimal speech object: words + audio path + duration."""

    def __init__(self, words, audio, duration):
        self.words, self.audio_path, self.duration = words, audio, duration


def _probe(path: Path, entries: str) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={entries}",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip().splitlines()[0]


def _make_audio(path: Path, seconds: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "anullsrc=r=24000:cl=mono", "-t", str(seconds), str(path)],
        check=True, capture_output=True, timeout=60,
    )
    return path


# --- CharacterProfile -----------------------------------------------------
def test_profile_is_stable_and_reused():
    a = animation.profile_for("Goku", voice_id="goku_v")
    b = animation.profile_for("Goku", voice_id="goku_v")
    assert a.id == b.id and a.hair == b.hair and a.clothes == b.clothes
    assert a.voice_id == "goku_v" and a.visual_style == "anime"
    assert a.palette()["outfit"] == a.clothes


def test_distinct_characters_get_distinct_looks():
    goku = animation.profile_for("Goku")
    vegeta = animation.profile_for("Vegeta")
    assert (goku.hair, goku.clothes) != (vegeta.hair, vegeta.clothes)


# --- SceneBreakdown -------------------------------------------------------
def test_breakdown_sizes_scenes_to_the_target_duration():
    lines = [("Goku", "Bonjour, tu veux te battre avec moi aujourd'hui ?"),
             ("Vegeta", "Oui, je vais te montrer ma vraie force.")]
    scenes = animation.breakdown(lines, 20.0, environment="parc")
    assert len(scenes) == 2
    assert abs(sum(s.duration for s in scenes) - 20.0) < 0.5
    assert all(s.lip_sync_required for s in scenes)
    assert scenes[0].environment == "parc"


def test_breakdown_reads_action_camera_and_emotion():
    lines = [("Goku", "Je marche vers la porte !"),
             ("Vegeta", "Tais-toi imbécile, je suis furieux.")]
    scenes = animation.breakdown(lines, 10.0)
    assert scenes[0].character_action == "marche"
    assert scenes[0].camera == "tracking"          # walking -> travelling camera
    assert scenes[0].emotion == "neutre"
    assert scenes[1].emotion == "colere"


# --- Providers ------------------------------------------------------------
def test_factory_defaults_to_the_local_provider(monkeypatch):
    monkeypatch.delenv("ANIMATION_PROVIDER", raising=False)
    assert animation.AnimationProviderFactory.create().name == "local"
    assert animation.AnimationProviderFactory.create("local").name == "local"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        animation.AnimationProviderFactory.create("does_not_exist")


def test_remote_provider_without_url_fails_loudly(monkeypatch):
    monkeypatch.delenv("ANIMATION_REMOTE_URL", raising=False)
    provider = providers.RemoteAnimationProvider()
    scene = animation.breakdown([("Goku", "Salut !")], 4.0)[0]
    profile = animation.profile_for("Goku")
    with pytest.raises(animation.AnimationFailed) as err:
        provider.generate_scene(profile, scene, [], Path("/tmp"))
    assert err.value.as_dict()["code"] == "ANIMATION_GENERATION_FAILED"


# --- Real animation -------------------------------------------------------
def test_local_provider_animates_frames_not_a_still_image(tmp_path):
    scene = animation.breakdown([("Goku", "Bonjour tout le monde !")], 2.0)[0]
    profile = animation.profile_for("Goku")
    spans = [("Bonjour", 0.0, 0.4), ("tout", 0.4, 0.8), ("le", 0.8, 1.0),
             ("monde", 1.0, 1.4)]
    out = providers.LocalAnimationProvider().generate_scene(
        profile, scene, spans, tmp_path / "scenes")
    assert out.video_path.exists()
    assert out.width == 1080 and out.height == 1920
    assert out.metadata["animated"] is True

    def frame(t):
        png = tmp_path / f"f{t}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t),
                        "-i", str(out.video_path), "-frames:v", "1", str(png)],
                       check=True, capture_output=True, timeout=60)
        return Image.open(png).convert("RGB").tobytes()

    frames = [frame(t) for t in (0.1, 0.4, 0.7, 1.0)]
    assert len(set(frames)) == len(frames), "scene frames are identical (not animated)"


def test_lip_sync_is_driven_by_the_spoken_words():
    spans = [("Bonjour", 0.0, 0.4), ("oui", 0.4, 0.8)]
    assert animation.providers.avatars.viseme("oui") == "u"
    # The mouth helper reads the word at the exact instant.
    from pipeline import avatars
    assert avatars._mouth_at(spans, 0.2) == avatars.viseme("Bonjour")
    assert avatars._mouth_at(spans, 0.6) == "u"


# --- E2E ------------------------------------------------------------------
def test_e2e_animated_reel_is_a_real_multi_scene_mp4(tmp_path):
    audio = _make_audio(tmp_path / "voice.mp3", 8.0)
    words = [
        _Word("Bonjour", 0.0, 0.5, "Goku"), _Word("Vegeta", 0.5, 1.1, "Goku"),
        _Word("Oui", 1.4, 1.8, "Vegeta"), _Word("montre", 1.8, 2.3, "Vegeta"),
        _Word("ta", 2.3, 2.5, "Vegeta"), _Word("force", 2.5, 3.1, "Vegeta"),
    ]
    speech = _Speech(words, audio, 8.0)
    lines = [("Goku", "Bonjour Vegeta"), ("Vegeta", "Oui, montre ta force")]
    out = tmp_path / "final.mp4"
    result = animation.generate_animated_reel(
        speech, lines, out, tmp_path / "work",
        target_duration=8.0, environment="parc", audio_path=audio,
    )
    assert out.exists() and out.stat().st_size > 0
    assert result.provider == "local"
    assert len(result.scenes) == 2
    assert _probe(out, "codec_name") == "h264"
    assert _probe(out, "width") == "1080" and _probe(out, "height") == "1920"
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True, check=True, timeout=60).stdout.strip())
    assert abs(duration - 8.0) < 0.6
    # Audio must really be present.
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True, timeout=60).stdout.strip()
    assert has_audio == "aac"


def test_scene_clips_are_not_identical(tmp_path):
    """Two scenes with different actions must not produce the same footage."""
    speech = _Speech([], tmp_path / "silent.mp3", 4.0)
    _make_audio(speech.audio_path, 4.0)
    lines = [("Goku", "Je marche vers la porte"), ("Goku", "Je m'assied sur le banc")]
    result = animation.generate_animated_reel(
        speech, lines, tmp_path / "f.mp4", tmp_path / "w",
        target_duration=4.0, environment="parc",
    )
    assert len(result.scene_clips) == 2
    assert result.scene_clips[0].read_bytes() != result.scene_clips[1].read_bytes()
