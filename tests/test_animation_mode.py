"""Wiring test: the "Animated Character" mode reaches the real renderer.

`jobs.produce` is exercised with a stub generator that records the keyword
arguments, so we assert the flags actually travel from the request to
`generate.generate` without rendering a video.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import jobs
from pipeline import script_writer, tts


@pytest.fixture()
def captured(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "output")

    monkeypatch.setattr(
        script_writer, "write_metadata",
        lambda script, topic: {"title": topic, "description": "d", "hashtags": [],
                               "thumbnail_prompt": "x"},
    )
    monkeypatch.setattr(jobs.thumbnail, "generate_thumbnail",
                        lambda *a, **k: Path(a[2]).write_bytes(b"jpg"))
    monkeypatch.setattr(jobs, "storage", type("S", (), {
        "upload": staticmethod(lambda path, key: f"/videos/{key}")}))

    def fake_synthesize(text, out_path, voice=tts.DEFAULT_VOICE, rate=tts.DEFAULT_RATE):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"mp3")
        return tts.Speech(audio_path=Path(out_path), duration=6.0, words=[
            tts.WordTiming(text=w, start=i * 0.3, end=i * 0.3 + 0.25)
            for i, w in enumerate(text.split())])

    monkeypatch.setattr(tts, "synthesize", fake_synthesize)

    seen: dict = {}

    def fake_generate(text, out_path, on_step=None, info_out=None, **kwargs):
        seen.update(kwargs)
        seen["text"] = text
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
        if info_out:
            Path(info_out).mkdir(parents=True, exist_ok=True)
        return Path(out_path)

    monkeypatch.setattr(jobs, "generate", fake_generate)
    return seen


def test_animated_flags_travel_from_request_to_generate(captured):
    jobs.produce("anim1", {
        "text": "Narrateur: Bonjour, bienvenue dans ce test animé complet.",
        "dialogue": True,
        "animated_characters": True,
        "character_style": "cartoon_2d",
        "animation_provider": "local",
        "duration": 20,
        "music": False,
        "use_stock": True,
    })
    assert captured["animated_characters"] is True
    assert captured["character_style"] == "cartoon_2d"
    assert captured["animation_provider"] == "local"


def test_3d_style_selects_the_software_3d_provider():
    """`cartoon_3d` must route to `local3d`; other styles stay on the 2D engine."""
    from generate import _resolve_provider

    assert _resolve_provider("cartoon_3d", None) == "local3d"
    assert _resolve_provider("anime", None) is None
    assert _resolve_provider("cartoon_2d", None) is None
    # An explicit provider always wins over the style default.
    assert _resolve_provider("cartoon_3d", "local") == "local"


def test_defaults_keep_the_existing_stock_mode(captured):
    jobs.produce("stock1", {
        "text": "Un texte de narration classique pour la vidéo faceless.",
        "duration": 20,
        "music": False,
    })
    assert captured["animated_characters"] is False
    assert captured["character_style"] == "anime"
    assert captured["animation_provider"] is None


def test_animation_info_files_are_archived(tmp_path):
    """`write_animation_info` writes the scene and character reports."""
    from pipeline import animation

    speech = type("S", (), {"words": [], "audio_path": tmp_path / "a.mp3", "duration": 3.0})()
    (tmp_path / "a.mp3").write_bytes(b"mp3")
    out = tmp_path / "final.mp4"
    result = animation.generate_animated_reel(
        speech, [("Goku", "Salut !")], out, tmp_path / "w",
        target_duration=3.0, environment="parc",
    )
    info = tmp_path / "info"
    animation.write_animation_info(result, info)
    scenes = json.loads((info / "animation_scenes.json").read_text(encoding="utf-8"))
    chars = json.loads((info / "characters.json").read_text(encoding="utf-8"))
    assert scenes[0]["speaker"] == "Goku"
    assert chars["Goku"]["name"] == "Goku"
