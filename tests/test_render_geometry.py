"""The render geometry must follow the environment.

A 1080x1920 encode is what OOM-kills a 512 MB Render free instance, so the
host is expected to lower VIDEO_WIDTH/VIDEO_HEIGHT. Every module that sizes a
frame or an ffmpeg filter has to read the same values, or subtitles end up
positioned for a resolution the video is not rendered at.
"""

from __future__ import annotations

import importlib

import pytest


def _reload(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import config

    importlib.reload(config)
    return config


MODULES = [
    "pipeline.visuals",
    "pipeline.compose",
    "pipeline.ai_images",
    "pipeline.overlay",
    "pipeline.thumbnail",
    "pipeline.subtitles",
]


@pytest.fixture()
def reload_modules():
    """Reload config and every geometry consumer, then restore the defaults."""
    import config

    def run(monkeypatch, **env):
        cfg = _reload(monkeypatch, **env)
        loaded = {}
        for name in MODULES:
            loaded[name] = importlib.reload(importlib.import_module(name))
        return cfg, loaded

    yield run

    # Restore the process-wide defaults so later tests keep 1080x1920.
    import os

    for key, value in (
        ("VIDEO_WIDTH", "1080"),
        ("VIDEO_HEIGHT", "1920"),
        ("VIDEO_FPS", "30"),
        ("FFMPEG_THREADS", "1"),
        ("FFMPEG_PRESET", "veryfast"),
    ):
        os.environ[key] = value
    importlib.reload(config)
    for name in MODULES:
        importlib.reload(importlib.import_module(name))


def test_geometry_env_overrides_every_module(monkeypatch, reload_modules):
    cfg, mods = reload_modules(
        monkeypatch, VIDEO_WIDTH="720", VIDEO_HEIGHT="1280", VIDEO_FPS="24"
    )
    assert (cfg.VIDEO_WIDTH, cfg.VIDEO_HEIGHT, cfg.VIDEO_FPS) == (720, 1280, 24)

    for name in ("pipeline.visuals", "pipeline.compose", "pipeline.ai_images",
                 "pipeline.overlay", "pipeline.thumbnail"):
        mod = mods[name]
        assert (mod.WIDTH, mod.HEIGHT) == (720, 1280), name

    assert mods["pipeline.subtitles"].PLAY_RES == (720, 1280)
    assert mods["pipeline.visuals"].FPS == 24
    assert mods["pipeline.compose"].FPS == 24


def test_default_geometry_is_full_portrait(monkeypatch, reload_modules):
    cfg, mods = reload_modules(monkeypatch)
    assert (cfg.VIDEO_WIDTH, cfg.VIDEO_HEIGHT) == (1080, 1920)
    assert mods["pipeline.subtitles"].PLAY_RES == (1080, 1920)


def test_ffmpeg_pressure_defaults_are_thrifty():
    import config

    assert config.FFMPEG_THREADS == 1
    assert config.FFMPEG_PRESET == "veryfast"


def test_ffmpeg_calls_cap_threads():
    """No encode may ask ffmpeg for an unbounded thread count."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "pipeline"
    offenders = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if '"loglevel", "error",' in line and "FFMPEG_THREADS" not in line:
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, "encodes without -threads cap:\n" + "\n".join(offenders)
