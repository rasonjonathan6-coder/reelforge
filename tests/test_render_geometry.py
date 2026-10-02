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
        ("VIDEO_TRANSITION", "0.6"),
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


def test_pipeline_modules_import_every_config_constant_they_use():
    """Regression: thumbnail.py used FFMPEG_THREADS without importing it.

    That NameError fired at the 92% metadata/thumbnail stage - the exact
    failure users reported. Comparing each module's AST against the names it
    actually has bound catches the whole class of bug without a render.
    """
    import ast
    from pathlib import Path

    import config

    constants = {
        name for name in dir(config)
        if name.isupper() and not name.startswith("_")
    }
    root = Path(__file__).resolve().parent.parent / "pipeline"
    problems = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.alias):
                bound.add(node.asname or node.name)
        used = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        missing = sorted((used & constants) - bound)
        if missing:
            problems.append(f"{path.name}: {', '.join(missing)}")
    assert not problems, "config constants used but not imported:\n" + "\n".join(problems)


def test_hard_cuts_join_with_concat(monkeypatch, reload_modules, tmp_path):
    """VIDEO_TRANSITION=0 must avoid xfade, which decodes every scene at once."""
    import subprocess

    _, mods = reload_modules(monkeypatch, VIDEO_TRANSITION="0")
    visuals = mods["pipeline.visuals"]
    assert visuals.TRANSITION == 0

    segments = []
    for index in range(3):
        seg = tmp_path / f"s{index}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=navy:size=108x192:rate=30:duration=2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg)],
            check=True, capture_output=True, timeout=120,
        )
        segments.append(seg)

    out = visuals._join_xfade_timed(segments, [1.0, 1.0, 1.0], 6.0, tmp_path / "joined.mp4")
    assert out.exists() and out.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert abs(float(probe.stdout.strip()) - 6.0) < 0.3
