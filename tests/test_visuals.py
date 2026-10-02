"""Tests for the stock-footage visuals pipeline.

These run fully offline: the providers are replaced by a fake candidate list
and the clip files are served from a local HTTP server, so the real download,
probe and ffmpeg montage code paths are exercised without touching Pexels or
Pixabay.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

from pipeline import visuals


@pytest.fixture()
def clip_server(tmp_path):
    """Serve three real mp4 clips of different orientations."""
    clips = tmp_path / "clips"
    clips.mkdir()
    for name, size, duration in (
        ("landscape.mp4", "1280x720", "4"),
        ("portrait.mp4", "1080x1920", "3"),
        ("square.mp4", "720x720", "2"),
    ):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"testsrc=size={size}:rate=30:duration={duration}",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             str(clips / name)],
            check=True,
        )

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(clips))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    catalog = [
        ("landscape.mp4", 1280, 720, 4.0),
        ("portrait.mp4", 1080, 1920, 3.0),
        ("square.mp4", 720, 720, 2.0),
    ]
    yield base, catalog
    httpd.shutdown()


def _fake_candidates(base, catalog):
    def build(name, query, limit=12):
        return [
            {"id": f"{name}{index}", "url": f"{base}/{file}",
             "duration": duration, "width": width, "height": height}
            for index, (file, width, height, duration) in enumerate(catalog)
        ]

    return build


def _duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


def _size(path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    width, height = proc.stdout.strip().split(",")
    return int(width), int(height)


def test_scene_queries_are_distinct_and_scene_derived():
    script = "La ville brille la nuit. Les voitures roulent vite. Le ciel devient rouge."
    queries = visuals.scene_queries("ville", script, 3)
    assert len(queries) == 3
    assert len(set(queries)) == 3
    assert all(query.strip() for query in queries)


def test_scene_queries_fall_back_to_topic():
    queries = visuals.scene_queries("space exploration", "", 3)
    assert len(queries) == 3
    assert all("space" in query or "exploration" in query for query in queries)


def test_select_prefers_portrait_and_skips_used():
    catalog = [
        {"id": "1", "url": "u1", "duration": 3.0, "width": 1280, "height": 720},
        {"id": "2", "url": "u2", "duration": 3.0, "width": 1080, "height": 1920},
    ]
    seen: set[str] = set()
    first = visuals._select(catalog, 1, seen)
    assert first[0]["id"] == "2"
    assert seen == {"2"}
    second = visuals._select(catalog, 1, seen)
    assert second[0]["id"] == "1"


def test_no_stock_keys_uses_local_fallback(tmp_path, monkeypatch):
    for key in ("PEXELS_API_KEY", "PIXABAY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    work = tmp_path / "work"
    result = visuals.build_background_info(
        8.0, work, query="city", use_stock=True, script="Une. Deux. Trois."
    )
    assert result.visual_source == "local_fallback"
    assert result.path.exists()
    assert result.message.startswith("Aucune vidéo stock")
    assert abs(_duration(result.path) - 8.0) < 0.2


def test_stock_clips_are_downloaded_and_montaged(tmp_path, monkeypatch, clip_server):
    base, catalog = clip_server
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.setattr(visuals, "_candidates", _fake_candidates(base, catalog))

    work = tmp_path / "work"
    result = visuals.build_background_info(
        20.0, work, query="city", use_stock=True,
        script="La ville brille. Les rues sont vides. Le ciel rougeoie. Tout dort.",
    )
    assert result.visual_source == "pexels"
    assert result.sources == ["pexels"]
    assert result.message == "Vidéos Pexels utilisées"
    assert len(result.clips) >= 3
    assert result.path.exists()
    assert _size(result.path) == (1080, 1920)
    assert abs(_duration(result.path) - 20.0) < 0.2
    # First pass populates the cache; every distinct scene came from the API.
    assert result.cache_stats["stored"] >= 1
    assert result.cache_stats["hits"] == 0
    assert result.scene_origins.count("pexels_api") >= 3
    assert "pexels_cache" not in result.scene_origins


def test_stock_clips_are_served_from_cache_on_second_run(tmp_path, monkeypatch, clip_server):
    base, catalog = clip_server
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)

    # Distinct ids so each scene gets its own entry (and its own cache key).
    def many_candidates(name, query, limit=12):
        return [
            {"id": f"clip{index}", "url": f"{base}/portrait.mp4",
             "duration": 3.0, "width": 1080, "height": 1920}
            for index in range(10)
        ]

    monkeypatch.setattr(visuals, "_candidates", many_candidates)

    script = "La ville brille. Les rues sont vides. Le ciel rougeoie. Tout dort."
    first = visuals.build_background_info(
        20.0, tmp_path / "work1", query="city", use_stock=True, script=script,
    )
    assert first.cache_stats["stored"] >= 1
    assert first.cache_stats["hits"] == 0

    # The provider must not be consulted again for the same scene queries.
    def forbidden(*args, **kwargs):
        raise AssertionError("cache miss: the provider was queried again")

    monkeypatch.setattr(visuals, "_candidates", forbidden)
    second = visuals.build_background_info(
        20.0, tmp_path / "work2", query="city", use_stock=True, script=script,
    )
    assert second.visual_source == "pexels"
    assert second.cache_stats["hits"] >= 1
    assert second.cache_stats["stored"] == 0
    assert all(origin == "pexels_cache" for origin in second.scene_origins)
    assert second.path.exists()
    assert abs(_duration(second.path) - 20.0) < 0.2


def test_provider_failure_falls_back_to_local(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)

    def boom(name, query, limit=12):
        raise RuntimeError("network down")

    monkeypatch.setattr(visuals, "_candidates", boom)
    result = visuals.build_background_info(
        8.0, tmp_path / "work", query="city", use_stock=True, script="Une. Deux."
    )
    assert result.visual_source == "local_fallback"


def test_zoompan_keeps_source_motion(tmp_path):
    """`d` must not swallow the clip's frames: a moving clip and a still image
    must not render identically under the scene filter."""
    from PIL import Image, ImageChops

    def still(path):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc=size=1080x1920:rate=30:duration=1",
             "-frames:v", "1", str(path)],
            check=True,
        )

    moving = tmp_path / "moving.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=1080x1920:rate=30:duration=6",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(moving)],
        check=True,
    )
    frozen = tmp_path / "frozen.png"
    still(frozen)

    filt = visuals._cover_filter()
    rendered = {}
    for name, source, extra in (("moving", moving, ["-stream_loop", "-1"]),
                                ("frozen", frozen, ["-loop", "1"])):
        out = tmp_path / f"out_{name}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *extra, "-i", str(source),
             "-t", "4", "-vf", filt, "-an", "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", str(out)],
            check=True,
        )
        frame = tmp_path / f"frame_{name}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", "2", "-i", str(out),
             "-frames:v", "1", str(frame)],
            check=True,
        )
        rendered[name] = Image.open(frame).convert("RGB")

    diff = ImageChops.difference(rendered["moving"], rendered["frozen"]).convert("L")
    changed = sum(diff.histogram()[21:]) / diff.size[0] / diff.size[1]
    assert changed > 0.05, "scene filter froze the clip: only the first frame survives"


def test_caller_clips_win_over_stock(tmp_path, monkeypatch, clip_server):
    base, catalog = clip_server
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    provided = sorted((tmp_path / "clips").glob("*.mp4"))
    result = visuals.build_background_info(
        10.0, tmp_path / "work", use_stock=True, clips=provided,
        script="Une. Deux. Trois.",
    )
    assert result.visual_source == "ai_clips"
    assert result.message == "Clips IA utilisés"
    assert abs(_duration(result.path) - 10.0) < 0.2
