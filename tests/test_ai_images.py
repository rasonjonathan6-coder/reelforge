"""Tests for the free AI-image background source.

The network is never touched: the image endpoint is replaced by a local HTTP
server serving a real PNG, so the download, the image-to-clip conversion and
the montage are all exercised for real.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

from pipeline import ai_images, visuals


@pytest.fixture()
def image_server(tmp_path):
    """Serve one real PNG for every /prompt/... request."""
    root = tmp_path / "www"
    root.mkdir()
    png = root / "image.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=480x832:rate=1", "-frames:v", "1", str(png)],
        check=True,
    )

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            self.path = "/image.png"
            super().do_GET()

        def log_message(self, *args):  # keep the test output quiet
            pass

    handler = functools.partial(Handler, directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())


def test_fetch_image_saves_bytes(tmp_path, image_server):
    dest = tmp_path / "out.png"
    result = ai_images.fetch_image("a cat", dest, base_url=image_server)
    assert result == dest
    assert dest.exists() and dest.stat().st_size > 0


def test_fetch_image_returns_none_when_unreachable(tmp_path):
    dest = tmp_path / "out.png"
    # Port 1 is never listening: the helper must degrade, not raise.
    assert ai_images.fetch_image("a cat", dest,
                                 base_url="http://127.0.0.1:1", attempts=1) is None


def test_image_to_clip_produces_vertical_video(tmp_path, image_server):
    image = tmp_path / "in.png"
    assert ai_images.fetch_image("a cat", image, base_url=image_server)
    clip = ai_images.image_to_clip(image, tmp_path / "clip.mp4", 3.0)
    assert clip.exists()
    assert abs(_duration(clip) - 3.0) < 0.3


def test_generate_scene_clips_one_per_scene(tmp_path, image_server, monkeypatch):
    monkeypatch.setenv("AI_IMAGE_BASE_URL", image_server)
    texts = ["premier plan", "deuxieme plan", "troisieme plan"]
    clips = ai_images.generate_scene_clips("sujet", texts, tmp_path / "work")
    assert len(clips) == len(texts)
    for clip in clips:
        assert clip.exists() and clip.stat().st_size > 0


def test_build_background_uses_ai_images(tmp_path, image_server, monkeypatch):
    monkeypatch.setenv("AI_IMAGE_BASE_URL", image_server)
    info = visuals.build_background_info(
        8.0, tmp_path / "work", topic="sujet", script="un script",
        visual_source="ai_images",
    )
    assert info.visual_source == "ai_images"
    assert info.path.exists()
    assert abs(_duration(info.path) - 8.0) < 1.0


def test_build_background_falls_back_when_images_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_IMAGE_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("AI_IMAGE_ATTEMPTS", "1")
    info = visuals.build_background_info(
        6.0, tmp_path / "work", topic="sujet", script="un script",
        visual_source="ai_images",
    )
    # No image came back: the local animated fallback must be reported honestly.
    assert info.visual_source == "local_fallback"
    assert info.path.exists()
