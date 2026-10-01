"""Tests for the persistent stock-clip cache.

The cache is exercised through its real code paths: real files on disk, real
ffprobe validation and real threading. Only the network download is replaced by
a local file copy, which is exactly the seam the cache exposes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from pipeline import stock_cache, visuals


@pytest.fixture(scope="module")
def clip_template(tmp_path_factory):
    """A real 2s mp4, copied by the fake fetchers so ffprobe is genuinely run."""
    path = tmp_path_factory.mktemp("clip") / "template.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x240:rate=15:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _fetch_from(template: Path, counter: dict | None = None):
    """Build a `fetch` callable that copies the template to the temp path."""

    def fetch():
        if counter is not None:
            counter["calls"] = counter.get("calls", 0) + 1
        dest = stock_cache.temp_path("fetch")
        shutil.copyfile(template, dest)
        return dest, {"source_url": "http://example.test/clip.mp4", "provider_id": "42"}

    return fetch


def test_cache_key_is_stable_and_query_specific():
    first = stock_cache.cache_key("pexels", "ville la nuit", "portrait")
    assert first == stock_cache.cache_key("pexels", "ville la nuit", "portrait")
    assert len(first) == 64
    assert first != stock_cache.cache_key("pexels", "ville le jour", "portrait")
    assert first != stock_cache.cache_key("pixabay", "ville la nuit", "portrait")
    assert first != stock_cache.cache_key("pexels", "ville la nuit", "landscape")


def test_first_request_misses_then_hits(clip_template):
    stats = stock_cache.CacheStats()
    fetch = _fetch_from(clip_template)

    path, status, meta = stock_cache.get(
        "pexels", "rue déserte", "portrait", fetch, visuals._validate_clip, stats=stats
    )
    assert status == stock_cache.MISS
    assert path is not None and path.exists()
    assert meta["provider_id"] == "42"
    assert stats.misses == 1 and stats.stored == 1 and stats.hits == 0

    again, status, _ = stock_cache.get(
        "pexels", "rue déserte", "portrait", fetch, visuals._validate_clip, stats=stats
    )
    assert status == stock_cache.HIT
    assert again == path
    assert stats.hits == 1 and stats.misses == 1


def test_hit_does_not_call_the_fetcher(clip_template):
    calls = {"calls": 0}
    fetch = _fetch_from(clip_template, calls)
    stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip)
    stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip)
    stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip)
    assert calls["calls"] == 1


def test_expired_entry_is_refetched(clip_template):
    calls = {"calls": 0}
    fetch = _fetch_from(clip_template, calls)
    stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip, ttl_days=30)

    stats = stock_cache.CacheStats()
    path, status, _ = stock_cache.get(
        "pexels", "q", "portrait", fetch, visuals._validate_clip,
        ttl_days=1e-9, stats=stats,
    )
    assert status == stock_cache.MISS
    assert stats.expired == 1
    assert calls["calls"] == 2
    assert path is not None and path.exists()


def test_corrupt_entry_is_detected_and_replaced(clip_template):
    fetch = _fetch_from(clip_template)
    path, _, _ = stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip)

    path.write_bytes(b"not a video at all")

    stats = stock_cache.CacheStats()
    refreshed, status, _ = stock_cache.get(
        "pexels", "q", "portrait", fetch, visuals._validate_clip, stats=stats
    )
    assert status == stock_cache.MISS
    assert stats.invalid == 1
    assert refreshed is not None
    assert refreshed.read_bytes() != b"not a video at all"
    assert visuals._validate_clip(refreshed) is not None


def test_disabled_cache_never_stores(clip_template, monkeypatch):
    monkeypatch.setattr(stock_cache, "PEXELS_CACHE_ENABLED", False)
    calls = {"calls": 0}
    fetch = _fetch_from(clip_template, calls)

    stats = stock_cache.CacheStats()
    path, status, _ = stock_cache.get(
        "pexels", "q", "portrait", fetch, visuals._validate_clip, stats=stats
    )
    assert status == stock_cache.DISABLED
    assert path is None
    assert calls["calls"] == 0
    assert stats.disabled == 1
    assert list(stock_cache.cache_dir().glob("*.json")) == []


def test_fetch_failure_reports_miss_and_leaves_no_entry():
    def fetch():
        return None, None

    path, status, _ = stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip)
    assert status == stock_cache.MISS
    assert path is None
    assert list(stock_cache.cache_dir().glob("*.mp4")) == []


def test_fetch_exception_reports_error_and_leaves_no_entry():
    def fetch():
        raise RuntimeError("network down")

    stats = stock_cache.CacheStats()
    path, status, _ = stock_cache.get(
        "pexels", "q", "portrait", fetch, visuals._validate_clip, stats=stats
    )
    assert status == stock_cache.ERROR
    assert path is None
    assert stats.errors == 1
    assert list(stock_cache.cache_dir().glob("*.mp4")) == []


def test_concurrent_requests_for_same_key_download_once(clip_template):
    calls = {"calls": 0}
    lock = threading.Lock()

    def fetch():
        with lock:
            calls["calls"] += 1
        time.sleep(0.2)  # widen the window a duplicate download could slip into
        dest = stock_cache.temp_path("fetch")
        shutil.copyfile(clip_template, dest)
        return dest, {"provider_id": "42"}

    results: list[str] = []

    def worker():
        _, status, _ = stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip)
        results.append(status)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls["calls"] == 1, "the same key must be downloaded exactly once"
    assert results.count(stock_cache.MISS) == 1
    assert results.count(stock_cache.HIT) == 4


def test_no_temp_files_survive_a_successful_store(clip_template):
    fetch = _fetch_from(clip_template)
    stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip)
    assert list(stock_cache.cache_dir().glob("*.tmp")) == []


def test_cleanup_removes_expired_entries(clip_template):
    fetch = _fetch_from(clip_template)
    path, _, _ = stock_cache.get("pexels", "q", "portrait", fetch, visuals._validate_clip)
    meta_path = stock_cache._paths(stock_cache.cache_key("pexels", "q", "portrait"))[1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created_at"] = time.time() - 90 * 86400
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    removed = stock_cache.cleanup(ttl_days=30)
    assert removed["expired"] == 1
    assert not path.exists()


def test_cleanup_caps_size_evicting_least_recently_used(clip_template):
    old_fetch = _fetch_from(clip_template)
    old_path, _, _ = stock_cache.get("pexels", "old", "portrait", old_fetch, visuals._validate_clip)
    new_path, _, _ = stock_cache.get("pexels", "new", "portrait", old_fetch, visuals._validate_clip)

    old_key = stock_cache.cache_key("pexels", "old", "portrait")
    old_meta = stock_cache._paths(old_key)[1]
    payload = json.loads(old_meta.read_text(encoding="utf-8"))
    payload["last_accessed_at"] = time.time() - 3600
    old_meta.write_text(json.dumps(payload), encoding="utf-8")

    size = old_path.stat().st_size
    removed = stock_cache.cleanup(ttl_days=30, max_gb=(size * 1.5) / 1024 ** 3)

    assert removed["oversize"] == 1
    assert not old_path.exists(), "the least recently used entry must go first"
    assert new_path.exists()


def test_stats_snapshot_counts_entries_and_bytes(clip_template):
    fetch = _fetch_from(clip_template)
    stock_cache.get("pexels", "a", "portrait", fetch, visuals._validate_clip)
    stock_cache.get("pexels", "b", "portrait", fetch, visuals._validate_clip)

    snapshot = stock_cache.stats_snapshot()
    assert snapshot["entries"] == 2
    assert snapshot["bytes"] > 0
    assert snapshot["enabled"] is True
    assert Path(snapshot["path"]) == stock_cache.cache_dir()
