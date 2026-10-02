"""Persistent local cache for stock-footage clips (Pexels/Pixabay).

Why: a reel asks the provider for one clip per scene, so a batch of similar
topics re-searches and re-downloads the same shots. This module keeps the
already-downloaded, already-validated files and hands them back on a repeat
request, cutting provider calls and bandwidth.

Design constraints kept deliberately simple:
  * standard library only (hashlib, json, os, tempfile, threading, time);
  * one small JSON sidecar per clip, named after the cache key;
  * the clip itself is validated with ffprobe *before* it is published;
  * writes are atomic (tmp file + os.replace), so an interrupted download can
    never be mistaken for a valid entry;
  * a per-key lock keeps two concurrent jobs from downloading the same clip.

The cache stores no credentials: only the provider URL and video metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from config import (
    PEXELS_CACHE_DIR,
    PEXELS_CACHE_ENABLED,
    PEXELS_CACHE_MAX_GB,
    PEXELS_CACHE_TTL_DAYS,
)

META_SUFFIX = ".json"
CLIP_SUFFIX = ".mp4"
TMP_SUFFIX = ".tmp"
_LOCK_SUFFIX = ".lock"

# Cache status of one scene's clip, reported back to the caller.
HIT = "hit"
MISS = "miss"
EXPIRED = "expired"
INVALID = "invalid"
DISABLED = "disabled"
ERROR = "error"


def enabled() -> bool:
    return bool(PEXELS_CACHE_ENABLED)


def cache_dir() -> Path:
    return Path(PEXELS_CACHE_DIR)


def cache_key(provider: str, query: str, orientation: str) -> str:
    """Stable SHA-256 over a canonical request.

    The raw query never becomes a filename: only the digest does.
    """
    canonical = json.dumps(
        {"provider": provider, "query": query, "orientation": orientation},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _paths(key: str) -> tuple[Path, Path]:
    folder = cache_dir()
    return folder / f"{key}{CLIP_SUFFIX}", folder / f"{key}{META_SUFFIX}"


@dataclass
class CacheStats:
    """Per-job cache counters, merged into the job metadata."""

    hits: int = 0
    misses: int = 0
    expired: int = 0
    invalid: int = 0
    disabled: int = 0
    errors: int = 0
    stored: int = 0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "expired": self.expired,
            "invalid": self.invalid,
            "disabled": self.disabled,
            "errors": self.errors,
            "stored": self.stored,
        }

    def merge(self, other: "CacheStats") -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.expired += other.expired
        self.invalid += other.invalid
        self.disabled += other.disabled
        self.errors += other.errors
        self.stored += other.stored


# One lock per cache key; the same key must never be downloaded twice at once.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


@contextmanager
def key_lock(key: str):
    """Serialize work on one cache key across threads of this process."""
    lock = _lock_for(key)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _read_meta(meta_path: Path) -> dict | None:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - unreadable sidecar means no entry
        return None


def _is_expired(meta: dict, ttl_days: int) -> bool:
    if ttl_days <= 0:
        return False
    try:
        created = float(meta.get("created_at"))
    except (TypeError, ValueError):
        return True
    return (time.time() - created) > ttl_days * 86400


def _log(action: str, key: str, **extra) -> None:
    """One-line, secret-free log: only the digest and non-sensitive fields."""
    tail = "".join(f" {name}={value}" for name, value in extra.items())
    print(f"[PexelsCache] {action} query_hash={key[:12]}{tail}")


def _lookup(key: str, validate, ttl_days: int) -> tuple[Path | None, str, dict | None]:
    """Inspect an entry: ('hit'|'expired'|'invalid'|'miss', meta)."""
    clip_path, meta_path = _paths(key)
    if not clip_path.exists() or not meta_path.exists():
        return None, MISS, None

    meta = _read_meta(meta_path)
    if meta is None:
        return None, INVALID, None
    if _is_expired(meta, ttl_days):
        return None, EXPIRED, meta

    info = validate(clip_path)
    if not info:
        return None, INVALID, meta
    return clip_path, HIT, meta


def _touch(meta_path: Path, meta: dict) -> None:
    meta["last_accessed_at"] = time.time()
    try:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 - access time is best-effort
        pass


def _publish(tmp_clip: Path, clip_path: Path, meta_path: Path, meta: dict) -> None:
    """Atomically move the validated clip in place, then write its sidecar."""
    os.replace(tmp_clip, clip_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def get(
    provider: str,
    query: str,
    orientation: str,
    fetch,
    validate,
    ttl_days: int | None = None,
    stats: CacheStats | None = None,
) -> tuple[Path | None, str, dict | None]:
    """Return a usable cached clip for this request, or fetch and store one.

    `fetch()` downloads the clip to a temp file and returns `(tmp_path, meta)`
    or `(None, None)` on failure. `validate(path)` returns a metadata dict (with
    at least width/height/duration) for a usable video, else None.

    Returns `(clip_path, status, meta)` where status is one of hit/miss/expired/
    invalid/disabled/error and meta carries the provider URL/id when known. A
    corrupt entry is deleted and re-fetched rather than failing the job.
    """
    ttl = PEXELS_CACHE_TTL_DAYS if ttl_days is None else ttl_days
    if not enabled():
        # Cache off: the caller runs its normal (uncached) fetch path.
        if stats is not None:
            stats.disabled += 1
        return None, DISABLED, None

    key = cache_key(provider, query, orientation)

    with key_lock(key):
        clip_path, status, meta = _lookup(key, validate, ttl)
        if status == HIT:
            if stats is not None:
                stats.hits += 1
            _touch(_paths(key)[1], meta or {})
            _log("HIT", key)
            return clip_path, HIT, meta

        if status in (EXPIRED, INVALID):
            if stats is not None:
                if status == EXPIRED:
                    stats.expired += 1
                else:
                    stats.invalid += 1
            _log("EXPIRED" if status == EXPIRED else "INVALID", key)
            _remove_entry(key)

        if stats is not None:
            stats.misses += 1

        try:
            tmp_clip, fresh_meta = fetch()
        except Exception as exc:  # noqa: BLE001 - caller falls back to next provider
            if stats is not None:
                stats.errors += 1
            _log("ERROR", key, reason=type(exc).__name__)
            return None, ERROR, None

        if not tmp_clip or not Path(tmp_clip).exists():
            _log("MISS", key)
            return None, MISS, None

        info = validate(tmp_clip)
        if not info:
            Path(tmp_clip).unlink(missing_ok=True)
            _log("INVALID", key, stage="download")
            return None, INVALID, None

        folder = cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        final_clip, final_meta = _paths(key)
        entry = {
            "cache_key": key,
            "provider": provider,
            "query": query,
            "orientation": orientation,
            "created_at": time.time(),
            "last_accessed_at": time.time(),
            "duration": info.get("duration"),
            "width": info.get("width"),
            "height": info.get("height"),
            "video_codec": info.get("video_codec"),
            "file_size": Path(tmp_clip).stat().st_size,
        }
        entry.update(fresh_meta or {})
        try:
            _publish(Path(tmp_clip), final_clip, final_meta, entry)
        except Exception as exc:  # noqa: BLE001 - a cache write must never be fatal
            Path(tmp_clip).unlink(missing_ok=True)
            if stats is not None:
                stats.errors += 1
            _log("ERROR", key, reason=type(exc).__name__)
            return None, ERROR, None

        if stats is not None:
            stats.stored += 1
        _log("STORE", key, provider=provider, orientation=orientation)
        return final_clip, MISS, entry


def _remove_entry(key: str) -> None:
    clip_path, meta_path = _paths(key)
    clip_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)


def cleanup(ttl_days: int | None = None, max_gb: float | None = None) -> dict:
    """Maintenance: drop expired/invalid entries, stray temp files, then cap size.

    Never runs automatically on every request; call it from the startup hook or
    by hand. Entries currently held by a lock are skipped so a running job is
    never pulled from under itself.
    """
    ttl = PEXELS_CACHE_TTL_DAYS if ttl_days is None else ttl_days
    limit = PEXELS_CACHE_MAX_GB if max_gb is None else max_gb
    folder = cache_dir()
    removed = {"expired": 0, "invalid": 0, "temp": 0, "oversize": 0}

    if not folder.exists():
        return removed

    entries: list[tuple[float, int, str, Path]] = []
    for meta_path in folder.glob(f"*{META_SUFFIX}"):
        key = meta_path.stem
        clip_path, _ = _paths(key)
        if not clip_path.exists():
            meta_path.unlink(missing_ok=True)
            removed["invalid"] += 1
            continue
        meta = _read_meta(meta_path)
        if meta is None or _is_expired(meta, ttl):
            _remove_entry(key)
            removed["expired"] += 1
            continue
        entries.append((float(meta.get("last_accessed_at") or 0), clip_path.stat().st_size, key, clip_path))

    # Stray temp files from an interrupted download.
    for stray in folder.glob(f"*{TMP_SUFFIX}"):
        stray.unlink(missing_ok=True)
        removed["temp"] += 1

    if limit and limit > 0:
        budget = int(limit * 1024 ** 3)
        total = sum(size for _, size, _, _ in entries)
        for _, size, key, clip_path in sorted(entries):  # least recently used first
            if total <= budget:
                break
            with _locks_guard:
                lock = _locks.get(key)
            if lock is not None and lock.locked():
                continue  # never delete an entry a job is using right now
            _remove_entry(key)
            total -= size
            removed["oversize"] += 1

    return removed


def stats_snapshot() -> dict:
    """Lightweight size/entry counts, for diagnostics and the API."""
    folder = cache_dir()
    entries = 0
    total = 0
    if folder.exists():
        for meta_path in folder.glob(f"*{META_SUFFIX}"):
            clip_path, _ = _paths(meta_path.stem)
            if clip_path.exists():
                entries += 1
                total += clip_path.stat().st_size
    return {"entries": entries, "bytes": total, "path": str(folder), "enabled": enabled()}


def temp_path(key: str) -> Path:
    """A unique, same-filesystem temp path for one download attempt."""
    folder = cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f"{key[:12]}-", suffix=TMP_SUFFIX, dir=str(folder))
    os.close(handle)
    return Path(name)
