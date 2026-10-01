"""Shared test isolation.

Every test that can touch the stock cache runs against a throwaway directory:
the suite must never read from or write into the repository's real cache.
"""

from __future__ import annotations

import pytest

from pipeline import stock_cache


@pytest.fixture(autouse=True)
def isolated_stock_cache(tmp_path, monkeypatch):
    cache = tmp_path / "stock-cache"
    monkeypatch.setattr(stock_cache, "PEXELS_CACHE_DIR", cache)
    monkeypatch.setattr(stock_cache, "PEXELS_CACHE_ENABLED", True)
    monkeypatch.setattr(stock_cache, "PEXELS_CACHE_TTL_DAYS", 30)
    monkeypatch.setattr(stock_cache, "PEXELS_CACHE_MAX_GB", 5.0)
    stock_cache._locks.clear()
    return cache
