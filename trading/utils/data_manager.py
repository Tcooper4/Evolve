"""
Centralized data lifecycle management for Evolve.
Handles caching strategy, TTLs, cleanup, and memory budgets.
"""

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Cache Configuration ──────────────────────────────────────────────────────
# (ttl_seconds, max_entries)
CACHE_CONFIG = {
    "price_realtime": (60, 500),  # 1 min — live prices
    "price_historical": (300, 200),  # 5 min — OHLCV history
    "news": (600, 100),  # 10 min — news articles
    "earnings": (3600, 500),  # 1 hour — earnings dates
    "short_interest": (3600, 500),  # 1 hour — SI data
    "insider_flow": (3600, 200),  # 1 hour — insider data
    "forecast": (300, 100),  # 5 min — model forecasts
    "market_pulse": (300, 1),  # 5 min — home page metrics
    "top_movers": (600, 1),  # 10 min — movers list
}


# In-memory cache with TTL
_cache: dict = {}
_cache_lock = threading.Lock()


def cache_get(key: str) -> Optional[Any]:
    """Get a value from the in-memory cache if not expired."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() < entry["expires"]:
            return entry["value"]
        if entry:
            # expired
            del _cache[key]
        return None


def cache_set(key: str, value: Any, ttl: int = 300) -> None:
    """Set a value in the in-memory cache with TTL."""
    with _cache_lock:
        # Enforce max cache size (evict oldest if over 1000 entries)
        if len(_cache) > 1000:
            oldest = sorted(_cache.items(), key=lambda x: x[1]["expires"])[:100]
            for k, _ in oldest:
                _cache.pop(k, None)
        _cache[key] = {"value": value, "expires": time.time() + ttl}


def cache_clear(prefix: str = "") -> int:
    """Clear cache entries matching prefix. Returns count cleared."""
    with _cache_lock:
        keys = [k for k in list(_cache.keys()) if k.startswith(prefix)]
        for k in keys:
            _cache.pop(k, None)
        return len(keys)


def cache_stats() -> dict:
    """Return simple stats on in-memory cache usage."""
    with _cache_lock:
        now = time.time()
        active = sum(1 for v in _cache.values() if now < v["expires"])
        expired = len(_cache) - active
        return {"total": len(_cache), "active": active, "expired": expired}


# ── Disk cache (for larger, slower-changing data) ────────────────────────────
DISK_CACHE_DIR = Path("data/cache")
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _disk_cache_path(key: str) -> Path:
    return DISK_CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.json"


def disk_cache_get(key: str) -> Optional[Any]:
    """Retrieve a value from disk cache if not expired."""
    path = _disk_cache_path(key)
    try:
        if path.exists():
            with path.open() as f:
                entry = json.load(f)
            if time.time() < entry.get("expires", 0):
                return entry.get("value")
            # expired
            path.unlink(missing_ok=True)
    except Exception:
        # Corrupted or unreadable; best effort only
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    return None


def disk_cache_set(key: str, value: Any, ttl: int = 3600) -> None:
    """Persist a value to disk cache with TTL in seconds."""
    path = _disk_cache_path(key)
    try:
        payload = {
            "value": value,
            "expires": time.time() + ttl,
            "key_hint": key[:50],
        }
        with path.open("w") as f:
            json.dump(payload, f, default=str)
    except Exception as e:
        logger.debug("Disk cache write failed for %s: %s", key, e)


# ── Log rotation ─────────────────────────────────────────────────────────────
def rotate_logs(
    log_dir: str = "logs", max_size_mb: int = 10, keep_files: int = 3
) -> dict:
    """Rotate log files larger than max_size_mb, keeping keep_files backups."""
    rotated: list = []
    log_path = Path(log_dir)
    if not log_path.exists():
        return {"rotated": 0, "freed_bytes": 0}

    freed = 0
    for log_file in log_path.glob("*.log"):
        size = log_file.stat().st_size
        if size > max_size_mb * 1024 * 1024:
            # Rotate: rename to .log.1, delete old backups
            for i in range(keep_files, 0, -1):
                old = log_file.with_suffix(f".log.{i}")
                if old.exists():
                    if i == keep_files:
                        freed += old.stat().st_size
                        old.unlink()
                    else:
                        old.rename(log_file.with_suffix(f".log.{i+1}"))
            log_file.rename(log_file.with_suffix(".log.1"))
            log_file.touch()  # create fresh empty log
            rotated.append(str(log_file))

    return {"rotated": len(rotated), "freed_bytes": freed, "files": rotated}


# ── Disk cache cleanup ───────────────────────────────────────────────────────
def cleanup_disk_cache(max_age_hours: int = 24) -> dict:
    """Remove expired disk cache files by modification time."""
    cutoff = time.time() - max_age_hours * 3600
    removed, freed = 0, 0
    for path in DISK_CACHE_DIR.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                freed += path.stat().st_size
                path.unlink()
                removed += 1
        except Exception:
            continue
    return {"removed": removed, "freed_bytes": freed}


# ── yfinance data budget ─────────────────────────────────────────────────────
_yf_request_times: list = []
_yf_lock = threading.Lock()


def yf_rate_limit_check(max_per_minute: int = 60) -> bool:
    """Returns True if safe to make a yfinance request."""
    now = time.time()
    with _yf_lock:
        _yf_request_times[:] = [t for t in _yf_request_times if now - t < 60]
        if len(_yf_request_times) >= max_per_minute:
            return False
        _yf_request_times.append(now)
        return True


def get_yf_request_count() -> int:
    """Return number of yfinance requests in the last 60 seconds."""
    now = time.time()
    with _yf_lock:
        return sum(1 for t in _yf_request_times if now - t < 60)


