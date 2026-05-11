"""
Disk-based cache for query result components.

Each mode (hierarchical, long_context) is cached independently, keyed by
(doc_id, query, mode_component). This means a `both`-mode query can reuse
previously-cached hierarchical or long_context results — we only re-run
the components we don't already have.

Cached entries are JSON files under data/query_cache/. TTL is 24h.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.config import settings

CACHE_TTL_HOURS = 24

_cache_dir: Optional[Path] = None


def _get_cache_dir() -> Path:
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = settings.data_dir / "query_cache"
        _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir


def _component_key(doc_id: str, query: str, component: str) -> str:
    """
    Hash inputs into a cache key.

    `component` is one of: "hierarchical", "long_context".
    Each mode's result is stored under its own key so they can be reused
    independently across `hierarchical`, `long_context`, and `both` requests.
    """
    raw = f"{doc_id}::{query.strip().lower()}::{component}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_component(doc_id: str, query: str, component: str) -> Optional[dict]:
    """
    Return a cached sub-result for one component, or None on miss/expired.
    """
    key = _component_key(doc_id, query, component)
    cache_file = _get_cache_dir() / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        cache_file.unlink(missing_ok=True)
        return None

    cached_at_str = data.get("__cached_at")
    if not cached_at_str:
        return None

    try:
        cached_at = datetime.fromisoformat(cached_at_str)
    except ValueError:
        return None

    if datetime.now() - cached_at > timedelta(hours=CACHE_TTL_HOURS):
        cache_file.unlink(missing_ok=True)
        return None

    data.pop("__cached_at", None)
    return data


def set_component(doc_id: str, query: str, component: str, payload: dict) -> None:
    """Store one component's sub-result."""
    key = _component_key(doc_id, query, component)
    cache_file = _get_cache_dir() / f"{key}.json"

    to_write = dict(payload)
    to_write["__cached_at"] = datetime.now().isoformat()
    cache_file.write_text(json.dumps(to_write, indent=2), encoding="utf-8")


def clear_cache() -> int:
    """Delete all cache entries. Returns count of files removed."""
    cache_dir = _get_cache_dir()
    count = 0
    for f in cache_dir.glob("*.json"):
        f.unlink(missing_ok=True)
        count += 1
    return count