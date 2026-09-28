"""
Auto-deletion job.

Per the privacy spec, biometric vectors must expire the moment an event
ends — this isn't a cleanup task bolted on later, it runs from the same
day the matching engine goes live. This demo runs it as an in-process
background thread on a short interval; in production this becomes a
scheduled job (e.g. a Cloud Scheduler / cron-triggered Lambda) hitting the
same purge_event() call, so no logic needs to move, just the trigger.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict

from .vector_store import store

logger = logging.getLogger("expiry")

# event_id -> event_ends_at, tracked so the sweep knows what to check
_watched_events: Dict[str, datetime] = {}
_lock = threading.Lock()


def watch_event(event_id: str, event_ends_at: datetime) -> None:
    with _lock:
        _watched_events[event_id] = event_ends_at


def _sweep_once() -> None:
    now = datetime.utcnow()
    with _lock:
        expired = [eid for eid, ends_at in _watched_events.items() if ends_at <= now]
        for eid in expired:
            _watched_events.pop(eid, None)
    for eid in expired:
        removed = store.purge_event(eid)
        logger.info("purged event=%s vectors_removed=%d reason=event_ended", eid, removed)


def start_background_sweeper(interval_seconds: int = 30) -> threading.Thread:
    """Started once at app startup (see main.py's lifespan hook)."""

    def _loop():
        while True:
            try:
                _sweep_once()
            except Exception:
                logger.exception("expiry sweep failed")
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, daemon=True, name="expiry-sweeper")
    t.start()
    return t


def sweep_now() -> None:
    """Exposed for tests and the /admin/sweep-now debug endpoint."""
    _sweep_once()
