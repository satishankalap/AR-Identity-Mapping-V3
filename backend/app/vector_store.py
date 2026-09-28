"""
Event-scoped vector store — registry of per-event indexes.

v1 did brute-force cosine similarity in a Python loop. v2 (this file) uses
NumpyIndex by default — a real, measured speedup (see benchmark.py) — with
FaissIndex available as a drop-in for when an event outgrows it. Both live
in vector_index_v2.py; this file just decides which one to instantiate.

Everything stays scoped by event_id so one event's attendees are never
matched against another event's database, and dropping an event's index is
how the auto-expiry job (see expiry.py) does its work.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Dict, List

from .models import IdentMethod, MatchResult, ProfileRecord
from .vector_index_v2 import NumpyIndex

# Swap point: change this single assignment to FaissIndex (from
# .vector_index_v2 import FaissIndex) once faiss-cpu is installed and an
# event's scale actually needs it. Nothing else in this file changes.
IndexBackend = NumpyIndex


class VectorStore:
    """Registry of per-event indexes. Thread-safe for the demo's single-process use."""

    def __init__(self):
        self._lock = threading.Lock()
        self._indexes: Dict[str, IndexBackend] = {}

    def _get_or_create(self, event_id: str) -> IndexBackend:
        if event_id not in self._indexes:
            self._indexes[event_id] = IndexBackend()
        return self._indexes[event_id]

    def register(
        self,
        event_id: str,
        external_id: str,
        name: str,
        role: str | None,
        company: str | None,
        ident_method: IdentMethod,
        embedding: List[float] | None,
        event_ends_at: datetime,
    ) -> ProfileRecord:
        if ident_method == IdentMethod.face_embedding and embedding is None:
            raise ValueError("face_embedding registrations require an embedding vector")

        record = ProfileRecord(
            profile_id=str(uuid.uuid4()),
            event_id=event_id,
            external_id=external_id,
            name=name,
            role=role,
            company=company,
            ident_method=ident_method,
            created_at=datetime.utcnow(),
            event_ends_at=event_ends_at,
        )
        with self._lock:
            if embedding is not None:
                self._get_or_create(event_id).add(embedding, record)
            else:
                # Marker/BLE profiles carry no vector — they're looked up by
                # external_id on the client side, not matched here.
                self._get_or_create(event_id)
        return record

    def match(self, event_id: str, embedding: List[float], top_k: int, min_similarity: float) -> List[MatchResult]:
        index = self._indexes.get(event_id)
        if index is None:
            return []
        with self._lock:
            hits = index.search(embedding, top_k, min_similarity)
        return [MatchResult(profile=record, similarity=sim) for record, sim in hits]

    def purge_event(self, event_id: str) -> int:
        """Called by the auto-expiry job. Drops every embedding for the event."""
        with self._lock:
            index = self._indexes.pop(event_id, None)
        return len(index) if index else 0

    def event_size(self, event_id: str) -> int:
        index = self._indexes.get(event_id)
        return len(index) if index else 0


store = VectorStore()
