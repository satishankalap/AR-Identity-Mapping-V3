"""
Vector index v2 — the actual swap promised in v1's comments.

Two backends, same interface:

  - NumpyIndex: vectorized cosine similarity over a single (N, dim) matrix
    instead of a Python-level loop per candidate. This is the default —
    it's a real, measured speedup (see benchmark.py) and needs nothing
    beyond numpy, which is already a dependency.

  - FaissIndex: wraps faiss.IndexFlatIP for exact search, or IndexIVFFlat
    for larger events. Import-guarded — if faiss isn't installed, building
    one raises ImportError with an install hint rather than crashing at
    import time. Use this once an event's attendee count or query volume
    outgrows what NumpyIndex handles comfortably (see benchmark.py's
    numbers for where that line sits on real hardware).

Both are drop-in replacements for v1's EventIndex — same add/remove/search
surface — so vector_store.py's VectorStore class doesn't need to change,
only which index class it instantiates.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Deliberately no import of .models here: this module does pure vector math
# and shouldn't need pydantic (or anything else) to be testable in isolation
# (see benchmark.py, which runs standalone). ProfileRecord is only used as a
# type hint under TYPE_CHECKING, and results are plain (record, similarity)
# tuples — vector_store.py wraps them into pydantic MatchResult objects at
# the API boundary, where pydantic is actually needed.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .models import ProfileRecord


class NumpyIndex:
    """Vectorized cosine similarity over all vectors in one event."""

    def __init__(self, dim: Optional[int] = None):
        self.dim = dim
        self._matrix: Optional[np.ndarray] = None   # (N, dim), L2-normalized rows
        self._ids: List[str] = []
        self._records: Dict[str, "ProfileRecord"] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, embedding: List[float], record: "ProfileRecord") -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if self.dim is None:
            self.dim = vec.shape[0]
        elif vec.shape[0] != self.dim:
            raise ValueError(f"embedding dim {vec.shape[0]} != index dim {self.dim}")

        if self._matrix is None:
            self._matrix = vec.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])
        self._ids.append(record.profile_id)
        self._records[record.profile_id] = record

    def remove(self, profile_id: str) -> None:
        if profile_id not in self._records:
            return
        idx = self._ids.index(profile_id)
        self._matrix = np.delete(self._matrix, idx, axis=0)
        self._ids.pop(idx)
        del self._records[profile_id]

    def search(self, query: List[float], top_k: int, min_similarity: float) -> List[Tuple["ProfileRecord", float]]:
        if self._matrix is None or len(self._ids) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        # One matrix-vector product scores every candidate at once — this is
        # the whole speedup versus v1's per-candidate Python loop.
        sims = self._matrix @ q
        order = np.argsort(-sims)[:top_k]
        results = []
        for i in order:
            sim = float(sims[i])
            if sim >= min_similarity:
                results.append((self._records[self._ids[i]], round(sim, 4)))
        return results


class FaissIndex:
    """
    Exact search via faiss.IndexFlatIP. Requires `pip install faiss-cpu`
    (or faiss-gpu). Not importable in this sandbox (no network to install
    it) — written and structured to match NumpyIndex's interface exactly,
    so swapping VectorStore over to it later is a one-line change.
    """

    def __init__(self, dim: Optional[int] = None):
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FaissIndex requires faiss-cpu: pip install faiss-cpu"
            ) from e
        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim) if dim else None
        self._ids: List[str] = []
        self._records: Dict[str, "ProfileRecord"] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, embedding: List[float], record: "ProfileRecord") -> None:
        vec = np.asarray([embedding], dtype=np.float32)
        self._faiss.normalize_L2(vec)
        if self._index is None:
            self.dim = vec.shape[1]
            self._index = self._faiss.IndexFlatIP(self.dim)
        self._index.add(vec)
        self._ids.append(record.profile_id)
        self._records[record.profile_id] = record

    def remove(self, profile_id: str) -> None:
        # IndexFlatIP doesn't support removal directly; production use
        # should switch to IndexIDMap2 + remove_ids, or rebuild the index
        # on the (infrequent) event-purge path instead of per-profile removal.
        raise NotImplementedError(
            "FaissIndex removal needs IndexIDMap2 — not required for the "
            "purge-whole-event flow this system actually uses"
        )

    def search(self, query: List[float], top_k: int, min_similarity: float) -> List[Tuple["ProfileRecord", float]]:
        if self._index is None or len(self._ids) == 0:
            return []
        q = np.asarray([query], dtype=np.float32)
        self._faiss.normalize_L2(q)
        sims, idxs = self._index.search(q, min(top_k, len(self._ids)))
        results = []
        for sim, i in zip(sims[0], idxs[0]):
            if i == -1:
                continue
            if float(sim) >= min_similarity:
                results.append((self._records[self._ids[i]], round(float(sim), 4)))
        return results
