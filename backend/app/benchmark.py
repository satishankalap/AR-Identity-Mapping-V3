"""
Measures the actual speedup from the v1 -> v2 vector index swap, so "swap to
a real vector index" isn't just a claim in a comment. Run directly:

    python -m app.benchmark
"""
from __future__ import annotations

import math
import random
import time
from typing import List

from .vector_index_v2 import NumpyIndex


class _FakeRecord:
    def __init__(self, profile_id: str):
        self.profile_id = profile_id


def _cosine_py(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _v1_search(entries, query, top_k, min_similarity):
    scored = [(_cosine_py(query, emb), rec) for emb, rec in entries]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [rec for sim, rec in scored[:top_k] if sim >= min_similarity]


def run_benchmark(n_vectors: int = 3000, dim: int = 128, n_queries: int = 20) -> None:
    random.seed(7)
    vectors = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(n_vectors)]
    records = [_FakeRecord(f"p{i}") for i in range(n_vectors)]
    queries = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(n_queries)]

    # --- v1: pure Python loop ---
    entries = list(zip(vectors, records))
    t0 = time.perf_counter()
    for q in queries:
        _v1_search(entries, q, top_k=5, min_similarity=0.0)
    v1_elapsed = time.perf_counter() - t0

    # --- v2: numpy vectorized index ---
    idx = NumpyIndex()
    for vec, rec in zip(vectors, records):
        idx.add(vec, rec)
    t0 = time.perf_counter()
    for q in queries:
        idx.search(q, top_k=5, min_similarity=0.0)
    v2_elapsed = time.perf_counter() - t0

    print(f"n_vectors={n_vectors} dim={dim} n_queries={n_queries}")
    print(f"v1 (python loop):     {v1_elapsed*1000:.1f}ms total, {v1_elapsed/n_queries*1000:.2f}ms/query")
    print(f"v2 (numpy vectorized): {v2_elapsed*1000:.1f}ms total, {v2_elapsed/n_queries*1000:.2f}ms/query")
    print(f"speedup: {v1_elapsed/v2_elapsed:.1f}x")


if __name__ == "__main__":
    run_benchmark()
