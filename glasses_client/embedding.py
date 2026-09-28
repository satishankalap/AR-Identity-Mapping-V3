"""
Embedding extraction — deliberately a stub.

Per the Phase 2 handoff, this codebase never implements the actual face
embedding model; that's the Applied ML lead's pipeline (per the project plan's
pod split), running either on-device or on the paired phone. This module
exists only to define the *interface* Phase 3's pipeline expects, so the
real model can be dropped in without touching state_machine.py, matching_api.py,
or pipeline.py.

Swap `extract()` for a call into the real model (e.g. a MobileFaceNet /
ArcFace-style on-device model exported to Core ML / TFLite) and everything
downstream keeps working unchanged.
"""
from __future__ import annotations

import hashlib
from typing import List


def extract(frame_id: str, dims: int = 128) -> List[float]:
    """
    STUB — returns a deterministic pseudo-embedding derived from frame_id so
    demos and tests are repeatable. This is NOT a real face embedding and
    must never be used for anything beyond wiring/integration testing.
    """
    digest = hashlib.sha256(frame_id.encode()).digest()
    # Expand the digest into `dims` floats in [-1, 1] — just enough structure
    # for cosine similarity to behave sensibly in a demo, nothing more.
    vals = []
    for i in range(dims):
        b = digest[i % len(digest)]
        vals.append((b / 255.0) * 2 - 1)
    return vals
