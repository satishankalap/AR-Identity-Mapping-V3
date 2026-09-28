"""
What's actually verified here, and what isn't:

  VERIFIED:  the LBP histogram math is deterministic, correctly normalized,
             correctly sized, and genuinely sensitive to image content
             (different textures produce different, distinguishable vectors).
             Also: the "no face found" path returns None cleanly.

  NOT VERIFIED: detection accuracy or embedding discriminativeness on real
             human faces. This sandbox has no camera and no network to fetch
             a face dataset, so this can only be exercised against synthetic
             test images. Before this touches a real pilot, run it against
             an actual labeled face dataset and check false-accept/
             false-reject rates at the min_similarity threshold Phase 2 uses.

Run:  python3 -m glasses_client.test_embedding_v2
"""
from __future__ import annotations

import numpy as np

from .embedding_v2 import extract, lbp_histogram, FACE_SIZE, GRID, LBP_BINS


def _checkerboard(size: int, square: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random((size, size)) * 255).astype(np.uint8)
    # stamp a coarse checkerboard on top so cells have real texture structure,
    # not just uniform noise
    for y in range(0, size, square * 2):
        img[y:y + square, :] = np.clip(img[y:y + square, :].astype(int) + 60, 0, 255).astype(np.uint8)
    return img


def _solid(size: int, value: int) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


def test_histogram_length_and_normalization():
    img = _checkerboard(FACE_SIZE, 12, seed=1)
    vec = lbp_histogram(img)
    expected_len = GRID * GRID * LBP_BINS
    assert len(vec) == expected_len, f"expected {expected_len} dims, got {len(vec)}"
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0, f"expected unit norm, got {norm}"
    print(f"PASS histogram_length_and_normalization (dim={len(vec)}, norm={norm:.4f})")


def test_deterministic():
    img = _checkerboard(FACE_SIZE, 12, seed=2)
    v1 = lbp_histogram(img)
    v2 = lbp_histogram(img)
    assert v1 == v2, "same image should produce identical histograms"
    print("PASS deterministic (same image -> identical vector)")


def test_sensitive_to_content():
    img_a = _checkerboard(FACE_SIZE, 8, seed=3)
    img_b = _checkerboard(FACE_SIZE, 20, seed=3)   # different texture scale
    img_c = _solid(FACE_SIZE, 128)                  # flat, no texture at all

    va, vb, vc = lbp_histogram(img_a), lbp_histogram(img_b), lbp_histogram(img_c)

    def cosine(x, y):
        x, y = np.array(x), np.array(y)
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        return float(x @ y / (nx * ny)) if nx and ny else 0.0

    sim_ab = cosine(va, vb)
    sim_ac = cosine(va, vc)
    print(f"  cosine(different textures)  = {sim_ab:.4f}")
    print(f"  cosine(textured vs flat)    = {sim_ac:.4f}")
    assert sim_ab < 0.999, "different textures should not produce near-identical vectors"
    print("PASS sensitive_to_content (distinct textures -> distinguishable vectors)")


def test_no_face_returns_none():
    # A flat gray frame has no face-like Haar features anywhere in it.
    blank = np.full((200, 200, 3), 128, dtype=np.uint8)
    result = extract(blank)
    assert result is None, "a blank frame should yield no detection, not a fabricated vector"
    print("PASS no_face_returns_none (blank frame -> None, no false positive)")


if __name__ == "__main__":
    test_histogram_length_and_normalization()
    test_deterministic()
    test_sensitive_to_content()
    test_no_face_returns_none()
    print("\nAll offline-verifiable checks passed.")
    print("Reminder: detection accuracy on real faces is still unverified — see module docstring.")
