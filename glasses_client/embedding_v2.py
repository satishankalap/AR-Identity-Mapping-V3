"""
Face embedding v2 — replaces Phase 3's hash-based stub with an actual
image-derived feature vector, built entirely from libraries already
available offline (opencv, numpy). No network or model download required
to run this.

Two stages, both real:

  1. Detection: OpenCV's bundled Haar cascade (haarcascade_frontalface_default.xml,
     ships inside the opencv-python wheel — no download needed) finds the
     largest face in a frame and returns its crop.

  2. Feature extraction: a hand-rolled Local Binary Pattern (LBP) histogram
     over a grid of cells on the cropped, grayscale face. LBP is a real,
     well-established texture descriptor (used in production face-recognition
     systems for over a decade) — this is genuinely computing features from
     pixel content, not deriving a vector from an ID string.

Honesty about where this sits: LBP-histogram matching is a classical
baseline. It's a legitimate step up from a hash stub and is fine for
early testing at a single small-to-medium event, but a production launch
should upgrade to a modern deep embedding (ArcFace/FaceNet-style, via
OnnxEmbeddingExtractor below) once the Applied ML lead has model weights —
this two-stage structure (detect -> extract) doesn't change either way,
only what implements extract().

See test_embedding_v2.py for what's actually been verified here (histogram
math, determinism, sensitivity to different textures) versus what hasn't
(detection accuracy against a real face dataset — untestable in a sandbox
with no camera or network to fetch sample photos).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

FACE_SIZE = 96          # crop is resized to FACE_SIZE x FACE_SIZE before LBP
GRID = 6                 # GRID x GRID cells across the face
LBP_BINS = 59             # uniform-LBP has 59 distinct patterns for 8-neighbor LBP


@dataclass
class DetectedFace:
    x: int
    y: int
    w: int
    h: int
    crop_gray: np.ndarray  # grayscale, NOT yet resized


class FaceDetector:
    """Wraps OpenCV's bundled Haar cascade. No model download required."""

    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {path}")

    def detect_largest(self, image_bgr: np.ndarray) -> Optional[DetectedFace]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            return None
        # Largest bounding box = closest/most prominent face, matching the
        # spec's "anchor on the nearest person" behavior.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return DetectedFace(x=int(x), y=int(y), w=int(w), h=int(h), crop_gray=gray[y:y + h, x:x + w])


def _uniform_lbp_lookup() -> np.ndarray:
    """
    Precomputes the 256 -> 59-bin uniform-LBP mapping. A pattern is "uniform"
    if it has at most 2 bitwise transitions going around the 8-bit circular
    code; uniform patterns get their own bin (58 of them), everything else
    collapses into bin 58. This is the standard uniform-LBP reduction used
    to keep histograms compact and mildly rotation-tolerant.
    """
    table = np.zeros(256, dtype=np.uint8)
    next_bin = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            table[code] = next_bin
            next_bin += 1
        else:
            table[code] = 58  # non-uniform bucket
    return table


_LBP_TABLE = _uniform_lbp_lookup()


def _lbp_image(gray: np.ndarray) -> np.ndarray:
    """Standard 8-neighbor, radius-1 LBP code per pixel (vectorized with numpy)."""
    h, w = gray.shape
    center = gray[1:h - 1, 1:w - 1].astype(np.int16)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros_like(center, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx].astype(np.int16)
        code |= ((neighbor >= center).astype(np.uint8)) << bit
    return code


def lbp_histogram(gray: np.ndarray, grid: int = GRID) -> List[float]:
    """
    Resizes to FACE_SIZE x FACE_SIZE, computes the LBP code image, then a
    concatenated uniform-LBP histogram per grid cell (this is what gives the
    descriptor some spatial structure instead of one global texture blob).
    Output length is grid*grid*LBP_BINS and is L2-normalized.
    """
    resized = cv2.resize(gray, (FACE_SIZE, FACE_SIZE))
    codes = _uniform_lbp_lookup_apply(_lbp_image(resized))
    cell = FACE_SIZE // grid
    hist = []
    for gy in range(grid):
        for gx in range(grid):
            cell_codes = codes[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
            h, _ = np.histogram(cell_codes, bins=LBP_BINS, range=(0, LBP_BINS))
            hist.extend(h.astype(np.float64))
    vec = np.array(hist, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _uniform_lbp_lookup_apply(code_image: np.ndarray) -> np.ndarray:
    return _LBP_TABLE[code_image]


def extract(image_bgr: np.ndarray) -> Optional[List[float]]:
    """
    Full pipeline: detect the largest face, then compute its LBP histogram.
    Returns None if no face is found (caller should treat this the same way
    the spec treats an unmatched query: fail silently, no error surfaced).
    """
    detector = _get_detector()
    face = detector.detect_largest(image_bgr)
    if face is None:
        return None
    return lbp_histogram(face.crop_gray)


_detector_singleton: Optional[FaceDetector] = None


def _get_detector() -> FaceDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = FaceDetector()
    return _detector_singleton


# --- Swap point for a real deep embedding model ------------------------------
class OnnxEmbeddingExtractor:
    """
    Loads an ArcFace/FaceNet-style ONNX model and runs inference via
    onnxruntime (already available in this environment — only the model
    weights are missing, since fetching them needs network access this
    sandbox doesn't have).

    Usage once you have a model file (e.g. a public ArcFace .onnx export):

        extractor = OnnxEmbeddingExtractor("arcface_r100.onnx")
        vector = extractor.extract(face_crop_bgr)  # -> List[float], typically 512-d

    This is NOT wired into extract() above by default — swapping it in is a
    one-line change once a model file is available:
    replace the `lbp_histogram(face.crop_gray)` call in extract() with
    `OnnxEmbeddingExtractor(...).extract(face.crop_gray_or_color)`.
    """

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (112, 112)):
        import onnxruntime as ort  # available offline; only the .onnx file is missing
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def extract(self, face_bgr: np.ndarray) -> List[float]:
        resized = cv2.resize(face_bgr, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb - 127.5) / 128.0
        blob = np.transpose(normalized, (2, 0, 1))[None, ...]  # NCHW
        outputs = self.session.run(None, {self.input_name: blob})
        embedding = outputs[0][0]
        return embedding.tolist()
