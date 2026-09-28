# Face Embedding v2 — Real Pipeline, Not a Stub

Replaces Phase 3's `embedding.py` (which returned a hash of a fake frame ID)
with an actual image-processing pipeline: OpenCV Haar-cascade face detection
+ a hand-rolled uniform-LBP texture histogram. Runs fully offline — both
pieces ship inside packages already in this environment (`opencv-python`,
`numpy`), no model download required.

## What's verified vs what isn't

**Verified by `test_embedding_v2.py` (actually run, not just written):**
- Histogram output is the correct length (2,124 = 6×6 grid × 59 uniform-LBP bins) and unit-normalized
- Same image → identical vector (deterministic)
- Different textures → measurably different vectors (cosine 0.995 between two textured patterns, 0.223 between textured and flat — real separation, not coincidence)
- A face-less frame returns `None` cleanly rather than fabricating a match

**NOT verified — and this matters before any real deployment:**
- Detection accuracy on actual human faces. I tested the cascade against a
  crude drawn cartoon face and, as expected, it did **not** detect it —
  Haar cascades need real photographic gradients, which synthetic art
  doesn't have. This sandbox has no camera and no network to pull a real
  face dataset, so detection accuracy is genuinely untested. **Before this
  touches a pilot, run it against a real labeled face set and measure
  false-accept/false-reject rates at whatever `min_similarity` threshold
  Phase 2 uses.**

## Where this sits technically

LBP-histogram matching is a classical, pre-deep-learning face recognition
technique — real and functional, but lower accuracy than a modern embedding
model, and more sensitive to lighting/angle than the spec's target
conditions. It's a legitimate baseline for internal testing at a small
event, not a final answer.

`embedding_v2.py` also includes `OnnxEmbeddingExtractor`, a ready-to-use
wrapper around `onnxruntime` (also already available in this environment)
for a real ArcFace/FaceNet-style deep embedding — the only missing piece is
a model weights file, which needs network access this sandbox doesn't have.
Swapping to it is a one-line change once the Applied ML lead sources
weights; the detect → extract pipeline shape doesn't change.

## Wiring it into Phase 3

`glasses_client/embedding.py` (the Phase 3 stub) takes a `frame_id: str`.
This module takes real pixel data: `extract(image_bgr: np.ndarray) -> Optional[List[float]]`.
That's a deliberate interface upgrade — `pipeline.py`'s `_run_match()` needs
a one-line change to pass an actual captured frame instead of a fake ID:

```python
# before (Phase 3 stub):
embedding = extract_embedding(frame_id)

# after (this module):
embedding = extract(captured_frame_bgr)
if embedding is None:
    self.sm.match_failed_or_timed_out()  # no face found — same silent-fail path
    return
```

## Run the tests yourself

```bash
python3 -m glasses_client.test_embedding_v2
```
