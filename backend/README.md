# Phase 2 — Matching Engine

This is the backend from the [project plan](/) Phase 2 milestone: an event-scoped
vector database with an opt-in face-embedding path, built so Phase 3 (hardware
port) has a real API to call instead of another mock.

## What's here

| File | Role |
|---|---|
| `app/models.py` | Request/response schemas. `IdentMethod` keeps marker, BLE, and face-embedding identification on one pipeline. Consent is a required field, not a comment. |
| `app/vector_store.py` | Event-scoped registry of per-event indexes. Delegates the actual search math to `vector_index_v2.py`. |
| `app/vector_index_v2.py` | `NumpyIndex` (default, vectorized cosine similarity) and `FaissIndex` (drop-in, needs `faiss-cpu`). See benchmark below. |
| `app/benchmark.py` | Measures v1-loop vs v2-numpy speed directly — run it yourself with `python3 -m app.benchmark`. |
| `app/expiry.py` | Background sweeper that purges an event's vectors the moment it ends. Runs from day one, not bolted on later. |
| `app/main.py` | FastAPI app wiring it together. |

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then, e.g.:

```bash
# Register a face-embedding profile (embedding would come from the client's
# own face-embedding model — this service never extracts embeddings itself)
curl -X POST localhost:8000/profiles -H "Content-Type: application/json" -d '{
  "event_id": "arxr2026",
  "external_id": "attendee-042",
  "name": "Sarah Chen",
  "role": "Lead Spatial Architect",
  "company": "Acme Inc.",
  "ident_method": "face",
  "embedding": [0.12, 0.87, 0.03, 0.41, 0.09, 0.55, 0.02, 0.18],
  "consent_given": true,
  "event_ends_at": "2026-12-01T20:00:00"
}'

# Query a match
curl -X POST localhost:8000/match -H "Content-Type: application/json" -d '{
  "event_id": "arxr2026",
  "embedding": [0.11, 0.85, 0.04, 0.40, 0.10, 0.54, 0.01, 0.19],
  "top_k": 1,
  "min_similarity": 0.85
}'
```

## What's intentionally out of scope here

- **Face embedding extraction itself.** This service stores and matches
  vectors; it doesn't run the model that turns a camera frame into one. See
  the companion `glasses_client/embedding_v2.py` for a real (offline-testable)
  detect+extract pipeline, or the `OnnxEmbeddingExtractor` swap point in that
  same file for a production deep-embedding model.
- **Auth, rate limiting, persistence.** This is in-memory and unauthenticated
  by design, for local iteration. Needs a real datastore and auth layer
  before it touches a live event.

## Privacy rules enforced in code, not just docs

- `POST /profiles` rejects any `face` registration where `consent_given` is
  not `true` — a 400, not a warning.
- Every profile carries `event_ends_at`; the background sweeper purges the
  entire event's vector index the moment that time passes, no manual
  cleanup step required.
- An unmatched query returns an empty list, not an error — an unregistered
  face fails silently, per the spec's edge-case handling.

## Vector index v2 — the FAISS-swap promise, delivered

`vector_index_v2.py` replaces the original per-candidate Python loop with
`NumpyIndex`, a vectorized cosine-similarity search over a single matrix.
Measured on this machine (`python3 -m app.benchmark`):

```
n_vectors=3000 dim=128 n_queries=20
v1 (python loop):      715.3ms total, 35.76ms/query
v2 (numpy vectorized):   1.9ms total,  0.10ms/query
speedup: 369.3x
```

`FaissIndex` (same file) wraps `faiss.IndexFlatIP` behind an identical
interface for when an event's scale genuinely needs it — it's written and
structured, but not importable in this sandbox (no network to install
`faiss-cpu`). Swapping to it later is the one-line change documented at the
top of `vector_store.py` (`IndexBackend = NumpyIndex` → `FaissIndex`).

`vector_store.py`'s public API (`register` / `match` / `purge_event`)
didn't change, so `main.py` needed zero edits for this upgrade.

## Handoff to Phase 3 (hardware port) calls `/match` with the vector the phone-side face
pipeline produces after a gaze-dwell trigger fires, and renders the AR card
from whatever `ProfileRecord` comes back. No API changes should be needed —
only what's calling it.
