# AR Identity Mapping — laptop test rig (v3.2)

## v3.2: fixes "left/right/up/down not recognized"

**Root cause.** The v3.1 enrollment measured head turn with a box-position
heuristic. When it fell back to the Haar-cascade backend (which it did
silently — this environment's newer `mediapipe` no longer ships the API v3.1
called), the frontal-face detector kept matching for roughly ±20° of turn and
the fallback returned "yaw = 0" for every one of those frames. So straight-on
capture worked (0 is a correct guess for "front"), but left/right/up/down
almost never crossed the gate — not because the threshold was mistuned, but
because the estimator wasn't measuring the angle at all through most of the
range you were asked to turn.

**Fix, two parts:**
1. **`head_pose.py` (new)** — real head-pose geometry: 6 facial landmarks →
   `cv2.solvePnP` → yaw/pitch/roll in actual degrees. This is the same
   technique production head-pose systems use, not a heuristic. The angle
   math is independently verified in `test_head_pose.py` against synthetic,
   exactly-known rotations (7/7 pass) — including the specific case that
   broke before: a modest 15° turn must read as clearly non-zero, and it does.
2. **Adaptive calibration, not one fixed threshold.** Enrollment now opens
   with a ~2.5-second sweep: "move your head around slowly." Whatever yaw/
   pitch range *you* reach on *your* camera becomes the gate for the rest of
   enrollment (55% of your own range, clamped to a sane floor/ceiling). A
   single fixed number tuned on one person's face and one webcam was never
   going to generalize to hundreds of people with different necks, glasses,
   desks, and cameras — this is what makes it actually work at scale instead
   of just on the machine it was tuned on.

If a machine genuinely has no way to get real landmarks (no internet for the
one-time model download, see below), enrollment now says so honestly and
switches to a degraded mode — several quality-checked captures with natural
small movement between them — instead of silently pretending to gate on
direction it can't measure, which is what caused the original bug.

**One-time setup this adds:** the landmark backend needs a small model file
downloaded once (`head_pose.ensure_model()` does this automatically the first
time `enroll` runs, given internet access). It's cached after that. If your
enrollment machine has no internet at all, download it manually from the URL
printed in the error and place it next to `head_pose.py` as
`face_landmarker.task`.

**New/changed files:** `head_pose.py`, `test_head_pose.py` (new); `guided_enroll.py`
rewritten (calibration phase, degrees-based gating, degraded-mode fallback);
`test_guided_enroll.py` rewritten for the new logic (7/7 pass, including a
regression test for a "moderate turner" and a false-positive guard so a tiny
2° twitch can't satisfy a directional pose); `cli.py`'s `enroll` command
mentions the calibration step and checks the model is available before
opening the camera.

---


This folder wires **three things you already had** (Phase 2 backend, Phase 3
gaze pipeline, real LBP embedding) plus the **new guided enrollment** into one
thing you can run end-to-end on a laptop with a webcam, before touching a
phone or glasses.

```
ar-integrated/
├── backend/                  Phase 2 — FastAPI matching service (unchanged logic)
└── glasses_client/            Phase 3 — gaze pipeline + NEW guided enrollment + NEW cli.py
```

## What changed to make this wire together (read this once)

| File | What changed | Why |
|---|---|---|
| `glasses_client/guided_enroll.py` | **NEW.** Face-ID-style 6-pose capture, one camera session, auto-capture, auto-close. | Your original ask: no more open-camera-3x / press-`s`. |
| `glasses_client/pipeline.py` | `on_gaze_hit` now takes a **real image** (or a zero-arg function that returns one) instead of only a `frame_id` string. Calls the real `embedding_v2.extract()`. Targeting ring shows **real** dwell %, not a hardcoded 50%. | Phase 3 shipped with a hash-based stub; this switches it to the real detect+embed pipeline you already built in `embedding_v2.py`, with zero changes needed in `state_machine.py`. |
| `glasses_client/state_machine.py` | Added `dwell_progress_pct()`. Nothing else touched. | Feeds the real ring % above. |
| `glasses_client/matching_api.py` | Added `register_profile()` (calls `POST /profiles`) and `event_size()` (calls `GET /events/{id}/size`). Default threshold is now a named constant `DEFAULT_MIN_SIMILARITY = 0.84`. | The client previously could only *query* matches, never *register* a face — needed for enrollment to actually reach the server. |
| `glasses_client/render_bridge.py` | Added `OpenCVRenderBridge`, a new class alongside the existing `ConsoleRenderBridge`. | Something to literally draw the ring/card on your webcam preview, standing in for the glasses display. |
| `glasses_client/cli.py` | **NEW.** `check` / `enroll` / `live` commands. | One command per step of the flow described below. |
| `glasses_client/demo.py` | One line changed per call: `frame_id=x` → `frame=x`, to match the renamed pipeline parameter. Behavior is identical — reran it, same output. | Keeps the original text-only demo (no camera) working. |
| `backend/` | **Untouched.** Its logic was tested directly (see below) and behaves correctly. | Nothing about matching or the privacy rules needed to change. |

Everything else (`vector_store.py`, `vector_index_v2.py`, `expiry.py`,
`embedding_v2.py`, `state_machine.py`'s core transitions) is exactly what you
already had — I didn't touch matching logic, privacy enforcement, or timings.

## What I actually ran here, and what I could not

This sandbox has **no camera and no internet**, so:
- ✅ Ran: `demo.py` (full scenario replay, output above), `test_embedding_v2.py`
  (4/4 pass), a new `test_guided_enroll.py` (7/7 pass, pure state-machine logic,
  no camera), and the vector-index search math directly (`NumpyIndex`, correct
  top-1 match) — all pass.
- ❌ Not run: the FastAPI server itself (`pip install fastapi` fails — no
  network here), and everything that needs an actual webcam (`enroll`, `live`).
  **You need to run those two yourself** — that's the whole point of this
  laptop rig.

---

## Step-by-step: run it on your laptop

### 0. Install dependencies
```bash
cd ar-integrated
pip install -r backend/requirements.txt
pip install opencv-python numpy mediapipe
```
The first time you run `enroll`, it downloads a small (~4MB) face-landmark
model automatically (needs internet **once**; cached after that). If that
machine has no internet, download it yourself from the URL the error message
prints and save it as `glasses_client/face_landmarker.task`.

If neither works, enrollment still runs — it switches to a degraded mode
that captures several good-quality frames without verifying direction, and
says so on screen, rather than silently pretending to.

### 1. Start the backend (Phase 2), in its own terminal
```bash
cd ar-integrated/backend
uvicorn app.main:app --reload
```
Leave this running. It's in-memory — restarting it wipes all registered faces.

### 2. Check the client can reach it, in a second terminal
```bash
cd ar-integrated
python3 -m glasses_client.cli check
```
Expect: `backend OK at http://localhost:8000; vectors indexed for 'arxr2026': 0`

### 3. Enroll your face
```bash
python3 -m glasses_client.cli enroll --name "Your Name" --role "ML Engineer" --company Acme
```
What happens:
1. You're shown a consent notice and must type `yes` — camera never opens otherwise.
2. **One** camera window opens. First, a ~2.5s **calibration sweep**: move your
   head around slowly (a filling cyan ring shows the countdown). This measures
   *your* comfortable turn range on *your* camera — nothing is captured yet.
3. Then follow the on-screen prompt: front → left → right → up → down → front
   again. Each pose auto-captures when you hold it steady (no key to press).
   The gate for "far enough left/right/up/down" is now based on what you
   showed it during calibration, not a fixed number — so you don't need to
   guess how far is "enough."
4. The window **closes itself** when all 6 are captured.
4. Each of the 6 saved photos is run through `embedding_v2.extract()` (face
   detect + LBP histogram) and sent to the backend with `POST /profiles`,
   once per view. You'll see a line per pose: registered, or "no face detected"
   if that particular photo's crop failed (common for hard side/up/down angles
   with the Haar detector — see the warning it prints if fewer than 3 land).
5. It prints the final vector count from the server (`GET /events/.../size`),
   so you can see the enrollment actually reached the backend, not just the
   local folder.

### 4. Run live recognition + overlay
```bash
python3 -m glasses_client.cli live
```
- A camera window opens. A white crosshair marks the "gaze point" (there's no
  real eye tracker on a laptop, so this is standing in for it — see below).
- Put your face so the crosshair sits on it and hold still. You'll see:
  - a **cyan ring** filling — this is the real `TARGETING` dwell timer from
    `state_machine.py`, not decoration
  - a brief **pulsing circle** — `MATCHING`, the one moment it calls `/match`
  - a **green box + name card** — `ANCHORED`, pulled straight from whatever
    `MatchedProfile` the backend returned
- The top-left line shows the live state, similarity score, threshold, and
  API latency, so you can see *why* something did or didn't match, not just
  that it didn't.
- Walk out of frame — after ~1.5s the card clears and it goes back to `STANDBY`.
- Have someone else (or a photo of someone else) sit in: it should stay in
  `STANDBY`/show "no match above threshold", never a card.
- `--debug-scores` shows you the actual similarity even when it's below
  threshold, which is the fastest way to tune `DEFAULT_MIN_SIMILARITY` in
  `matching_api.py`.
- `q` or `Esc` quits.

### 5. Tune the threshold, if enrollment succeeds but live doesn't match
```bash
python3 -m glasses_client.cli live --debug-scores
```
Watch the `sim=` number while your own face is in frame vs. a stranger's.
Pick a threshold between the two, then either pass `--threshold 0.80` on the
`live` command or change `DEFAULT_MIN_SIMILARITY` in `matching_api.py`.

---

## Honest limitations before you build the mobile app

- **Gaze simulation, not real gaze.** `cli.py`'s "gaze hit = crosshair on
  face" is a laptop stand-in. On a phone, this becomes wherever you point the
  camera (or, on real glasses, the actual gaze-duration API) — nothing in
  `pipeline.py` or `state_machine.py` needs to change, only what calls
  `on_gaze_hit`.
- **LBP embedding is a classical baseline**, not a deep model — same caveat
  the original `EMBEDDING_V2_README.md` already gave you. Expect it to be
  more sensitive to lighting and angle than a production ArcFace/FaceNet
  model. `OnnxEmbeddingExtractor` in `embedding_v2.py` is the documented
  upgrade path once you have model weights; nothing else in this chain
  changes shape when you swap it in.
- **No liveness detection.** A printed photo or a phone screen showing a
  photo can currently pass. Worth adding (blink detection, or a depth check
  on real hardware) before any real pilot with strangers.
- **Haar detector is frontal-biased.** Expect some of the left/right/up/down
  enrollment frames to fail detection — this is normal and the `enroll`
  command tells you when it happens. It's the same limitation the original
  `EMBEDDING_V2_README.md` already flagged.
- **Backend is in-memory, unauthenticated, single-process** — fine for this
  laptop rig, not for a real event, exactly as its own README already says.

## Moving to a mobile app next

`cli.py`'s three commands map directly onto the three screens a mobile app
needs: a **consent + enrollment** screen (`enroll`), a **live camera view**
(`live`), and a **backend reachability/settings** check (`check`). The
underlying calls (`run_guided_enrollment`, `GazePipeline.on_gaze_hit`,
`MatchingAPIClient.register_profile` / `.match`) are plain Python with a
stdlib-only HTTP client, so the logic ports directly to Kotlin/Swift call-for-
call — only the camera capture and on-screen drawing need native replacements,
exactly like `render_bridge.py`'s `RenderBridge` seam already anticipates.
