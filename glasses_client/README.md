# Phase 3 — Hardware Port Client

The gaze-dwell pipeline from the project plan's Phase 3 milestone, built and
tested end-to-end against a fake in-process matching backend so it runs
anywhere with just the standard library — swap in the real Phase 2 API and
nothing else changes.

## What's here

| File | Role |
|---|---|
| `state_machine.py` | Standby → Targeting → Matching → Anchored, with the spec's exact timings (150ms saccade floor, 500ms dwell, 700ms match timeout). |
| `matching_api.py` | stdlib-only HTTP client for Phase 2's `/match` endpoint. |
| `embedding.py` | **Stub.** Defines the interface the real on-device face-embedding model plugs into — this file must be replaced before any real deployment, never used as-is. |
| `render_bridge.py` | The seam to the actual glasses SDK (Android XR / Jetpack Compose XR, Vuzix, Meta's `mwdat-display`). Ships a `ConsoleRenderBridge` for local dev. |
| `pipeline.py` | Wires the four pieces above into the loop: gaze hit → dwell → one frame → one match call → anchored card. |
| `demo.py` | Runs three scenarios end-to-end: a successful match, an unregistered face (fails silently, per spec), and a quick saccade (never triggers at all). |

## Run it

```bash
python3 -m glasses_client.demo
```

No install step — everything here is standard library only. Sample output:

```
--- Scenario 1: registered attendee, full dwell ---
[HUD] targeting ring on attendee-042: 50%
...
[HUD] pulsing match indicator on attendee-042
[HUD] card anchored on attendee-042: Sarah Chen — Lead Spatial Architect · Acme Inc.

--- Scenario 2: unregistered face, fails silently ---
...
[HUD] pulsing match indicator on unknown-person
[HUD] cleared unknown-person

--- Scenario 3: quick saccade, never triggers ---
[HUD] targeting ring on attendee-017: 50%
[HUD] cleared attendee-017
```

## Wiring it to the real Phase 2 service

`demo.py` uses `FakeMatchingAPIClient`, an in-process stand-in. To point the
pipeline at the actual Phase 2 backend instead:

```python
from glasses_client.matching_api import MatchingAPIClient
from glasses_client.pipeline import GazePipeline
from glasses_client.render_bridge import ConsoleRenderBridge  # or a real hardware bridge

pipeline = GazePipeline(
    event_id="arxr2026",
    matching_client=MatchingAPIClient(base_url="https://your-phase2-host"),
    render_bridge=ConsoleRenderBridge(),
)
```

Nothing in `state_machine.py` or `pipeline.py` needs to change.

## Wiring it to real hardware

Two seams, both already isolated:

1. **Sensor input** — whatever gives you gaze-ray/head-pose data (Android
   XR's gaze-duration APIs, or your own depth-buffer intersection code) should
   call `pipeline.on_gaze_hit(target_id, frame_id)` on each tick and
   `pipeline.on_gaze_lost()` when the ray no longer hits anything. The state
   machine already filters out saccades under 150ms — the sensor layer
   doesn't need its own debounce logic.
2. **Rendering** — implement `RenderBridge`'s three methods against the real
   SDK (Jetpack Compose XR, Vuzix's display API, or `mwdat-display`) instead
   of `ConsoleRenderBridge`.

## What's still a stub, on purpose

- **`embedding.py`** returns a deterministic hash-based vector, not a real
  face embedding. This is wiring/integration-test scaffolding only — the
  Applied ML lead's real model (per the plan's pod split) replaces this file
  entirely before anything touches a live camera.
- **Camera capture itself** isn't here — this pipeline assumes something
  upstream (the OS camera API on the glasses or paired phone) hands it a
  `frame_id` once the dwell timer fires. It never polls the camera
  continuously, matching the "single frame per trigger" battery strategy
  from the spec.
