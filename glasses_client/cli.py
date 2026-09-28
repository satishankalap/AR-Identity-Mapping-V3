"""
One entry point for the whole laptop test flow.

  python -m glasses_client.cli check                       # is the backend up?
  python -m glasses_client.cli enroll --name "Alice Kim" --role "ML Engineer" --company Acme
  python -m glasses_client.cli live [--debug-scores]       # look at the camera, dwell, get the card

Run from the folder that CONTAINS glasses_client/ (the project root).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import cv2

from .embedding_v2 import FaceDetector, extract
from .guided_enroll import Config, POSE_SEQUENCE, run_guided_enrollment
from .matching_api import DEFAULT_MIN_SIMILARITY, MatchingAPIClient, MatchingAPIError
from .pipeline import GazePipeline
from .render_bridge import OpenCVRenderBridge
from .state_machine import State


# ------------------------------------------------------------------ check
def cmd_check(a):
    c = MatchingAPIClient(a.url)
    try:
        print(f"backend OK at {a.url}; vectors indexed for '{a.event}': {c.event_size(a.event)}")
    except MatchingAPIError as e:
        print(f"backend NOT reachable at {a.url}: {e}\nStart it: cd backend && uvicorn app.main:app --reload")


# ----------------------------------------------------------------- enroll
def cmd_enroll(a):
    print("\nCONSENT: your face will be captured (6 photos) and converted to a numeric template\n"
          "stored on the matching server for this event only. It is auto-deleted when the event\n"
          f"ends ({a.hours}h from now in this test).\n"
          "The camera opens once: first a ~2.5s calibration sweep (move your head around so the\n"
          "system learns YOUR comfortable turn range), then 6 guided poses, auto-captured.")
    consent = input("Type 'yes' to agree: ").strip().lower() == "yes"
    if not consent:
        print("No consent -> camera never opened. Exiting."); return

    if a.pose_backend in ("auto", "landmarks"):
        try:
            from .head_pose import ensure_model
            ensure_model()
        except Exception as e:
            print(f"[enroll] {e}")
            if a.pose_backend == "landmarks":
                return
            print("[enroll] continuing with the degraded (no-direction) fallback instead.")

    result = run_guided_enrollment(consent_granted=True, user_id=a.external_id, out_dir=a.out,
                                   camera_index=a.camera, backend=a.pose_backend)
    if not result.success:
        print("Enrollment failed:", result.reason); return
    print(f"\nCaptured {len(result.frames)} frames. Camera is closed. Extracting templates...")

    client, ok = MatchingAPIClient(a.url), 0
    ends = datetime.utcnow() + timedelta(hours=a.hours)
    for pose, img in result.frames.items():
        vec = extract(img)                       # Haar detect + LBP histogram (embedding_v2)
        if vec is None:
            print(f"  {pose.value:14s} no face detected by embedding pipeline -> skipped"); continue
        client.register_profile(a.event, a.external_id, a.name, a.role, a.company,
                                vec, consent_given=True, event_ends_at=ends)
        ok += 1
        print(f"  {pose.value:14s} template registered ({len(vec)} dims)")
    print(f"\n{ok}/{len(result.frames)} views registered. Server now holds "
          f"{client.event_size(a.event)} vectors for event '{a.event}'.")
    if ok < 3:
        print("WARNING: fewer than 3 usable views - side/up/down views often fail Haar (frontal-only). "
              "Try better light / smaller head turns, or upgrade to the ONNX deep embedding.")


# ------------------------------------------------------------------- live
def cmd_live(a):
    client = MatchingAPIClient(a.url, timeout_s=a.timeout)
    render = OpenCVRenderBridge()
    pipe = GazePipeline(a.event, client, render, min_similarity=a.threshold,
                        debug_scores=a.debug_scores)
    detector = FaceDetector()
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print("camera unavailable"); return
    last_seen, cooldown_until, last_box, t0 = 0.0, 0.0, None, time.time()
    try:
        while True:
            ok, raw = cap.read()
            if not ok: break
            now = time.time()
            h, w = raw.shape[:2]
            # 1. detect the largest face (on real glasses: depth/SLAM person tracking)
            face = detector.detect_largest(raw)
            box = (face.x, face.y, face.w, face.h) if face else None
            # 2. "gaze" simulation: the centre crosshair is where the wearer looks.
            #    Gaze HIT = crosshair inside the face box.
            gaze_hit = bool(face and face.x <= w // 2 <= face.x + face.w and face.y <= h // 2 <= face.y + face.h)

            state = pipe.sm.state
            if state == State.ANCHORED:
                # camera would be idle here; laptop keeps tracking the box only to move the card
                if face: last_seen, last_box = now, box
                elif now - last_seen > a.walkaway_s: pipe.on_target_walked_away("face-1")
            elif now < cooldown_until:
                pass                                                  # cool-down after a failed match
            elif gaze_hit:
                last_seen, last_box = now, box
                pipe.on_gaze_hit("face-1", frame=lambda r=raw: r.copy())   # frame read only at dwell end
            else:
                pipe.on_gaze_lost()

            # after a failed/no-match dwell, pause before re-triggering on the same face
            if state != State.STANDBY and pipe.sm.state == State.STANDBY and pipe.last_note and \
               not pipe.last_note.startswith("matched"):
                cooldown_until = now + a.cooldown_s

            # 3. draw. The raw frame is un-mirrored (what the camera sees, what gets embedded);
            #    the displayed view is mirrored (selfie-style), so any box drawn on it must be
            #    flipped too. Prefer the freshest box (this frame's), else the last one we had
            #    while the state machine is still holding a target.
            view = cv2.flip(raw, 1)
            src_box = box if (box and pipe.sm.state != State.STANDBY) else \
                     (last_box if pipe.sm.state != State.STANDBY else None)

            def mirror_box(b):
                bx, by, bw, bh = b
                return (w - bx - bw, by, bw, bh)

            cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
            render.draw(view, mirror_box(src_box) if src_box else None, now - t0)
            sim = "-" if pipe.last_similarity is None else f"{pipe.last_similarity:.3f}"
            lat = "-" if pipe.last_latency_ms is None else f"{pipe.last_latency_ms:.0f}ms"
            cv2.putText(view, f"state={pipe.sm.state.name}  sim={sim}  thr={a.threshold:.2f}  api={lat}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(view, pipe.last_note[:70], (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(view, "put your face on the crosshair | q=quit", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("AR identity - live test", view)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")): break
    finally:
        cap.release(); cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--event", default="arxr2026")
    ap.add_argument("--camera", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    e = sub.add_parser("enroll")
    e.add_argument("--name", required=True); e.add_argument("--role"); e.add_argument("--company")
    e.add_argument("--external-id", default="attendee-001"); e.add_argument("--out", default="enroll_out")
    e.add_argument("--hours", type=int, default=8); e.add_argument("--pose-backend", default="auto",
                   choices=["auto", "landmarks", "haar"])
    l = sub.add_parser("live")
    l.add_argument("--threshold", type=float, default=DEFAULT_MIN_SIMILARITY)
    l.add_argument("--debug-scores", action="store_true", help="always show best score, even below threshold")
    l.add_argument("--timeout", type=float, default=0.7, help="match API timeout (s)")
    l.add_argument("--cooldown-s", type=float, default=2.0); l.add_argument("--walkaway-s", type=float, default=1.5)
    a = ap.parse_args()
    {"check": cmd_check, "enroll": cmd_enroll, "live": cmd_live}[a.cmd](a)


if __name__ == "__main__":
    main()
