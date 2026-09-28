"""
guided_enroll.py -- Face-ID-style guided enrollment (Phase 3), v3.2.

v3.2 fixes the real bug from testing v3.1: left/right/up/down almost never
triggered. Root cause (see head_pose.py's docstring for the full diagnosis):
the old estimator measured turn direction with a crude box-position heuristic
that stayed "front" for most of the range you were asked to turn through, and
used ONE fixed threshold for every person/camera/lighting setup. Two fixes:

  1. Real geometry: yaw/pitch/roll in degrees via solvePnP against 6 facial
     landmarks (head_pose.py) -- math independently verified against known
     synthetic rotations in test_head_pose.py, not just eyeballed.
  2. No fixed threshold. Before the 6 poses, a ~2.5s CALIBRATION sweep asks
     you to move your head around once; whatever range YOU actually reach on
     YOUR camera becomes the gate for the rest of enrollment (a fraction of
     it, clamped to a sane floor/ceiling). This is what makes it generalize
     across hundreds of people with different necks, glasses, desks, and
     webcams, instead of one number tuned on one face.

If no reliable pose backend is available (mediapipe missing, or its model
file couldn't be downloaded -- see head_pose.ensure_model), enrollment
degrades honestly instead of pretending: it captures several quality-checked
frames spaced across natural small head movement, and tells you on screen
that direction isn't being verified, rather than silently gating on a
heuristic that doesn't actually measure direction (which is what caused the
original bug).

Usage
  python guided_enroll.py --user alice --out enroll_out
  from guided_enroll import run_guided_enrollment
  result = run_guided_enrollment(consent_granted=True, user_id="alice")
"""
from __future__ import annotations
import os, time, argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ------------------------------------------------------------------ config
class Pose(str, Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    FRONT_CONFIRM = "front_confirm"

POSE_SEQUENCE = [Pose.FRONT, Pose.LEFT, Pose.RIGHT, Pose.UP, Pose.DOWN, Pose.FRONT_CONFIRM]

PROMPTS = {
    Pose.FRONT: "Look straight at the camera",
    Pose.LEFT: "Slowly turn your head to the LEFT",
    Pose.RIGHT: "Slowly turn your head to the RIGHT",
    Pose.UP: "Tilt your head UP",
    Pose.DOWN: "Tilt your head DOWN",
    Pose.FRONT_CONFIRM: "Look straight again to finish",
}

@dataclass
class Config:
    hold_frames: int = 8
    front_tol_deg: float = 6.0          # how close to "centre" still counts as front
    turn_fraction: float = 0.55         # gate = this fraction of what THIS user reached while calibrating
    min_yaw_gate_deg: float = 8.0       # never require less than this (avoids noise-triggering)
    max_yaw_gate_deg: float = 35.0      # never require more than this (avoids demanding a near-profile turn)
    min_pitch_gate_deg: float = 6.0
    max_pitch_gate_deg: float = 25.0
    calibration_seconds: float = 2.5
    degraded_capture_count: int = 6     # frames captured when no reliable pose signal exists
    degraded_min_gap_s: float = 0.8     # spacing between degraded-mode captures
    min_face_frac: float = 0.12
    max_face_frac: float = 0.9
    min_sharpness: float = 25.0
    min_brightness: float = 45.0
    max_brightness: float = 215.0
    timeout_per_pose_s: float = 25.0
    total_timeout_s: float = 150.0
    mirror_preview: bool = True


@dataclass
class PoseReading:
    face_found: bool
    yaw: float = 0.0          # degrees; + = turned to their right. 0 if reliable_pose is False.
    pitch: float = 0.0        # degrees; + = looking down.
    face_frac: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    n_faces: int = 1


def _quality(gray: np.ndarray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())


# ---------------------------------------------------------------- backends
class LandmarkPose:
    """Primary backend: real degrees from head_pose.LandmarkPoseEstimator."""
    name, reliable_pose = "landmarks", True

    def __init__(self):
        from .head_pose import LandmarkPoseEstimator
        self._est = LandmarkPoseEstimator()

    def read(self, frame_bgr) -> PoseReading:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sharp, bright = _quality(gray)
        hp = self._est.read(frame_bgr)
        if not hp.found:
            return PoseReading(False, sharpness=sharp, brightness=bright)
        face_frac = (hp.bbox[3] / h) if hp.bbox else 0.0
        return PoseReading(True, hp.yaw_deg, hp.pitch_deg, face_frac, sharp, bright, hp.n_faces)


class HaarPose:
    """Fallback when no landmark backend is available. Detects a face and
    checks framing/quality, but does NOT claim to know which way you're
    turned -- yaw/pitch always read 0. Marked reliable_pose=False so
    EnrollmentSession skips directional gating instead of faking it."""
    name, reliable_pose = "haar", False

    def __init__(self):
        d = cv2.data.haarcascades
        self.front = cv2.CascadeClassifier(d + "haarcascade_frontalface_default.xml")

    def read(self, frame_bgr) -> PoseReading:
        h, w = frame_bgr.shape[:2]
        gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        sharp, bright = _quality(gray)
        faces = self.front.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) == 0:
            return PoseReading(False, sharpness=sharp, brightness=bright)
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return PoseReading(True, 0.0, 0.0, fh / h, sharp, bright, len(faces))


def make_backend(prefer: str = "auto"):
    if prefer in ("auto", "landmarks"):
        try:
            b = LandmarkPose(); print("[enroll] pose backend: landmarks (solvePnP, degrees) - reliable direction gating"); return b
        except Exception as e:
            if prefer == "landmarks": raise
            print(f"[enroll] landmark backend unavailable ({e})")
    print("[enroll] pose backend: haar (fallback) - can only confirm a face is present, "
          "NOT which way you're turned. Enrollment will use degraded capture mode.")
    return HaarPose()


# ------------------------------------------------------------ state machine
class Phase(str, Enum):
    CALIBRATING = "calibrating"
    POSING = "posing"
    DEGRADED = "degraded"
    DONE = "done"
    FAILED = "failed"

class EnrollmentSession:
    """Pure logic (no camera) so it's unit-testable. Feed one PoseReading per frame."""

    def __init__(self, cfg: Config = Config(), reliable_pose: bool = True,
                clock: Callable[[], float] = time.monotonic):
        self.cfg, self.clock, self.reliable_pose = cfg, clock, reliable_pose
        self.phase = Phase.CALIBRATING if reliable_pose else Phase.DEGRADED
        self.idx, self.streak = 0, 0
        self.captured: List[Pose] = []
        self.started = self.phase_started = clock()
        self.status = "running"
        self.message = ("Move your head around slowly - up, down, left, right - to calibrate"
                        if reliable_pose else
                        "Direction can't be verified on this setup - hold still for each capture")
        self.baseline_yaw = self.baseline_pitch = None
        self.yaw_min = self.yaw_max = self.pitch_min = self.pitch_max = 0.0
        self.yaw_gate = self.pitch_gate = 0.0
        self._last_capture_t = 0.0

    @property
    def current(self) -> Optional[Pose]:
        if self.phase == Phase.POSING and self.idx < len(POSE_SEQUENCE):
            return POSE_SEQUENCE[self.idx]
        return None

    @property
    def progress(self) -> float:
        if self.phase == Phase.CALIBRATING:
            return min(1.0, (self.clock() - self.phase_started) / self.cfg.calibration_seconds)
        if self.phase == Phase.DEGRADED:
            return len(self.captured) / self.cfg.degraded_capture_count
        return self.streak / self.cfg.hold_frames

    def _quality_problem(self, r: PoseReading) -> Optional[str]:
        c = self.cfg
        if not r.face_found: return "No face found - move into view"
        if r.n_faces > 1: return "Only one person in frame, please"
        if r.face_frac < c.min_face_frac: return "Move closer"
        if r.face_frac > c.max_face_frac: return "Move back a little"
        if r.brightness < c.min_brightness: return "Too dark - find more light"
        if r.brightness > c.max_brightness: return "Too bright - reduce glare"
        if r.sharpness < c.min_sharpness: return "Hold still - image is blurry"
        return None

    def _gate(self, achieved_min: float, achieved_max: float, floor: float, ceil_: float) -> float:
        span = max(abs(achieved_min), abs(achieved_max))
        return float(np.clip(span * self.cfg.turn_fraction, floor, ceil_))

    def _finish_calibration(self) -> None:
        c = self.cfg
        self.yaw_gate = self._gate(self.yaw_min, self.yaw_max, c.min_yaw_gate_deg, c.max_yaw_gate_deg)
        self.pitch_gate = self._gate(self.pitch_min, self.pitch_max, c.min_pitch_gate_deg, c.max_pitch_gate_deg)
        self.phase, self.phase_started, self.idx = Phase.POSING, self.clock(), 0
        self.message = PROMPTS[POSE_SEQUENCE[0]]

    def _pose_matches(self, pose: Pose, r: PoseReading) -> bool:
        dy, dp = r.yaw - self.baseline_yaw, r.pitch - self.baseline_pitch
        t = self.cfg.front_tol_deg
        if pose in (Pose.FRONT, Pose.FRONT_CONFIRM): return abs(dy) <= t and abs(dp) <= t
        if pose == Pose.LEFT:  return dy <= -self.yaw_gate and abs(dp) <= self.pitch_gate
        if pose == Pose.RIGHT: return dy >= self.yaw_gate and abs(dp) <= self.pitch_gate
        if pose == Pose.UP:    return dp <= -self.pitch_gate and abs(dy) <= self.yaw_gate
        if pose == Pose.DOWN:  return dp >= self.pitch_gate and abs(dy) <= self.yaw_gate
        return False

    def update(self, r: PoseReading) -> bool:
        """Returns True on the frame where a photo should be saved."""
        if self.status != "running":
            return False
        now = self.clock()
        if now - self.started > self.cfg.total_timeout_s or \
           now - self.phase_started > self.cfg.timeout_per_pose_s * (2 if self.phase == Phase.CALIBRATING else 1):
            self.status, self.message = "failed", "Timed out - please try again"
            return False

        problem = self._quality_problem(r)

        # ---- calibration: no capture, just track the range while quality is OK
        if self.phase == Phase.CALIBRATING:
            if not problem:
                if self.baseline_yaw is None:
                    self.baseline_yaw, self.baseline_pitch = r.yaw, r.pitch
                    self.yaw_min = self.yaw_max = r.yaw
                    self.pitch_min = self.pitch_max = r.pitch
                self.yaw_min, self.yaw_max = min(self.yaw_min, r.yaw), max(self.yaw_max, r.yaw)
                self.pitch_min, self.pitch_max = min(self.pitch_min, r.pitch), max(self.pitch_max, r.pitch)
            else:
                self.message = problem
            if self.clock() - self.phase_started >= self.cfg.calibration_seconds and self.baseline_yaw is not None:
                self._finish_calibration()
            elif not problem:
                self.message = "Keep moving your head around..."
            return False

        # ---- degraded mode: no direction signal, just spaced-out quality captures
        if self.phase == Phase.DEGRADED:
            if problem:
                self.message = problem
                return False
            self.message = "Hold still..."
            if now - self._last_capture_t < self.cfg.degraded_min_gap_s:
                return False
            self.streak += 1
            if self.streak < self.cfg.hold_frames:
                return False
            self.streak, self._last_capture_t = 0, now
            self.captured.append(Pose.FRONT)  # label is nominal in degraded mode
            if len(self.captured) >= self.cfg.degraded_capture_count:
                self.status, self.phase, self.message = "complete", Phase.DONE, "Enrollment complete"
            else:
                self.message = "Move your head slightly, then hold still again"
            return True

        # ---- posing: directional gate using the calibrated thresholds
        pose = self.current
        if problem:
            self.streak, self.message = 0, problem
            return False
        if not self._pose_matches(pose, r):
            self.streak, self.message = 0, PROMPTS[pose]
            return False
        self.streak += 1
        self.message = "Great, hold it..."
        if self.streak < self.cfg.hold_frames:
            return False

        self.captured.append(pose)
        self.idx += 1
        self.streak, self.phase_started = 0, now
        if self.idx >= len(POSE_SEQUENCE):
            self.status, self.phase, self.message = "complete", Phase.DONE, "Enrollment complete"
        else:
            self.message = PROMPTS[POSE_SEQUENCE[self.idx]]
        return True


# ------------------------------------------------------------------ overlay
def _draw(frame, sess: EnrollmentSession):
    h, w = frame.shape[:2]
    cx, cy, R = w // 2, h // 2, int(min(w, h) * 0.38)
    if sess.phase == Phase.CALIBRATING:
        cv2.ellipse(frame, (cx, cy), (R, R), 0, -90, -90 + 360 * sess.progress, (0, 200, 255), 6)
        cv2.ellipse(frame, (cx, cy), (R, R), 0, -90 + 360 * sess.progress, 270, (70, 70, 70), 6)
    else:
        total = sess.cfg.degraded_capture_count if sess.phase == Phase.DEGRADED else len(POSE_SEQUENCE)
        done = len(sess.captured)
        for i in range(total):
            a0, a1 = -90 + i * 360 / total + 2, -90 + (i + 1) * 360 / total - 2
            if i < done: col, span = (80, 220, 100), a1 - a0
            elif i == done and sess.phase != Phase.DEGRADED:
                col, span = (0, 200, 255), (a1 - a0) * sess.progress
            else: col, span = (90, 90, 90), 0
            cv2.ellipse(frame, (cx, cy), (R, R), 0, a0, a1, (60, 60, 60), 6)
            if span > 0: cv2.ellipse(frame, (cx, cy), (R, R), 0, a0, a0 + span, col, 6)
        pose = sess.current
        if pose:
            d = {Pose.LEFT: (-1, 0), Pose.RIGHT: (1, 0), Pose.UP: (0, -1), Pose.DOWN: (0, 1)}.get(pose)
            if d:
                cv2.arrowedLine(frame, (cx, cy), (cx + d[0]*(R-25), cy + d[1]*(R-25)),
                                (0, 200, 255), 4, tipLength=0.25)
    cv2.putText(frame, sess.message, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    tot = sess.cfg.degraded_capture_count if sess.phase == Phase.DEGRADED else len(POSE_SEQUENCE)
    cv2.putText(frame, f"{len(sess.captured)}/{tot}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)


# ------------------------------------------------------------------- runner
@dataclass
class EnrollmentResult:
    success: bool
    frames: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    reason: str = ""

def run_guided_enrollment(consent_granted: bool, user_id: str = "user",
                          out_dir: str = "enroll_out", camera_index: int = 0,
                          cfg: Config = Config(), backend: str = "auto",
                          show_window: bool = True,
                          on_complete: Optional[Callable[[EnrollmentResult], None]] = None
                          ) -> EnrollmentResult:
    if not consent_granted:
        return EnrollmentResult(False, reason="consent not granted")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return EnrollmentResult(False, reason="camera unavailable")
    est = make_backend(backend)
    sess = EnrollmentSession(cfg, reliable_pose=est.reliable_pose)
    result, seen = EnrollmentResult(False), 0
    os.makedirs(out_dir, exist_ok=True)
    try:
        while sess.status == "running":
            ok, frame = cap.read()
            if not ok:
                result.reason = "camera read failed"; break
            raw = frame
            frame = cv2.flip(raw, 1) if cfg.mirror_preview else raw
            reading = est.read(frame)
            pose_before = sess.current or Pose.FRONT
            if sess.update(reading):
                clean = raw.copy()
                seen += 1
                path = os.path.join(out_dir, f"{user_id}_{seen-1}_{pose_before.value}.jpg")
                cv2.imwrite(path, clean)
                result.frames[f"{pose_before.value}_{seen}"], result.paths[f"{pose_before.value}_{seen}"] = clean, path
            if show_window:
                _draw(frame, sess)
                cv2.imshow("Face enrollment", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    result.reason = "cancelled by user"; break
    finally:
        cap.release()
        if show_window: cv2.destroyAllWindows()
    result.success = sess.status == "complete"
    if not result.success and not result.reason:
        result.reason = sess.message
    if result.success and on_complete:
        on_complete(result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="user"); ap.add_argument("--out", default="enroll_out")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", choices=["auto", "landmarks", "haar"], default="auto")
    a = ap.parse_args()
    r = run_guided_enrollment(True, a.user, a.out, a.camera, backend=a.backend)
    print("OK" if r.success else f"FAILED: {r.reason}", list(r.paths.values()))
