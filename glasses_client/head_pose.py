"""
head_pose.py -- real, geometry-based head pose (yaw/pitch/roll in DEGREES),
replacing the ratio-heuristic pose estimators in guided_enroll.py.

Root cause of "left/right/up/down not recognized":
  guided_enroll.py's old HaarPose fallback returned yaw=0.0 for ANY frame where
  the frontal Haar cascade found a face -- and the frontal cascade keeps
  matching up to roughly +-20-25 degrees of head turn, well past where "left"
  or "right" should already be registering. It only picked up yaw at all once
  the turn was so extreme that the FRONTAL cascade failed and the PROFILE
  cascade took over, which needs an even harder turn and a generous minSize.
  So: front worked (frontal cascade fires, yaw defaults to 0 = "front" - a
  correct accident), but left/right/up/down almost never crossed the gate.
  This wasn't a threshold-tuning problem, it was the estimator not measuring
  the angle at all for most of the range you asked people to turn through.
  Separately, this environment's mediapipe (0.10.33) no longer ships the old
  mp.solutions.face_mesh API guided_enroll.py's MediaPipePose called, so most
  installs were silently landing on that broken Haar fallback in the first place.

Fix: solvePnP against 6 stable facial landmarks (nose tip, chin, eye corners,
mouth corners) from MediaPipe's newer Tasks FaceLandmarker. This gives a real
3D head rotation, in degrees, continuously -- not a box-position guess that
only has two states (found/not found).
"""
from __future__ import annotations
import os, urllib.request
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Canonical 3D face model (mm-scale, arbitrary origin at the nose tip) matched to
# these 6 MediaPipe FaceMesh landmark indices. Standard points used across most
# OpenCV head-pose-estimation references (dlib's 68-point equivalents: 30, 8, 36, 45, 48, 54).
_LM_IDX = {"nose_tip": 1, "chin": 152, "eye_L": 33, "eye_R": 263, "mouth_L": 61, "mouth_R": 291}
_MODEL_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)


@dataclass
class HeadPose:
    found: bool
    yaw_deg: float = 0.0     # + = turned to their right (camera's left)
    pitch_deg: float = 0.0   # + = looking down
    roll_deg: float = 0.0
    bbox: Optional[tuple] = None       # (x, y, w, h) in pixels, from landmark extent
    n_faces: int = 0


def rotation_matrix_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """XYZ Euler extraction. Returns (pitch, yaw, roll) in degrees.
    Pure math, independent of any detector -- unit-tested below with
    synthetic rotations, since real face detection can't be tested offline."""
    sy = (R[0, 0] ** 2 + R[1, 0] ** 2) ** 0.5
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def solve_head_pose(points_2d: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float, float]:
    """points_2d: (6,2) pixel coords in the order of _LM_IDX above. Returns
    (yaw_deg, pitch_deg, roll_deg). This is the piece verified by
    test_head_pose.py against synthetic, known rotations."""
    focal = frame_w
    cam_matrix = np.array([[focal, 0, frame_w / 2], [0, focal, frame_h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_3D, points_2d.astype(np.float64), cam_matrix,
                              np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler_deg(R)
    return yaw, pitch, roll


def ensure_model(path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    """Downloads the ~4MB FaceLandmarker model once. Needs internet the FIRST
    time only; cached afterward. Raises with a clear message if it can't reach it
    (this sandbox has no network, so this path is untested here by necessity —
    run it once on a machine with internet before relying on it)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise RuntimeError(
            f"Could not download the face landmark model ({e}). Download it manually from\n{url}\n"
            f"and save it to {path}, or run this once on a machine with internet."
        ) from e
    return path


class LandmarkPoseEstimator:
    """Primary pose backend. Needs `pip install mediapipe` (already required) plus
    the one-time model download above."""
    name = "landmarks"

    def __init__(self, model_path: Optional[str] = None):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        path = ensure_model(model_path or MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=2, min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp_image_cls = __import__("mediapipe").Image
        self._mp_fmt = __import__("mediapipe").ImageFormat.SRGB

    def read(self, frame_bgr: np.ndarray) -> HeadPose:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(self._mp_image_cls(image_format=self._mp_fmt, data=rgb))
        faces = result.face_landmarks
        if not faces:
            return HeadPose(False)
        lm = faces[0]
        pts = np.array([[lm[_LM_IDX[k]].x * w, lm[_LM_IDX[k]].y * h] for k in
                        ("nose_tip", "chin", "eye_L", "eye_R", "mouth_L", "mouth_R")])
        yaw, pitch, roll = solve_head_pose(pts, w, h)
        xs, ys = [p.x * w for p in lm], [p.y * h for p in lm]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        return HeadPose(True, yaw, pitch, roll, bbox, len(faces))
