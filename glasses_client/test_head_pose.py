"""
Verifies the solvePnP/Euler-angle MATH in head_pose.py against synthetic,
exactly-known rotations. This is the part that's testable without a camera:
we manufacture 2D landmark projections for a known yaw/pitch and check we
recover it. Real face-landmark accuracy still needs a real webcam.
"""
import numpy as np
import cv2
from .head_pose import _MODEL_3D, solve_head_pose

W, H = 640, 480
FOCAL = W


def project(yaw_deg, pitch_deg, roll_deg=0.0, tz=800):
    """Rotate the canonical 3D model by the given Euler angles, place it in
    front of a pinhole camera, and project to 2D -- the inverse of what
    solve_head_pose does, so recovering the same angles back is a real check."""
    y, p, r = np.radians([pitch_deg, yaw_deg, roll_deg])
    Rx = np.array([[1,0,0],[0,np.cos(y),-np.sin(y)],[0,np.sin(y),np.cos(y)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    R = Rz @ Ry @ Rx
    cam = np.array([[FOCAL,0,W/2],[0,FOCAL,H/2],[0,0,1]], dtype=np.float64)
    tvec = np.array([0, 0, tz], dtype=np.float64)
    pts2d, _ = cv2.projectPoints(_MODEL_3D, cv2.Rodrigues(R)[0], tvec, cam, np.zeros((4,1)))
    return pts2d.reshape(-1, 2)


def _check(yaw, pitch, roll=0.0, tol=2.0):
    pts = project(yaw, pitch, roll)
    y2, p2, r2 = solve_head_pose(pts, W, H)
    assert abs(y2 - yaw) < tol, f"yaw {yaw} -> recovered {y2:.2f}"
    assert abs(p2 - pitch) < tol, f"pitch {pitch} -> recovered {p2:.2f}"


def test_front():         _check(0, 0)
def test_left_30():       _check(-30, 0)
def test_right_30():      _check(30, 0)
def test_up_20():         _check(0, -20)
def test_down_20():       _check(0, 20)
def test_combined():      _check(20, -15)
def test_small_angles_still_separable():
    # the actual bug we're fixing: does a MODEST turn read as non-zero yaw?
    pts_front, pts_15 = project(0, 0), project(15, 0)
    y0, _, _ = solve_head_pose(pts_front, W, H)
    y15, _, _ = solve_head_pose(pts_15, W, H)
    assert abs(y0) < 2.0 and (y15 - y0) > 10.0, (y0, y15)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
