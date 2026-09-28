from .guided_enroll import *

class Clock:
    t = 0.0
    def __call__(self): return self.t

def R(yaw=0.0, pitch=0.0, **k):
    d = dict(face_found=True, face_frac=0.4, sharpness=120, brightness=120, n_faces=1)
    d.update(k); return PoseReading(yaw=yaw, pitch=pitch, **d)

def hold(s, r, n=8):
    return [s.update(r) for _ in range(n)]

def run_calibration(s, clock, sweep):
    """Feed a calibration sweep (list of (yaw,pitch)) until the phase advances.
    Stops the instant it does, so leftover sweep entries don't leak into POSING
    and pre-build a streak there (harmless in real use, just noisy for tests)."""
    while s.phase == Phase.CALIBRATING:
        for yaw, pitch in sweep:
            s.update(R(yaw, pitch))
            if s.phase != Phase.CALIBRATING:
                return s
        clock.t += 0.3
    return s

def test_calibration_then_full_flow_moderate_turner():
    # this user only turns ~18 degrees in real use -- the OLD fixed-threshold
    # design (yaw_thresh ~0.12 ratio, tuned on someone else) is exactly the
    # kind of case that used to silently fail. Adaptive gating must still work.
    c = Clock(); cfg = Config(calibration_seconds=1.0)
    s = EnrollmentSession(cfg, reliable_pose=True, clock=c)
    run_calibration(s, c, [(0, 0), (-18, 0), (18, 0), (0, -14), (0, 14), (0, 0)])
    assert s.phase == Phase.POSING
    assert 8.0 <= s.yaw_gate <= 35.0 and 6.0 <= s.pitch_gate <= 25.0
    for r in [R(0.5, 0), R(-18, 0), R(18, 0), R(0, -14), R(0, 14), R(0, 0)]:
        assert hold(s, r)[-1] is True
    assert s.status == "complete" and len(s.captured) == 6

def test_small_turn_does_not_falsely_pass_as_left():
    # a turn smaller than the gate must NOT register -- gate isn't just "any nonzero yaw"
    c = Clock(); cfg = Config(calibration_seconds=1.0)
    s = EnrollmentSession(cfg, reliable_pose=True, clock=c)
    run_calibration(s, c, [(0, 0), (-20, 0), (20, 0), (0, -15), (0, 15)])
    hold(s, R(0, 0))                          # captures FRONT
    assert s.current == Pose.LEFT
    assert not any(hold(s, R(-2, 0)))         # barely turned - should NOT satisfy LEFT
    assert s.current == Pose.LEFT

def test_person_who_barely_moves_gets_floor_threshold_not_zero():
    # if calibration range is tiny, the gate must clamp to the floor, not collapse
    # to ~0 (which would false-trigger on head jitter / camera noise)
    c = Clock(); cfg = Config(calibration_seconds=1.0)
    s = EnrollmentSession(cfg, reliable_pose=True, clock=c)
    run_calibration(s, c, [(0, 0), (1, 0), (-1, 0)])
    assert s.yaw_gate == cfg.min_yaw_gate_deg

def test_degraded_mode_when_pose_unreliable():
    cfg = Config(hold_frames=3, degraded_capture_count=4, degraded_min_gap_s=0.0)
    s = EnrollmentSession(cfg, reliable_pose=False, clock=Clock())
    assert s.phase == Phase.DEGRADED
    caps = 0
    for _ in range(4):
        assert not any(hold(s, R(), n=2))
        caps += hold(s, R(), n=1)[-1]
    # keep feeding until done (frames needed = hold_frames per capture x captures)
    while s.status == "running":
        if s.update(R()): pass
    assert s.status == "complete" and len(s.captured) == cfg.degraded_capture_count

def test_streak_resets_on_bad_quality_during_posing():
    c = Clock(); cfg = Config(calibration_seconds=1.0)
    s = EnrollmentSession(cfg, reliable_pose=True, clock=c)
    run_calibration(s, c, [(0, 0), (-20, 0), (20, 0), (0, -15), (0, 15)])
    hold(s, R(0, 0), 5)
    s.update(R(0, 0, sharpness=5))
    assert s.streak == 0 and "blurry" in s.message

def test_no_camera_without_consent():
    assert run_guided_enrollment(False).reason == "consent not granted"

def test_wrong_pose_does_not_capture_and_resets_streak():
    c = Clock(); cfg = Config(calibration_seconds=1.0)
    s = EnrollmentSession(cfg, reliable_pose=True, clock=c)
    run_calibration(s, c, [(0, 0), (-20, 0), (20, 0), (0, -15), (0, 15)])
    hold(s, R(0, 0))
    assert s.current == Pose.LEFT
    assert not any(hold(s, R(20, 0)))          # turned the WRONG way
    assert s.current == Pose.LEFT

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
