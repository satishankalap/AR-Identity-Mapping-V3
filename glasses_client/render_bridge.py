"""
Render bridge — the seam between this pipeline and whatever glasses SDK is
actually driving the display.

Per the doc's hardware compatibility notes:
  - Android XR (Samsung Galaxy Glasses, XREAL): Jetpack Compose XR's
    Glanceable Information Architecture APIs render the card and expose
    gaze-duration triggers natively — an AndroidXRRenderBridge would call
    into those instead of doing the dwell timing itself.
  - Vuzix / Magic Leap 2 / Brilliant Labs Frame: open Android-based SDKs,
    draw directly via their display APIs.
  - Meta Ray-Ban / Display glasses: mwdat-display for rendering,
    mwdat-camera's single-frame-trigger mode instead of continuous capture.

This file ships a ConsoleRenderBridge so the pipeline is fully testable
without any hardware attached. Swapping hardware means writing one new
class that implements the same three methods below — nothing in
pipeline.py, state_machine.py, or matching_api.py changes.
"""
from __future__ import annotations

from typing import Protocol


class RenderBridge(Protocol):
    def show_targeting_ring(self, target_id: str, progress_pct: float) -> None: ...
    def show_matching_indicator(self, target_id: str) -> None: ...
    def show_anchored_card(self, target_id: str, name: str, role: str | None, company: str | None) -> None: ...
    def clear(self, target_id: str) -> None: ...


class ConsoleRenderBridge:
    """Reference implementation for local dev/demo — logs what would be drawn."""

    def show_targeting_ring(self, target_id: str, progress_pct: float) -> None:
        print(f"[HUD] targeting ring on {target_id}: {progress_pct:.0f}%")

    def show_matching_indicator(self, target_id: str) -> None:
        print(f"[HUD] pulsing match indicator on {target_id}")

    def show_anchored_card(self, target_id: str, name: str, role: str | None, company: str | None) -> None:
        line2 = " · ".join(x for x in (role, company) if x)
        print(f"[HUD] card anchored on {target_id}: {name}" + (f" — {line2}" if line2 else ""))

    def clear(self, target_id: str) -> None:
        print(f"[HUD] cleared {target_id}")


class OpenCVRenderBridge:
    """NEW in v3.1 - laptop stand-in for the glasses display. The pipeline calls the
    same four methods; this one just remembers what should be on screen, and draw()
    paints it onto the webcam frame. On real glasses/phone you replace this class
    with one that calls the XR SDK - nothing else changes."""

    def __init__(self):
        self.ring_pct = 0.0
        self.matching = False
        self.card = None            # (name, role, company) once anchored

    def show_targeting_ring(self, target_id, progress_pct):
        self.ring_pct, self.matching = progress_pct, False

    def show_matching_indicator(self, target_id):
        self.matching, self.ring_pct = True, 100.0

    def show_anchored_card(self, target_id, name, role, company):
        self.card, self.matching, self.ring_pct = (name, role, company), False, 0.0

    def clear(self, target_id):
        self.ring_pct, self.matching, self.card = 0.0, False, None

    # ---- drawing (needs cv2; imported lazily so the module stays stdlib-only otherwise)
    def draw(self, frame, box, tick: float = 0.0):
        import cv2, math
        if box is None:
            return
        x, y, w, h = box
        cx, cy = x + w // 2, y + h // 2
        if self.card:
            name, role, company = self.card
            lines = [name, " · ".join(v for v in (role, company) if v)]
            cw, ch = max(220, 12 * max(len(l) for l in lines)), 24 * len(lines) + 16
            px = min(x + w + 14, frame.shape[1] - cw - 4); py = max(y, 4)
            ov = frame.copy()
            cv2.rectangle(ov, (px, py), (px + cw, py + ch), (30, 30, 30), -1)
            cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 100), 2)
            cv2.line(frame, (x + w, cy), (px, py + ch // 2), (80, 220, 100), 1)
            cv2.putText(frame, lines[0], (px + 8, py + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if lines[1]:
                cv2.putText(frame, lines[1], (px + 8, py + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
        elif self.matching:
            r = int(max(w, h) * 0.6 + 6 * math.sin(tick * 10))          # pulsing ring
            cv2.circle(frame, (cx, cy), r, (0, 200, 255), 3)
        elif self.ring_pct > 0:
            r = int(max(w, h) * 0.6)
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, 3.6 * self.ring_pct, (0, 200, 255), 4)
