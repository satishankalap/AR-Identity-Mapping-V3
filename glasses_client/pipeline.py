"""
Wires state_machine + embedding + matching_api + render_bridge into the on-demand
identification loop:   gaze hit -> dwell -> single frame capture -> vector match -> anchored card

v3.1 CHANGES vs v3.0
  * on_gaze_hit(target_id, frame) now accepts a REAL image (numpy BGR array) or a
    zero-arg callable returning one. The callable is only invoked at the moment the
    dwell completes, so the camera is read exactly once per trigger (battery strategy).
    A plain string still works and uses the old hash stub, so demo.py is unchanged.
  * Real embedding: embedding_v2.extract() (Haar detect + LBP). "No face in frame"
    -> silent fail, same path as "no match".
  * Targeting ring shows REAL dwell progress (was hard-coded 50%).
  * min_similarity is configurable; debug_scores=True asks the server for the best
    score even below threshold (so you can SEE near-misses while calibrating) and
    applies the threshold locally instead.
  * Records last_similarity / last_latency_ms for the debug HUD.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Union

import numpy as np

from .embedding import extract as stub_extract            # legacy string-based stub
from .embedding_v2 import extract as real_extract         # real image-based pipeline
from .matching_api import DEFAULT_MIN_SIMILARITY, MatchingAPIClient, MatchingAPIError
from .render_bridge import RenderBridge
from .state_machine import GazeStateMachine, State

Frame = Union[str, np.ndarray, Callable[[], np.ndarray]]


class GazePipeline:
    def __init__(self, event_id: str, matching_client: MatchingAPIClient,
                 render_bridge: RenderBridge, min_similarity: float = DEFAULT_MIN_SIMILARITY,
                 debug_scores: bool = False):
        self.event_id = event_id
        self.matching_client = matching_client
        self.render = render_bridge
        self.min_similarity = min_similarity
        self.debug_scores = debug_scores
        self.sm = GazeStateMachine(on_state_change=self._on_state_change)
        self._current_target: Optional[str] = None
        self.last_similarity: Optional[float] = None
        self.last_latency_ms: Optional[float] = None
        self.last_note: str = ""

    def _on_state_change(self, old: State, new: State) -> None:
        if new == State.STANDBY and self._current_target:
            self.render.clear(self._current_target)
            self._current_target = None

    def on_gaze_hit(self, target_id: str, frame: Frame) -> None:
        self._current_target = target_id
        prev = self.sm.state
        self.sm.gaze_hit(target_id)
        if self.sm.state == State.TARGETING:
            self.render.show_targeting_ring(target_id, self.sm.dwell_progress_pct())
        if prev != State.MATCHING and self.sm.state == State.MATCHING:
            self._run_match(target_id, frame)

    def on_gaze_lost(self) -> None:
        self.sm.gaze_lost()

    def on_target_walked_away(self, target_id: str) -> None:
        if target_id == self._current_target:
            self.sm.target_walked_away()

    def _embed(self, frame) -> Optional[list]:
        if isinstance(frame, str):
            return stub_extract(frame)
        return real_extract(frame)          # None when no face is found

    def _run_match(self, target_id: str, frame: Frame) -> None:
        self.render.show_matching_indicator(target_id)
        t0 = time.monotonic()
        image = frame() if callable(frame) else frame       # <- the ONE camera read
        embedding = self._embed(image)
        if embedding is None:
            self.last_similarity, self.last_note = None, "no face found in captured frame"
            self.sm.match_failed_or_timed_out()
            return
        try:
            server_thr = 0.0 if self.debug_scores else self.min_similarity
            results = self.matching_client.match(self.event_id, embedding, top_k=1,
                                                 min_similarity=server_thr)
        except MatchingAPIError as e:
            self.last_similarity, self.last_note = None, f"backend error: {e}"
            self.sm.match_failed_or_timed_out()
            return
        finally:
            self.last_latency_ms = (time.monotonic() - t0) * 1000

        self.last_similarity = results[0].similarity if results else None
        if not results or results[0].similarity < self.min_similarity:
            self.last_note = "no match above threshold"
            self.sm.match_failed_or_timed_out()          # spec: silent fail
            return
        top = results[0]
        self.last_note = f"matched {top.name}"
        self.sm.match_succeeded()
        self.render.show_anchored_card(target_id, top.name, top.role, top.company)
