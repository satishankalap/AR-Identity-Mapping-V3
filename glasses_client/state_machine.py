"""
State machine for on-demand identification: Standby -> Targeting -> Matching -> Anchored.

Timings match the spec exactly so the phone/glasses split can be tuned without
touching this file:
  - saccade filter:     gaze must hold on a target for > SACCADE_FLOOR_MS
                         before Targeting even begins (spec: rapid eye
                         movements under 150ms never trigger anything)
  - dwell timer:         DWELL_MS of continued gaze -> fires a single-frame capture
  - match window:        MATCH_TIMEOUT_MS to get a result back before giving up
  - anchor persistence:  once Anchored, the camera goes back to idle; only
                         low-power spatial tracking (SLAM) keeps the card pinned

This class has no camera, network, or rendering code in it on purpose — it only
decides *when* those things should happen. Wire it to real sensors via the
callbacks passed into `GazePipeline` (see pipeline.py).
"""
from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable, Optional


SACCADE_FLOOR_MS = 150   # gaze holds shorter than this are ignored entirely
DWELL_MS = 500           # continuous gaze required before a frame is captured
MATCH_TIMEOUT_MS = 700   # how long to wait for a match result before reverting


class State(Enum):
    STANDBY = auto()     # camera idle, only low-power gaze/head-pose tracking runs
    TARGETING = auto()   # gaze has held on a target, dwell ring is filling
    MATCHING = auto()    # single frame captured, waiting on the match result
    ANCHORED = auto()    # matched — camera off again, spatial anchor holds the card


class GazeStateMachine:
    def __init__(
        self,
        on_state_change: Optional[Callable[[State, State], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.state = State.STANDBY
        self._target_since: Optional[float] = None
        self._on_state_change = on_state_change
        self._clock = clock

    def _transition(self, new_state: State) -> None:
        if new_state == self.state:
            return
        old = self.state
        self.state = new_state
        if self._on_state_change:
            self._on_state_change(old, new_state)

    def gaze_hit(self, target_id: str) -> None:
        """Call every tick the gaze ray intersects a head/face bounding box."""
        now = self._clock()
        if self.state == State.STANDBY:
            self._target_since = now
            self._transition(State.TARGETING)
            return

        if self.state == State.TARGETING:
            held_ms = (now - self._target_since) * 1000
            if held_ms < SACCADE_FLOOR_MS:
                return  # too short to count as intentional yet
            if held_ms >= DWELL_MS:
                self._transition(State.MATCHING)

    def dwell_progress_pct(self) -> float:
        """0-100 while TARGETING (drives the real targeting ring); 0 otherwise."""
        if self.state != State.TARGETING or self._target_since is None:
            return 0.0
        return min(100.0, (self._clock() - self._target_since) * 1000 / DWELL_MS * 100)

    def gaze_lost(self) -> None:
        """Call when the gaze ray no longer intersects any target."""
        if self.state in (State.TARGETING,):
            self._target_since = None
            self._transition(State.STANDBY)
        # Once MATCHING or ANCHORED, losing gaze doesn't revert anything —
        # the spatial anchor persists independent of where the wearer looks next.

    def match_succeeded(self) -> None:
        if self.state == State.MATCHING:
            self._transition(State.ANCHORED)

    def match_failed_or_timed_out(self) -> None:
        # Spec: unregistered attendees fail silently — back to Standby,
        # no error surfaced to the wearer.
        if self.state == State.MATCHING:
            self._target_since = None
            self._transition(State.STANDBY)

    def target_walked_away(self) -> None:
        """Call when depth/SLAM tracking loses the anchored person (occlusion, left the room)."""
        if self.state == State.ANCHORED:
            self._transition(State.STANDBY)
