"""
End-to-end demo of the Phase 3 pipeline, self-contained (no live Phase 2
server required) so it runs anywhere with just the standard library.

It uses a FakeMatchingAPIClient that mimics the real MatchingAPIClient's
interface but matches in-process against a tiny registered set, using the
same embedding stub as the "camera." Point PipelineDemo at a real
MatchingAPIClient(base_url="http://localhost:8000") instead to run against
the actual Phase 2 service once it's deployed — nothing else changes.

Run:  python -m glasses_client.demo
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from .embedding import extract as extract_embedding
from .matching_api import MatchedProfile
from .pipeline import GazePipeline
from .render_bridge import ConsoleRenderBridge
from .state_machine import DWELL_MS


@dataclass
class _FakeProfile:
    profile_id: str
    name: str
    role: str
    company: str
    embedding: List[float]


class FakeMatchingAPIClient:
    """Duck-types MatchingAPIClient.match() against an in-memory roster."""

    def __init__(self):
        self._roster: Dict[str, _FakeProfile] = {}

    def register(self, external_id: str, name: str, role: str, company: str) -> None:
        # Registration embedding is derived from external_id, same as how the
        # demo's "camera" derives a query embedding from whatever it's looking at.
        self._roster[external_id] = _FakeProfile(
            profile_id=external_id,
            name=name,
            role=role,
            company=company,
            embedding=extract_embedding(external_id),
        )

    def match(self, event_id: str, embedding: List[float], top_k: int = 1,
              min_similarity: float = 0.85) -> List[MatchedProfile]:
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb) if na and nb else 0.0

        scored = sorted(
            ((cosine(embedding, p.embedding), p) for p in self._roster.values()),
            key=lambda t: t[0], reverse=True,
        )
        return [
            MatchedProfile(p.profile_id, p.name, p.role, p.company, round(sim, 4))
            for sim, p in scored[:top_k] if sim >= min_similarity
        ]


def run_demo() -> None:
    matching_client = FakeMatchingAPIClient()
    matching_client.register("attendee-042", "Sarah Chen", "Lead Spatial Architect", "Acme Inc.")
    matching_client.register("attendee-017", "Marcus Reyes", "VP Product", "Nimbus Labs")

    pipeline = GazePipeline(
        event_id="arxr2026",
        matching_client=matching_client,
        render_bridge=ConsoleRenderBridge(),
    )

    print("--- Scenario 1: registered attendee, full dwell ---")
    # Keep hitting the same target until the dwell timer completes (or we
    # give up after a generous cap) — real sensor ticks aren't evenly spaced
    # either, so driving off elapsed time rather than a fixed tick count
    # matches how this behaves on real hardware.
    _run_until_settled(pipeline, "attendee-042")
    pipeline.on_gaze_lost()
    # Anchored cards persist independent of gaze (per spec) — simulate the
    # wearer's attention actually moving to someone new before Scenario 2.
    pipeline.on_target_walked_away("attendee-042")

    print("\n--- Scenario 2: unregistered face, fails silently ---")
    _run_until_settled(pipeline, "unknown-person")
    pipeline.on_gaze_lost()

    print("\n--- Scenario 3: quick saccade, never triggers ---")
    pipeline.on_gaze_hit("attendee-017", frame="attendee-017")
    pipeline.on_gaze_lost()  # gone before the saccade floor even elapses
    print("(ring flashes briefly, but note: no 'matching indicator' or 'card anchored' line above — the saccade filter blocked the dwell from ever completing)")


def _run_until_settled(pipeline: GazePipeline, target_id: str, tick_s: float = 0.08, max_ticks: int = 20) -> None:
    from .state_machine import State
    for _ in range(max_ticks):
        pipeline.on_gaze_hit(target_id, frame=target_id)
        if pipeline.sm.state in (State.ANCHORED, State.STANDBY):
            break
        time.sleep(tick_s)


if __name__ == "__main__":
    run_demo()
