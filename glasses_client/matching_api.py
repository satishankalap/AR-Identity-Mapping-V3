"""
HTTP client for the Phase 2 matching engine — stdlib only, so it runs on a
paired phone's Python runtime (or ports trivially to Kotlin/Swift) without
needing a dependency manager for a single API call.

This is the piece that changes when Phase 2's storage backend changes
(FAISS/Milvus swap, auth, etc.) — nothing here or in the state machine or
pipeline should need to change, only the URL and any auth headers.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

# Threshold you tuned on your own machine (FAR/FRR run). Used as the client default.
DEFAULT_MIN_SIMILARITY = 0.84


@dataclass
class MatchedProfile:
    profile_id: str
    name: str
    role: Optional[str]
    company: Optional[str]
    similarity: float


class MatchingAPIError(Exception):
    pass


class MatchingAPIClient:
    def __init__(self, base_url: str, timeout_s: float = 0.7):
        # timeout defaults to MATCH_TIMEOUT_MS from state_machine.py — a slow
        # backend should look like "no match" to the wearer, not a hang.
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def match(self, event_id: str, embedding: List[float], top_k: int = 1,
              min_similarity: float = DEFAULT_MIN_SIMILARITY) -> List[MatchedProfile]:
        payload = json.dumps({
            "event_id": event_id,
            "embedding": embedding,
            "top_k": top_k,
            "min_similarity": min_similarity,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/match",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            raise MatchingAPIError(str(e)) from e

        return [
            MatchedProfile(
                profile_id=m["profile"]["profile_id"],
                name=m["profile"]["name"],
                role=m["profile"].get("role"),
                company=m["profile"].get("company"),
                similarity=m["similarity"],
            )
            for m in body.get("matches", [])
        ]

    def register_profile(self, event_id: str, external_id: str, name: str,
                         role: Optional[str], company: Optional[str],
                         embedding: List[float], consent_given: bool,
                         event_ends_at: Optional[datetime] = None,
                         timeout_s: float = 5.0) -> dict:
        """POST /profiles for ONE embedding (call once per enrolled view).
        NEW in v3.1. The backend also enforces consent (HTTP 400) - this client-side
        check just fails faster and never sends a biometric without consent."""
        if not consent_given:
            raise MatchingAPIError("refusing to register a face embedding without consent")
        ends = event_ends_at or (datetime.utcnow() + timedelta(hours=8))
        payload = json.dumps({
            "event_id": event_id, "external_id": external_id, "name": name,
            "role": role, "company": company, "ident_method": "face",
            "embedding": embedding, "consent_given": True,
            "event_ends_at": ends.replace(microsecond=0).isoformat(),   # naive UTC, as backend expects
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/profiles", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise MatchingAPIError(f"HTTP {e.code}: {e.read().decode(errors='ignore')[:200]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise MatchingAPIError(str(e)) from e

    def event_size(self, event_id: str, timeout_s: float = 2.0) -> int:
        """GET /events/{id}/size - handy to confirm enrollment landed."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/events/{event_id}/size", timeout=timeout_s) as r:
                return json.loads(r.read())["vectors_indexed"]
        except (urllib.error.URLError, TimeoutError) as e:
            raise MatchingAPIError(str(e)) from e
