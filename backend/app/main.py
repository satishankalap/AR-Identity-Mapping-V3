"""
Phase 2 — Matching engine API.

Endpoints:
  POST /profiles          register a profile for an event (marker, BLE, or opt-in face embedding)
  POST /match              query an embedding against an event's index
  GET  /events/{id}/size   debug: how many vectors are currently indexed for an event
  POST /admin/sweep-now    debug: force the auto-expiry sweep instead of waiting on the timer

Run it:
  pip install -r requirements.txt
  uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException

from .expiry import start_background_sweeper, sweep_now, watch_event
from .models import MatchQuery, MatchResponse, RegisterProfile, ProfileRecord
from .vector_store import store


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_background_sweeper(interval_seconds=30)
    yield


app = FastAPI(title="AR Identity Mapping — Matching Engine", version="0.1.0", lifespan=lifespan)


@app.post("/profiles", response_model=ProfileRecord)
def register_profile(payload: RegisterProfile):
    if payload.ident_method.value == "face" and not payload.consent_given:
        # Enforced here, not just documented — a face_embedding profile
        # cannot exist in this system without explicit consent on the record.
        raise HTTPException(status_code=400, detail="consent_given must be true for face_embedding profiles")

    if payload.event_ends_at <= datetime.utcnow():
        raise HTTPException(status_code=400, detail="event_ends_at must be in the future")

    record = store.register(
        event_id=payload.event_id,
        external_id=payload.external_id,
        name=payload.name,
        role=payload.role,
        company=payload.company,
        ident_method=payload.ident_method,
        embedding=payload.embedding,
        event_ends_at=payload.event_ends_at,
    )
    watch_event(payload.event_id, payload.event_ends_at)
    return record


@app.post("/match", response_model=MatchResponse)
def match(query: MatchQuery):
    results = store.match(
        event_id=query.event_id,
        embedding=query.embedding,
        top_k=query.top_k,
        min_similarity=query.min_similarity,
    )
    # Deliberately returns an empty list rather than a 404 on no match —
    # per the spec's edge-case handling, an unmatched face fails silently,
    # it never surfaces as an error to the wearer.
    return MatchResponse(matches=results)


@app.get("/events/{event_id}/size")
def event_size(event_id: str):
    return {"event_id": event_id, "vectors_indexed": store.event_size(event_id)}


@app.post("/admin/sweep-now")
def admin_sweep_now():
    sweep_now()
    return {"status": "swept"}
