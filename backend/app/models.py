"""
Request/response schemas for the Phase 2 matching engine.

Identification method is a first-class field (`IdentMethod`), not an
afterthought — this is what lets marker-based and opt-in face-embedding
identification share one pipeline while keeping the privacy rules
(see README) enforced at the schema level rather than by convention.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, conlist


class IdentMethod(str, Enum):
    marker = "marker"          # QR / AR badge marker — Phase 1, no biometric data
    ble = "ble"                # Bluetooth proximity opt-in
    face_embedding = "face"    # Opt-in biometric vector — gated by consent, see below


class RegisterProfile(BaseModel):
    event_id: str
    external_id: str = Field(..., description="ID from the client app (badge ID, user ID, etc.)")
    name: str
    role: Optional[str] = None
    company: Optional[str] = None
    ident_method: IdentMethod
    # Required only for ident_method == face_embedding. The service never
    # extracts embeddings itself — the client app (or its ML pipeline) does
    # that and sends the resulting vector. This service only stores + matches.
    embedding: Optional[conlist(float, min_length=8)] = None
    consent_given: bool = Field(
        ..., description="Must be true for face_embedding; enforced server-side, not just here."
    )
    event_ends_at: datetime = Field(..., description="Profile (and any embedding) is purged at this time.")


class ProfileRecord(BaseModel):
    profile_id: str
    event_id: str
    external_id: str
    name: str
    role: Optional[str] = None
    company: Optional[str] = None
    ident_method: IdentMethod
    created_at: datetime
    event_ends_at: datetime


class MatchQuery(BaseModel):
    event_id: str
    embedding: conlist(float, min_length=8)
    top_k: int = 1
    min_similarity: float = Field(0.85, ge=0.0, le=1.0)


class MatchResult(BaseModel):
    profile: ProfileRecord
    similarity: float


class MatchResponse(BaseModel):
    matches: List[MatchResult]
