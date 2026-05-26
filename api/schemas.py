# api/schemas.py

"""
API Schemas — Pydantic Request/Response Models

WHY PYDANTIC SCHEMAS?
    FastAPI uses these to:
    1. Auto-validate incoming requests
       Wrong type → 422 error before your code runs
    2. Auto-serialize outgoing responses
    3. Auto-generate /docs Swagger UI
    4. Give IDE autocomplete on all fields
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─── Request Models ───────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Incoming query from client."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question",
        examples=["What is the refund policy?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for multi-turn conversation.",
        examples=["user_123_session_1"],
    )
    top_k: Optional[int] = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )


class IngestRequest(BaseModel):
    """Request to ingest a document."""

    file_path: str = Field(
        ...,
        description="Local path to document",
        examples=["my_documents/policy.pdf"],
    )
    upload_to_gcs: bool = Field(
        default=False,
        description="Archive to Google Cloud Storage",
    )


# ─── Response Models ──────────────────────────────────────────────────────────


class SourceDocument(BaseModel):
    """A single retrieved source chunk."""

    text: str
    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    """Full response returned to client."""

    answer: str = Field(description="Generated answer")
    intent: str = Field(description="rag / chitchat / out_of_scope / blocked")
    session_id: Optional[str] = Field(description="Session ID echoed back")
    sources: list[SourceDocument] = Field(
        default=[],
        description="Source chunks used to answer",
    )
    metadata: dict = Field(
        default={},
        description="Latency, token counts, model info",
    )
    guardrail_triggered: bool = Field(
        default=False,
        description="True if guardrails blocked the query",
    )


class IngestResponse(BaseModel):
    """Response after document ingestion."""

    status: str
    file: str
    chunks_uploaded: int
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: dict
