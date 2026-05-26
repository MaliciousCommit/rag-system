# observability/logfire_config.py

"""
Pydantic Logfire Configuration — Span Tracing

WHY LOGFIRE?
    While LangSmith traces the AGENT (what nodes ran),
    Logfire traces the API (what HTTP requests came in).

    Together they give you the full picture:

    LangSmith:   Planner(50ms) → Retriever(800ms) → Responder(1200ms)
    Logfire:     POST /query → 2100ms → 200 OK

    Logfire also traces:
    - FastAPI request/response cycles
    - Database calls
    - External HTTP calls (to Qdrant, Groq)
    - Custom spans you add manually

HOW IT WORKS:
    logfire.instrument_fastapi(app) → auto-traces all endpoints
    logfire.instrument_httpx()      → auto-traces HTTP calls
    with logfire.span("my_op"):     → manual custom spans
"""

import os
import logging
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
logger = logging.getLogger(__name__)

# Track if logfire is configured
_logfire_configured = False


def setup_logfire(app=None) -> bool:
    """
    Configure Pydantic Logfire tracing.

    Args:
        app: FastAPI app instance (for auto-instrumentation)

    Returns:
        True if configured successfully, False if skipped
    """
    global _logfire_configured

    token = os.getenv("LOGFIRE_TOKEN")

    if not token:
        logger.warning("[Logfire] LOGFIRE_TOKEN not set — span tracing disabled")
        return False

    try:
        import logfire

        # ── Configure Logfire ──────────────────────────────────────────────────
        logfire.configure(
            token=token,
            service_name="rag-system",
            service_version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "development"),
        )

        # ── Auto-instrument FastAPI ────────────────────────────────────────────
        # Automatically creates spans for every HTTP request
        # You see: method, path, status code, duration
        if app is not None:
            logfire.instrument_fastapi(app)
            logger.info("[Logfire] FastAPI instrumented ✅")

        # ── Auto-instrument HTTPX ──────────────────────────────────────────────
        # Traces all outbound HTTP calls (to Qdrant, Groq, HuggingFace)
        # You see: which external services are called, how long they take
        logfire.instrument_httpx()
        logger.info("[Logfire] HTTPX instrumented ✅")

        _logfire_configured = True
        logger.info("[Logfire] Span tracing enabled ✅")
        logger.info("[Logfire] View spans at: https://logfire.pydantic.dev")
        return True

    except ImportError:
        logger.warning("[Logfire] logfire package not installed")
        return False
    except Exception as e:
        logger.error(f"[Logfire] Setup failed: {e}")
        return False


def create_span(name: str, attributes: Optional[dict] = None):
    """
    Create a manual Logfire span for custom operations.

    Usage:
        with create_span("qdrant_search", {"query": "refund policy"}):
            results = client.query_points(...)

    WHY MANUAL SPANS?
        Auto-instrumentation covers HTTP calls.
        Manual spans cover YOUR business logic:
        - How long does chunking take?
        - How long does reranking take?
        - Which queries take longest?
    """
    global _logfire_configured

    if not _logfire_configured:
        # Return a no-op context manager if logfire not configured
        from contextlib import nullcontext

        return nullcontext()

    try:
        import logfire

        return logfire.span(name, **(attributes or {}))
    except Exception:
        from contextlib import nullcontext

        return nullcontext()
