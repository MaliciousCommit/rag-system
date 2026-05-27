# api/main.py

"""
FastAPI Application — RAG System REST API

ENDPOINTS:
    GET  /health              → System health check
    POST /query               → Main RAG query endpoint
    POST /ingest              → Trigger document ingestion (background)
    DELETE /session/{id}      → Clear conversation history
    GET  /docs                → Auto Swagger UI (built into FastAPI)

WHY FASTAPI?
    1. Async — handles concurrent requests
    2. Pydantic built-in — automatic validation
    3. /docs auto-generated — test without Postman
    4. Type hints — full Pylance support
    5. Industry standard for Python APIs
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager

# from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from observability.langsmith_config import setup_langsmith
from observability.logfire_config import setup_logfire, create_span

from api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    SourceDocument,
)
from gateway.nemo_guard import check_input_safety, get_blocked_response
from rag_core.graph import run_query
from ingestion.pipeline import ingest_file


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── In-Memory Session Store ──────────────────────────────────────────────────
# Stores conversation history per session_id
# Resets on server restart — Phase 7 replaces with Redis
session_store: dict[str, list] = {}


# ─── App Lifecycle ────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("🚀 RAG System API starting up...")

    # ✅ NO pre-loading — saves RAM at startup
    # Models load lazily on first request
    # Railway free tier can't afford startup loading

    langsmith_ok = setup_langsmith()
    logfire_ok = setup_logfire(app)

    logger.info(
        f"[Observability] LangSmith: {'✅' if langsmith_ok else '⚠️ disabled'} | "
        f"Logfire: {'✅' if logfire_ok else '⚠️ disabled'}"
    )

    logger.info("✅ API ready — visit /docs")
    logger.info("=" * 55)

    yield

    logger.info("🛑 Shutting down...")


# ─── App Instance ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG System API",
    description="""
## Production-Grade RAG System

Built with LangGraph + Groq + Qdrant + FlashRank

### Features
- 🧠 **LangGraph** agentic pipeline (Planner → Retriever → Responder)
- 🔍 **Qdrant** vector search + **FlashRank** reranking
- 🛡️ **Guardrails** rule-based safety layer
- 🔀 **Portkey** LLM gateway with automatic fallback
- 📄 Multi-format ingestion (PDF, DOCX, PPTX, HTML)
- 💬 Multi-turn conversation with session management
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allows Streamlit UI (Phase 4) to call this API ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Phase 7: restrict to your Cloud Run URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Cloud Run calls this every 30s to verify container is alive.
    Must return 200 — anything else triggers container restart.
    """
    components: dict[str, str] = {}

    # Check Qdrant connection
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=5,
        )
        client.get_collections()
        components["qdrant"] = "healthy"
    except Exception as e:
        components["qdrant"] = f"unhealthy: {str(e)[:50]}"

    # Check keys exist
    components["groq"] = "configured" if os.getenv("GROQ_API_KEY") else "❌ missing"
    components["portkey"] = (
        "configured" if os.getenv("PORTKEY_API_KEY") else "not configured"
    )

    # Overall status
    status = "healthy" if "❌" not in str(components) else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        components=components,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: QueryRequest):
    """
    Main RAG query endpoint with full observability.

    Flow:
        1. Guardrails check     ← Logfire span
        2. Load session history
        3. Run LangGraph agent  ← Logfire span
        4. Save session
        5. Return response
    """
    start_time = time.time()
    logger.info(f"[API] /query → '{request.query[:60]}'")

    # ── Step 1: Guardrails Check ──────────────────────────────────────────────
    with create_span("guardrails_check", {"query": request.query[:50]}):
        is_safe, reason = check_input_safety(request.query)

    if not is_safe:
        logger.warning(f"[API] Guardrail triggered: {reason}")
        return QueryResponse(
            answer=get_blocked_response(),
            intent="blocked",
            session_id=request.session_id,
            sources=[],
            metadata={"blocked_reason": reason},
            guardrail_triggered=True,
        )

    # ── Step 2: Load Session History ──────────────────────────────────────────
    session_id = request.session_id or "default"
    chat_history = session_store.get(session_id, [])
    logger.info(f"[API] Session '{session_id}' | {len(chat_history) // 2} prior turns")

    # ── Step 3: Run LangGraph Agent ───────────────────────────────────────────
    try:
        with create_span("langgraph_agent", {"session_id": session_id}):
            result = run_query(
                query=request.query,
                chat_history=chat_history,
            )
    except Exception as e:
        logger.error(f"[API] Agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # ── Step 4: Save Session ──────────────────────────────────────────────────
    session_store[session_id] = result.get("chat_history", [])

    # ── Step 5: Format Sources ────────────────────────────────────────────────
    sources = []
    for doc in result.get("reranked_docs", []):
        payload = doc.get("metadata", {})
        sources.append(
            SourceDocument(
                text=doc.get("text", "")[:300],
                source=payload.get("source", "unknown"),
                page=int(payload.get("page", 0)),
                score=round(float(doc.get("rerank_score", 0.0)), 4),
            )
        )

    # ── Step 6: Return Response ───────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 2)
    logger.info(f"[API] ✅ Done in {elapsed}s | intent={result.get('intent')}")

    return QueryResponse(
        answer=result.get("response", ""),
        intent=result.get("intent", "unknown"),
        session_id=session_id,
        sources=sources,
        metadata={
            **result.get("metadata", {}),
            "latency_seconds": elapsed,
            "history_turns": len(chat_history) // 2,
        },
        guardrail_triggered=False,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_endpoint(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger document ingestion in the background.

    WHY background_tasks?
        1000-page PDF takes minutes to ingest.
        Running synchronously = HTTP timeout.
        BackgroundTasks: API returns instantly,
        ingestion continues in background.
        Check server logs to see progress.
    """
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=404, detail=f"File not found: {request.file_path}"
        )

    background_tasks.add_task(
        ingest_file,
        request.file_path,
        request.upload_to_gcs,
    )

    filename = os.path.basename(request.file_path)
    logger.info(f"[API] Ingestion started in background: {filename}")

    return IngestResponse(
        status="started",
        file=filename,
        chunks_uploaded=0,
        message=f"Ingestion started for '{filename}'. Check server logs for progress.",
    )


@app.delete("/session/{session_id}", tags=["Session"])
async def clear_session(session_id: str):
    """Clear conversation history for a specific session."""
    if session_id in session_store:
        del session_store[session_id]
        logger.info(f"[API] Session '{session_id}' cleared")
        return {"message": f"Session '{session_id}' cleared ✅"}
    return {"message": f"Session '{session_id}' not found"}


@app.get("/sessions", tags=["Session"])
async def list_sessions():
    """List all active sessions and their turn counts."""
    return {
        session_id: {
            "turns": len(history) // 2,
            "messages": len(history),
        }
        for session_id, history in session_store.items()
    }


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-restart on code changes (dev only)
        log_level="info",
    )
