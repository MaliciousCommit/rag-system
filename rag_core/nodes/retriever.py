# rag_core/nodes/retriever.py

"""
Retriever Node — Two-Stage Document Retrieval

STAGE 1 — Vector Search (Qdrant):
    Converts the query to an embedding vector and finds the most
    semantically similar document chunks in the vector database.
    Fast but imperfect — returns top-K candidates.

STAGE 2 — Reranking (FlashRank):
    Takes the top-K candidates from Stage 1 and re-scores them
    using a cross-encoder model (more accurate than vector similarity).
    Returns the top-N truly relevant documents.

WHY TWO STAGES?
    Vector search is approximate — cosine similarity captures broad semantics
    but misses fine-grained relevance. The reranker is a cross-encoder: it reads
    BOTH the query and document together, giving much more accurate scores.

    But cross-encoders are slow — you can't run them on your entire database.
    So: use fast vector search to get 20 candidates, then use the slow
    cross-encoder to pick the best 5. Best of both worlds.

    This is what separates production RAG from toy RAG.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest

from rag_core.state import GraphState

load_dotenv()
logger = logging.getLogger(__name__)


# ─── Singleton Clients ────────────────────────────────────────────────────────
# WHY module-level singletons?
#   Loading a SentenceTransformer model takes 2-5 seconds.
#   If we loaded it inside the function, EVERY query would pay that cost.
#   By loading once at module import time, subsequent calls are instant.
#   This is called the "singleton pattern" — one instance, shared everywhere.

_embedding_model = None
_qdrant_client = None
_reranker = None


def get_embedding_model() -> SentenceTransformer:
    """Load embedding model once and cache it."""
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        logger.info(f"[Retriever] Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        logger.info("[Retriever] Embedding model loaded ✅")
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client once and cache it."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        logger.info("[Retriever] Qdrant client initialized ✅")
    return _qdrant_client


def get_reranker() -> Ranker:
    """Initialize FlashRank reranker once and cache it."""
    global _reranker
    if _reranker is None:
        logger.info("[Retriever] Loading FlashRank reranker...")
        # ms-marco-MiniLM-L-12-v2 is the best balance of speed vs accuracy
        # It's a cross-encoder trained on Microsoft's MARCO passage ranking dataset
        _reranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank"
        )
        logger.info("[Retriever] Reranker loaded ✅")
    return _reranker


def embed_query(query: str) -> List[float]:
    """
    Convert query text to embedding vector.

    WHY BGE (BAAI/bge-small-en-v1.5)?
        - Free, runs locally (no API cost per embedding)
        - 384 dimensions — small enough to be fast, large enough to be accurate
        - Optimized for retrieval tasks (BGE = Beijing Academy of AI, Embedding)
        - "bge-small" = 33M params, runs on CPU in ~10ms

    The query_prefix is important for BGE models — it improves retrieval quality.
    """
    model = get_embedding_model()

    # BGE models expect this prefix for query encoding (not document encoding)
    # This is documented in the BGE model card on HuggingFace
    query_with_prefix = (
        f"Represent this sentence for searching relevant passages: {query}"
    )

    embedding = model.encode(query_with_prefix, normalize_embeddings=True)
    return embedding.tolist()


def vector_search(query_vector: List[float], top_k: int = 20) -> List[dict]:
    """
    Stage 1: Search Qdrant for semantically similar documents.
    Uses query_points() API (qdrant-client >= 1.7.0).
    """
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=0.2,
        )

        docs = []
        for hit in response.points:
            # ✅ Defensive guard — payload typed as dict | None
            payload = hit.payload or {}

            docs.append(
                {
                    "id": str(hit.id),
                    "text": payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                }
            )

        logger.info(f"[Retriever] Qdrant returned {len(docs)} candidates")
        return docs

    except Exception as e:
        logger.error(f"[Retriever] Qdrant search failed: {e}")
        return []


def rerank_documents(query: str, docs: List[dict], top_n: int = 8) -> List[dict]:
    """
    Stage 2: Rerank documents using FlashRank cross-encoder.

    Args:
        query: Original user query (NOT the embedded version)
        docs: Candidate documents from vector search
        top_n: How many to keep after reranking

    Returns:
        Top-N most relevant documents, reordered by cross-encoder score

    WHY top_n=8?
        The Responder's context window isn't infinite.
        8 high-quality chunks give enough context without overwhelming the LLM.
        More isn't always better — too many chunks = confused LLM.
    """
    if not docs:
        logger.warning("[Retriever] No docs to rerank")
        return []

    ranker = get_reranker()

    # FlashRank expects passages as list of dicts with "id" and "text"
    passages = [{"id": doc["id"], "text": doc["text"]} for doc in docs]

    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)

    # reranked is ordered best-first, take top_n
    top_docs = []
    for item in reranked[:top_n]:
        # Find the original doc to preserve its metadata
        original = next((d for d in docs if d["id"] == item.get("id")), None)
        if original:
            top_docs.append(
                {
                    **original,
                    "rerank_score": item.get("score", 0.0),
                }
            )

    logger.info(f"[Retriever] Reranked {len(docs)} → kept top {len(top_docs)}")
    return top_docs


def retriever_node(state: GraphState) -> dict:
    """
    Retriever Node — Orchestrates two-stage retrieval.

    Args:
        state: GraphState with 'query' field populated

    Returns:
        dict with 'retrieved_docs' and 'reranked_docs'
    """
    query = state["query"]
    logger.info(f"[Retriever] Processing query: '{query}'")

    # STAGE 1: Embed query and search Qdrant
    query_vector = embed_query(query)
    retrieved_docs = vector_search(query_vector, top_k=20)

    # Handle empty retrieval gracefully
    if not retrieved_docs:
        logger.warning("[Retriever] No documents found in Qdrant")
        return {
            "retrieved_docs": [],
            "reranked_docs": [],
            "metadata": {"retrieval_count": 0, "rerank_count": 0},
        }

    # STAGE 2: Rerank candidates with FlashRank
    reranked_docs = rerank_documents(query, retrieved_docs, top_n=8)

    return {
        "retrieved_docs": retrieved_docs,
        "reranked_docs": reranked_docs,
        "metadata": {
            "retrieval_count": len(retrieved_docs),
            "rerank_count": len(reranked_docs),
            "top_rerank_score": reranked_docs[0]["rerank_score"]
            if reranked_docs
            else 0,
        },
    }
