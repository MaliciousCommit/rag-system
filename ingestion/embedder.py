# ingestion/embedder.py

"""
Embedder — Convert Chunks to Vectors and Upload to Qdrant

This is the final step in the ingestion pipeline.
Takes chunks from the Chunker, embeds them with HuggingFace,
and upserts them into Qdrant Cloud.

BATCHING STRATEGY:
    We embed in batches of 64 (not one at a time).
    WHY?
        sentence-transformers is optimized for batch processing.
        Batch of 64 is ~10x faster than 64 individual calls.
        For 1000 chunks: ~2 seconds vs ~20 seconds.
"""

import os
import uuid
import logging
from typing import List
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Singletons — load once, reuse across calls
_embedding_model = None
_qdrant_client = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        logger.info(f"[Embedder] Loading model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    return _qdrant_client


def ensure_collection_exists(collection_name: str, vector_size: int) -> None:
    """Create Qdrant collection if it doesn't already exist."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            f"[Embedder] Created collection '{collection_name}' ({vector_size}d)"
        )
    else:
        logger.info(f"[Embedder] Collection '{collection_name}' already exists ✅")


def embed_and_upload(chunks: List[dict], batch_size: int = 64) -> int:
    """
    Embed chunks and upload to Qdrant in batches.

    Args:
        chunks:     Output from chunker.chunk_documents()
        batch_size: How many chunks to embed at once

    Returns:
        Number of chunks successfully uploaded
    """
    if not chunks:
        logger.warning("[Embedder] No chunks to embed")
        return 0

    model = get_embedding_model()
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")

    # Get vector dimension and ensure collection exists
    vector_size = model.get_sentence_embedding_dimension()
    if vector_size is None:
        vector_size = len(model.encode("test", normalize_embeddings=True))
    ensure_collection_exists(collection_name, vector_size)

    total_uploaded = 0

    # Process in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        batch_texts = [chunk["text"] for chunk in batch]

        # Embed entire batch at once — much faster than one by one
        logger.info(
            f"[Embedder] Embedding batch {batch_start // batch_size + 1} "
            f"({len(batch)} chunks)..."
        )
        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Build Qdrant points
        points = []
        for chunk, embedding in zip(batch, embeddings):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "file_type": chunk["file_type"],
                        "page": chunk["page"],
                        "section": chunk.get("section", ""),
                        "slide_title": chunk.get("slide_title", ""),
                        "chunk_index": chunk["chunk_index"],
                    },
                )
            )

        # Upload batch to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        total_uploaded += len(points)
        logger.info(f"[Embedder] Uploaded {total_uploaded}/{len(chunks)} chunks")

    logger.info(f"[Embedder] ✅ Done — {total_uploaded} chunks in Qdrant")
    return total_uploaded
