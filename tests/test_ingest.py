# tests/test_ingest.py

"""
Test Data Ingestion Script

Adds sample documents to Qdrant so we can test the RAG pipeline.
This is NOT the production ingestion pipeline (that's Phase 2).
This is just enough to verify Phase 1 works end-to-end.

Run this ONCE before running main.py:
    python tests/test_ingest.py
"""

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─── Sample Documents ─────────────────────────────────────────────────────────
# These simulate what real ingested documents look like after chunking.
# In Phase 2, these will come from your actual PDFs/DOCX/HTML files.

SAMPLE_DOCUMENTS = [
    {
        "text": """Refund Policy: Customers are eligible for a full refund within 30 days
        of purchase. After 30 days, only store credit is available. Digital products
        are non-refundable once downloaded. To initiate a refund, contact
        support@company.com with your order number.""",
        "source": "policy_handbook.pdf",
        "page": 12,
    },
    {
        "text": """Premium Subscription Benefits: Premium subscribers get access to all
        features including unlimited storage, priority support, and early access to
        new features. Premium costs $29/month or $290/year (save 2 months).
        Cancel anytime from your account settings.""",
        "source": "pricing_guide.pdf",
        "page": 3,
    },
    {
        "text": """Password Reset Instructions: To reset your password, click 'Forgot Password'
        on the login page. Enter your registered email address. You will receive a
        reset link valid for 24 hours. If you don't receive the email, check your
        spam folder or contact support.""",
        "source": "user_manual.pdf",
        "page": 7,
    },
    {
        "text": """Data Privacy: We collect only essential data required to provide our service.
        Your data is encrypted at rest using AES-256 and in transit using TLS 1.3.
        We never sell your personal data to third parties.
        You can request data deletion anytime under GDPR Article 17.""",
        "source": "privacy_policy.pdf",
        "page": 1,
    },
    {
        "text": """Q3 2024 Financial Summary: Total revenue reached $4.2M, up 23% YoY.
        Operating costs were $2.8M. Net profit margin improved to 33%.
        Key growth drivers: premium subscriptions (+45%) and enterprise deals (+67%).
        Q4 guidance: $4.8M-$5.1M revenue expected.""",
        "source": "q3_report.pdf",
        "page": 2,
    },
]


def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        print(f"✅ Collection '{collection_name}' already exists")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,  # Must match embedding model output dimension
            distance=Distance.COSINE,  # Cosine similarity for text embeddings
        ),
    )
    print(f"✅ Created collection '{collection_name}' with {vector_size} dimensions")


def ingest_documents(documents: list):
    """Embed and upload documents to Qdrant."""

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")

    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    # ✅ FIX: get_sentence_embedding_dimension() returns int | None
    # If None, fall back to encoding a test string and measuring the output
    vector_size = model.get_sentence_embedding_dimension()
    if vector_size is None:
        vector_size = len(model.encode("test", normalize_embeddings=True))

    print(f"Embedding dimension: {vector_size}")
    # BAAI/bge-small-en-v1.5 → should print 384

    # Create collection
    create_collection(client, collection_name, vector_size)  # ✅ now guaranteed int

    # Embed and upload each document
    print(f"\nIngesting {len(documents)} documents...")
    points = []

    for doc in documents:
        embedding = model.encode(doc["text"], normalize_embeddings=True)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "text": doc["text"],
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", 0),
            },
        )
        points.append(point)
        print(f"  ✅ Embedded: {doc['source']} (page {doc.get('page', '?')})")

    client.upsert(
        collection_name=collection_name,
        points=points,
    )

    print(f"\n🎉 Successfully ingested {len(points)} documents into Qdrant!")
    print(f"   Collection : {collection_name}")
    print(f"   Dimensions : {vector_size}")
    print(f"   URL        : {os.getenv('QDRANT_URL')}")


if __name__ == "__main__":
    ingest_documents(SAMPLE_DOCUMENTS)
