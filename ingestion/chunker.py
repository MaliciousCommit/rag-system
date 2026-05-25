# ingestion/chunker.py

"""
Text Chunker — Split Documents into Retrieval-Ready Chunks

WHY CHUNKING?
    LLMs have context limits. A 50-page PDF can't fit in one prompt.
    More importantly, vector search works best on FOCUSED chunks.
    A 200-word chunk about "refund policy" will match a refund query
    far better than a 5000-word chapter that mentions refunds once.

CHUNK SIZE STRATEGY:
    chunk_size=512    → ~400 words. Enough context per chunk.
    chunk_overlap=100 → 100 chars overlap between chunks.

    WHY OVERLAP?
        "The refund period is 30" [chunk 1 ends]
        "days from purchase date" [chunk 2 starts]

        Without overlap: both chunks have incomplete sentences.
        With overlap: the key phrase spans both chunks and appears
        fully in at least one → retrieval finds complete information.

    VISUAL:
    |──── chunk 1 (512) ────|
                        |── overlap ──|──── chunk 2 (512) ────|
                                                          |── overlap ──|──── chunk 3 ────|
"""

import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loaders import Document

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> List[dict]:
    """
    Split a list of Documents into smaller overlapping chunks.

    Args:
        documents:     Output from any loader function
        chunk_size:    Max characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of chunk dicts ready for embedding + Qdrant upload
        Each dict: {
            "text"      : str,   → the chunk text (goes into Qdrant)
            "source"    : str,   → original filename
            "file_type" : str,   → pdf/docx/pptx/html
            "page"      : int,   → source page/slide number
            "chunk_index": int,  → position within the document
        }

    WHY return dicts instead of Documents?
        Dicts are easier to serialize, log, and upload to Qdrant.
        The chunk is the terminal unit — no more transformation needed.
    """

    # RecursiveCharacterTextSplitter — the gold standard for RAG chunking
    # WHY "Recursive"?
    #   It tries to split on "\n\n" first (paragraphs)
    #   If chunks are still too big → splits on "\n" (lines)
    #   If still too big → splits on " " (words)
    #   If still too big → splits on "" (characters)
    #   This preserves natural language boundaries as long as possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_index = 0

    for doc in documents:
        if not doc.page_content.strip():
            continue

        # Split this document's text into chunks
        raw_chunks = splitter.split_text(doc.page_content)

        for raw_chunk in raw_chunks:
            if not raw_chunk.strip():
                continue

            chunk = {
                "text": raw_chunk.strip(),
                "source": doc.metadata.get("source", "unknown"),
                "file_path": doc.metadata.get("file_path", ""),
                "file_type": doc.metadata.get("file_type", "unknown"),
                "page": doc.metadata.get("page", 0),
                "section": doc.metadata.get("section", ""),
                "slide_title": doc.metadata.get("slide_title", ""),
                "chunk_index": chunk_index,
                "chunk_size": len(raw_chunk),
            }
            all_chunks.append(chunk)
            chunk_index += 1

    logger.info(
        f"[Chunker] {len(documents)} documents → {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return all_chunks
