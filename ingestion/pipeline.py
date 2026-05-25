# ingestion/pipeline.py

"""
Ingestion Pipeline — Master Orchestrator

This is the single entry point for ingesting any document.
It orchestrates: Load → Chunk → Embed → Upload to Qdrant → Archive to GCS

USAGE:
    # Ingest a single file
    from ingestion.pipeline import ingest_file
    ingest_file("path/to/document.pdf")

    # Ingest an entire folder
    from ingestion.pipeline import ingest_folder
    ingest_folder("path/to/documents/")
"""

import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from ingestion.loaders import load_document
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_upload
from ingestion.gcs_handler import upload_raw_file, upload_processed_chunks

load_dotenv()
logger = logging.getLogger(__name__)

# File types we support
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html", ".htm"}


def ingest_file(file_path: str, upload_to_gcs: bool = False) -> dict:
    """
    Full ingestion pipeline for a single file.

    Args:
        file_path:     Path to the document to ingest
        upload_to_gcs: Whether to archive to GCS (False for local dev)

    Returns:
        Result dict with stats and status
    """
    path = Path(file_path)

    # ── Validation ────────────────────────────────────────────────────────────
    if not path.exists():
        logger.error(f"[Pipeline] File not found: {file_path}")
        return {"status": "error", "error": "File not found", "file": file_path}

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(f"[Pipeline] Unsupported file type: {path.suffix}")
        return {"status": "skipped", "reason": f"Unsupported type: {path.suffix}"}

    logger.info(f"[Pipeline] ── Starting ingestion: {path.name} ──")

    result = {
        "file": path.name,
        "status": "success",
        "documents_loaded": 0,
        "chunks_created": 0,
        "chunks_uploaded": 0,
        "gcs_raw_uri": None,
        "gcs_processed_uri": None,
    }

    try:
        # ── STEP 1: Load ──────────────────────────────────────────────────────
        logger.info(f"[Pipeline] Step 1/4 — Loading {path.name}")
        documents = load_document(file_path)

        if not documents:
            logger.warning(f"[Pipeline] No content extracted from {path.name}")
            result["status"] = "empty"
            return result

        result["documents_loaded"] = len(documents)
        logger.info(f"[Pipeline] Loaded {len(documents)} document sections")

        # ── STEP 2: Chunk ─────────────────────────────────────────────────────
        logger.info("[Pipeline] Step 2/4 — Chunking")
        chunks = chunk_documents(documents)

        if not chunks:
            logger.warning("[Pipeline] No chunks produced")
            result["status"] = "empty"
            return result

        result["chunks_created"] = len(chunks)

        # ── STEP 3: Embed + Upload to Qdrant ──────────────────────────────────
        logger.info("[Pipeline] Step 3/4 — Embedding + uploading to Qdrant")
        uploaded_count = embed_and_upload(chunks)
        result["chunks_uploaded"] = uploaded_count

        # ── STEP 4: Archive to GCS (optional) ────────────────────────────────
        if upload_to_gcs:
            logger.info("[Pipeline] Step 4/4 — Archiving to GCS")
            raw_uri = upload_raw_file(file_path)
            processed_uri = upload_processed_chunks(chunks, path.name)
            result["gcs_raw_uri"] = raw_uri
            result["gcs_processed_uri"] = processed_uri
        else:
            logger.info("[Pipeline] Step 4/4 — GCS upload skipped (local dev mode)")

        logger.info(
            f"[Pipeline] ✅ Complete: {path.name} | "
            f"{result['chunks_uploaded']} chunks ingested"
        )
        return result

    except Exception as e:
        logger.error(f"[Pipeline] ❌ Failed on {path.name}: {e}")
        result["status"] = "error"
        result["error"] = str(e)
        return result


def ingest_folder(folder_path: str, upload_to_gcs: bool = False) -> List[dict]:
    """
    Ingest all supported documents in a folder.

    Args:
        folder_path:   Path to folder containing documents
        upload_to_gcs: Whether to archive to GCS

    Returns:
        List of result dicts, one per file
    """
    folder = Path(folder_path)

    if not folder.exists():
        logger.error(f"[Pipeline] Folder not found: {folder_path}")
        return []

    # Find all supported files recursively
    files = [f for f in folder.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning(f"[Pipeline] No supported files found in {folder_path}")
        return []

    logger.info(f"[Pipeline] Found {len(files)} files to ingest")

    results = []
    for i, file_path in enumerate(files, 1):
        logger.info(f"[Pipeline] Processing file {i}/{len(files)}: {file_path.name}")
        result = ingest_file(str(file_path), upload_to_gcs=upload_to_gcs)
        results.append(result)

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    total_chunks = sum(r.get("chunks_uploaded", 0) for r in results)

    logger.info(
        f"[Pipeline] ── Folder ingestion complete ──\n"
        f"  Files   : {len(files)} total | {success} success | {failed} failed\n"
        f"  Chunks  : {total_chunks} uploaded to Qdrant"
    )
    return results
