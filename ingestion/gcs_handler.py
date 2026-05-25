# ingestion/gcs_handler.py

"""
GCS Handler — Google Cloud Storage Integration

WHY GCS?
    Stores the original raw files AND processed chunks permanently.

    Raw bucket    → untouched originals (audit trail, reprocessing)
    Processed bucket → cleaned text JSON (faster re-ingestion)

    Without GCS: files only exist on your local machine.
    With GCS: files are backed up, shareable, and accessible
              from Cloud Run in production.

BUCKET STRUCTURE:
    rag-system-raw-docs/
        └── uploads/
            ├── policy_handbook.pdf
            ├── user_manual.docx
            └── q3_report.pptx

    rag-system-processed-docs/
        └── chunks/
            ├── policy_handbook_chunks.json
            ├── user_manual_chunks.json
            └── q3_report_chunks.json
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_gcs_client():
    """
    Initialize GCS client.
    Reads credentials from GOOGLE_APPLICATION_CREDENTIALS env var
    or from gcloud auth application-default login (local dev).
    """
    try:
        from google.cloud import storage

        return storage.Client(project=os.getenv("GCP_PROJECT_ID"))
    except Exception as e:
        logger.error(f"[GCS] Failed to initialize client: {e}")
        raise


def upload_raw_file(local_path: str) -> Optional[str]:
    """
    Upload a raw document to the GCS raw bucket.

    Args:
        local_path: Path to the local file

    Returns:
        GCS URI (gs://bucket/path) on success, None on failure
    """
    try:
        client = get_gcs_client()
        bucket_name = os.getenv("GCS_RAW_BUCKET", "rag-system-raw-docs")
        bucket = client.bucket(bucket_name)

        file_name = Path(local_path).name
        blob_path = f"uploads/{file_name}"
        blob = bucket.blob(blob_path)

        blob.upload_from_filename(local_path)
        gcs_uri = f"gs://{bucket_name}/{blob_path}"

        logger.info(f"[GCS] Uploaded raw file → {gcs_uri}")
        return gcs_uri

    except Exception as e:
        logger.error(f"[GCS] Failed to upload {local_path}: {e}")
        return None


def upload_processed_chunks(chunks: List[dict], source_filename: str) -> Optional[str]:
    """
    Upload processed chunks JSON to the GCS processed bucket.

    Args:
        chunks:          List of chunk dicts from chunker
        source_filename: Original filename (used to name the JSON)

    Returns:
        GCS URI on success, None on failure
    """
    try:
        client = get_gcs_client()
        bucket_name = os.getenv("GCS_PROCESSED_BUCKET", "rag-system-processed-docs")
        bucket = client.bucket(bucket_name)

        # Name the JSON after the source file
        stem = Path(source_filename).stem
        blob_path = f"chunks/{stem}_chunks.json"
        blob = bucket.blob(blob_path)

        # Serialize chunks to JSON
        json_content = json.dumps(chunks, indent=2, ensure_ascii=False)
        blob.upload_from_string(json_content, content_type="application/json")

        gcs_uri = f"gs://{bucket_name}/{blob_path}"
        logger.info(f"[GCS] Uploaded {len(chunks)} chunks → {gcs_uri}")
        return gcs_uri

    except Exception as e:
        logger.error(f"[GCS] Failed to upload chunks for {source_filename}: {e}")
        return None


def list_raw_files() -> List[str]:
    """List all files in the raw GCS bucket."""
    try:
        client = get_gcs_client()
        bucket_name = os.getenv("GCS_RAW_BUCKET", "rag-system-raw-docs")
        blobs = client.list_blobs(bucket_name, prefix="uploads/")
        return [blob.name for blob in blobs if not blob.name.endswith("/")]
    except Exception as e:
        logger.error(f"[GCS] Failed to list files: {e}")
        return []
