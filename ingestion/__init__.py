# ingestion/__init__.py

"""
Ingestion package — exposes the main pipeline functions.
Import from here instead of individual modules.

Usage:
    from ingestion import ingest_file, ingest_folder
"""

from ingestion.pipeline import ingest_file, ingest_folder

__all__ = ["ingest_file", "ingest_folder"]
