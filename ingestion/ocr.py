# ingestion/ocr.py

"""
OCR — Google Document AI for Scanned PDFs

WHY OCR?
    Many real-world PDFs are scanned images — not text-based.
    pypdf returns empty strings on these.
    Google Document AI uses ML to read text from images.

    Examples of scanned docs:
    - Old policy documents photographed and saved as PDF
    - Legal contracts signed and scanned
    - Medical records
    - Invoices from older systems

WHEN TO USE OCR vs normal PDF loader:
    The pipeline auto-detects by checking if pypdf extracts < 50 chars.
    If too little text → fall back to Document AI OCR.

SETUP REQUIRED (Phase 7 — skip for now in local dev):
    1. Enable Document AI API in GCP console
    2. Create a Document OCR processor
    3. Set DOCUMENT_AI_PROCESSOR_ID in .env
    4. Set GOOGLE_APPLICATION_CREDENTIALS in .env
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from ingestion.loaders import Document

load_dotenv()
logger = logging.getLogger(__name__)


def load_pdf_ocr(file_path: str) -> List[Document]:
    """
    Extract text from a scanned PDF using Google Document AI.

    Args:
        file_path: Path to the scanned PDF

    Returns:
        List[Document] with extracted text per page

    NOTE: Requires GCP credentials and Document AI processor.
          In local dev, set SKIP_OCR=true in .env to bypass.
    """

    # Allow skipping OCR in local dev without GCP credentials
    if os.getenv("SKIP_OCR", "false").lower() == "true":
        logger.warning("[OCR] SKIP_OCR=true — returning empty. Set up GCP for OCR.")
        return []

    try:
        from google.cloud import documentai
        from google.api_core.client_options import ClientOptions

        project_id = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_REGION", "asia").split("-")[
            0
        ]  # "asia" from "asia-south1"
        processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")

        if not processor_id:
            logger.error("[OCR] DOCUMENT_AI_PROCESSOR_ID not set in .env")
            return []

        # Initialize Document AI client
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        client = documentai.DocumentProcessorServiceClient(client_options=opts)

        processor_name = client.processor_path(
            project_id or "",
            location,
            processor_id or "",
        )

        # Read the PDF file
        with open(file_path, "rb") as f:
            pdf_content = f.read()

        # Build the request
        raw_document = documentai.RawDocument(
            content=pdf_content,
            mime_type="application/pdf",
        )
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document,
        )

        logger.info(f"[OCR] Sending {file_path} to Document AI...")
        result = client.process_document(request=request)
        document = result.document

        logger.info(f"[OCR] Received response — {len(document.pages)} pages")

        # Extract text per page
        documents = []
        full_text = document.text

        for page in document.pages:
            page_num = page.page_number

            # Extract text segments for this page using layout references
            page_text_parts = []
            for block in page.blocks:
                segment = block.layout.text_anchor
                for seg in segment.text_segments:
                    start = int(seg.start_index) if seg.start_index else 0
                    end = int(seg.end_index)
                    page_text_parts.append(full_text[start:end])

            page_text = " ".join(page_text_parts).strip()

            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": os.path.basename(file_path),
                            "file_path": file_path,
                            "file_type": "pdf_ocr",
                            "page": page_num,
                            "ocr": True,
                        },
                    )
                )

        logger.info(f"[OCR] Extracted {len(documents)} pages via Document AI")
        return documents

    except Exception as e:
        logger.error(f"[OCR] Document AI failed: {e}")
        return []
