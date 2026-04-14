"""
PDF → base64 image conversion service.

Design decisions:
- JPEG at quality=85 is used instead of PNG.
  A4 page at 150 DPI as PNG ≈ 2–5 MB base64; same page as JPEG ≈ 150–400 KB base64.
  This is a 10–15× reduction in payload size per page, which is the single
  biggest driver of Azure GPT-4o vision call latency and timeouts.
- DPI defaults to 150 (configurable via IMAGE_DPI env var).
  150 DPI is sufficient for GPT-4o "high" detail — the model internally
  down-samples images anyway.
"""

import base64
import logging
from io import BytesIO

import fitz
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


def pdf_to_base64_images(pdf_path: str, dpi: int | None = None) -> list[str]:
    """Render every page of a PDF to a JPEG base64 string.

    Args:
        pdf_path:  Absolute path to the PDF file.
        dpi:       Rendering resolution.  Defaults to ``settings.image_dpi``.

    Returns:
        List of base64-encoded JPEG strings, one per page.
    """
    effective_dpi = dpi or settings.image_dpi
    output: list[str] = []

    with fitz.open(pdf_path) as doc:
        total = len(doc)
        logger.info("PDF opened: %s | pages=%d | dpi=%d", pdf_path, total, effective_dpi)

        for page_idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=effective_dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            buf = BytesIO()
            # JPEG at quality=85 gives a huge size reduction vs PNG with minimal
            # visual quality loss for OCR purposes.
            img.save(buf, format="JPEG", quality=85, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            output.append(b64)
            logger.info(
                "  Page %d/%d rendered → JPEG size=%.1f KB | base64=%.1f KB",
                page_idx, total,
                buf.tell() / 1024,
                len(b64) / 1024,
            )

    logger.info("PDF→JPEG conversion complete: %d pages", len(output))
    return output


def chunk_pages(images: list[str], chunk_size: int | None = None) -> list[list[str]]:
    """Split page images into batches for API calls.

    Args:
        images:     List of base64 image strings.
        chunk_size: Pages per batch.  Defaults to ``settings.page_batch_size``.

    Returns:
        List of batches, each batch being a list of base64 strings.
    """
    size = chunk_size or settings.page_batch_size
    batches = [images[i: i + size] for i in range(0, len(images), size)]
    logger.info("Chunked %d pages into %d batches of ≤%d pages each", len(images), len(batches), size)
    return batches
