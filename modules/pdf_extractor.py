"""Get text out of a PDF report.

A PDF holding a real text layer is read directly. A scanned PDF has no text
layer, so its pages are rendered to images and passed to the OCR module.

pypdf is used for the text layer. PyMuPDF is used instead when it is installed,
because it keeps column spacing better; the project does not require it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

MIN_CHARS_FOR_TEXT_LAYER = 120


@dataclass
class PdfText:
    text: str
    pages: int
    used_ocr: bool
    engine: str
    notes: list


def _read_with_pymupdf(data: bytes):
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    n = doc.page_count
    doc.close()
    return "\n".join(pages), n


def _read_with_pypdf(data: bytes):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages), len(reader.pages)


def extract_text(source, ocr_if_needed: bool = True, dpi: int = 300) -> PdfText:
    """Read a PDF given as a path, bytes or file-like object."""
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
    elif isinstance(source, bytes):
        data = source
    else:
        data = source.read()

    notes = []
    engine = "pymupdf"
    out = _read_with_pymupdf(data)
    if out is None:
        engine = "pypdf"
        out = _read_with_pypdf(data)
    text, n_pages = out

    if len(text.strip()) >= MIN_CHARS_FOR_TEXT_LAYER:
        return PdfText(text=text, pages=n_pages, used_ocr=False, engine=engine, notes=notes)

    notes.append("This PDF has little or no text layer, so it looks like a scan.")
    if not ocr_if_needed:
        return PdfText(text=text, pages=n_pages, used_ocr=False, engine=engine, notes=notes)

    from .image_ocr import ocr_pdf_pages
    ocr_text, ocr_notes = ocr_pdf_pages(data, dpi=dpi)
    notes.extend(ocr_notes)
    return PdfText(text=ocr_text, pages=n_pages, used_ocr=True,
                   engine=engine + "+tesseract", notes=notes)


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"
