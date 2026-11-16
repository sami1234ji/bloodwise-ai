"""Read text from photographs and scans of reports.

Tesseract is a separate program, not a Python package. `pip install pytesseract`
installs only the wrapper. If the program is missing the module says so plainly
rather than failing with an obscure error.

Preprocessing targets the things that go wrong with a phone photograph of a
report: low contrast, shadows, noise and a few degrees of rotation.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

TESSERACT_CONFIG = "--oem 3 --psm 6"


class TesseractMissing(RuntimeError):
    pass


def tesseract_available() -> bool:
    if shutil.which("tesseract"):
        return True
    try:
        import pytesseract
        cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
        return bool(cmd and shutil.which(cmd))
    except Exception:
        return False


def require_tesseract() -> None:
    if not tesseract_available():
        raise TesseractMissing(
            "The Tesseract OCR program was not found. Installing pytesseract is not "
            "enough: Tesseract itself must be installed.\n"
            "  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and "
            "set the TESSERACT_CMD environment variable to tesseract.exe\n"
            "  macOS:   brew install tesseract\n"
            "  Linux:   sudo apt install tesseract-ocr")


def _configure():
    import os
    import pytesseract
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


@dataclass
class OcrResult:
    text: str
    mean_confidence: float
    notes: list


def deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten a page that is a few degrees out of true."""
    try:
        import cv2
    except ImportError:
        return gray
    inverted = 255 - gray
    coords = np.column_stack(np.where(inverted > 100))
    if coords.shape[0] < 50:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.4 or abs(angle) > 20:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess(image: Image.Image) -> Image.Image:
    """Grayscale, upscale small images, denoise, even out lighting, deskew."""
    image = ImageOps.exif_transpose(image).convert("L")
    if min(image.size) < 1000:
        scale = 1000 / min(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)),
                             Image.LANCZOS)
    arr = np.array(image)
    try:
        import cv2
        arr = cv2.fastNlMeansDenoising(arr, None, 10, 7, 21)
        arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
        arr = deskew(arr)
    except ImportError:
        arr = np.where(arr > arr.mean(), 255, 0).astype(np.uint8)
    return Image.fromarray(arr)


def ocr_image(source, preprocess_image: bool = True) -> OcrResult:
    """OCR a PIL image, a path or image bytes."""
    require_tesseract()
    pytesseract = _configure()

    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)

    notes = []
    prepared = preprocess(image) if preprocess_image else image.convert("L")

    data = pytesseract.image_to_data(prepared, config=TESSERACT_CONFIG,
                                     output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
    mean_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    text = pytesseract.image_to_string(prepared, config=TESSERACT_CONFIG)

    if preprocess_image and len(text.strip()) < 40:
        notes.append("Very little text was found after preprocessing; "
                     "the original image was tried as well.")
        raw = pytesseract.image_to_string(image.convert("L"), config=TESSERACT_CONFIG)
        if len(raw.strip()) > len(text.strip()):
            text = raw
    if mean_conf and mean_conf < 0.6:
        notes.append(f"Average OCR confidence was {mean_conf:.0%}. Check every value "
                     "on the review screen before continuing.")
    return OcrResult(text=text, mean_confidence=mean_conf, notes=notes)


def ocr_pdf_pages(pdf_bytes: bytes, dpi: int = 300):
    """Render PDF pages to images and OCR each one."""
    require_tesseract()
    notes = []
    images = []
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
    except Exception as exc:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            zoom = dpi / 72
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
            doc.close()
        except Exception:
            raise RuntimeError(
                "Could not render the PDF pages for OCR. pdf2image needs the Poppler "
                "tools installed (Linux: poppler-utils, macOS: brew install poppler, "
                f"Windows: add Poppler's bin folder to PATH). Original error: {exc}")

    texts = []
    for i, image in enumerate(images, 1):
        page = ocr_image(image)
        texts.append(page.text)
        notes.extend(f"page {i}: {n}" for n in page.notes)
    return "\n".join(texts), notes
