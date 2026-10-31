
import io
import os
import random
import time
from pathlib import Path

import fitz
from PIL import Image, ImageOps, UnidentifiedImageError
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def _get_gemini_client():
    """Create a Gemini client using GEMINI_API_KEY from the .env file."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your project's .env file."
        )

    return genai.Client(api_key=api_key)


def _get_model():
    """Return the Gemini model configured for this project."""
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

    if model.startswith("models/"):
        model = model[len("models/"):]

    if not model:
        raise RuntimeError(
            "GEMINI_MODEL is empty. Set a supported model in your .env file."
        )

    return model


# ---------------------------------------------------------------------
# Gemini retry handling
# ---------------------------------------------------------------------

def _is_temporary_api_error(exc):
    """Identify errors that may succeed when retried."""
    message = str(exc).lower()

    temporary_markers = (
        "503",
        "429",
        "500",
        "502",
        "504",
        "unavailable",
        "resource_exhausted",
        "high demand",
        "temporarily overloaded",
        "deadline exceeded",
        "timed out",
        "timeout",
        "connection reset",
        "connection error",
        "service unavailable",
        "internal server error",
    )

    status_code = getattr(exc, "code", None)
    if callable(status_code):
        try:
            status_code = status_code()
        except Exception:
            status_code = None

    if status_code is None:
        status_code = getattr(exc, "status_code", None)

    try:
        if int(status_code) in (408, 429, 500, 502, 503, 504):
            return True
    except (TypeError, ValueError):
        pass

    return any(marker in message for marker in temporary_markers)


def _generate_with_retries(client, model, prompt, image_bytes, mime_type):
    """
    Call Gemini with bounded retries for transient API failures.
    Four attempts maximum; permanent errors are not retried.
    """
    last_error = None

    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )

            result = (response.text or "").strip()

            if not result:
                raise RuntimeError(
                    "Gemini returned an empty transcription. "
                    "Try a clearer image or a different supported model."
                )

            return result

        except Exception as exc:
            last_error = exc

            if not _is_temporary_api_error(exc) or attempt == 3:
                break

            # Wait approximately 2, 4, and 8 seconds before retries.
            delay = (2 ** (attempt + 1)) + random.uniform(0, 0.5)
            time.sleep(delay)

    raise RuntimeError(
        f"Gemini OCR failed using model '{model}'. "
        f"Details: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------
# Image OCR
# ---------------------------------------------------------------------

def _transcribe_image(image_bytes: bytes, mime_type: str) -> str:
    """Transcribe printed or handwritten text from an image using Gemini."""
    if not image_bytes:
        raise ValueError("The image contains no data.")

    prompt = """
You are a careful handwriting transcription assistant for an educational
assessment system.

Transcribe all visible text in the supplied image as accurately as possible.

Instructions:
1. Preserve question numbers, headings, bullets, and paragraph order.
2. Preserve mathematical expressions, numbers, symbols, and equations.
3. Keep separate answers on separate lines where possible.
4. Do not solve questions, grade answers, correct mistakes, or summarize.
5. Do not invent words that are not visible.
6. If a word is unreadable, write [unclear].
7. If content is cut off or obscured, indicate that clearly.
8. Preserve the student's original spelling and wording.
9. Return only the transcription, without introductory commentary.
"""

    client = _get_gemini_client()
    model = _get_model()

    return _generate_with_retries(
        client=client,
        model=model,
        prompt=prompt,
        image_bytes=image_bytes,
        mime_type=mime_type,
    )


def _ocr_image_bytes(data: bytes) -> str:
    """Read an image, normalize its orientation, and transcribe it."""
    if not data:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(io.BytesIO(data)) as source:
            # Correct camera orientation and normalize color format.
            image = ImageOps.exif_transpose(source)
            image = image.convert("RGB")

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

        return _transcribe_image(png_bytes, "image/png")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not read the uploaded image. "
            f"Please upload a valid, readable image. Details: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# PDF text extraction and scanned-page OCR
# ---------------------------------------------------------------------

def extract_pdf_bytes(data: bytes) -> str:
    """
    Extract selectable text from digital PDFs.
    Use Gemini vision for pages without sufficient embedded text.
    """
    if not data:
        raise ValueError("The uploaded PDF is empty.")

    page_texts = []

    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.page_count == 0:
                raise ValueError("The uploaded PDF contains no pages.")

            for page_number, page in enumerate(document, start=1):
                embedded_text = page.get_text("text").strip()

                # Use embedded text when the PDF page is digital.
                if len(embedded_text) >= 15:
                    page_texts.append(
                        f"[Page {page_number}]\n{embedded_text}"
                    )
                    continue

                # Render scanned or handwritten pages for Gemini vision.
                pixmap = page.get_pixmap(
                    dpi=200,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                image_bytes = pixmap.tobytes("png")

                try:
                    recognized_text = _transcribe_image(
                        image_bytes,
                        "image/png",
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"OCR failed on PDF page {page_number} "
                        f"of {document.page_count}. Details: {exc}"
                    ) from exc

                page_texts.append(
                    f"[Page {page_number} OCR]\n{recognized_text}"
                )

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the uploaded PDF. Details: {exc}"
        ) from exc

    return "\n\n".join(page_texts).strip()


# ---------------------------------------------------------------------
# Main entry point used by app.py
# ---------------------------------------------------------------------

def extract_uploaded_file(name: str, data: bytes) -> str:
    """
    Extract text from supported TXT, PDF, and image uploads.

    Keep this function name unchanged because app.py imports it.
    """
    if not data:
        raise ValueError(
            f"The uploaded file '{name}' is empty. "
            "Please choose the file again and upload it."
        )

    suffix = Path(name).suffix.lower()

    if suffix == ".txt":
        try:
            return data.decode("utf-8-sig", errors="replace").strip()
        except Exception as exc:
            raise RuntimeError(
                f"Could not read the text file '{name}': {exc}"
            ) from exc

    if suffix == ".pdf":
        return extract_pdf_bytes(data)

    supported_image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if suffix in supported_image_extensions:
        return _ocr_image_bytes(data)

    raise ValueError(
        f"Unsupported file type '{suffix or 'unknown'}'. "
        "Supported formats: PDF, TXT, PNG, JPG, JPEG, WEBP, BMP, TIF, TIFF."
    )
