
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

image_path = Path("test_handwriting.jpg")

try:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path.resolve()}"
        )

    if image_path.stat().st_size == 0:
        raise ValueError("The image file is empty.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    client = genai.Client(api_key=api_key)

    print("Reading handwritten image with Gemini...")

    image_bytes = image_path.read_bytes()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
            """
            Transcribe the handwritten notes in this image.
            Preserve headings, bullet points, and paragraph order.
            Do not summarize, correct, or invent missing words.
            Mark uncertain words as [unclear].
            Return only the transcription.
            """,
        ],
    )

    print("\n--- OCR RESULT ---\n")
    print(response.text or "No text was returned.")

except Exception as exc:
    print("\n--- OCR ERROR ---\n")
    print(f"{type(exc).__name__}: {exc}")
