"""Page-aware PDF loading with beginner-friendly errors."""

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from beginner_pdf_rag.models import Page


class PdfLoadError(RuntimeError):
    """Raised when a PDF cannot be prepared for retrieval."""


def load_pdf(path: str | Path) -> list[Page]:
    """Extract normalized text from each page of a readable PDF."""
    source = Path(path)
    if source.suffix.lower() != ".pdf":
        raise PdfLoadError("Choose a file with a .pdf extension.")
    if not source.is_file():
        raise PdfLoadError("Select an existing PDF file and try again.")

    try:
        with source.open("rb") as input_file:
            reader = PdfReader(input_file)
            if reader.is_encrypted:
                raise PdfLoadError(
                    "The PDF is encrypted. Remove its password and try again."
                )
            pages = [
                Page(number, " ".join((page.extract_text() or "").split()))
                for number, page in enumerate(reader.pages, start=1)
            ]
    except OSError:
        raise PdfLoadError(
            "Could not read the PDF. Check file permissions and try again."
        ) from None
    except PyPdfError:
        raise PdfLoadError(
            "Could not parse the PDF. Check that it is not corrupted and try again."
        ) from None

    if not any(page.text for page in pages):
        raise PdfLoadError(
            "No extractable text was found. Run OCR on this PDF and try again."
        )
    return pages
