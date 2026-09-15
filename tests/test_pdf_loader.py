from pathlib import Path

import pytest

from beginner_pdf_rag.pdf_loader import PdfLoadError, load_pdf
from tests.pdf_fixtures import make_blank_pdf, make_encrypted_pdf, make_text_pdf


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_pdf_preserves_one_based_page_numbers():
    pages = load_pdf(FIXTURES / "two-pages.pdf")

    assert [(page.number, page.text) for page in pages] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]


def test_image_only_pdf_has_an_ocr_instruction(tmp_path):
    path = make_blank_pdf(tmp_path / "blank.pdf")

    with pytest.raises(PdfLoadError, match="OCR"):
        load_pdf(path)


def test_missing_pdf_explains_how_to_select_an_existing_file(tmp_path):
    with pytest.raises(PdfLoadError, match="existing PDF"):
        load_pdf(tmp_path / "missing.pdf")


def test_non_pdf_path_is_rejected_with_a_clear_next_action(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a PDF")

    with pytest.raises(PdfLoadError, match=".pdf"):
        load_pdf(path)


def test_load_pdf_normalizes_repeated_whitespace_within_each_page(tmp_path):
    path = make_text_pdf(tmp_path / "spacing.pdf", ["First   page text"])

    pages = load_pdf(path)

    assert [(page.number, page.text) for page in pages] == [(1, "First page text")]


def test_encrypted_pdf_instructs_the_user_to_remove_its_password(tmp_path):
    path = make_encrypted_pdf(tmp_path / "locked.pdf")

    with pytest.raises(PdfLoadError, match="password"):
        load_pdf(path)
