import pytest

from beginner_pdf_rag.chunking import chunk_pages
from beginner_pdf_rag.models import Page


def test_chunks_keep_page_and_overlap():
    chunks = chunk_pages([Page(7, "abcdefghij")], chunk_size=6, overlap=2)

    assert [(item.page, item.index, item.text) for item in chunks] == [
        (7, 0, "abcdef"),
        (7, 1, "efghij"),
        (7, 2, "ij"),
    ]


def test_chunks_skip_blank_pages_and_restart_indexes_per_page():
    chunks = chunk_pages([Page(1, "  "), Page(2, "abc"), Page(3, "xy")], 2, 0)

    assert [(item.page, item.index, item.text) for item in chunks] == [
        (2, 0, "ab"),
        (2, 1, "c"),
        (3, 0, "xy"),
    ]


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_chunk_parameters_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], chunk_size, overlap)
