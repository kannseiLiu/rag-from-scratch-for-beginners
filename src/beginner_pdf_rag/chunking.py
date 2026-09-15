"""Transparent overlapping chunks that retain source page metadata."""

from collections.abc import Sequence

from beginner_pdf_rag.models import Chunk, Page


def chunk_pages(
    pages: Sequence[Page], chunk_size: int = 1200, overlap: int = 200
) -> list[Chunk]:
    """Split nonblank page text into separately indexed overlapping chunks."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be an integer greater than zero")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError("overlap must be an integer smaller than chunk_size")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        for index, start in enumerate(range(0, len(page.text), step)):
            chunks.append(Chunk(page.number, index, page.text[start : start + chunk_size]))
    return chunks
