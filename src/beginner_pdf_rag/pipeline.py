"""Provider-independent indexing and question answering for PDF chunks."""

from dataclasses import dataclass
from pathlib import Path

from beginner_pdf_rag.chunking import chunk_pages
from beginner_pdf_rag.models import Chunk, SearchResult
from beginner_pdf_rag.pdf_loader import load_pdf
from beginner_pdf_rag.providers import Embedder, Generator
from beginner_pdf_rag.retrieval import rank_chunks


class RagError(RuntimeError):
    """Raised when a document is not ready to answer a question."""


@dataclass(frozen=True)
class RagAnswer:
    text: str
    sources: tuple[SearchResult, ...]


class PdfRag:
    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def index(self, pdf_path: str | Path) -> int:
        """Load, chunk, and embed a PDF, returning its number of chunks."""
        chunks = chunk_pages(load_pdf(pdf_path), self._chunk_size, self._overlap)
        if not chunks:
            raise RagError("No text chunks were created from this PDF.")

        try:
            embeddings = self._embedder.embed([chunk.text for chunk in chunks])
        except ValueError as error:
            raise RagError(f"Could not prepare document embeddings: {error}") from None
        if len(embeddings) != len(chunks):
            raise RagError("The embedding provider returned an unexpected number of vectors.")

        self._chunks = chunks
        self._embeddings = embeddings
        return len(chunks)

    def ask(self, question: str, top_k: int = 3) -> RagAnswer:
        """Answer a question from the indexed PDF and retain its retrieved sources."""
        if not self._chunks:
            raise RagError("Please index a PDF before asking a question.")
        if not isinstance(question, str) or not question.strip():
            raise RagError("Question must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RagError("top_k must be a positive integer.")

        try:
            query_embeddings = self._embedder.embed([question])
            if len(query_embeddings) != 1:
                raise RagError("The embedding provider returned an invalid query embedding.")
            results = rank_chunks(
                query_embeddings[0], self._chunks, self._embeddings, top_k
            )
            context = "\n\n".join(
                f"[Page {result.chunk.page} | score={result.score:.3f}]\n"
                f"{result.chunk.text}"
                for result in results
            )
            answer = self._generator.generate(question, context)
        except ValueError as error:
            raise RagError(f"Could not answer this question: {error}") from None
        return RagAnswer(text=answer, sources=tuple(results))
