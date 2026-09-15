"""Transparent cosine-similarity retrieval over embedded PDF chunks."""

from collections.abc import Sequence

import numpy as np

from beginner_pdf_rag.models import Chunk, SearchResult


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    """Convert a sequence to a validated, one-dimensional float vector."""
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric vector") from error

    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite numbers")
    return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two finite, non-zero vectors."""
    left_vector = _as_vector(left, "left vector")
    right_vector = _as_vector(right, "right vector")

    if left_vector.shape != right_vector.shape:
        raise ValueError("vectors must have the same dimension")
    if not np.any(left_vector) or not np.any(right_vector):
        raise ValueError("cannot compute similarity for a zero vector")

    return float(
        np.dot(left_vector, right_vector)
        / (np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    )


def rank_chunks(
    query_embedding: Sequence[float],
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    top_k: int,
) -> list[SearchResult]:
    """Rank chunks by cosine similarity, preserving input order for ties."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if len(chunks) != len(embeddings):
        raise ValueError("embedding count must match chunk count")

    query_vector = _as_vector(query_embedding, "query vector")
    if not chunks:
        return []

    scored: list[tuple[float, int, Chunk]] = []
    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        score = cosine_similarity(query_vector, embedding)
        scored.append((score, index, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [SearchResult(chunk=chunk, score=score) for score, _, chunk in scored[:top_k]]
