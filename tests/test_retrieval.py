import numpy as np
import pytest

from beginner_pdf_rag.models import Chunk
from beginner_pdf_rag.retrieval import cosine_similarity, rank_chunks


def test_cosine_similarity_and_top_k_ranking():
    chunks = [Chunk(1, 0, "cats"), Chunk(2, 0, "retrieval"), Chunk(3, 0, "cooking")]
    results = rank_chunks(
        [1.0, 0.0],
        chunks,
        [[0.8, 0.2], [1.0, 0.0], [0.0, 1.0]],
        top_k=2,
    )

    assert [item.chunk.text for item in results] == ["retrieval", "cats"]
    assert results[0].score == pytest.approx(1.0)


def test_zero_vectors_and_dimension_mismatch_are_actionable():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_cosine_similarity_rejects_empty_nonfinite_and_non1d_vectors():
    with pytest.raises(ValueError, match="empty"):
        cosine_similarity([], [])
    with pytest.raises(ValueError, match="finite"):
        cosine_similarity([1.0, float("nan")], [1.0, 0.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        cosine_similarity([[1.0, 0.0]], [1.0, 0.0])


@pytest.mark.parametrize("top_k", [True, False, 1.5, "1", None, 0, -1])
def test_rank_chunks_rejects_non_integer_or_nonpositive_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        rank_chunks([1.0], [Chunk(1, 0, "text")], [[1.0]], top_k=top_k)


def test_rank_chunks_rejects_embedding_count_mismatch():
    with pytest.raises(ValueError, match="embedding count"):
        rank_chunks([1.0], [Chunk(1, 0, "text")], [], top_k=1)


def test_rank_chunks_rejects_invalid_embedding_values_and_dimensions():
    chunk = Chunk(1, 0, "text")
    with pytest.raises(ValueError, match="finite"):
        rank_chunks([1.0], [chunk], [[float("inf")]], top_k=1)
    with pytest.raises(ValueError, match="dimension"):
        rank_chunks([1.0, 0.0], [chunk], [[1.0]], top_k=1)
    with pytest.raises(ValueError, match="empty"):
        rank_chunks([1.0], [chunk], [[]], top_k=1)


def test_rank_chunks_uses_original_order_for_equal_scores_and_does_not_mutate_inputs():
    query = np.array([1.0, 0.0])
    chunks = [Chunk(1, 0, "first"), Chunk(2, 0, "second"), Chunk(3, 0, "third")]
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    original_query = query.copy()
    original_chunks = chunks.copy()
    original_embeddings = embeddings.copy()

    results = rank_chunks(query, chunks, embeddings, top_k=3)

    assert [item.chunk.text for item in results] == ["first", "second", "third"]
    np.testing.assert_array_equal(query, original_query)
    assert chunks == original_chunks
    np.testing.assert_array_equal(embeddings, original_embeddings)


def test_rank_chunks_allows_no_chunks_when_embedding_count_also_zero():
    assert rank_chunks([1.0], [], [], top_k=1) == []
