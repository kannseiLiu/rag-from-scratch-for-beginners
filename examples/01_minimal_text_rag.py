"""The complete RAG algorithm on three short passages (requires Ollama)."""

import numpy as np
import ollama


DOCUMENTS = [
    "Dense Passage Retrieval represents questions and passages as vectors.",
    "A retriever finds the most relevant passages before a language model answers.",
    "The DPR paper evaluates retrieval on open-domain question-answering datasets.",
]
QUESTION = "What does a retriever do in RAG?"


def embed(texts: list[str]) -> list[list[float]]:
    """Turn text into vectors with the local embedding model."""
    response = ollama.embed(model="nomic-embed-text", input=texts)
    return response["embeddings"]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Measure how closely two embedding vectors point in the same direction."""
    left_vector = np.array(left)
    right_vector = np.array(right)
    return float(
        np.dot(left_vector, right_vector)
        / (np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    )


def main() -> None:
    document_embeddings = embed(DOCUMENTS)
    question_embedding = embed([QUESTION])[0]

    scored = [
        (cosine_similarity(question_embedding, embedding), document)
        for document, embedding in zip(DOCUMENTS, document_embeddings, strict=True)
    ]
    top_k = 2
    retrieved = sorted(scored, reverse=True)[:top_k]
    context = "\n".join(f"- {document}" for _, document in retrieved)
    prompt = f"""Answer using only this context.

Context:
{context}

Question: {QUESTION}
Answer:"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"]
    print("Top-K passages:")
    print(context)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
