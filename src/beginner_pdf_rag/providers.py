"""Small SDK adapters for local Ollama and hosted OpenAI models."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, Protocol

import ollama
from openai import OpenAI

from beginner_pdf_rag.config import Settings


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Generator(Protocol):
    def generate(self, question: str, context: str) -> str: ...


_GROUNDED_INSTRUCTIONS = (
    "Answer the question using ONLY the provided document context. "
    "If the context does not support an answer, say exactly: "
    "I cannot find this information in the document. "
    "When the context supports an answer, cite the supporting page as [Page N]."
)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value


def _require_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise ValueError("texts must not be empty and must be a sequence of strings.")
    values = list(texts)
    if not values or any(not isinstance(text, str) or not text.strip() for text in values):
        raise ValueError("texts must not be empty and must be a sequence of strings.")
    return values


def _normalize_vector(value: Any, provider: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{provider} returned an empty or invalid embedding vector.")
    try:
        vector = [float(number) for number in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{provider} returned an invalid embedding vector.") from error
    if not all(isfinite(number) for number in vector):
        raise ValueError(f"{provider} returned a non-finite embedding vector.")
    return vector


def _normalize_embeddings(
    response: Any, field: str, expected_count: int, provider: str
) -> list[list[float]]:
    data = _field(response, field)
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence) or not data:
        raise ValueError(f"{provider} returned empty embedding data.")
    if len(data) != expected_count:
        raise ValueError(f"{provider} returned an embedding count that does not match the input.")

    items = list(data)
    if field == "embeddings":
        return [_normalize_vector(item, provider) for item in items]

    indexes = [_field(item, "index") for item in items]
    if any(index is not None for index in indexes):
        if (
            any(not isinstance(index, int) or isinstance(index, bool) for index in indexes)
            or set(indexes) != set(range(expected_count))
        ):
            raise ValueError(f"{provider} returned invalid embedding indexes.")
        items = [item for _, item in sorted(zip(indexes, items, strict=True))]

    vectors = [_normalize_vector(_field(item, "embedding"), provider) for item in items]
    return vectors


def _ollama_error(error: Exception, model: str) -> RuntimeError:
    if _field(error, "status_code") == 404:
        return RuntimeError(
            f"Ollama model {model!r} is unavailable. Run `ollama pull {model}`."
        )
    if isinstance(error, (ConnectionError, OSError)):
        return RuntimeError("Cannot connect to Ollama. Start Ollama and try again.")
    return RuntimeError(
        "Ollama request failed. Start Ollama and confirm the selected model is installed."
    )


def _openai_error() -> RuntimeError:
    return RuntimeError("OpenAI request failed; check API configuration and try again.")


def _normalize_text(value: Any, provider: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{provider} returned an empty response.")
    return value


class OllamaEmbedder:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = _require_texts(texts)
        try:
            response = self._client.embed(model=self._model, input=values)
        except Exception as error:
            raise _ollama_error(error, self._model) from None
        return _normalize_embeddings(response, "embeddings", len(values), "Ollama")


class OllamaGenerator:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, question: str, context: str) -> str:
        _require_text(question, "question")
        _require_text(context, "context")
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _GROUNDED_INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion:\n{question}",
                    },
                ],
                stream=False,
            )
        except Exception as error:
            raise _ollama_error(error, self._model) from None
        return _normalize_text(_field(_field(response, "message"), "content"), "Ollama")


class OpenAIEmbedder:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = _require_texts(texts)
        try:
            response = self._client.embeddings.create(model=self._model, input=values)
        except Exception:
            raise _openai_error() from None
        return _normalize_embeddings(response, "data", len(values), "OpenAI")


class OpenAIGenerator:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def generate(self, question: str, context: str) -> str:
        _require_text(question, "question")
        _require_text(context, "context")
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=_GROUNDED_INSTRUCTIONS,
                input=f"Context:\n{context}\n\nQuestion:\n{question}",
            )
        except Exception:
            raise _openai_error() from None
        return _normalize_text(_field(response, "output_text"), "OpenAI")


def build_providers(settings: Settings) -> tuple[Embedder, Generator]:
    if settings.provider == "ollama":
        client = ollama.Client(host=settings.ollama_base_url)
        return (
            OllamaEmbedder(client, settings.embedding_model),
            OllamaGenerator(client, settings.chat_model),
        )
    if settings.provider == "openai":
        client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        return (
            OpenAIEmbedder(client, settings.embedding_model),
            OpenAIGenerator(client, settings.chat_model),
        )
    raise ValueError("Provider must be 'ollama' or 'openai'.")
