from types import SimpleNamespace

import httpx
import ollama
import openai
import pytest

import beginner_pdf_rag.providers as providers
from beginner_pdf_rag.config import Settings
from beginner_pdf_rag.providers import (
    OllamaEmbedder,
    OllamaGenerator,
    OpenAIEmbedder,
    OpenAIGenerator,
    ProviderError,
    build_providers,
)


class FakeOllamaClient:
    def __init__(self, embeddings=None, chat_response=None, error=None):
        self.embeddings = embeddings
        self.chat_response = chat_response
        self.error = error
        self.embed_call = None
        self.chat_call = None

    def embed(self, **kwargs):
        self.embed_call = kwargs
        if self.error:
            raise self.error
        return {"embeddings": self.embeddings}

    def chat(self, **kwargs):
        self.chat_call = kwargs
        if self.error:
            raise self.error
        return self.chat_response


class FakeOpenAIClient:
    def __init__(self, embeddings=None, output_text=None, error=None):
        self.embeddings_call = None
        self.responses_call = None
        self._embeddings = embeddings
        self._output_text = output_text
        self._error = error
        self.embeddings = SimpleNamespace(create=self.create_embeddings)
        self.responses = SimpleNamespace(create=self.create_response)

    def create_embeddings(self, **kwargs):
        self.embeddings_call = kwargs
        if self._error:
            raise self._error
        return {"data": [{"embedding": item} for item in self._embeddings]}

    def create_response(self, **kwargs):
        self.responses_call = kwargs
        if self._error:
            raise self._error
        return SimpleNamespace(output_text=self._output_text)


def test_ollama_embedder_preserves_batch_order():
    client = FakeOllamaClient(embeddings=[[1.0, 0.0], [0.0, 1.0]])

    result = OllamaEmbedder(client, "nomic-embed-text").embed(["a", "b"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert client.embed_call == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_ollama_generator_supports_mapping_chat_responses_and_grounding():
    client = FakeOllamaClient(chat_response={"message": {"content": "Answer [Page 2]"}})

    answer = OllamaGenerator(client, "qwen3:4b").generate(
        "Question?", "[Page 2]\nEvidence"
    )

    assert answer == "Answer [Page 2]"
    assert client.chat_call["model"] == "qwen3:4b"
    assert client.chat_call["stream"] is False
    assert "ONLY" in client.chat_call["messages"][0]["content"]
    assert "[Page 2]" in client.chat_call["messages"][1]["content"]


def test_openai_embedder_normalizes_object_responses_in_input_order():
    client = FakeOpenAIClient(embeddings=[[1, 0], [0, 1]])
    client.embeddings.create = lambda **kwargs: SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0, 1]),
            SimpleNamespace(index=0, embedding=[1, 0]),
        ]
    )

    result = OpenAIEmbedder(client, "text-embedding-3-small").embed(["a", "b"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]


def test_openai_generator_passes_grounded_instructions():
    client = FakeOpenAIClient(output_text="Answer [Page 2]")

    answer = OpenAIGenerator(client, "gpt-4.1-mini").generate(
        "Question?", "[Page 2]\nEvidence"
    )

    assert answer == "Answer [Page 2]"
    request = client.responses_call
    assert "ONLY" in request["instructions"]
    assert "I cannot find this information in the document." in request["instructions"]
    assert "[Page N]" in request["instructions"]
    assert "[Page 2]" in request["input"]


@pytest.mark.parametrize(
    ("provider", "embedder_type", "generator_type"),
    [
        ("ollama", OllamaEmbedder, OllamaGenerator),
        ("openai", OpenAIEmbedder, OpenAIGenerator),
    ],
)
def test_build_providers_selects_configured_provider(
    provider, embedder_type, generator_type
):
    settings = Settings(
        provider=provider,
        ollama_base_url="http://127.0.0.1:11434",
        embedding_model="embedding-model",
        chat_model="chat-model",
        openai_api_key="secret-key",
        openai_base_url="https://api.openai.com/v1",
    )

    embedder, generator = build_providers(settings)

    assert isinstance(embedder, embedder_type)
    assert isinstance(generator, generator_type)


@pytest.mark.parametrize(
    ("adapter", "method", "args"),
    [
        (OllamaEmbedder(FakeOllamaClient(embeddings=[]), "model"), "embed", ([],)),
        (OpenAIEmbedder(FakeOpenAIClient(embeddings=[]), "model"), "embed", ([],)),
        (OllamaGenerator(FakeOllamaClient(), "model"), "generate", ("", "context")),
        (OpenAIGenerator(FakeOpenAIClient(), "model"), "generate", ("question", "")),
    ],
)
def test_adapters_reject_empty_input_before_call(adapter, method, args):
    with pytest.raises(ValueError, match="must not be empty"):
        getattr(adapter, method)(*args)


@pytest.mark.parametrize(
    ("adapter", "method", "args"),
    [
        (OllamaEmbedder(FakeOllamaClient(embeddings=[]), "model"), "embed", (["a"],)),
        (OpenAIEmbedder(FakeOpenAIClient(embeddings=[]), "model"), "embed", (["a"],)),
        (
            OllamaGenerator(FakeOllamaClient(chat_response={"message": {"content": ""}}), "model"),
            "generate",
            ("question", "context"),
        ),
        (OpenAIGenerator(FakeOpenAIClient(output_text=""), "model"), "generate", ("question", "context")),
    ],
)
def test_adapters_reject_empty_provider_responses(adapter, method, args):
    with pytest.raises(ValueError, match="empty"):
        getattr(adapter, method)(*args)


def test_embedder_rejects_wrong_embedding_count():
    client = FakeOllamaClient(embeddings=[[1.0, 0.0]])

    with pytest.raises(ValueError, match="count"):
        OllamaEmbedder(client, "model").embed(["a", "b"])


@pytest.mark.parametrize(
    "adapter",
    [
        OllamaEmbedder(FakeOllamaClient(embeddings=[[1.0, 0.0], [1.0]]), "model"),
        OpenAIEmbedder(FakeOpenAIClient(embeddings=[[1.0, 0.0], [1.0]]), "model"),
    ],
)
def test_embedders_reject_ragged_batches(adapter):
    with pytest.raises(ValueError, match="same dimension"):
        adapter.embed(["a", "b"])


@pytest.mark.parametrize(
    "embedding",
    [[1.0, object()], [1.0, float("nan")], [1.0, float("inf")]],
)
def test_embedder_rejects_nonnumeric_and_nonfinite_vectors(embedding):
    client = FakeOllamaClient(embeddings=[embedding])

    with pytest.raises(ValueError, match="invalid|non-finite"):
        OllamaEmbedder(client, "model").embed(["a"])


def test_openai_embedder_rejects_wrong_embedding_count():
    client = FakeOpenAIClient(embeddings=[[1.0, 0.0]])

    with pytest.raises(ValueError, match="count"):
        OpenAIEmbedder(client, "model").embed(["a", "b"])


@pytest.mark.parametrize(
    "indexes",
    [["zero", 1], [0, 0]],
)
def test_openai_embedder_rejects_invalid_or_duplicate_indexes(indexes):
    client = FakeOpenAIClient()
    client.embeddings.create = lambda **kwargs: SimpleNamespace(
        data=[
            SimpleNamespace(index=indexes[0], embedding=[1.0, 0.0]),
            SimpleNamespace(index=indexes[1], embedding=[0.0, 1.0]),
        ]
    )

    with pytest.raises(ValueError, match="indexes"):
        OpenAIEmbedder(client, "model").embed(["a", "b"])


def test_ollama_errors_explain_starting_service_or_pulling_model():
    connection_client = FakeOllamaClient(error=ConnectionError("private request text"))
    missing_model_client = FakeOllamaClient(
        error=ollama.ResponseError("private request text", status_code=404)
    )

    with pytest.raises(ProviderError, match="Start Ollama") as connection_error:
        OllamaEmbedder(connection_client, "nomic-embed-text").embed(["a"])
    with pytest.raises(ProviderError, match="ollama pull nomic-embed-text") as model_error:
        OllamaEmbedder(missing_model_client, "nomic-embed-text").embed(["a"])

    assert "private request text" not in str(connection_error.value)
    assert "private request text" not in str(model_error.value)


def test_ollama_response_error_is_safe_and_actionable():
    client = FakeOllamaClient(
        error=ollama.ResponseError("private request text", status_code=500)
    )

    with pytest.raises(ProviderError, match="Ollama request failed") as error:
        OllamaEmbedder(client, "model").embed(["a"])

    assert "private request text" not in str(error.value)


def test_ollama_programming_errors_propagate():
    client = FakeOllamaClient(error=TypeError("client bug"))

    with pytest.raises(TypeError, match="client bug"):
        OllamaEmbedder(client, "model").embed(["a"])


def _response_error(error_type, status_code):
    request = httpx.Request("POST", "https://api.example.test/v1/responses")
    response = httpx.Response(status_code, request=request)
    return error_type("secret-key and private request text", response=response, body={})


@pytest.mark.parametrize(
    "error",
    [
        openai.APIConnectionError(
            message="secret-key and private request text",
            request=httpx.Request("POST", "https://api.example.test/v1/embeddings"),
        ),
        _response_error(openai.AuthenticationError, 401),
        _response_error(openai.RateLimitError, 429),
        _response_error(openai.APIStatusError, 500),
    ],
)
def test_openai_sdk_errors_are_safe_and_actionable(error):
    client = FakeOpenAIClient(error=error)

    with pytest.raises(ProviderError, match="check API configuration") as error:
        OpenAIEmbedder(client, "model").embed(["a"])

    assert "secret-key" not in str(error.value)
    assert "private request text" not in str(error.value)


def test_openai_programming_errors_propagate():
    client = FakeOpenAIClient(error=AttributeError("client bug"))

    with pytest.raises(AttributeError, match="client bug"):
        OpenAIEmbedder(client, "model").embed(["a"])


def test_build_providers_uses_exact_factory_arguments(monkeypatch):
    ollama_calls = []
    openai_calls = []
    ollama_client = object()
    openai_client = object()
    monkeypatch.setattr(
        providers.ollama,
        "Client",
        lambda **kwargs: ollama_calls.append(kwargs) or ollama_client,
    )
    monkeypatch.setattr(
        providers,
        "OpenAI",
        lambda **kwargs: openai_calls.append(kwargs) or openai_client,
    )
    ollama_settings = Settings(
        provider="ollama",
        ollama_base_url="http://ollama.test:11434",
        embedding_model="ollama-embed",
        chat_model="ollama-chat",
        openai_api_key=None,
        openai_base_url="https://api.example.test/v1",
    )
    openai_settings = Settings(
        provider="openai",
        ollama_base_url="http://ollama.test:11434",
        embedding_model="openai-embed",
        chat_model="openai-chat",
        openai_api_key="secret-key",
        openai_base_url="https://api.example.test/v1",
    )

    ollama_embedder, ollama_generator = build_providers(ollama_settings)
    openai_embedder, openai_generator = build_providers(openai_settings)

    assert ollama_calls == [{"host": "http://ollama.test:11434"}]
    assert openai_calls == [
        {"api_key": "secret-key", "base_url": "https://api.example.test/v1"}
    ]
    assert (ollama_embedder._client, ollama_embedder._model) == (
        ollama_client,
        "ollama-embed",
    )
    assert (ollama_generator._client, ollama_generator._model) == (
        ollama_client,
        "ollama-chat",
    )
    assert (openai_embedder._client, openai_embedder._model) == (
        openai_client,
        "openai-embed",
    )
    assert (openai_generator._client, openai_generator._model) == (
        openai_client,
        "openai-chat",
    )
