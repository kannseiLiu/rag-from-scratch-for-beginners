from types import SimpleNamespace

import pytest

from beginner_pdf_rag.config import Settings
from beginner_pdf_rag.providers import (
    OllamaEmbedder,
    OllamaGenerator,
    OpenAIEmbedder,
    OpenAIGenerator,
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


class MissingOllamaModelError(Exception):
    status_code = 404


def test_ollama_errors_explain_starting_service_or_pulling_model():
    connection_client = FakeOllamaClient(error=ConnectionError("private request text"))
    missing_model_client = FakeOllamaClient(error=MissingOllamaModelError("private request text"))

    with pytest.raises(RuntimeError, match="Start Ollama") as connection_error:
        OllamaEmbedder(connection_client, "nomic-embed-text").embed(["a"])
    with pytest.raises(RuntimeError, match="ollama pull nomic-embed-text") as model_error:
        OllamaEmbedder(missing_model_client, "nomic-embed-text").embed(["a"])

    assert "private request text" not in str(connection_error.value)
    assert "private request text" not in str(model_error.value)


def test_openai_connection_error_is_safe_and_actionable():
    client = FakeOpenAIClient(error=ConnectionError("secret-key and private request text"))

    with pytest.raises(RuntimeError, match="check API configuration") as error:
        OpenAIEmbedder(client, "model").embed(["a"])

    assert "secret-key" not in str(error.value)
    assert "private request text" not in str(error.value)
