from dataclasses import FrozenInstanceError

import pytest

import beginner_pdf_rag.config as config
from beginner_pdf_rag.config import Settings, SettingsError
from beginner_pdf_rag.models import Chunk, Page, SearchResult


PROVIDER_ENVIRONMENT_VARIABLES = (
    "RAG_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_CHAT_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_CHAT_MODEL",
)


@pytest.fixture(autouse=True)
def isolate_provider_environment(monkeypatch):
    for variable in PROVIDER_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda: None)


def test_models_are_immutable_data_objects():
    page = Page(number=1, text="A page")
    chunk = Chunk(page=1, index=0, text="A chunk")
    result = SearchResult(chunk=chunk, score=0.9)

    assert (page.number, page.text) == (1, "A page")
    assert (chunk.page, chunk.index, chunk.text) == (1, 0, "A chunk")
    assert (result.chunk, result.score) == (chunk, 0.9)
    with pytest.raises(FrozenInstanceError):
        page.text = "Changed"
    with pytest.raises(FrozenInstanceError):
        chunk.index = 1
    with pytest.raises(FrozenInstanceError):
        result.score = 0.8


def test_settings_are_immutable():
    settings = Settings.from_env("ollama")

    with pytest.raises(FrozenInstanceError):
        settings.provider = "openai"


def test_settings_loads_dotenv_in_production(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "load_dotenv", lambda: calls.append(True))

    Settings.from_env("ollama")

    assert calls == [True]


def test_ollama_defaults_do_not_require_an_api_key():
    settings = Settings.from_env("ollama")

    assert settings.provider == "ollama"
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.chat_model == "qwen3:4b"
    assert settings.openai_api_key is None
    assert settings.openai_base_url == "https://api.openai.com/v1"


def test_openai_requires_a_nonblank_api_key_and_setup_instruction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    with pytest.raises(SettingsError, match="OPENAI_API_KEY") as error:
        Settings.from_env("openai")

    assert "cp .env.example .env" in str(error.value)


def test_openai_reads_its_models_and_base_url_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embedding-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "chat-test")

    settings = Settings.from_env("openai")

    assert settings.provider == "openai"
    assert settings.embedding_model == "embedding-test"
    assert settings.chat_model == "chat-test"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_base_url == "https://example.test/v1"


def test_unknown_provider_explains_supported_choices_and_setup(monkeypatch):
    with pytest.raises(SettingsError, match="ollama.*openai") as error:
        Settings.from_env("local")

    assert "cp .env.example .env" in str(error.value)
