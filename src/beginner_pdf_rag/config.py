import os
from dataclasses import dataclass

from dotenv import load_dotenv


class SettingsError(ValueError):
    """Raised when the provider configuration is incomplete or invalid."""


@dataclass(frozen=True)
class Settings:
    provider: str
    ollama_base_url: str
    embedding_model: str
    chat_model: str
    openai_api_key: str | None
    openai_base_url: str

    @classmethod
    def from_env(cls, provider: str) -> "Settings":
        load_dotenv()

        if provider not in {"ollama", "openai"}:
            raise SettingsError(
                "Provider must be 'ollama' or 'openai'. "
                "Run `cp .env.example .env` to create your configuration."
            )

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if provider == "ollama":
            return cls(
                provider=provider,
                ollama_base_url=ollama_base_url,
                embedding_model=os.getenv(
                    "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
                ),
                chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b"),
                openai_api_key=None,
                openai_base_url=openai_base_url,
            )

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SettingsError(
                "OPENAI_API_KEY is required for the openai provider. "
                "Run `cp .env.example .env` and set OPENAI_API_KEY."
            )
        return cls(
            provider=provider,
            ollama_base_url=ollama_base_url,
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            openai_api_key=api_key,
            openai_base_url=openai_base_url,
        )
