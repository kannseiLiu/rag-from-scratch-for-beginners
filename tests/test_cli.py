from dataclasses import dataclass

import pytest

from beginner_pdf_rag.models import Chunk, SearchResult


@dataclass(frozen=True)
class FakeAnswer:
    text: str
    sources: tuple[SearchResult, ...]


def test_cli_uses_ollama_when_rag_provider_is_missing(monkeypatch, capsys):
    from beginner_pdf_rag import cli

    calls = []
    monkeypatch.delenv("RAG_PROVIDER", raising=False)
    monkeypatch.setattr(cli.Settings, "from_env", lambda provider: calls.append(provider) or object())
    monkeypatch.setattr(cli, "build_providers", lambda settings: (object(), object()))

    class FakeRag:
        def __init__(self, embedder, generator):
            pass

        def index(self, pdf):
            calls.append(("index", pdf))
            return 1

        def ask(self, question, top_k):
            calls.append(("ask", question, top_k))
            return FakeAnswer("Grounded answer", ())

    monkeypatch.setattr(cli, "PdfRag", FakeRag)

    assert cli.main(["ask", "--pdf", "paper.pdf", "--question", "What changed?"]) == 0
    assert calls == ["ollama", ("index", "paper.pdf"), ("ask", "What changed?", 3)]
    assert "Indexed 1 chunks." in capsys.readouterr().out


def test_cli_uses_environment_or_explicit_provider_and_prints_sources(monkeypatch, capsys):
    from beginner_pdf_rag import cli

    providers = []
    monkeypatch.setenv("RAG_PROVIDER", "openai")
    monkeypatch.setattr(cli.Settings, "from_env", lambda provider: providers.append(provider) or object())
    monkeypatch.setattr(cli, "build_providers", lambda settings: (object(), object()))

    source = SearchResult(Chunk(page=4, index=0, text="evidence"), score=0.875)

    class FakeRag:
        def __init__(self, embedder, generator):
            pass

        def index(self, pdf):
            return 1

        def ask(self, question, top_k):
            return FakeAnswer("Grounded answer [Page 4]", (source,))

    monkeypatch.setattr(cli, "PdfRag", FakeRag)

    assert cli.main(
        [
            "ask",
            "--pdf",
            "paper.pdf",
            "--question",
            "What changed?",
            "--provider",
            "ollama",
            "--top-k",
            "1",
            "--show-sources",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert providers == ["ollama"]
    assert "Grounded answer [Page 4]" in output
    assert "Sources" in output
    assert "4 | 0.875" in output


def test_cli_uses_rag_provider_when_present(monkeypatch, capsys):
    from beginner_pdf_rag import cli

    providers = []
    monkeypatch.setenv("RAG_PROVIDER", "openai")
    monkeypatch.setattr(cli.Settings, "from_env", lambda provider: providers.append(provider) or object())
    monkeypatch.setattr(cli, "build_providers", lambda settings: (object(), object()))

    class FakeRag:
        def __init__(self, embedder, generator):
            pass

        def index(self, pdf):
            return 1

        def ask(self, question, top_k):
            return FakeAnswer("Grounded answer", ())

    monkeypatch.setattr(cli, "PdfRag", FakeRag)

    assert cli.main(["ask", "--pdf", "paper.pdf", "--question", "What changed?"]) == 0
    assert providers == ["openai"]
    assert "Grounded answer" in capsys.readouterr().out


@pytest.mark.parametrize("provider", ["openai", "invalid"])
def test_cli_prints_expected_provider_errors_without_tracebacks(monkeypatch, capsys, provider):
    from beginner_pdf_rag import cli
    from beginner_pdf_rag.config import SettingsError

    monkeypatch.setattr(
        cli.Settings,
        "from_env",
        lambda configured: (_ for _ in ()).throw(SettingsError("Set up the provider.")),
    )

    assert cli.main(
        ["ask", "--pdf", "paper.pdf", "--question", "What changed?", "--provider", provider]
    ) == 1

    error = capsys.readouterr().err
    assert "Set up the provider." in error
    assert "Traceback" not in error


def test_cli_prints_pdf_errors_without_tracebacks(monkeypatch, capsys):
    from beginner_pdf_rag import cli
    from beginner_pdf_rag.pdf_loader import PdfLoadError

    monkeypatch.setattr(cli.Settings, "from_env", lambda provider: object())
    monkeypatch.setattr(cli, "build_providers", lambda settings: (object(), object()))

    class FakeRag:
        def __init__(self, embedder, generator):
            pass

        def index(self, pdf):
            raise PdfLoadError("Select an existing PDF file and try again.")

    monkeypatch.setattr(cli, "PdfRag", FakeRag)

    assert cli.main(["ask", "--pdf", "missing.pdf", "--question", "What changed?"]) == 1

    error = capsys.readouterr().err
    assert "existing PDF" in error
    assert "Traceback" not in error


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true"])
def test_cli_rejects_nonpositive_or_noninteger_top_k(value):
    from beginner_pdf_rag import cli

    with pytest.raises(SystemExit) as error:
        cli.main(["ask", "--pdf", "paper.pdf", "--question", "What changed?", "--top-k", value])

    assert error.value.code == 2
