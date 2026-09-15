"""Command-line entrypoint for asking questions about one PDF."""

import argparse
import os
import sys
from collections.abc import Sequence

from beginner_pdf_rag.config import Settings, SettingsError
from beginner_pdf_rag.pdf_loader import PdfLoadError
from beginner_pdf_rag.pipeline import PdfRag, RagError
from beginner_pdf_rag.providers import build_providers


def _positive_int(value: str) -> int:
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question about a PDF.")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="index a PDF and ask one question")
    ask.add_argument("--pdf", required=True, metavar="PATH")
    ask.add_argument("--question", required=True, metavar="TEXT")
    ask.add_argument("--provider", default=os.getenv("RAG_PROVIDER", "ollama"))
    ask.add_argument("--top-k", type=_positive_int, default=3, metavar="N")
    ask.add_argument("--show-sources", action="store_true")
    return parser


def _print_sources(answer_sources: Sequence) -> None:
    print("\nSources")
    print("Page | Score")
    print("---- | -----")
    for result in answer_sources:
        print(f"{result.chunk.page} | {result.score:.3f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``pdf-rag ask`` command and return its shell status."""
    args = _parser().parse_args(argv)
    try:
        settings = Settings.from_env(args.provider)
        embedder, generator = build_providers(settings)
        rag = PdfRag(embedder, generator)
        chunk_count = rag.index(args.pdf)
        answer = rag.ask(args.question, args.top_k)
    except (SettingsError, PdfLoadError, RagError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Indexed {chunk_count} chunks.")
    print(answer.text)
    if args.show_sources:
        _print_sources(answer.sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
