"""Ask the bundled DPR paper with local Ollama models."""

import argparse
from pathlib import Path

from beginner_pdf_rag.config import Settings
from beginner_pdf_rag.pipeline import PdfRag
from beginner_pdf_rag.providers import build_providers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("data/dpr-paper.pdf"))
    parser.add_argument(
        "--question",
        default="What datasets are used to evaluate DPR?",
    )
    args = parser.parse_args()

    settings = Settings.from_env("ollama")
    embedder, generator = build_providers(settings)
    rag = PdfRag(embedder, generator)
    print(f"Indexed {rag.index(args.pdf)} chunks.")
    print(rag.ask(args.question).text)


if __name__ == "__main__":
    main()
