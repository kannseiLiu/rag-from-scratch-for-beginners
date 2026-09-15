"""Artifact contracts for the progressive examples and bundled paper."""

import ast
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "examples/01_minimal_text_rag.py",
    "examples/02_pdf_rag_ollama.py",
    "examples/03_pdf_rag_openai.py",
)
PAPER = ROOT / "data/dpr-paper.pdf"
EXPECTED_SHA256 = "2abfc7991519be2b8ce30215905516fdb593218b7bf59e50ece0d8a088a7c4d5"
SOURCE_URL = "https://aclanthology.org/2020.emnlp-main.550.pdf"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def test_expected_examples_exist_and_parse() -> None:
    actual_examples = sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / "examples").glob("*.py")
    )
    assert actual_examples == list(EXAMPLES)
    for relative_path in EXAMPLES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        ast.parse(source, filename=relative_path)


def test_sample_paper_is_the_verified_acl_pdf() -> None:
    assert PAPER.read_bytes().startswith(b"%PDF")
    assert hashlib.sha256(PAPER.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_paper_attribution_records_title_authors_source_and_license() -> None:
    attribution = (ROOT / "data/README.md").read_text(encoding="utf-8")
    for required_text in (
        "Dense Passage Retrieval for Open-Domain Question Answering",
        "Vladimir Karpukhin",
        SOURCE_URL,
        "https://aclanthology.org/2020.emnlp-main.550/",
        "CC BY 4.0",
        LICENSE_URL,
        EXPECTED_SHA256,
    ):
        assert required_text in attribution


def test_tracked_user_facing_text_has_no_personal_absolute_path() -> None:
    absolute_path_marker = "/Use" + "rs/"
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
            "*.md",
            "*.toml",
            "*.yaml",
            "*.yml",
            "*.ini",
            "*.cfg",
            "*.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    offenders = [
        path
        for path in tracked
        if not path.startswith((".superpowers/", "docs/superpowers/"))
        and absolute_path_marker in (ROOT / path).read_text(encoding="utf-8")
    ]
    assert offenders == []
