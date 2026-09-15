"""Contract tests for the beginner documentation set.

These tests intentionally check reader-facing promises rather than implementation
details.  The README is the canonical command source for the tutorial.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_required_documentation_files_exist() -> None:
    for path in ("README.md", "docs/concepts.md", "docs/model-guide.md", "docs/troubleshooting.md", "LICENSE"):
        assert (ROOT / path).is_file(), path


def test_readme_contains_runnable_zero_background_lesson() -> None:
    readme = _read("README.md")
    required_phrases = (
        "Python 3.11",
        "git clone",
        "python3 -m venv .venv",
        "python -m venv .venv",
        "pip install -e .",
        "pip install -e .[test]",
        "Ollama",
        "ollama pull nomic-embed-text",
        "ollama pull qwen3:4b",
        "pdf-rag ask",
        "Indexed",
        "OPENAI_API_KEY",
        ".env",
        "Page",
        "OCR",
        "替换",
        "pytest",
        "局限",
        "路线图",
        "Hugging Face",
        "ACL",
        "MIT",
        "故障排查",
    )
    missing = [phrase for phrase in required_phrases if phrase not in readme]
    assert not missing, f"README is missing: {missing}"
    assert "```mermaid" in readme
    assert "15 分钟" in readme
    assert re.search(r"pdf-rag ask[^\n]*--pdf[^\n]*--question", readme)


def test_supporting_guides_cover_required_beginner_concepts() -> None:
    concepts = _read("docs/concepts.md")
    for term in ("token", "chunk", "overlap", "embedding", "向量", "余弦相似度", "Top-K", "context window", "prompt", "幻觉", "cosine_similarity"):
        assert term in concepts, term
    assert "cos" in concepts and "sqrt" in concepts
    assert "二维" in concepts or "2D" in concepts

    model_guide = _read("docs/model-guide.md")
    for term in ("隐私", "安装", "成本", "离线", "速度", "硬件", "embedding model", "chat model", "Ollama", "OpenAI"):
        assert term in model_guide, term

    troubleshooting = _read("docs/troubleshooting.md")
    for term in ("command not found", "Python", "Ollama", "401", "429", "API key", "OCR", "内存", "无关", "无法回答"):
        assert term in troubleshooting, term


def test_all_relative_markdown_links_point_to_existing_files() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "data/README.md"]
    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            target = target.split("#", 1)[0].strip()
            if not target or re.match(r"(?:https?|mailto):", target):
                continue
            assert (markdown_file.parent / target).resolve().is_file(), f"{markdown_file}: {target}"
