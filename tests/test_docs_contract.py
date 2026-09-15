"""Executable reader-facing contracts for the beginner documentation."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]
CANONICAL_REPO = "https://github.com/kannseiLiu/rag-from-scratch-for-beginners.git"
CLI_COMMAND = 'pdf-rag ask --pdf data/dpr-paper.pdf --question "What datasets are used to evaluate DPR?" --show-sources'


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _fenced_blocks(markdown: str, language: str | None = None) -> list[str]:
    pattern = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)
    return [body for lang, body in pattern.findall(markdown) if language is None or lang.strip() == language]


def _slug(heading: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff -]", "", heading.lower()).strip().replace(" ", "-")


def _env_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _read(".env.example").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_required_documentation_files_exist() -> None:
    for path in ("README.md", "docs/concepts.md", "docs/model-guide.md", "docs/troubleshooting.md", "LICENSE"):
        assert (ROOT / path).is_file(), path


def test_quick_start_blocks_are_exact_and_platform_specific() -> None:
    readme = _read("README.md")
    mac_blocks = _fenced_blocks(readme, "sh")
    windows_blocks = _fenced_blocks(readme, "powershell")
    mac_setup = next(block for block in mac_blocks if "python3 -m venv .venv" in block)
    windows_setup = next(block for block in windows_blocks if "py -3.11 -m venv .venv" in block)

    def commands(block: str) -> list[str]:
        return [line.strip() for line in block.splitlines() if line.strip() and not line.lstrip().startswith("#")]

    assert commands(mac_setup) == [
        f"git clone {CANONICAL_REPO} pdf-rag",
        "cd pdf-rag",
        "python3 -m venv .venv",
        "source .venv/bin/activate",
        "python -m pip install --upgrade pip",
        'python -m pip install -e ".[test]"',
    ]
    assert commands(windows_setup) == [
        f"git clone {CANONICAL_REPO} pdf-rag",
        "Set-Location pdf-rag",
        "py -3.11 -m venv .venv",
        ".\\.venv\\Scripts\\Activate.ps1",
        "python -m pip install --upgrade pip",
        'python -m pip install -e ".[test]"',
    ]
    assert "<本仓库地址>" not in readme


def test_cli_and_console_script_are_tied_to_implementation() -> None:
    readme = _read("README.md")
    cli_source = _read("src/beginner_pdf_rag/cli.py")
    project = tomllib.loads(_read("pyproject.toml"))
    assert CLI_COMMAND in readme
    assert readme.count(CLI_COMMAND) == 1
    assert project["project"]["scripts"]["pdf-rag"] == "beginner_pdf_rag.cli:main"
    assert "def main(" in cli_source
    assert 'ask.add_argument("--show-sources", action="store_true")' in cli_source


def test_documented_provider_models_and_urls_match_env_and_settings_source() -> None:
    env = _env_values()
    config_source = _read("src/beginner_pdf_rag/config.py")
    cli_source = _read("src/beginner_pdf_rag/cli.py")
    expected_defaults = {
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
        "OLLAMA_CHAT_MODEL": "qwen3:4b",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_CHAT_MODEL": "gpt-4.1-mini",
    }
    for key, value in expected_defaults.items():
        assert env[key] == value
        assert re.search(rf'os\.getenv\(\s*"{re.escape(key)}",\s*"{re.escape(value)}"\s*\)', config_source)
        assert key in _read("README.md") and value in _read("README.md")
    assert env["RAG_PROVIDER"] == "ollama"
    assert 'os.getenv("RAG_PROVIDER", "ollama")' in cli_source
    assert 'choices=("ollama", "openai")' in cli_source
    assert "OPENAI_API_KEY=replace-me" in _read(".env.example")


def test_secrets_are_safe_and_env_file_is_ignored() -> None:
    assert ".env" in _read(".gitignore").splitlines()
    relevant = "\n".join(_read(path) for path in ("README.md", ".env.example", "docs/model-guide.md", "docs/troubleshooting.md"))
    assert not re.search(r"OPENAI_API_KEY\s*=\s*sk-[A-Za-z0-9_-]{16,}", relevant)
    assert not re.search(r"\bsk-[A-Za-z0-9_-]{16,}\b", relevant)


def test_readme_has_page_citation_and_expected_sources_contract() -> None:
    readme = _read("README.md")
    output = next(block for block in _fenced_blocks(readme, "text") if "Indexed 61 chunks." in block)
    assert "[Page 2]" in output
    assert "Sources" in output
    assert "Page | Score" in output
    assert re.search(r"---- \| -----\n\d+ \| 0\.\d{3}", output)
    assert "--show-sources" in CLI_COMMAND
    assert "更换模型不影响分块数" in readme
    assert "更换 PDF 或提取出的文本可能改变分块数" in readme
    assert "换 PDF 或模型不会改变分块数" not in readme


def test_windows_test_command_uses_activated_python() -> None:
    readme = _read("README.md")
    assert "Windows PowerShell" in readme
    assert any("python -m pytest -q" in block for block in _fenced_blocks(readme, "powershell"))


def test_readme_headings_anchors_and_local_links_are_valid() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "data/README.md"]
    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8")
        for target in link_pattern.findall(text):
            target = target.strip()
            if re.match(r"(?:https?|mailto):", target):
                continue
            path_part, _, fragment = target.partition("#")
            if path_part:
                assert (markdown_file.parent / path_part).resolve().is_file(), f"{markdown_file}: {target}"
            if fragment:
                target_text = text if not path_part else (markdown_file.parent / path_part).read_text(encoding="utf-8")
                headings = {_slug(match) for match in re.findall(r"^#{1,6}\s+(.+?)\s*$", target_text, re.MULTILINE)}
                ids = set(re.findall(r'id=["\']([^"\']+)["\']', target_text))
                assert fragment in headings or fragment in ids, f"{markdown_file}: #{fragment}"
    for heading in ("RAG 的一条完整路径", "15 分钟跑通：本地 Ollama", "OpenAI-compatible API 路径", "换成自己的 PDF", "测试", "局限与路线图", "贡献、归属与许可证", "故障排查"):
        assert re.search(rf"^##\s+{re.escape(heading)}\s*$", _read("README.md"), re.MULTILINE)


def test_license_and_third_party_paper_are_separated() -> None:
    readme = _read("README.md")
    license_text = _read("LICENSE")
    data_readme = _read("data/README.md")
    assert "MIT License" in readme and "[MIT License](LICENSE)" in readme
    assert "CC BY 4.0" in readme and "https://creativecommons.org/licenses/by/4.0/" in readme
    assert "https://aclanthology.org/2020.emnlp-main.550/" in readme
    assert "repository-authored source code and" in license_text
    assert "third-party material" in license_text and "CC BY 4.0" in license_text
    assert "https://creativecommons.org/licenses/by/4.0/" in data_readme
    assert "https://aclanthology.org/2020.emnlp-main.550.pdf" in data_readme


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
