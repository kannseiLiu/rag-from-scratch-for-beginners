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
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".json"}
)
CONFIG_FILENAMES = frozenset({".env.example"})

# This historical plan spells out the personal-path marker as a contract
# example. It is not tutorial content or configuration, so scanning that one
# record would make this check self-referential. No other tracked planning
# documents are excluded.
PLANNING_RECORDS_WITH_LITERAL_CONTRACT_EXAMPLES = frozenset(
    {Path("docs/superpowers/plans/2026-09-15-beginner-pdf-rag-tutorial.md")}
)


def _is_contract_text(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name in CONFIG_FILENAMES


def _tracked_contract_paths() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    return sorted(
        path
        for raw_path in tracked
        if raw_path
        for path in [Path(raw_path.decode("utf-8"))]
        if _is_contract_text(path)
        and path not in PLANNING_RECORDS_WITH_LITERAL_CONTRACT_EXAMPLES
    )


def _personal_path_offenders(contents: dict[Path, str]) -> list[Path]:
    absolute_path_marker = "/Use" + "rs/"
    return [path for path, content in contents.items() if absolute_path_marker in content]


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
        "## Citation (ACL)",
        "Dense Passage Retrieval for Open-Domain Question Answering",
        "Vladimir Karpukhin",
        "Barlas Oguz",
        "Sewon Min",
        "Patrick Lewis",
        "Ledell Wu",
        "Sergey Edunov",
        "Danqi Chen",
        "Wen-tau Yih",
        "2020",
        "EMNLP",
        "6769–6781",
        "https://doi.org/10.18653/v1/2020.emnlp-main.550",
        SOURCE_URL,
        "https://aclanthology.org/2020.emnlp-main.550/",
        "CC BY 4.0",
        LICENSE_URL,
        EXPECTED_SHA256,
    ):
        assert required_text in attribution


def test_tracked_contract_paths_include_dotenv_example() -> None:
    paths = _tracked_contract_paths()
    assert Path(".env.example") in paths
    assert all(_is_contract_text(path) for path in paths)


def test_personal_path_detector_rejects_path_in_dotenv_example() -> None:
    absolute_path_marker = "/Use" + "rs/"
    contents = {Path(".env.example"): f"PDF_PATH={absolute_path_marker}private.pdf"}
    assert _personal_path_offenders(contents) == [Path(".env.example")]


def test_tracked_user_facing_text_has_no_personal_absolute_path() -> None:
    contents = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in _tracked_contract_paths()
    }
    assert _personal_path_offenders(contents) == []
