"""Offline contracts for GitHub contribution guidance."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_github_templates_exist() -> None:
    for path in (
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/learning_question.yml",
        ".github/pull_request_template.md",
        "CONTRIBUTING.md",
    ):
        assert (ROOT / path).is_file(), path


def test_bug_report_collects_reproduction_context_without_private_data() -> None:
    template = _read(".github/ISSUE_TEMPLATE/bug_report.yml")

    for field_id in ("os", "python_version", "provider", "model_names", "steps", "error"):
        assert f"id: {field_id}" in template
    assert "API key" in template
    assert "私人 PDF" in template
    assert "原文" in template


def test_learning_question_points_to_one_tutorial_step() -> None:
    template = _read(".github/ISSUE_TEMPLATE/learning_question.yml")

    assert "id: tutorial_step" in template
    assert "哪一步" in template
    assert "已尝试" in template
    assert "API key" in template
    assert "私人 PDF" in template


def test_contribution_guides_cover_setup_tests_style_and_privacy() -> None:
    guide = _read("CONTRIBUTING.md")
    pull_request = _read(".github/pull_request_template.md")

    for expected in (
        'python -m pip install -e ".[test]"',
        "python -m pytest -q",
        "代码风格",
        "API key",
        "私人 PDF",
    ):
        assert expected in guide
    assert "最小" in guide
    assert "python -m pytest -q" in pull_request
    assert "API key" in pull_request
    assert "私人 PDF" in pull_request
