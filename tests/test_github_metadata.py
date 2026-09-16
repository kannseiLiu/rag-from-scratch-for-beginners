"""Offline contracts for GitHub contribution guidance."""

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _parse_issue_form(source: str) -> dict[str, Any]:
    try:
        form = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise AssertionError("Issue Form must contain valid YAML") from exc

    assert isinstance(form, dict), "Issue Form must be a YAML mapping"
    assert isinstance(form.get("name"), str) and form["name"].strip()
    assert isinstance(form.get("description"), str) and form["description"].strip()
    assert isinstance(form.get("body"), list) and form["body"]

    field_ids: list[str] = []
    for field in form["body"]:
        assert isinstance(field, dict), "each Issue Form body item must be a mapping"
        assert field.get("type") in {"markdown", "input", "textarea", "dropdown", "checkboxes"}
        assert isinstance(field.get("attributes"), dict)
        if field["type"] != "markdown":
            assert isinstance(field.get("id"), str) and field["id"].strip()
            field_ids.append(field["id"])

    assert len(field_ids) == len(set(field_ids)), "Issue Form field ids must be unique"
    return form


def _required(field: dict[str, object]) -> bool:
    validations = field.get("validations", {})
    return isinstance(validations, dict) and validations.get("required") is True


def _field_map(path: str) -> dict[str, dict[str, object]]:
    form = _parse_issue_form(_read(path))
    return {
        field["id"]: field
        for field in form["body"]
        if isinstance(field, dict) and isinstance(field.get("id"), str)
    }


def test_github_templates_exist() -> None:
    for path in (
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/learning_question.yml",
        ".github/pull_request_template.md",
        "CONTRIBUTING.md",
    ):
        assert (ROOT / path).is_file(), path


@pytest.mark.parametrize(
    "invalid_form",
    (
        "# name: Hidden\n# description: Hidden\n# body: []\n",
        "name: Broken\ndescription: Missing body\n",
        "name: [unterminated\n",
    ),
)
def test_issue_form_parser_rejects_commented_incomplete_or_invalid_yaml(invalid_form: str) -> None:
    with pytest.raises(AssertionError):
        _parse_issue_form(invalid_form)


def test_bug_report_has_valid_required_fields_and_privacy_confirmation() -> None:
    fields = _field_map(".github/ISSUE_TEMPLATE/bug_report.yml")
    expected = {
        "os": "input",
        "python_version": "input",
        "provider": "dropdown",
        "model_names": "input",
        "steps": "textarea",
        "error": "textarea",
        "expected": "textarea",
        "privacy": "checkboxes",
    }

    assert {field_id: field["type"] for field_id, field in fields.items()} == expected
    assert all(_required(fields[field_id]) for field_id in expected if field_id != "privacy")
    assert fields["provider"]["attributes"]["options"] == ["Ollama", "OpenAI-compatible API"]
    privacy_options = fields["privacy"]["attributes"]["options"]
    assert len(privacy_options) == 1
    assert privacy_options[0]["required"] is True
    assert re.search(r"API key", privacy_options[0]["label"], re.IGNORECASE)
    assert "私人 PDF" in privacy_options[0]["label"]


def test_learning_question_has_required_step_context_and_privacy_confirmation() -> None:
    fields = _field_map(".github/ISSUE_TEMPLATE/learning_question.yml")
    expected = {
        "tutorial_step": "dropdown",
        "question": "textarea",
        "attempted": "textarea",
        "environment": "textarea",
        "privacy": "checkboxes",
    }

    assert {field_id: field["type"] for field_id, field in fields.items()} == expected
    assert all(_required(fields[field_id]) for field_id in ("tutorial_step", "question", "attempted"))
    assert not _required(fields["environment"])
    assert len(fields["tutorial_step"]["attributes"]["options"]) >= 8
    privacy_options = fields["privacy"]["attributes"]["options"]
    assert len(privacy_options) == 1
    assert privacy_options[0]["required"] is True
    assert re.search(r"API key", privacy_options[0]["label"], re.IGNORECASE)
    assert "私人 PDF" in privacy_options[0]["label"]


def test_contribution_guide_covers_setup_tests_style_and_privacy() -> None:
    guide = _read("CONTRIBUTING.md")
    headings = "\n".join(re.findall(r"^##\s+(.+)$", guide, re.MULTILINE))

    for topic in ("安装", "测试", "风格", "隐私", "Pull Request"):
        assert topic in headings
    assert 'python -m pip install -e ".[test]"' in guide
    assert "python -m pytest -q" in guide
    for sensitive_term in ("API key", ".env", "私人 PDF"):
        assert sensitive_term in guide


def test_pull_request_template_has_verification_and_privacy_checklist() -> None:
    pull_request = _read(".github/pull_request_template.md")
    headings = "\n".join(re.findall(r"^##\s+(.+)$", pull_request, re.MULTILINE))

    assert "验证" in headings
    assert "检查" in headings
    assert "python -m pytest -q" in pull_request
    assert pull_request.count("- [ ]") >= 5
    for sensitive_term in ("API key", ".env", "私人 PDF"):
        assert sensitive_term in pull_request


def test_env_ignore_rules_protect_local_secrets_but_keep_example() -> None:
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", "production.env", ".env.local"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    example = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", ".env.example"],
        cwd=ROOT,
        check=False,
    )

    assert ignored.returncode == 0
    assert ignored.stdout.splitlines() == ["production.env", ".env.local"]
    assert example.returncode == 1


def test_pdf_files_are_marked_binary_for_git_diff() -> None:
    attributes = _read(".gitattributes")
    assert "*.pdf binary" in attributes.splitlines()

    result = subprocess.run(
        ["git", "check-attr", "diff", "--", "data/dpr-paper.pdf", "tests/fixtures/two-pages.pdf"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "data/dpr-paper.pdf: diff: unset",
        "tests/fixtures/two-pages.pdf: diff: unset",
    ]
