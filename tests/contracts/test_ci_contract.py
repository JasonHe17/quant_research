from __future__ import annotations

from pathlib import Path


def test_github_actions_ci_workflow_exists() -> None:
    workflow = Path(".github/workflows/ci.yml")
    text = workflow.read_text(encoding="utf-8")

    assert workflow.exists()
    assert "python -m pip install -e \".[dev]\"" in text
    assert "python -m pytest -q" in text
    assert 'python-version: ["3.11", "3.12"]' in text
