from __future__ import annotations

from pathlib import Path


def test_readme_links_framework_pipeline_document() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doc = Path("docs/architecture/framework_pipeline.md")

    assert "docs/architecture/framework_pipeline.md" in readme
    assert doc.exists()
    assert "Quant Research Framework Pipeline v0" in doc.read_text(encoding="utf-8")


def test_readme_links_framework_v1_acceptance_document() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doc = Path("docs/validation/framework_v1_acceptance.md")

    assert "docs/validation/framework_v1_acceptance.md" in readme
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "Framework v1 Acceptance Plan" in text
    assert "Failure Gates" in text


def test_readme_links_production_strategy_framework_document() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doc = Path("docs/architecture/production_strategy_framework.md")

    assert "docs/architecture/production_strategy_framework.md" in readme
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "Production Strategy Framework and Roadmap" in text
    assert "Technical Roadmap" in text


def test_readme_links_factor_admission_document() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doc = Path("docs/validation/factor_admission.md")

    assert "docs/validation/factor_admission.md" in readme
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "Factor Admission Plan" in text
    assert "Default Gates" in text


def test_readme_links_factor_development_standard_document() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doc = Path("docs/validation/factor_development_standard.md")

    assert "docs/validation/factor_development_standard.md" in readme
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "Factor Development Standard" in text
    assert "Unified Candidate Review Format" in text
    assert "run_ml_challenger_standard_workflow.py" in text


def test_readme_links_candidate_factor_portfolio_document() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doc = Path("docs/strategy/candidate_factor_portfolios.md")

    assert "docs/strategy/candidate_factor_portfolios.md" in readme
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "Candidate Factor Portfolio Experiments" in text
    assert "Score Construction" in text


def test_readme_links_ml_factor_challenger_document() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    doc = Path("docs/strategy/ml_factor_challenger.md")

    assert "docs/strategy/ml_factor_challenger.md" in readme
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "ML Factor Challenger" in text
    assert "Standard No-Leak Workflow" in text
