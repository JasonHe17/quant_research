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
