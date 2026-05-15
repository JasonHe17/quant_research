"""Factor interfaces and orchestration."""

from quant_research.factors.base import Factor, FactorContext, FactorResult
from quant_research.factors.engine import FactorEngine
from quant_research.factors.evaluation import (
    SingleFactorEvaluationConfig,
    SingleFactorEvaluationResult,
    evaluate_single_factors,
)
from quant_research.factors.registry import (
    FactorRegistry,
    FactorRegistryEntry,
    FactorRegistryIssue,
    FactorRegistryValidationReport,
    load_factor_registry,
    render_factor_registry_markdown,
    validate_factor_registry,
    write_factor_registry_report,
)
from quant_research.factors.review import (
    build_factor_candidate_review,
    load_optional_json,
    render_factor_candidate_review_markdown,
    write_factor_candidate_review,
)

__all__ = [
    "Factor",
    "FactorContext",
    "FactorEngine",
    "FactorResult",
    "SingleFactorEvaluationConfig",
    "SingleFactorEvaluationResult",
    "FactorRegistry",
    "FactorRegistryEntry",
    "FactorRegistryIssue",
    "FactorRegistryValidationReport",
    "evaluate_single_factors",
    "load_factor_registry",
    "render_factor_registry_markdown",
    "validate_factor_registry",
    "write_factor_registry_report",
    "build_factor_candidate_review",
    "load_optional_json",
    "render_factor_candidate_review_markdown",
    "write_factor_candidate_review",
]
