from __future__ import annotations

from pathlib import Path

from quant_research.factors import (
    FactorRegistry,
    FactorRegistryEntry,
    build_factor_candidate_review,
    load_factor_registry,
    validate_factor_registry,
)


def test_seed_factor_registry_is_governance_clean() -> None:
    registry = load_factor_registry("configs/factors/factor_registry.json")

    report = validate_factor_registry(registry)

    assert report.status == "pass"
    assert report.summary["entry_count"] >= 1
    assert report.summary["status_counts"]["candidate"] >= 1


def test_factor_registry_rejects_duplicate_feature_columns() -> None:
    base = _entry("alpha_a", "shared_feature")
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(base, _entry("alpha_b", "shared_feature")),
    )

    report = validate_factor_registry(registry)

    assert report.status == "fail"
    assert any(issue.code == "duplicate_feature_column" for issue in report.issues)


def test_factor_registry_requires_active_point_in_time_safety() -> None:
    entry = _entry("alpha_bad", "alpha_bad_feature", point_in_time_safe=False)
    registry = FactorRegistry(registry_name="test", version=1, entries=(entry,))

    report = validate_factor_registry(registry)

    assert report.status == "fail"
    assert any(issue.code == "point_in_time_not_confirmed" for issue in report.issues)


def test_factor_candidate_review_uses_registry_and_admission_report() -> None:
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(_entry("alpha_good", "alpha_good_feature"),),
    )
    admission_report = {
        "generated_at": "2026-05-15T00:00:00+00:00",
        "summary": {"candidate_count": 1},
        "thresholds": {},
        "factors": [
            {
                "feature": "alpha_good_feature",
                "admission_status": "candidate",
                "direction": "long",
                "spearman_rank_ic_mean": 0.02,
                "spearman_rank_ic_t_stat": 3.0,
                "cost_adjusted_top_minus_bottom_label": 0.001,
            }
        ],
    }

    review = build_factor_candidate_review(
        registry,
        factor_id="alpha_good",
        admission_report=admission_report,
    )

    assert review["status"] == "ready_for_portfolio_review"
    assert review["single_factor_admission"]["rows"][0]["feature"] == "alpha_good_feature"


def test_factor_registry_loader_reads_json() -> None:
    registry = load_factor_registry(Path("configs/factors/factor_registry.json"))

    assert registry.get("intraday_volatility_5m_w24").expected_direction == "invert"


def _entry(
    factor_id: str,
    feature_column: str,
    *,
    point_in_time_safe: bool = True,
) -> FactorRegistryEntry:
    return FactorRegistryEntry(
        factor_id=factor_id,
        display_name=factor_id,
        family="momentum",
        status="candidate",
        expected_direction="long",
        feature_columns=(feature_column,),
        required_inputs=("close_price",),
        frequency="5m",
        description="test factor",
        hypothesis="test hypothesis",
        implementation={
            "module": "quant_research.tests",
            "function": "build_test_factor",
        },
        evaluation={"admission_status": "candidate"},
        a_share_constraints={
            "long_only": True,
            "price_limit_aware": True,
            "st_aware": True,
            "t_plus_one_safe": True,
        },
        references=("tests",),
        point_in_time_safe=point_in_time_safe,
        live_available=True,
    )
