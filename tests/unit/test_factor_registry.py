from __future__ import annotations

from pathlib import Path

from quant_research.factors import (
    FactorRegistry,
    FactorRegistryEntry,
    build_factor_candidate_review,
    find_factor_research_memory_matches,
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


def test_factor_registry_requires_research_memory_for_rejected_factors() -> None:
    entry = _entry("alpha_reject", "alpha_reject_feature", status="reject")
    registry = FactorRegistry(registry_name="test", version=1, entries=(entry,))

    report = validate_factor_registry(registry)

    assert report.status == "fail"
    assert any(issue.code == "missing_research_memory" for issue in report.issues)


def test_factor_registry_accepts_structured_research_memory() -> None:
    entry = _entry(
        "alpha_watch",
        "alpha_watch_feature",
        status="watchlist",
        research_memory={
            "decision_reason": "unstable_years",
            "negative_findings": "Useful IC but unstable annual slices.",
            "similar_to": ["alpha_parent"],
            "retry_conditions": "Retry only with a regime gate or orthogonal transform.",
            "evidence_artifacts": ["runs/test/factor_admission_report.json"],
        },
    )
    registry = FactorRegistry(registry_name="test", version=1, entries=(entry,))

    report = validate_factor_registry(registry)

    assert report.status == "pass"


def test_factor_registry_accepts_portfolio_native_evaluation_role() -> None:
    entry = _entry(
        "alpha_risk_penalty",
        "alpha_risk_penalty_feature",
        evaluation_role="risk_penalty",
    )
    registry = FactorRegistry(registry_name="test", version=1, entries=(entry,))

    report = validate_factor_registry(registry)

    assert report.status == "pass"
    assert report.entries[0]["evaluation_role"] == "risk_penalty"


def test_factor_registry_rejects_unknown_evaluation_role() -> None:
    entry = _entry(
        "alpha_bad_role",
        "alpha_bad_role_feature",
        evaluation_role="not_a_role",
    )
    registry = FactorRegistry(registry_name="test", version=1, entries=(entry,))

    report = validate_factor_registry(registry)

    assert report.status == "fail"
    assert any(issue.code == "unknown_evaluation_role" for issue in report.issues)


def test_factor_research_memory_matches_rejected_similar_factor() -> None:
    rejected = _entry(
        "intraday_vwap_deviation_5m_w48",
        "intraday_vwap_deviation_5m_w48",
        family="reversal",
        status="reject",
        required_inputs=("close_price", "volume", "turnover"),
        lookback_bars=48,
        research_memory={
            "decision_reason": "weak_hit_rate",
            "negative_findings": "Negative after cost-aware portfolio validation.",
            "similar_to": ["intraday_range_position_5m_w48"],
            "retry_conditions": "Retry only with a materially different transform.",
            "evidence_artifacts": ["runs/test/factor_candidate_review.json"],
        },
    )
    registry = FactorRegistry(registry_name="test", version=1, entries=(rejected,))

    matches = find_factor_research_memory_matches(
        registry,
        factor_id="intraday_vwap_reversal_redesign_5m_w48",
        family="reversal",
        required_inputs=("close_price", "volume", "turnover"),
        lookback_bars=48,
        keywords=("vwap", "deviation", "reversal"),
    )

    assert len(matches) == 1
    assert matches[0].factor_id == "intraday_vwap_deviation_5m_w48"
    assert matches[0].blocking is True
    assert "family" in matches[0].matched_fields
    assert "required_inputs" in matches[0].matched_fields


def test_factor_research_memory_filters_low_similarity() -> None:
    rejected = _entry(
        "intraday_liquidity_noise_5m_w12",
        "intraday_liquidity_noise_5m_w12",
        family="liquidity",
        status="reject",
        required_inputs=("bid_ask_spread",),
        lookback_bars=12,
        research_memory={
            "decision_reason": "weak_ic",
            "negative_findings": "No stable IC.",
            "similar_to": [],
            "retry_conditions": "Retry only with better microstructure inputs.",
            "evidence_artifacts": ["runs/test/factor_admission_report.json"],
        },
    )
    registry = FactorRegistry(registry_name="test", version=1, entries=(rejected,))

    matches = find_factor_research_memory_matches(
        registry,
        factor_id="overnight_gap_reversal",
        family="reversal",
        required_inputs=("open_price", "close_price"),
        lookback_bars=96,
        keywords=("overnight", "gap"),
        min_score=0.35,
    )

    assert matches == ()


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


def test_factor_candidate_review_summarizes_candidate_portfolio_backtest() -> None:
    registry = FactorRegistry(
        registry_name="test",
        version=1,
        entries=(_entry("alpha_good", "alpha_good_feature"),),
    )
    admission_report = {
        "factors": [
            {
                "feature": "alpha_good_feature",
                "admission_status": "candidate",
                "direction": "long",
            }
        ],
    }
    portfolio_summary = {
        "candidate_features": ["alpha_good_feature"],
        "methods": {"equal": {"row_count": 3}},
        "backtest_summary": [
            {
                "method": "equal",
                "policy": "single",
                "total_return": -0.10,
                "max_drawdown": -0.20,
                "gross_turnover": 12.0,
            }
        ],
    }

    review = build_factor_candidate_review(
        registry,
        factor_id="alpha_good",
        admission_report=admission_report,
        portfolio_validation=portfolio_summary,
    )

    summary = review["portfolio_validation"]
    assert review["status"] == "blocked"
    assert summary["summary_type"] == "candidate_factor_portfolio"
    assert summary["overall_status"] == "fail"
    assert summary["result_count"] == 1
    assert summary["primary_result"]["total_return"] == -0.10


def test_factor_registry_loader_reads_json() -> None:
    registry = load_factor_registry(Path("configs/factors/factor_registry.json"))

    assert registry.get("intraday_volatility_5m_w24").expected_direction == "invert"


def _entry(
    factor_id: str,
    feature_column: str,
    *,
    point_in_time_safe: bool = True,
    status: str = "candidate",
    family: str = "momentum",
    required_inputs: tuple[str, ...] = ("close_price",),
    lookback_bars: int | None = None,
    research_memory: dict[str, object] | None = None,
    evaluation_role: str = "alpha_rank",
) -> FactorRegistryEntry:
    return FactorRegistryEntry(
        factor_id=factor_id,
        display_name=factor_id,
        family=family,
        status=status,
        expected_direction="long",
        feature_columns=(feature_column,),
        required_inputs=required_inputs,
        frequency="5m",
        description="test factor",
        hypothesis="test hypothesis",
        implementation={
            "module": "quant_research.tests",
            "function": "build_test_factor",
        },
        evaluation={"admission_status": "candidate"},
        research_memory=research_memory or {},
        lookback_bars=lookback_bars,
        a_share_constraints={
            "long_only": True,
            "price_limit_aware": True,
            "st_aware": True,
            "t_plus_one_safe": True,
        },
        references=("tests",),
        point_in_time_safe=point_in_time_safe,
        live_available=True,
        evaluation_role=evaluation_role,
    )
