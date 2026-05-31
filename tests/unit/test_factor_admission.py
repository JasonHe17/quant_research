from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research.validation import (
    FactorAdmissionThresholds,
    build_factor_admission_report,
    write_factor_admission_outputs,
)


def test_factor_admission_classifies_candidates_watchlist_and_rejects(
    tmp_path: Path,
) -> None:
    factor_summary = pd.DataFrame(
        [
            {"feature": "alpha_good", "coverage": 0.99, "timestamp_count": 12, "sample_count": 1200},
            {"feature": "alpha_costly", "coverage": 0.99, "timestamp_count": 12, "sample_count": 1200},
            {"feature": "alpha_weak", "coverage": 0.99, "timestamp_count": 12, "sample_count": 1200},
        ]
    )
    by_timestamp = pd.DataFrame(
        [
            *_rows("alpha_good", [0.03, 0.02, 0.04, 0.03, 0.02, 0.04], 0.003, 0.20),
            *_rows("alpha_costly", [0.03, 0.02, 0.04, 0.03, 0.02, 0.04], 0.00005, 0.90),
            *_rows("alpha_weak", [0.0001, -0.0001, 0.0, 0.0001], 0.002, 0.20),
        ]
    )

    report = build_factor_admission_report(
        benchmark_summary={
            "status": "completed",
            "acceptance": {"overall_status": "pass", "failed_count": 0, "warning_count": 0},
            "backtests": {},
        },
        factor_summary=factor_summary,
        by_timestamp=by_timestamp,
        thresholds=FactorAdmissionThresholds(
            min_timestamp_count=4,
            min_years_observed=3,
            min_stable_years=2,
            min_abs_rank_ic_mean=0.001,
            min_abs_rank_ic_t_stat=1.0,
            min_directional_ic_hit_rate=0.5,
            cost_bps=13.0,
        ),
    )

    statuses = {row["feature"]: row["admission_status"] for row in report["factors"]}
    assert statuses == {
        "alpha_good": "candidate",
        "alpha_costly": "watchlist",
        "alpha_weak": "reject",
    }
    assert report["summary"]["candidate_count"] == 1

    artifacts = write_factor_admission_outputs(report, output_dir=tmp_path)

    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["csv"]).exists()
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    assert "Factor Admission Report" in markdown
    assert "alpha_good" in markdown


def test_factor_admission_supports_inverted_direction() -> None:
    factor_summary = pd.DataFrame(
        [{"feature": "alpha_inverse", "coverage": 1.0, "timestamp_count": 6, "sample_count": 600}]
    )
    by_timestamp = pd.DataFrame(
        _rows("alpha_inverse", [-0.03, -0.02, -0.04, -0.03, -0.02, -0.04], -0.003, 0.10)
    )

    report = build_factor_admission_report(
        benchmark_summary={"status": "completed", "acceptance": {}, "backtests": {}},
        factor_summary=factor_summary,
        by_timestamp=by_timestamp,
        thresholds=FactorAdmissionThresholds(
            min_timestamp_count=6,
            min_years_observed=3,
            min_stable_years=2,
            min_abs_rank_ic_t_stat=1.0,
            min_directional_ic_hit_rate=0.5,
        ),
    )

    row = report["factors"][0]
    assert row["direction"] == "invert"
    assert row["admission_status"] == "candidate"


def test_factor_admission_uses_registry_expected_direction_when_available() -> None:
    factor_summary = pd.DataFrame(
        [{"feature": "alpha_prior", "coverage": 1.0, "timestamp_count": 6, "sample_count": 600}]
    )
    by_timestamp = pd.DataFrame(
        _rows("alpha_prior", [-0.03, -0.02, -0.04, -0.03, -0.02, -0.04], -0.003, 0.10)
    )

    report = build_factor_admission_report(
        benchmark_summary={"status": "completed", "acceptance": {}, "backtests": {}},
        factor_summary=factor_summary,
        by_timestamp=by_timestamp,
        thresholds=FactorAdmissionThresholds(
            min_timestamp_count=6,
            min_years_observed=3,
            min_stable_years=2,
            min_abs_rank_ic_t_stat=1.0,
            min_directional_ic_hit_rate=0.5,
        ),
        feature_expected_directions={"alpha_prior": "long"},
    )

    row = report["factors"][0]
    assert row["direction"] == "long"
    assert row["expected_direction"] == "long"
    assert row["direction_source"] == "registry_expected_direction"
    assert row["directional_ic_hit_rate"] == pytest.approx(0.0)
    assert row["admission_status"] == "reject"


def test_factor_admission_allows_sparse_event_overlay_role() -> None:
    factor_summary = pd.DataFrame(
        [
            {
                "feature": "alpha_eod",
                "coverage": 0.12,
                "timestamp_count": 6,
                "sample_count": 600,
            }
        ]
    )
    by_timestamp = pd.DataFrame(
        _rows("alpha_eod", [0.03, 0.02, 0.04, 0.03, 0.02, 0.04], 0.003, 0.20)
    )

    report = build_factor_admission_report(
        benchmark_summary={"status": "completed", "acceptance": {}, "backtests": {}},
        factor_summary=factor_summary,
        by_timestamp=by_timestamp,
        thresholds=FactorAdmissionThresholds(
            min_timestamp_count=6,
            min_years_observed=3,
            min_stable_years=2,
            min_abs_rank_ic_t_stat=1.0,
            min_directional_ic_hit_rate=0.5,
        ),
        feature_roles={"alpha_eod": "event_overlay"},
    )

    row = report["factors"][0]
    assert row["evaluation_role"] == "event_overlay"
    assert row["admission_status"] == "candidate"
    assert row["failed_checks"] == []
    assert row["informational_failed_checks"] == ["coverage"]
    assert report["summary"]["role_counts"]["event_overlay"]["candidate"] == 1


def test_factor_admission_treats_state_allocator_rank_ic_as_diagnostic() -> None:
    factor_summary = pd.DataFrame(
        [{"feature": "market_state", "coverage": 1.0, "timestamp_count": 6, "sample_count": 600}]
    )
    by_timestamp = pd.DataFrame(
        _rows("market_state", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], -0.001, 0.99)
    )

    report = build_factor_admission_report(
        benchmark_summary={"status": "completed", "acceptance": {}, "backtests": {}},
        factor_summary=factor_summary,
        by_timestamp=by_timestamp,
        thresholds=FactorAdmissionThresholds(
            min_timestamp_count=6,
            min_years_observed=3,
            min_stable_years=2,
            min_abs_rank_ic_mean=0.001,
            min_abs_rank_ic_t_stat=1.0,
            min_directional_ic_hit_rate=0.5,
        ),
        feature_roles={"market_state": "state_allocator"},
    )

    row = report["factors"][0]
    assert row["evaluation_role"] == "state_allocator"
    assert row["admission_status"] == "candidate"
    assert "abs_rank_ic_mean" in row["informational_failed_checks"]
    assert "cost_adjusted_spread" in row["informational_failed_checks"]


def _rows(
    feature: str,
    ic_values: list[float],
    spread: float,
    turnover: float,
) -> list[dict[str, object]]:
    timestamps = [
        "2023-01-03T09:35:00+08:00",
        "2023-06-01T09:35:00+08:00",
        "2024-01-03T09:35:00+08:00",
        "2024-06-01T09:35:00+08:00",
        "2025-01-03T09:35:00+08:00",
        "2025-06-01T09:35:00+08:00",
    ]
    rows = []
    for index, ic in enumerate(ic_values):
        rows.append(
            {
                "feature": feature,
                "timestamp": timestamps[index],
                "sample_count": 100,
                "spearman_rank_ic": ic,
                "top_minus_bottom_label": spread,
                "top_n_turnover": turnover,
            }
        )
    return rows
