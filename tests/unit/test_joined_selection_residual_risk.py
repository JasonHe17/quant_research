from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from examples.analyze_joined_selection_residual_risk import (
    analyze_joined_selection_residual_risk,
)


def test_analyze_joined_selection_residual_risk_joins_monthly_diagnostics(
    tmp_path: Path,
) -> None:
    validation_dir = tmp_path / "validation"
    diagnostics_dir = (
        validation_dir / "full_base" / "scores" / "decorrelated" / "diagnostics"
    )
    health_dir = validation_dir / "full_base" / "factor_health"
    diagnostics_dir.mkdir(parents=True)
    health_dir.mkdir(parents=True)
    event_summary_path = tmp_path / "monthly_event_state_summary.csv"
    event_perf_path = tmp_path / "monthly_event_state_performance.csv"
    schedule_path = tmp_path / "gross_exposure_schedule.csv"
    output_dir = tmp_path / "out"

    pd.DataFrame(
        [
            {
                "scenario": "full_base",
                "method": "decorrelated",
                "policy": "partial_rebalance_daily",
                "month": "2024-01",
                "return": -0.10,
                "end_equity": 900000.0,
                "max_drawdown": -0.15,
                "trade_count": 10,
                "total_transaction_cost": 100.0,
                "gross_traded_notional": 2000.0,
            },
            {
                "scenario": "full_base",
                "method": "decorrelated",
                "policy": "partial_rebalance_daily",
                "month": "2024-02",
                "return": 0.05,
                "end_equity": 945000.0,
                "max_drawdown": -0.03,
                "trade_count": 8,
                "total_transaction_cost": 80.0,
                "gross_traded_notional": 1500.0,
            },
        ]
    ).to_csv(validation_dir / "validation_monthly_summary.csv", index=False)
    _write_contribution(
        diagnostics_dir / "factor_contribution_2024_01.csv",
        top_labels=(-0.01, -0.02),
        largest_features=("feature_a", "feature_b"),
    )
    _write_contribution(
        diagnostics_dir / "factor_contribution_2024_02.csv",
        top_labels=(0.01, 0.02),
        largest_features=("feature_a", "feature_a"),
    )
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02 09:35:00",
                "feature": "feature_a",
                "health_score": 0.2,
                "recommended_weight_scale": 0.25,
                "weight_scale": 1.0,
                "rolling_rank_ic": -0.01,
                "rolling_top_label": -0.02,
                "rolling_bottom_label": 0.01,
                "rolling_top_minus_bottom_label": -0.03,
                "health_state": "impaired",
            },
            {
                "timestamp": "2024-01-02 09:40:00",
                "feature": "feature_b",
                "health_score": 0.4,
                "recommended_weight_scale": 0.50,
                "weight_scale": 1.0,
                "rolling_rank_ic": 0.01,
                "rolling_top_label": -0.01,
                "rolling_bottom_label": 0.00,
                "rolling_top_minus_bottom_label": -0.01,
                "health_state": "watch",
            },
            {
                "timestamp": "2024-02-01 09:35:00",
                "feature": "feature_a",
                "health_score": 0.8,
                "recommended_weight_scale": 1.0,
                "weight_scale": 1.0,
                "rolling_rank_ic": 0.03,
                "rolling_top_label": 0.01,
                "rolling_bottom_label": -0.01,
                "rolling_top_minus_bottom_label": 0.02,
                "health_state": "healthy",
            },
            {
                "timestamp": "2024-02-01 09:40:00",
                "feature": "feature_b",
                "health_score": 0.9,
                "recommended_weight_scale": 1.0,
                "weight_scale": 1.0,
                "rolling_rank_ic": 0.02,
                "rolling_top_label": 0.02,
                "rolling_bottom_label": -0.01,
                "rolling_top_minus_bottom_label": 0.03,
                "health_state": "healthy",
            },
        ]
    ).to_csv(health_dir / "factor_health_schedule.csv", index=False)
    pd.DataFrame(
        [
            {
                "month": "2024-01",
                "timestamp_count": 2,
                "market_mean_label": -0.01,
                "score_rank_ic_mean": -0.1,
                "score_top_n_mean_label": -0.02,
                "score_top_minus_universe_label": -0.01,
                "score_top_minus_bottom_label": -0.03,
                "event_intensity_mean": 1.0,
                "limit_pressure_rate_mean": 0.2,
                "state_share_limit_diffusion": 0.30,
                "state_share_limit_diffusion_extreme": 0.20,
                "state_share_post_shock_stabilization": 0.10,
                "state_share_shock_elevated": 0.05,
                "state_share_shock_extreme": 0.00,
            },
            {
                "month": "2024-02",
                "timestamp_count": 2,
                "market_mean_label": 0.01,
                "score_rank_ic_mean": 0.1,
                "score_top_n_mean_label": 0.02,
                "score_top_minus_universe_label": 0.01,
                "score_top_minus_bottom_label": 0.03,
                "event_intensity_mean": 0.2,
                "limit_pressure_rate_mean": 0.0,
                "state_share_limit_diffusion": 0.00,
                "state_share_limit_diffusion_extreme": 0.00,
                "state_share_post_shock_stabilization": 0.10,
                "state_share_shock_elevated": 0.00,
                "state_share_shock_extreme": 0.00,
            },
        ]
    ).to_csv(event_summary_path, index=False)
    pd.DataFrame(
        [
            {
                "month": "2024-01",
                "event_state": "limit_diffusion",
                "timestamp_count": 1,
                "score_top_n_mean_label": -0.03,
                "month_state_share": 0.5,
            },
            {
                "month": "2024-01",
                "event_state": "calm",
                "timestamp_count": 1,
                "score_top_n_mean_label": -0.01,
                "month_state_share": 0.5,
            },
            {
                "month": "2024-02",
                "event_state": "calm",
                "timestamp_count": 2,
                "score_top_n_mean_label": 0.02,
                "month_state_share": 1.0,
            },
        ]
    ).to_csv(event_perf_path, index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02 09:35:00",
                "source_event_state": "calm",
                "effective_event_state": "limit_diffusion",
                "gross_exposure_scale": 0.0,
                "event_state_gate_reason": "blocked_event_state",
            },
            {
                "timestamp": "2024-01-02 09:40:00",
                "source_event_state": "calm",
                "effective_event_state": "calm",
                "gross_exposure_scale": 1.0,
                "event_state_gate_reason": "full_event_state",
            },
            {
                "timestamp": "2024-02-01 09:35:00",
                "source_event_state": "calm",
                "effective_event_state": "calm",
                "gross_exposure_scale": 1.0,
                "event_state_gate_reason": "full_event_state",
            },
            {
                "timestamp": "2024-02-01 09:40:00",
                "source_event_state": "calm",
                "effective_event_state": "calm",
                "gross_exposure_scale": 1.0,
                "event_state_gate_reason": "full_event_state",
            },
        ]
    ).to_csv(schedule_path, index=False)

    summary = analyze_joined_selection_residual_risk(
        argparse.Namespace(
            validation_dir=str(validation_dir),
            scenario="full_base",
            method="decorrelated",
            policy="partial_rebalance_daily",
            event_state_summary=str(event_summary_path),
            event_state_performance=str(event_perf_path),
            exposure_schedule=str(schedule_path),
            factor_health_schedule=None,
            output_dir=str(output_dir),
            loss_threshold=0.0,
            drawdown_threshold=-0.10,
            report_months=2,
        )
    )

    residual = pd.read_csv(output_dir / "monthly_residual_risk.csv")
    assert summary["loss_month_count"] == 1
    assert summary["drawdown_month_count"] == 1
    assert residual.loc[0, "month"] == "2024-01"
    assert residual.loc[0, "loss_month"]
    assert residual.loc[0, "event_state_toxic_share"] == 0.5
    assert residual.loc[0, "gate_blocked_scale_share"] == 0.5
    assert residual.loc[0, "contribution_negative_top_label_share"] == 1.0
    assert residual.loc[0, "health_impaired_share"] == 0.5
    assert (output_dir / "residual_risk_report.md").exists()


def _write_contribution(
    path: Path,
    *,
    top_labels: tuple[float, float],
    largest_features: tuple[str, str],
) -> None:
    pd.DataFrame(
        [
            {
                "timestamp": f"2024-01-02 09:{35 + index * 5}:00",
                "top_n": 50,
                "label_column": "forward_return",
                "top_score_mean_label": top_labels[index],
                "largest_contribution_feature": largest_features[index],
                "largest_abs_contribution_share": 0.5,
                "top_two_abs_contribution_share": 0.8,
                "total_abs_contribution": 10.0,
            }
            for index in range(2)
        ]
    ).to_csv(path, index=False)
