from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from examples.compare_policy_backtests import compare_policy_backtests


def test_compare_policy_backtests_writes_overall_and_monthly_outputs(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "comparison"
    _write_backtest(
        baseline,
        final_equity=1_010_000.0,
        equity=[("2024-01-31T15:00:00+08:00", 1_010_000.0)],
        trade_cost=100.0,
    )
    _write_backtest(
        candidate,
        final_equity=1_020_000.0,
        equity=[("2024-01-31T15:00:00+08:00", 1_020_000.0)],
        trade_cost=80.0,
    )

    summary = compare_policy_backtests(
        argparse.Namespace(
            baseline_dir=str(baseline),
            candidate_dir=str(candidate),
            output_dir=str(output),
            baseline_name="base",
            candidate_name="gate",
            initial_cash=1_000_000.0,
            start=None,
            end=None,
        )
    )

    monthly = pd.read_csv(output / "monthly_comparison.csv")
    assert summary["artifacts"]["monthly_comparison"] == str(
        output / "monthly_comparison.csv"
    )
    assert monthly.loc[0, "return_baseline"] == pytest.approx(0.01)
    assert monthly.loc[0, "return_candidate"] == pytest.approx(0.02)
    assert monthly.loc[0, "return_delta"] == pytest.approx(0.01)
    assert monthly.loc[0, "total_transaction_cost_delta"] == pytest.approx(-20.0)


def _write_backtest(
    path: Path,
    *,
    final_equity: float,
    equity: list[tuple[str, float]],
    trade_cost: float,
) -> None:
    path.mkdir(parents=True)
    summary = {
        "metrics": {
            "total_return": final_equity / 1_000_000.0 - 1.0,
            "max_drawdown": 0.0,
            "gross_turnover": 1.0,
            "trade_count": 1,
            "total_transaction_cost": trade_cost,
            "final_equity": final_equity,
        },
        "policy_diagnostics": {
            "average_target_gross_exposure": 1.0,
            "planned_gross_turnover": 1.0,
            "risk_reduction_count": 0,
        },
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {"timestamp": timestamp, "equity": value}
            for timestamp, value in equity
        ]
    ).to_csv(path / "equity_curve.csv", index=False)
    pd.DataFrame(
        [
            {
                "timestamp": equity[0][0],
                "notional": 10_000.0,
                "total_cost": trade_cost,
            }
        ]
    ).to_csv(path / "trades.csv", index=False)
