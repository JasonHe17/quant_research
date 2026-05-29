from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    path = Path("examples/build_backtest_adaptive_state_switch.py")
    examples_dir = str(path.parent.resolve())
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location(
        "build_backtest_adaptive_state_switch",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selection_schedule_switch_penalty_can_keep_previous_method() -> None:
    module = _load_module()
    backtests = {
        "baseline": _backtest(
            [
                ("2024-02-17", 100.0),
                ("2024-02-26", 100.0),
                ("2024-03-19", 100.0),
                ("2024-03-28", 100.0),
            ]
        ),
        "candidate_a": _backtest(
            [
                ("2024-02-17", 100.0),
                ("2024-02-26", 105.0),
                ("2024-03-19", 100.0),
                ("2024-03-28", 104.0),
            ]
        ),
        "candidate_b": _backtest(
            [
                ("2024-02-17", 100.0),
                ("2024-02-26", 101.0),
                ("2024-03-19", 100.0),
                ("2024-03-28", 107.0),
            ]
        ),
    }

    without_penalty = module._selection_schedule(
        backtests,
        candidates=("candidate_a", "candidate_b"),
        start="2024-03-01T00:00:00+08:00",
        end="2024-04-30T23:59:59+08:00",
        frequency="monthly",
        lookback_days=10,
        embargo_days=3,
        min_equity_points=2,
        min_objective_edge=0.0,
        return_weight=1.0,
        drawdown_penalty=0.0,
        turnover_penalty=0.0,
        cost_penalty=0.0,
        switch_penalty=0.0,
        fallback_to_baseline=True,
    )
    with_penalty = module._selection_schedule(
        backtests,
        candidates=("candidate_a", "candidate_b"),
        start="2024-03-01T00:00:00+08:00",
        end="2024-04-30T23:59:59+08:00",
        frequency="monthly",
        lookback_days=10,
        embargo_days=3,
        min_equity_points=2,
        min_objective_edge=0.0,
        return_weight=1.0,
        drawdown_penalty=0.0,
        turnover_penalty=0.0,
        cost_penalty=0.0,
        switch_penalty=0.04,
        fallback_to_baseline=True,
    )

    assert without_penalty["selected_method"].tolist() == [
        "candidate_a",
        "candidate_b",
    ]
    assert with_penalty["selected_method"].tolist() == [
        "candidate_a",
        "candidate_a",
    ]
    assert with_penalty["previous_selected_method"].tolist() == [
        "baseline",
        "candidate_a",
    ]
    assert with_penalty["switch_penalty_applied"].tolist() == [0.04, 0.0]


def test_daily_schedule_attaches_selected_policy_overrides() -> None:
    module = _load_module()
    selector = pd.DataFrame(
        [
            {
                "period_start": "2024-03-01",
                "period_end": "2024-03-02",
                "selected_method": "candidate_a",
                "baseline_fallback": False,
                "selected_objective": 0.10,
                "baseline_objective": 0.05,
                "selected_objective_edge": 0.05,
            }
        ]
    )

    schedule = module._daily_schedule(
        selector,
        start="2024-03-01T00:00:00+08:00",
        end="2024-03-31T23:59:59+08:00",
        policy_by_method={
            "candidate_a": {
                "policy_exit_rank": 50,
                "policy_force_source_transition_exits": False,
            },
        },
    )

    assert schedule["trade_date"].tolist() == ["2024-03-01", "2024-03-02"]
    assert schedule["selected_method"].tolist() == ["candidate_a", "candidate_a"]
    assert schedule["policy_exit_rank"].tolist() == [50, 50]
    assert schedule["policy_force_source_transition_exits"].tolist() == [False, False]


def _backtest(equity: list[tuple[str, float]]) -> dict[str, pd.DataFrame]:
    return {
        "equity": pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(f"{trade_date}T15:00:00+08:00"),
                    "trade_date": pd.Timestamp(trade_date).date(),
                    "equity": value,
                }
                for trade_date, value in equity
            ]
        ),
        "trades": pd.DataFrame(columns=["timestamp", "trade_date", "notional", "total_cost"]),
    }
