from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from examples.run_tree_score_backtest import (  # noqa: E402
    TreeScoreBacktestParams,
    _decision_kline_windows,
    _build_target_weights,
    _decision_market_context,
    _write_outputs,
)
from quant_research.backtest import (  # noqa: E402
    DECISION_TRACE_COLUMNS,
    DecisionReportConfig,
    render_decision_report,
)


def test_render_decision_report_writes_static_html(tmp_path: Path) -> None:
    output_path = render_decision_report(
        summary=_summary_payload(),
        decision_trace=_decision_trace(),
        market_context=_market_context(),
        kline_windows=_kline_windows(),
        policy_diagnostics=_policy_diagnostics(),
        trades=_trades(),
        equity_curve=_equity_curve(),
        output_path=tmp_path / "decision_report.html",
        config=DecisionReportConfig(
            max_instruments=10,
            max_timestamps=10,
            max_decisions=20,
        ),
    )

    text = output_path.read_text(encoding="utf-8")

    assert "Backtest Decision Review" in text
    assert "Decision Heatmap" in text
    assert "Largest Decisions" in text
    assert "Executed Trades" in text
    assert "K-Line Decision Explorer" in text
    assert "decision-kline-payload" in text
    assert "600000.SH" in text
    assert "open_price" in text


def test_tree_score_target_builder_can_emit_decision_trace(tmp_path: Path) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.8, "rank": 2},
            {"signal_time": "t1", "instrument_id": "inst-b", "score": 0.9, "rank": 1},
            {"signal_time": "t1", "instrument_id": "inst-a", "score": 0.8, "rank": 2},
        ]
    )

    result = _build_target_weights(
        ranked,
        _tree_score_params(tmp_path),
        include_decision_trace=True,
    )

    assert list(result.decision_trace.columns) == list(DECISION_TRACE_COLUMNS)
    assert result.decision_trace["action"].tolist() == ["entry", "hold"]
    assert result.decision_trace["rank"].tolist() == [1, 2]
    assert result.order_intents["instrument_id"].tolist() == ["inst-a"]


def test_write_outputs_writes_decision_artifacts_and_report(tmp_path: Path) -> None:
    params = replace(
        _tree_score_params(tmp_path),
        render_decision_report=True,
        decision_report_max_instruments=10,
        decision_report_max_timestamps=10,
        decision_report_max_decisions=20,
    )
    result = {
        "summary": _summary_payload(),
        "decision_trace": _decision_trace(),
        "market_context": _market_context(),
        "kline_windows": _kline_windows(),
        "policy_diagnostics": _policy_diagnostics(),
        "order_intents": pd.DataFrame(),
        "trades": _trades(),
        "equity_curve": _equity_curve(),
    }

    _write_outputs(result, params)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "decision_trace.parquet").exists()
    assert (tmp_path / "decision_market_context.parquet").exists()
    assert (tmp_path / "decision_kline_windows.parquet").exists()
    assert (tmp_path / "policy_diagnostics.parquet").exists()
    assert (tmp_path / "decision_report.html").exists()
    assert len(pd.read_parquet(tmp_path / "decision_trace.parquet")) == 3


def test_decision_market_context_links_decision_to_execution_bar_and_trade() -> None:
    context = _decision_market_context(
        _decision_trace().head(1),
        pd.DataFrame(
            [
                {
                    "exec_time": "2024-01-02T09:40:00+08:00",
                    "instrument_id": "inst-a",
                    "canonical_code": "600000.SH",
                    "open_price": 10.0,
                    "high_price": 10.2,
                    "low_price": 9.9,
                    "close_price": 10.1,
                    "volume": 10000,
                    "turnover": 100000.0,
                    "tradable_bar": True,
                    "limit_up_open": False,
                    "limit_down_open": False,
                    "target_weight": 0.5,
                }
            ]
        ),
        _trades(),
    )

    row = context.iloc[0]
    assert row["canonical_code"] == "600000.SH"
    assert row["exec_time"] == "2024-01-02T09:40:00+08:00"
    assert row["open_price"] == 10.0
    assert row["bar_return"] == pytest.approx(0.01)
    assert row["executed_side"] == "buy"
    assert row["executed_notional"] == 1000.0


def test_decision_kline_windows_center_on_execution_bar() -> None:
    context = _decision_market_context(
        _decision_trace().head(1),
        _executions(),
        _trades(),
    )
    windows = _decision_kline_windows(
        _decision_trace().head(1),
        _bars(),
        context,
        pre_bars=1,
        post_bars=1,
    )

    assert len(windows) == 3
    assert windows["bar_offset"].tolist() == [-1, 0, 1]
    exec_row = windows.loc[windows["is_execution_bar"]].iloc[0]
    assert exec_row["bar_time"] == "2024-01-02T09:40:00+08:00"
    assert exec_row["marker_side"] == "buy"
    assert exec_row["marker_price"] == 10.0


def _summary_payload() -> dict[str, object]:
    return {
        "params": {
            "start": "2024-01-02",
            "end": "2024-01-03",
            "trade_policy": "rank_buffer_drop",
            "top_n": 2,
            "data_access_mode": "data_portal",
        },
        "metrics": {
            "total_return": 0.04,
            "max_drawdown": -0.01,
            "trade_count": 2,
            "final_equity": 1_040_000.0,
        },
        "policy_diagnostics": {
            "decision_timestamp_count": 2,
            "planned_gross_turnover": 1.5,
        },
    }


def _decision_trace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-a",
                "action": "entry",
                "current_weight": 0.0,
                "aim_weight": 0.5,
                "target_weight": 0.5,
                "delta_weight": 0.5,
                "rank": 1,
                "score": 0.9,
                "expected_edge_bps": 12.0,
                "estimated_cost_bps": 3.0,
                "priority": 1,
                "decision_reason": "entry_rank",
                "constraint_flags": "",
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-b",
                "action": "entry",
                "current_weight": 0.0,
                "aim_weight": 0.5,
                "target_weight": 0.5,
                "delta_weight": 0.5,
                "rank": 2,
                "score": 0.8,
                "expected_edge_bps": 10.0,
                "estimated_cost_bps": 3.0,
                "priority": 2,
                "decision_reason": "entry_rank",
                "constraint_flags": "",
            },
            {
                "timestamp": "2024-01-03T09:35:00+08:00",
                "instrument_id": "inst-a",
                "action": "exit",
                "current_weight": 0.5,
                "aim_weight": 0.0,
                "target_weight": 0.0,
                "delta_weight": -0.5,
                "rank": 8,
                "score": 0.1,
                "expected_edge_bps": 0.0,
                "estimated_cost_bps": 3.0,
                "priority": 1,
                "decision_reason": "exit_rank",
                "constraint_flags": "",
            },
        ]
    )


def _policy_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "planned_gross_turnover": 1.0,
                "target_gross_exposure": 1.0,
            },
            {
                "timestamp": "2024-01-03T09:35:00+08:00",
                "planned_gross_turnover": 0.5,
                "target_gross_exposure": 0.5,
            },
        ]
    )


def _market_context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-a",
                "canonical_code": "600000.SH",
                "raw_name": "sample-a",
                "exec_time": "2024-01-02T09:40:00+08:00",
                "open_price": 10.0,
                "high_price": 10.2,
                "low_price": 9.9,
                "close_price": 10.1,
                "volume": 10000,
                "turnover": 100000.0,
                "bar_return": 0.01,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "executed_side": "buy",
                "executed_shares": 100,
                "executed_notional": 1000.0,
                "avg_trade_price": 10.0,
                "total_cost": 1.0,
                "trade_count": 1,
            }
        ]
    )


def _kline_windows() -> pd.DataFrame:
    context = _decision_market_context(
        _decision_trace().head(1),
        _executions(),
        _trades(),
    )
    return _decision_kline_windows(
        _decision_trace().head(1),
        _bars(),
        context,
        pre_bars=1,
        post_bars=1,
    )


def _executions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exec_time": "2024-01-02T09:40:00+08:00",
                "instrument_id": "inst-a",
                "canonical_code": "600000.SH",
                "raw_name": "sample-a",
                "open_price": 10.0,
                "high_price": 10.2,
                "low_price": 9.9,
                "close_price": 10.1,
                "volume": 10000,
                "turnover": 100000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            }
        ]
    )


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bar_end_time": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-a",
                "canonical_code": "600000.SH",
                "raw_name": "sample-a",
                "open_price": 9.8,
                "high_price": 10.0,
                "low_price": 9.7,
                "close_price": 9.9,
                "volume": 9000,
                "turnover": 89000.0,
            },
            {
                "bar_end_time": "2024-01-02T09:40:00+08:00",
                "instrument_id": "inst-a",
                "canonical_code": "600000.SH",
                "raw_name": "sample-a",
                "open_price": 10.0,
                "high_price": 10.2,
                "low_price": 9.9,
                "close_price": 10.1,
                "volume": 10000,
                "turnover": 100000.0,
            },
            {
                "bar_end_time": "2024-01-02T09:45:00+08:00",
                "instrument_id": "inst-a",
                "canonical_code": "600000.SH",
                "raw_name": "sample-a",
                "open_price": 10.1,
                "high_price": 10.4,
                "low_price": 10.0,
                "close_price": 10.3,
                "volume": 12000,
                "turnover": 122000.0,
            },
        ]
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:40:00+08:00",
                "instrument_id": "inst-a",
                "side": "buy",
                "shares": 100,
                "price": 10.0,
                "notional": 1_000.0,
                "total_cost": 1.0,
            }
        ]
    )


def _equity_curve() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": "2024-01-02T09:40:00+08:00", "equity": 1_000_000.0},
            {"timestamp": "2024-01-03T09:40:00+08:00", "equity": 1_040_000.0},
        ]
    )


def _tree_score_params(tmp_path: Path) -> TreeScoreBacktestParams:
    return TreeScoreBacktestParams(
        predictions_path=tmp_path,
        catalog_path=tmp_path / "catalog.duckdb",
        start="t0",
        end="t1",
        top_n=1,
        initial_cash=1_000_000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        sell_stamp_tax_bps=0.0,
        min_commission=0.0,
        lot_size=100,
        trade_policy="rank_buffer_drop",
        rebalance_every_n_bars=1,
        hold_rank_buffer=None,
        policy_entry_rank=1,
        policy_exit_rank=2,
        policy_max_entries_per_rebalance=None,
        policy_max_exits_per_rebalance=None,
        policy_min_hold_bars=0,
        policy_min_expected_edge_bps=None,
        policy_estimated_cost_bps=0.0,
        policy_no_trade_weight_band=0.0,
        policy_partial_rebalance_rate=1.0,
        policy_max_gross_turnover_per_rebalance=None,
        policy_total_gross_turnover_budget=None,
        policy_turnover_budget_period="path",
        policy_turnover_budget_pacing=0.0,
        policy_gross_exposure_scale=1.0,
        policy_gross_exposure_scale_path=None,
        policy_drawdown_brake_threshold=None,
        policy_drawdown_brake_reduced_scale=0.5,
        optimizer_candidate_rank=None,
        optimizer_score_to_edge_bps=100.0,
        optimizer_min_net_edge_bps=0.0,
        optimizer_risk_penalty_multiplier=1.0,
        optimizer_weighting="utility",
        optimizer_max_name_weight=None,
        optimizer_max_gross_exposure_increase_per_rebalance=None,
        min_trade_weight=0.0,
        exclude_st=True,
        limit_up_bps=None,
        limit_down_bps=None,
        max_bar_turnover_participation=None,
        allow_same_bar_capacity=False,
        data_access_mode="data_portal",
        streaming_chunk="month",
        streaming_chunk_padding_days=0,
        output_dir=tmp_path,
    )
