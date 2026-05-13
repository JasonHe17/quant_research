from __future__ import annotations

import pandas as pd
import pytest
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from quant_research.strategies import (
    FiveMinuteCrossSectionalConfig,
    FiveMinuteCrossSectionalStrategy,
    RankBufferDropConfig,
    RankBufferDropPolicy,
)
from examples.run_tree_score_backtest import (
    TreeScoreBacktestParams,
    _build_target_weights,
    _score_rank_limit,
)


def test_five_minute_cross_sectional_strategy_selects_top_signals_and_applies_t1() -> None:
    signals = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-new",
                "signal": 0.9,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-held",
                "signal": 0.2,
            },
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-low",
                "signal": 0.1,
            },
        ]
    )
    current = pd.DataFrame(
        [
            {
                "instrument_id": "inst-held",
                "current_weight": 0.5,
                "sellable_weight": 0.1,
            }
        ]
    )

    result = FiveMinuteCrossSectionalStrategy().build(
        signals,
        FiveMinuteCrossSectionalConfig(
            name="cn-5m",
            top_n=1,
            weighting="equal",
            max_weight=1.0,
        ),
        current_positions=current,
    )

    assert result.selected_signals["instrument_id"].tolist() == ["inst-new"]
    assert result.target_weights["target_weight"].tolist() == [1.0]
    held_order = result.constrained_orders.loc[
        result.constrained_orders["instrument_id"] == "inst-held"
    ].iloc[0]
    assert held_order["desired_delta_weight"] == pytest.approx(-0.5)
    assert held_order["executable_delta_weight"] == pytest.approx(-0.1)
    assert held_order["blocked_sell_weight"] == pytest.approx(0.4)
    assert result.diagnostics.loc[0, "blocked_sell_weight"] == pytest.approx(0.4)


def test_five_minute_cross_sectional_strategy_requires_explicit_parameters() -> None:
    with pytest.raises(ValueError, match="top_n"):
        FiveMinuteCrossSectionalConfig(name="bad", top_n=0, weighting="equal")

    with pytest.raises(ValueError, match="weighting"):
        FiveMinuteCrossSectionalConfig(name="bad", top_n=1, weighting="bad")  # type: ignore[arg-type]


def test_five_minute_cross_sectional_strategy_rejects_negative_signal_weights() -> None:
    signals = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T09:35:00+08:00",
                "instrument_id": "inst-1",
                "signal": -0.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="non-negative"):
        FiveMinuteCrossSectionalStrategy().build(
            signals,
            FiveMinuteCrossSectionalConfig(
                name="cn-5m",
                top_n=1,
                weighting="signal",
            ),
            current_positions=pd.DataFrame(
                columns=["instrument_id", "current_weight", "sellable_weight"]
            ),
        )


def test_rank_buffer_drop_policy_holds_existing_name_until_exit_rank() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.8, 2),
        ]
    )
    state = pd.DataFrame(
        [{"instrument_id": "inst-held", "current_weight": 1.0, "holding_bars": 5}]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(target_count=1, entry_rank=1, exit_rank=2)
    ).decide(forecasts, state)

    assert result.policy_state["instrument_id"].tolist() == ["inst-held"]
    assert result.portfolio_intent.loc[
        result.portfolio_intent["instrument_id"] == "inst-held", "reason"
    ].iloc[0] == "hold_buffer"
    assert "inst-new" not in result.policy_state["instrument_id"].tolist()


def test_rank_buffer_drop_policy_exits_beyond_exit_rank_and_enters_replacement() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.1, 3),
        ]
    )
    state = pd.DataFrame(
        [{"instrument_id": "inst-held", "current_weight": 1.0, "holding_bars": 5}]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(target_count=1, entry_rank=1, exit_rank=2)
    ).decide(forecasts, state)

    targets = result.trade_decisions.set_index("instrument_id")["target_weight"]
    reasons = result.trade_decisions.set_index("instrument_id")["decision_reason"]
    assert targets["inst-held"] == pytest.approx(0.0)
    assert targets["inst-new"] == pytest.approx(1.0)
    assert reasons["inst-held"] == "exit_rank"
    assert reasons["inst-new"] == "entry_rank"


def test_rank_buffer_drop_policy_uses_no_trade_band_for_small_resizes() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-a", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-b", 0.8, 2),
        ]
    )
    state = pd.DataFrame(
        [
            {"instrument_id": "inst-a", "current_weight": 0.49, "holding_bars": 5},
            {"instrument_id": "inst-b", "current_weight": 0.51, "holding_bars": 5},
        ]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(
            target_count=2,
            entry_rank=2,
            exit_rank=2,
            no_trade_weight_band=0.02,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-a", "target_weight"] == pytest.approx(0.49)
    assert decisions.loc["inst-b", "target_weight"] == pytest.approx(0.51)
    assert decisions["decision_reason"].tolist() == [
        "below_weight_band",
        "below_weight_band",
    ]
    assert result.order_intents.empty


def test_rank_buffer_drop_policy_blocks_unsellable_t1_reduction() -> None:
    forecasts = pd.DataFrame(
        [_forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.1, 3)]
    )
    state = pd.DataFrame(
        [
            {
                "instrument_id": "inst-held",
                "current_weight": 1.0,
                "sellable_weight": 0.0,
                "holding_bars": 5,
            }
        ]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(target_count=1, entry_rank=1, exit_rank=2)
    ).decide(forecasts, state)

    decision = result.trade_decisions.iloc[0]
    assert decision["target_weight"] == pytest.approx(1.0)
    assert decision["decision_reason"] == "t1_sell_blocked"
    assert decision["constraint_flags"] == "sellable_weight_limited"


def test_rank_buffer_drop_policy_partially_rebalances_and_caps_turnover() -> None:
    forecasts = pd.DataFrame(
        [
            _forecast("2024-01-02T09:35:00+08:00", "inst-new", 0.9, 1),
            _forecast("2024-01-02T09:35:00+08:00", "inst-held", 0.1, 3),
        ]
    )
    state = pd.DataFrame(
        [{"instrument_id": "inst-held", "current_weight": 1.0, "holding_bars": 5}]
    )

    result = RankBufferDropPolicy(
        RankBufferDropConfig(
            target_count=1,
            entry_rank=1,
            exit_rank=2,
            partial_rebalance_rate=0.5,
            max_gross_turnover_per_rebalance=0.5,
        )
    ).decide(forecasts, state)

    decisions = result.trade_decisions.set_index("instrument_id")
    assert decisions.loc["inst-held", "target_weight"] == pytest.approx(0.75)
    assert decisions.loc["inst-new", "target_weight"] == pytest.approx(0.25)
    assert set(result.trade_decisions["constraint_flags"]) == {"turnover_scaled"}


def _forecast(timestamp: str, instrument_id: str, score: float, rank: int) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "instrument_id": instrument_id,
        "score": score,
        "rank": rank,
    }


def test_tree_score_target_builder_supports_rank_buffer_drop_policy(tmp_path) -> None:
    ranked = pd.DataFrame(
        [
            {"signal_time": "t0", "instrument_id": "inst-a", "score": 0.9, "rank": 1},
            {"signal_time": "t0", "instrument_id": "inst-b", "score": 0.8, "rank": 2},
            {"signal_time": "t1", "instrument_id": "inst-b", "score": 0.9, "rank": 1},
            {"signal_time": "t1", "instrument_id": "inst-a", "score": 0.8, "rank": 2},
        ]
    )

    result = _build_target_weights(ranked, _tree_score_params(tmp_path))

    by_time = {
        time: group["instrument_id"].tolist()
        for time, group in result.target_weights.groupby("signal_time", sort=True)
    }
    assert by_time == {"t0": ["inst-a"], "t1": ["inst-a"]}
    assert result.diagnostics["hold_count"].sum() == 1
    assert result.diagnostics["planned_gross_turnover"].sum() == pytest.approx(1.0)


def test_tree_score_rank_limit_includes_policy_exit_rank(tmp_path) -> None:
    params = _tree_score_params(tmp_path)

    assert _score_rank_limit(params) == 2


def _tree_score_params(tmp_path) -> TreeScoreBacktestParams:
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
        min_trade_weight=0.0,
        exclude_st=True,
        limit_up_bps=None,
        limit_down_bps=None,
        max_bar_turnover_participation=None,
        data_access_mode="data_portal",
        streaming_chunk="month",
        streaming_chunk_padding_days=0,
        output_dir=tmp_path,
    )
