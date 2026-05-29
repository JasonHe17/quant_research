"""Backtest tree-model prediction scores with A-share T+1 execution."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.metrics.performance import total_return
from quant_research.metrics.risk import max_drawdown
from quant_research.strategies import (
    CostAwareOptimizerConfig,
    CostAwareOptimizerPolicy,
    RankBufferDropConfig,
    RankBufferDropPolicy,
    StrategyPolicyResult,
    empty_portfolio_state,
)

from run_baseline_a_real_backtest import (
    BacktestParams,
    SimulationState,
    _append_frame_csv,
    _empty_execution_constraint_counts,
    _empty_trade_metric_totals,
    _execution_event_constraint_counts,
    _execution_constraint_counts,
    _final_positions,
    _load_bars,
    _load_bars_from_files,
    _merge_execution_constraint_counts,
    _merge_trade_metric_totals,
    _simulate,
    _trade_metrics,
    _trade_metric_totals,
    _trade_metrics_from_totals,
    _streaming_work_units,
)


@dataclass(frozen=True, slots=True)
class TreeScoreBacktestParams:
    predictions_path: Path
    catalog_path: Path
    start: str
    end: str
    top_n: int
    initial_cash: float
    commission_bps: float
    slippage_bps: float
    sell_stamp_tax_bps: float
    min_commission: float
    lot_size: int
    trade_policy: str
    rebalance_every_n_bars: int
    hold_rank_buffer: int | None
    policy_entry_rank: int | None
    policy_exit_rank: int | None
    policy_max_entries_per_rebalance: int | None
    policy_max_exits_per_rebalance: int | None
    policy_min_hold_bars: int
    policy_min_expected_edge_bps: float | None
    policy_estimated_cost_bps: float
    policy_no_trade_weight_band: float
    policy_partial_rebalance_rate: float
    policy_max_gross_turnover_per_rebalance: float | None
    policy_total_gross_turnover_budget: float | None
    policy_turnover_budget_period: str
    policy_turnover_budget_pacing: float
    policy_gross_exposure_scale: float
    policy_gross_exposure_scale_path: Path | None
    policy_drawdown_brake_threshold: float | None
    policy_drawdown_brake_reduced_scale: float
    policy_cost_pressure_threshold_bps: float | None
    policy_cost_pressure_reduced_scale: float
    policy_cost_pressure_max_gross_turnover_per_rebalance: float | None
    policy_reset_on_source_change: bool
    policy_force_source_transition_exits: bool
    policy_source_transition_exit_rate: float
    policy_source_transition_turnover_cap: float | None
    policy_source_column: str
    optimizer_candidate_rank: int | None
    optimizer_score_to_edge_bps: float
    optimizer_min_net_edge_bps: float
    optimizer_risk_penalty_multiplier: float
    optimizer_target_cap_mode: str
    optimizer_weighting: str
    optimizer_max_name_weight: float | None
    optimizer_max_gross_exposure_increase_per_rebalance: float | None
    min_trade_weight: float
    exclude_st: bool
    limit_up_bps: float | None
    limit_down_bps: float | None
    max_bar_turnover_participation: float | None
    allow_same_bar_capacity: bool
    data_access_mode: str
    streaming_chunk: str
    streaming_chunk_padding_days: int
    output_dir: Path


@dataclass(frozen=True, slots=True)
class PolicyBudgetState:
    """Mutable path-level policy budget carried across streaming chunks."""

    remaining_turnover_budget: float | None = None
    budget_period_key: str | None = None


@dataclass(frozen=True, slots=True)
class TargetWeightBuildResult:
    """Target weights plus state and diagnostics from a strategy policy pass."""

    target_weights: pd.DataFrame
    policy_state: pd.DataFrame
    budget_state: PolicyBudgetState
    diagnostics: pd.DataFrame
    source_state: str | None = None
    source_by_instrument: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class BarTimeIndex:
    """Row lookup index for repeated time-window slices inside one bars chunk."""

    bar_times: list[object]
    next_time_by_time: dict[object, object]
    row_indices_by_time: dict[object, object]


def main() -> None:
    params = _parse_args()
    result = run_tree_score_backtest(params)
    _write_outputs(result, params)
    print(json.dumps(result["summary"], ensure_ascii=True, indent=2, sort_keys=True))


def run_tree_score_backtest(params: TreeScoreBacktestParams) -> dict[str, object]:
    backtest_params = BacktestParams(
        catalog_path=params.catalog_path,
        start=params.start,
        end=params.end,
        top_n=params.top_n,
        initial_cash=params.initial_cash,
        lookback_bars=1,
        min_avg_turnover=None,
        liquidity_window_bars=1,
        commission_bps=params.commission_bps,
        slippage_bps=params.slippage_bps,
        lot_size=params.lot_size,
        max_symbols=None,
        output_dir=params.output_dir,
        sell_stamp_tax_bps=params.sell_stamp_tax_bps,
        min_commission=params.min_commission,
        min_trade_weight=params.min_trade_weight,
        exclude_st=params.exclude_st,
        limit_up_bps=params.limit_up_bps,
        limit_down_bps=params.limit_down_bps,
        max_bar_turnover_participation=params.max_bar_turnover_participation,
        allow_same_bar_capacity=params.allow_same_bar_capacity,
        data_access_mode=params.data_access_mode,
        streaming_chunk=params.streaming_chunk,
        streaming_chunk_padding_days=params.streaming_chunk_padding_days,
    )
    params.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("trades.csv", "equity_curve.csv"):
        path = params.output_dir / filename
        if path.exists():
            path.unlink()
    if params.data_access_mode == "fast_parquet":
        return _run_tree_score_backtest_streaming(params, backtest_params)
    if params.policy_drawdown_brake_threshold is not None:
        raise ValueError("drawdown brake requires --data-access-mode fast_parquet")
    if params.policy_cost_pressure_threshold_bps is not None:
        raise ValueError("cost pressure brake requires --data-access-mode fast_parquet")

    ranked_signals = _load_ranked_score_signals(params)
    target_build = _build_target_weights(ranked_signals, params)
    signals = target_build.target_weights
    if signals.empty:
        raise ValueError("no score signals loaded for requested period")
    bars = _load_bars(backtest_params)
    if bars.empty:
        raise ValueError("no bars loaded for requested period")
    executions = _build_tree_score_executions(bars, signals)
    if executions.empty:
        raise ValueError("no executable tree score signals after next-bar shift")
    execution_constraint_counts = _execution_constraint_counts(executions)
    execution_events: list[dict[str, object]] = []
    trades, equity_curve, _, state = _simulate(
        executions,
        backtest_params,
        diagnostics=execution_events,
    )
    _merge_execution_constraint_counts(
        execution_constraint_counts,
        _execution_event_constraint_counts(pd.DataFrame(execution_events)),
    )
    if not trades.empty:
        _append_frame_csv(trades, params.output_dir / "trades.csv")
    if not equity_curve.empty:
        _append_frame_csv(equity_curve, params.output_dir / "equity_curve.csv")
    final_positions = _final_positions(state.lots)
    final_positions.to_csv(params.output_dir / "final_positions.csv", index=False)
    equity_values = [params.initial_cash] + equity_curve["equity"].astype(float).tolist()
    metrics = {
        "total_return": total_return(params.initial_cash, equity_values[-1]),
        "max_drawdown": max_drawdown(equity_values),
        "trade_count": float(len(trades)),
        "final_equity": float(equity_values[-1]),
    }
    metrics.update(_trade_metrics(trades, equity_values))
    summary = {
        "params": {
            "predictions_path": str(params.predictions_path),
            "catalog_path": str(params.catalog_path),
            "start": params.start,
            "end": params.end,
            "top_n": params.top_n,
            "initial_cash": params.initial_cash,
            "commission_bps": params.commission_bps,
            "slippage_bps": params.slippage_bps,
            "sell_stamp_tax_bps": params.sell_stamp_tax_bps,
            "min_commission": params.min_commission,
            "lot_size": params.lot_size,
            "trade_policy": params.trade_policy,
            "rebalance_every_n_bars": params.rebalance_every_n_bars,
            "hold_rank_buffer": params.hold_rank_buffer,
            "policy_entry_rank": params.policy_entry_rank,
            "policy_exit_rank": params.policy_exit_rank,
            "policy_max_entries_per_rebalance": params.policy_max_entries_per_rebalance,
            "policy_max_exits_per_rebalance": params.policy_max_exits_per_rebalance,
            "policy_min_hold_bars": params.policy_min_hold_bars,
            "policy_min_expected_edge_bps": params.policy_min_expected_edge_bps,
            "policy_estimated_cost_bps": params.policy_estimated_cost_bps,
            "policy_no_trade_weight_band": params.policy_no_trade_weight_band,
            "policy_partial_rebalance_rate": params.policy_partial_rebalance_rate,
            "policy_max_gross_turnover_per_rebalance": params.policy_max_gross_turnover_per_rebalance,
            "policy_total_gross_turnover_budget": params.policy_total_gross_turnover_budget,
            "policy_turnover_budget_period": params.policy_turnover_budget_period,
            "policy_turnover_budget_pacing": params.policy_turnover_budget_pacing,
            "policy_gross_exposure_scale": params.policy_gross_exposure_scale,
            "policy_gross_exposure_scale_path": (
                str(params.policy_gross_exposure_scale_path)
                if params.policy_gross_exposure_scale_path is not None
                else None
            ),
            "policy_drawdown_brake_threshold": params.policy_drawdown_brake_threshold,
            "policy_drawdown_brake_reduced_scale": (
                params.policy_drawdown_brake_reduced_scale
            ),
            "policy_cost_pressure_threshold_bps": params.policy_cost_pressure_threshold_bps,
            "policy_cost_pressure_reduced_scale": (
                params.policy_cost_pressure_reduced_scale
            ),
            "policy_cost_pressure_max_gross_turnover_per_rebalance": (
                params.policy_cost_pressure_max_gross_turnover_per_rebalance
            ),
            "policy_reset_on_source_change": params.policy_reset_on_source_change,
            "policy_force_source_transition_exits": (
                params.policy_force_source_transition_exits
            ),
            "policy_source_transition_exit_rate": params.policy_source_transition_exit_rate,
            "policy_source_transition_turnover_cap": (
                params.policy_source_transition_turnover_cap
            ),
            "policy_source_column": params.policy_source_column,
            "optimizer_candidate_rank": params.optimizer_candidate_rank,
            "optimizer_score_to_edge_bps": params.optimizer_score_to_edge_bps,
            "optimizer_min_net_edge_bps": params.optimizer_min_net_edge_bps,
            "optimizer_risk_penalty_multiplier": params.optimizer_risk_penalty_multiplier,
            "optimizer_target_cap_mode": params.optimizer_target_cap_mode,
            "optimizer_weighting": params.optimizer_weighting,
            "optimizer_max_name_weight": params.optimizer_max_name_weight,
            "optimizer_max_gross_exposure_increase_per_rebalance": (
                params.optimizer_max_gross_exposure_increase_per_rebalance
            ),
            "min_trade_weight": params.min_trade_weight,
            "exclude_st": params.exclude_st,
            "limit_up_bps": params.limit_up_bps,
            "limit_down_bps": params.limit_down_bps,
            "max_bar_turnover_participation": params.max_bar_turnover_participation,
            "allow_same_bar_capacity": params.allow_same_bar_capacity,
            "data_access_mode": params.data_access_mode,
            "streaming_chunk": params.streaming_chunk,
            "streaming_chunk_padding_days": params.streaming_chunk_padding_days,
        },
        "bar_count": int(len(bars)),
        "signal_count": int(len(signals)),
        "execution_row_count": int(len(executions)),
        "execution_constraint_counts": execution_constraint_counts,
        "instrument_count": int(bars["instrument_id"].nunique()),
        "policy_diagnostics": _summarize_policy_diagnostics(target_build.diagnostics),
        "metrics": metrics,
    }
    return {
        "summary": summary,
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve,
        "final_positions": final_positions,
    }


def _run_tree_score_backtest_streaming(
    params: TreeScoreBacktestParams,
    backtest_params: BacktestParams,
) -> dict[str, object]:
    if params.hold_rank_buffer is not None and params.trade_policy == "naive_top_n":
        raise ValueError("streaming score backtests do not support hold-rank buffer yet")
    if (
        params.policy_drawdown_brake_threshold is not None
        or params.policy_cost_pressure_threshold_bps is not None
    ):
        return _run_tree_score_backtest_streaming_rebalance_drawdown(
            params,
            backtest_params,
        )
    state = SimulationState(
        cash=float(params.initial_cash),
        lots={},
        previous_date=None,
        last_prices={},
    )
    equity_values = [params.initial_cash]
    trade_count = 0
    trade_metric_totals = _empty_trade_metric_totals()
    execution_constraint_counts = _empty_execution_constraint_counts()
    total_bars = 0
    total_signals = 0
    total_executions = 0
    instruments: set[str] = set()
    policy_state = empty_portfolio_state()
    policy_source_state: str | None = None
    policy_source_by_instrument: dict[str, str] = {}
    budget_state = _initial_policy_budget_state(params)
    policy_diagnostics: list[pd.DataFrame] = []
    for work_unit in _streaming_work_units(backtest_params):
        bars = _load_bars_from_files(
            backtest_params,
            work_unit.files,
            start=work_unit.load_start,
            end=work_unit.load_end,
        )
        if bars.empty:
            continue
        in_signal_window = (
            (bars["bar_end_time"] >= work_unit.signal_start)
            & (bars["bar_end_time"] <= work_unit.signal_end)
        )
        total_bars += int(in_signal_window.sum())
        instruments.update(
            str(value)
            for value in bars.loc[in_signal_window, "instrument_id"].unique()
        )
        ranked_signals = _load_ranked_score_signals(
            params,
            start=work_unit.signal_start,
            end=work_unit.signal_end,
        )
        target_build = _build_target_weights(
            ranked_signals,
            params,
            policy_state=policy_state,
            budget_state=budget_state,
            source_state=policy_source_state,
            source_by_instrument=policy_source_by_instrument,
        )
        signals = target_build.target_weights
        policy_state = target_build.policy_state
        policy_source_state = target_build.source_state
        policy_source_by_instrument = target_build.source_by_instrument or {}
        budget_state = target_build.budget_state
        if not target_build.diagnostics.empty:
            policy_diagnostics.append(target_build.diagnostics)
        total_signals += len(signals)
        executions = _build_tree_score_executions(
            bars,
            signals,
            tracked_instruments=set(state.lots),
            sparse=True,
        )
        if executions.empty:
            continue
        total_executions += len(executions)
        _merge_execution_constraint_counts(
            execution_constraint_counts,
            _execution_constraint_counts(executions),
        )
        period_execution_events: list[dict[str, object]] = []
        period_trades, period_equity, _, state = _simulate(
            executions,
            backtest_params,
            state=state,
            diagnostics=period_execution_events,
        )
        _merge_execution_constraint_counts(
            execution_constraint_counts,
            _execution_event_constraint_counts(pd.DataFrame(period_execution_events)),
        )
        if not period_trades.empty:
            trade_count += len(period_trades)
            _merge_trade_metric_totals(
                trade_metric_totals,
                _trade_metric_totals(period_trades),
            )
            _append_frame_csv(period_trades, params.output_dir / "trades.csv")
        if not period_equity.empty:
            equity_values.extend(period_equity["equity"].astype(float).tolist())
            _append_frame_csv(period_equity, params.output_dir / "equity_curve.csv")
        del bars, ranked_signals, signals, executions, period_trades, period_equity
    if len(equity_values) == 1:
        raise ValueError("no executable score signals after next-bar shift")
    final_positions = _final_positions(state.lots)
    final_positions.to_csv(params.output_dir / "final_positions.csv", index=False)
    metrics = {
        "total_return": total_return(params.initial_cash, equity_values[-1]),
        "max_drawdown": max_drawdown(equity_values),
        "trade_count": float(trade_count),
        "final_equity": float(equity_values[-1]),
    }
    metrics.update(_trade_metrics_from_totals(trade_metric_totals, equity_values))
    summary = _summary_payload(
        params,
        bar_count=total_bars,
        signal_count=total_signals,
        execution_row_count=total_executions,
        execution_constraint_counts=execution_constraint_counts,
        instrument_count=len(instruments),
        policy_diagnostics=_summarize_policy_diagnostics(_concat_policy_diagnostics(policy_diagnostics)),
        metrics=metrics,
    )
    return {
        "summary": summary,
        "metrics": metrics,
        "trades": pd.DataFrame(),
        "equity_curve": pd.DataFrame(),
        "final_positions": final_positions,
    }


def _run_tree_score_backtest_streaming_rebalance_drawdown(
    params: TreeScoreBacktestParams,
    backtest_params: BacktestParams,
) -> dict[str, object]:
    state = SimulationState(
        cash=float(params.initial_cash),
        lots={},
        previous_date=None,
        last_prices={},
    )
    equity_values = [params.initial_cash]
    trade_count = 0
    trade_metric_totals = _empty_trade_metric_totals()
    execution_constraint_counts = _empty_execution_constraint_counts()
    total_bars = 0
    total_signals = 0
    total_executions = 0
    instruments: set[str] = set()
    policy_state = empty_portfolio_state()
    policy_source_state: str | None = None
    policy_source_by_instrument: dict[str, str] = {}
    budget_state = _initial_policy_budget_state(params)
    policy_diagnostics: list[pd.DataFrame] = []
    peak_equity = float(params.initial_cash)
    last_simulated_time: object | None = None
    scale_by_timestamp = _load_policy_gross_exposure_schedule(params)

    def run_execution_batch(executions: pd.DataFrame) -> None:
        nonlocal state, trade_count, total_executions
        if executions.empty:
            return
        total_executions += len(executions)
        _merge_execution_constraint_counts(
            execution_constraint_counts,
            _execution_constraint_counts(executions),
        )
        period_execution_events: list[dict[str, object]] = []
        period_trades, period_equity, _, state = _simulate(
            executions,
            backtest_params,
            state=state,
            diagnostics=period_execution_events,
        )
        _merge_execution_constraint_counts(
            execution_constraint_counts,
            _execution_event_constraint_counts(pd.DataFrame(period_execution_events)),
        )
        if not period_trades.empty:
            trade_count += len(period_trades)
            _merge_trade_metric_totals(
                trade_metric_totals,
                _trade_metric_totals(period_trades),
            )
            _append_frame_csv(period_trades, params.output_dir / "trades.csv")
        if not period_equity.empty:
            equity_values.extend(period_equity["equity"].astype(float).tolist())
            _append_frame_csv(period_equity, params.output_dir / "equity_curve.csv")

    for work_unit in _streaming_work_units(backtest_params):
        bars = _load_bars_from_files(
            backtest_params,
            work_unit.files,
            start=work_unit.load_start,
            end=work_unit.load_end,
        )
        if bars.empty:
            continue
        in_signal_window = (
            (bars["bar_end_time"] >= work_unit.signal_start)
            & (bars["bar_end_time"] <= work_unit.signal_end)
        )
        total_bars += int(in_signal_window.sum())
        instruments.update(
            str(value)
            for value in bars.loc[in_signal_window, "instrument_id"].unique()
        )
        ranked_signals = _load_ranked_score_signals(
            params,
            start=work_unit.signal_start,
            end=work_unit.signal_end,
        )
        if ranked_signals.empty:
            continue
        grouped_signals = list(ranked_signals.groupby("signal_time", sort=True))
        bar_time_index = _bar_time_index(bars)
        bar_times = bar_time_index.bar_times
        for index, (signal_time, group) in enumerate(grouped_signals):
            mark_rows = _price_execution_rows_from_index(
                bars,
                bar_time_index,
                tracked_instruments=set(state.lots),
                start_exclusive=last_simulated_time,
                end_inclusive=signal_time,
            )
            run_execution_batch(mark_rows)
            if not mark_rows.empty:
                last_simulated_time = mark_rows["exec_time"].max()

            current_equity = float(equity_values[-1])
            peak_equity = max(peak_equity, current_equity)
            drawdown_brake_scale = _drawdown_brake_scale(
                params,
                current_equity=current_equity,
                peak_equity=peak_equity,
            )
            cost_pressure_scale = _cost_pressure_scale(params, trade_metric_totals)
            cost_pressure_turnover_cap = _cost_pressure_turnover_cap(
                params,
                trade_metric_totals,
            )
            realized_cost_bps = _realized_transaction_cost_bps(
                trade_metric_totals,
                params.initial_cash,
            )
            gross_exposure_scale_cap = _min_optional_scale(
                drawdown_brake_scale,
                cost_pressure_scale,
            )
            target_build = _build_target_weights(
                group,
                params,
                policy_state=policy_state,
                budget_state=budget_state,
                source_state=policy_source_state,
                source_by_instrument=policy_source_by_instrument,
                gross_exposure_scale_cap=gross_exposure_scale_cap,
                scale_by_timestamp=scale_by_timestamp,
                drawdown_brake_scale=drawdown_brake_scale,
                cost_pressure_scale=cost_pressure_scale,
                cost_pressure_turnover_cap=cost_pressure_turnover_cap,
                realized_cost_bps=realized_cost_bps,
            )
            signals = target_build.target_weights
            policy_state = target_build.policy_state
            policy_source_state = target_build.source_state
            policy_source_by_instrument = target_build.source_by_instrument or {}
            budget_state = target_build.budget_state
            if not target_build.diagnostics.empty:
                policy_diagnostics.append(target_build.diagnostics)
            total_signals += len(signals)

            next_signal_time = (
                grouped_signals[index + 1][0]
                if index + 1 < len(grouped_signals)
                else None
            )
            segment_end = _next_segment_end(
                signal_time,
                next_signal_time=next_signal_time,
                bar_times=bar_times,
            )
            executions = _build_segment_tree_score_executions(
                bars,
                bar_time_index,
                signals,
                tracked_instruments=set(state.lots),
                start_exclusive=signal_time,
                end_inclusive=segment_end,
            )
            run_execution_batch(executions)
            if not executions.empty:
                last_simulated_time = executions["exec_time"].max()
        del bars, ranked_signals

    if len(equity_values) == 1:
        raise ValueError("no executable score signals after next-bar shift")
    final_positions = _final_positions(state.lots)
    final_positions.to_csv(params.output_dir / "final_positions.csv", index=False)
    metrics = {
        "total_return": total_return(params.initial_cash, equity_values[-1]),
        "max_drawdown": max_drawdown(equity_values),
        "trade_count": float(trade_count),
        "final_equity": float(equity_values[-1]),
    }
    metrics.update(_trade_metrics_from_totals(trade_metric_totals, equity_values))
    summary = _summary_payload(
        params,
        bar_count=total_bars,
        signal_count=total_signals,
        execution_row_count=total_executions,
        execution_constraint_counts=execution_constraint_counts,
        instrument_count=len(instruments),
        policy_diagnostics=_summarize_policy_diagnostics(
            _concat_policy_diagnostics(policy_diagnostics)
        ),
        metrics=metrics,
    )
    return {
        "summary": summary,
        "metrics": metrics,
        "trades": pd.DataFrame(),
        "equity_curve": pd.DataFrame(),
        "final_positions": final_positions,
    }


def _load_ranked_score_signals(
    params: TreeScoreBacktestParams,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    rank_limit = _score_rank_limit(params)
    start = start or params.start
    end = end or params.end
    scan_target = _prediction_scan_target(params.predictions_path)
    connection = duckdb.connect()
    try:
        available_columns = _prediction_columns(connection, scan_target)
        forecast_columns = [
            column
            for column in (
                "expected_edge_bps",
                "expected_return_bps",
                "risk_penalty_bps",
                "health_risk_bps",
                "optimizer_risk_penalty_bps",
                "max_target_weight",
                "target_weight_cap",
                "optimizer_max_target_weight",
                "entry_eligible",
            )
            if column in available_columns
        ]
        if (
            params.policy_source_column in available_columns
            and params.policy_source_column not in forecast_columns
        ):
            forecast_columns.append(params.policy_source_column)
        ranked_select_columns = "".join(
            f",\n                    p.{column}" for column in forecast_columns
        )
        output_select_columns = "".join(
            f",\n                {column}" for column in forecast_columns
        )
        query = """
            WITH signal_times AS (
                SELECT
                    timestamp,
                    row_number() OVER (ORDER BY timestamp) AS time_rank
                FROM (
                    SELECT DISTINCT timestamp
                    FROM read_parquet(?)
                    WHERE timestamp >= ?
                      AND timestamp <= ?
                      AND score IS NOT NULL
                )
            ),
            ranked AS (
                SELECT
                    p.timestamp AS signal_time,
                    p.instrument_id,
                    p.score,
                    t.time_rank{ranked_select_columns},
                    row_number() OVER (
                        PARTITION BY p.timestamp
                        ORDER BY p.score DESC, p.instrument_id ASC
                    ) AS rank
                FROM read_parquet(?) p
                JOIN signal_times t
                  ON p.timestamp = t.timestamp
                WHERE p.timestamp >= ?
                  AND p.timestamp <= ?
                  AND p.score IS NOT NULL
                  AND ((t.time_rank - 1) % ?) = 0
            )
            SELECT
                signal_time,
                instrument_id,
                score,
                rank{output_select_columns}
            FROM ranked
            WHERE rank <= ?
            ORDER BY signal_time, rank
        """.format(
            ranked_select_columns=ranked_select_columns,
            output_select_columns=output_select_columns,
        )
        return connection.execute(
            query,
            [
                scan_target,
                start,
                end,
                scan_target,
                start,
                end,
                params.rebalance_every_n_bars,
                rank_limit,
            ],
        ).fetchdf()
    finally:
        connection.close()


def _score_rank_limit(params: TreeScoreBacktestParams) -> int:
    limits = [params.top_n]
    if params.hold_rank_buffer is not None:
        limits.append(params.hold_rank_buffer)
    if params.trade_policy == "rank_buffer_drop":
        if params.policy_entry_rank is not None:
            limits.append(params.policy_entry_rank)
        if params.policy_exit_rank is not None:
            limits.append(params.policy_exit_rank)
    if params.trade_policy == "cost_aware_optimizer":
        if params.optimizer_candidate_rank is not None:
            limits.append(params.optimizer_candidate_rank)
        if params.policy_exit_rank is not None:
            limits.append(params.policy_exit_rank)
    return max(limits)


def _prediction_columns(connection: duckdb.DuckDBPyConnection, scan_target: str) -> set[str]:
    schema = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet(?) LIMIT 0",
        [scan_target],
    ).fetchdf()
    return set(schema["column_name"].astype(str))


def _prediction_scan_target(path: Path) -> str:
    if path.is_dir():
        return str(path / "*.parquet")
    return str(path)


def _build_target_weights(
    ranked_signals: pd.DataFrame,
    params: TreeScoreBacktestParams,
    *,
    policy_state: pd.DataFrame | None = None,
    budget_state: PolicyBudgetState | None = None,
    source_state: str | None = None,
    source_by_instrument: dict[str, str] | None = None,
    gross_exposure_scale_cap: float | None = None,
    scale_by_timestamp: dict[str, float] | None = None,
    drawdown_brake_scale: float | None = None,
    cost_pressure_scale: float | None = None,
    cost_pressure_turnover_cap: float | None = None,
    realized_cost_bps: float | None = None,
) -> TargetWeightBuildResult:
    if params.trade_policy == "naive_top_n":
        return TargetWeightBuildResult(
            target_weights=_build_buffered_target_weights(ranked_signals, params),
            policy_state=policy_state
            if policy_state is not None
            else empty_portfolio_state(),
            budget_state=budget_state
            if budget_state is not None
            else _initial_policy_budget_state(params),
            diagnostics=pd.DataFrame(),
            source_state=source_state,
            source_by_instrument=source_by_instrument or {},
        )
    if params.trade_policy not in {"rank_buffer_drop", "cost_aware_optimizer"}:
        raise ValueError(f"unsupported trade policy: {params.trade_policy}")
    if scale_by_timestamp is None:
        scale_by_timestamp = _load_policy_gross_exposure_schedule(params)
    base_scale = _effective_gross_exposure_scale(
        params.policy_gross_exposure_scale,
        gross_exposure_scale_cap,
    )
    default_policy = _policy_for_params(params, gross_exposure_scale=base_scale)
    diagnostics: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    state = policy_state if policy_state is not None else empty_portfolio_state()
    current_source_state = source_state
    current_source_by_instrument = dict(source_by_instrument or {})
    grouped_signals, group_count = _signal_groups(ranked_signals)
    current_budget_state = (
        budget_state if budget_state is not None else _initial_policy_budget_state(params)
    )
    remaining_turnover_budget = current_budget_state.remaining_turnover_budget
    budget_period_key = current_budget_state.budget_period_key
    for index, (signal_time, group) in enumerate(grouped_signals):
        refreshed_budget = _policy_budget_state_for_signal(
            params,
            signal_time=signal_time,
            remaining_turnover_budget=remaining_turnover_budget,
            budget_period_key=budget_period_key,
        )
        remaining_turnover_budget = refreshed_budget.remaining_turnover_budget
        budget_period_key = refreshed_budget.budget_period_key
        policy = default_policy
        effective_policy_scale = base_scale
        policy_turnover_cap = _policy_turnover_cap_for_signal(
            params,
            remaining_turnover_budget=remaining_turnover_budget,
            remaining_decision_count=group_count - index,
        )
        path_turnover_cap = _path_turnover_cap_for_signal(
            params,
            remaining_turnover_budget=remaining_turnover_budget,
            remaining_decision_count=group_count - index,
        )
        if cost_pressure_turnover_cap is not None:
            policy_turnover_cap = _min_optional_turnover_cap(
                policy_turnover_cap,
                cost_pressure_turnover_cap,
            )
            path_turnover_cap = _min_optional_turnover_cap(
                path_turnover_cap,
                cost_pressure_turnover_cap,
            )
        if scale_by_timestamp:
            scale = scale_by_timestamp.get(
                _timestamp_key(signal_time),
                params.policy_gross_exposure_scale,
            )
            scale = _effective_gross_exposure_scale(scale, gross_exposure_scale_cap)
            effective_policy_scale = scale
            policy = _policy_for_params(
                params,
                gross_exposure_scale=scale,
                turnover_cap=policy_turnover_cap,
            )
        elif policy_turnover_cap != params.policy_max_gross_turnover_per_rebalance:
            policy = _policy_for_params(
                params,
                gross_exposure_scale=base_scale,
                turnover_cap=policy_turnover_cap,
            )
        forecast_columns = [
            column
            for column in (
                "signal_time",
                "instrument_id",
                "score",
                "rank",
                "expected_edge_bps",
                "expected_return_bps",
                "risk_penalty_bps",
                "health_risk_bps",
                "optimizer_risk_penalty_bps",
                "max_target_weight",
                "target_weight_cap",
                "optimizer_max_target_weight",
                "entry_eligible",
                params.policy_source_column,
            )
            if column in group.columns
        ]
        forecasts = group.loc[:, forecast_columns].rename(
            columns={"signal_time": "timestamp"}
        )
        signal_source = _signal_source_for_group(
            group,
            params.policy_source_column,
        )
        source_change_reset = False
        if (
            params.policy_reset_on_source_change
            and signal_source is not None
            and current_source_state is not None
            and signal_source != current_source_state
        ):
            state = empty_portfolio_state()
            current_source_by_instrument = {}
            source_change_reset = True
        if signal_source is not None:
            current_source_state = signal_source
        source_transition_forced_exit_count = 0
        if (
            params.policy_force_source_transition_exits
            and signal_source is not None
            and not state.empty
        ):
            forecasts, source_transition_forced_exit_count = (
                _remove_cross_source_held_forecasts(
                    forecasts,
                    state,
                    source_by_instrument=current_source_by_instrument,
                    signal_source=signal_source,
                )
            )
            if source_transition_forced_exit_count > 0:
                transition_turnover_cap = _min_optional_turnover_cap(
                    policy_turnover_cap,
                    params.policy_source_transition_turnover_cap,
                )
                policy = _policy_for_params(
                    params,
                    gross_exposure_scale=effective_policy_scale,
                    turnover_cap=transition_turnover_cap,
                    partial_rebalance_rate=params.policy_source_transition_exit_rate,
                    no_trade_weight_band=0.0,
                )
        result = policy.decide(forecasts, state)
        if path_turnover_cap is not None:
            result = _enforce_path_turnover_cap(result, path_turnover_cap)
        state = result.policy_state
        diagnostics_frame = result.diagnostics.copy()
        if remaining_turnover_budget is not None:
            planned_turnover = float(diagnostics_frame.loc[0, "planned_gross_turnover"])
            budget_before = remaining_turnover_budget
            remaining_turnover_budget = max(budget_before - planned_turnover, 0.0)
            diagnostics_frame["dynamic_turnover_cap"] = (
                0.0 if path_turnover_cap is None else float(path_turnover_cap)
            )
            diagnostics_frame["turnover_path_budget_before"] = budget_before
            diagnostics_frame["turnover_path_budget_after"] = remaining_turnover_budget
            diagnostics_frame["turnover_budget_period"] = params.policy_turnover_budget_period
            diagnostics_frame["turnover_budget_period_key"] = budget_period_key
        drawdown_diag_scale = drawdown_brake_scale
        if (
            drawdown_diag_scale is None
            and gross_exposure_scale_cap is not None
            and cost_pressure_scale is None
        ):
            drawdown_diag_scale = gross_exposure_scale_cap
        if drawdown_diag_scale is not None:
            diagnostics_frame["drawdown_brake_scale"] = float(drawdown_diag_scale)
            diagnostics_frame["drawdown_brake_active"] = bool(
                drawdown_diag_scale < 1.0
            )
        if cost_pressure_scale is not None:
            diagnostics_frame["cost_pressure_scale"] = float(cost_pressure_scale)
            diagnostics_frame["cost_pressure_active"] = bool(cost_pressure_scale < 1.0)
        if cost_pressure_turnover_cap is not None:
            diagnostics_frame["cost_pressure_turnover_cap"] = float(
                cost_pressure_turnover_cap
            )
            diagnostics_frame["cost_pressure_turnover_active"] = True
        if realized_cost_bps is not None:
            diagnostics_frame["realized_cost_bps"] = float(realized_cost_bps)
        if signal_source is not None:
            diagnostics_frame["signal_source"] = signal_source
        diagnostics_frame["source_change_reset"] = bool(source_change_reset)
        diagnostics_frame["source_transition_forced_exit_count"] = int(
            source_transition_forced_exit_count
        )
        diagnostics.append(diagnostics_frame)
        target = result.portfolio_intent.loc[
            result.portfolio_intent["policy_target_weight"].astype(float) > 0,
            [
                "timestamp",
                "instrument_id",
                "score",
                "rank",
                "policy_target_weight",
            ],
        ].copy()
        if target.empty:
            current_source_by_instrument = _next_source_by_instrument(
                result.policy_state,
                previous_source_by_instrument=current_source_by_instrument,
                signal_source=signal_source,
            )
            continue
        target = target.rename(
            columns={
                "timestamp": "signal_time",
                "policy_target_weight": "target_weight",
            }
        )
        target["bar_end_time"] = signal_time
        targets.append(
            target.loc[
                :,
                [
                    "signal_time",
                    "bar_end_time",
                    "instrument_id",
                    "score",
                    "rank",
                    "target_weight",
                ],
            ]
        )
        current_source_by_instrument = _next_source_by_instrument(
            result.policy_state,
            previous_source_by_instrument=current_source_by_instrument,
            signal_source=signal_source,
        )
    if targets:
        target_weights = pd.concat(targets, ignore_index=True)
    else:
        target_weights = _empty_target_weight_frame()
    return TargetWeightBuildResult(
        target_weights=target_weights,
        policy_state=state,
        budget_state=PolicyBudgetState(
            remaining_turnover_budget,
            budget_period_key,
        ),
        diagnostics=_concat_policy_diagnostics(diagnostics),
        source_state=current_source_state,
        source_by_instrument=current_source_by_instrument,
    )


def _signal_groups(
    ranked_signals: pd.DataFrame,
) -> tuple[list[tuple[object, pd.DataFrame]], int]:
    if ranked_signals.empty:
        return [], 0
    signal_times = ranked_signals["signal_time"].drop_duplicates().tolist()
    if len(signal_times) == 1:
        return [(signal_times[0], ranked_signals)], 1
    groups = list(ranked_signals.groupby("signal_time", sort=True))
    return groups, len(groups)


def _signal_source_for_group(group: pd.DataFrame, column: str) -> str | None:
    if not column or column not in group.columns:
        return None
    values = group[column].dropna().astype(str).unique().tolist()
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(
            f"signal source column {column!r} has multiple values for one timestamp"
        )
    return values[0]


def _remove_cross_source_held_forecasts(
    forecasts: pd.DataFrame,
    state: pd.DataFrame,
    *,
    source_by_instrument: dict[str, str],
    signal_source: str,
) -> tuple[pd.DataFrame, int]:
    if forecasts.empty or state.empty or not source_by_instrument:
        return forecasts, 0
    held_ids = state.loc[
        state["current_weight"].astype(float) > 0,
        "instrument_id",
    ].astype(str)
    forced_ids = {
        instrument_id
        for instrument_id in held_ids
        if source_by_instrument.get(instrument_id)
        and source_by_instrument[instrument_id] != signal_source
    }
    if not forced_ids:
        return forecasts, 0
    keep = ~forecasts["instrument_id"].astype(str).isin(forced_ids)
    filtered = forecasts.loc[keep].copy()
    if filtered.empty:
        return forecasts, 0
    return filtered, len(forced_ids)


def _next_source_by_instrument(
    state: pd.DataFrame,
    *,
    previous_source_by_instrument: dict[str, str],
    signal_source: str | None,
) -> dict[str, str]:
    if state.empty:
        return {}
    output: dict[str, str] = {}
    for row in state.itertuples(index=False):
        instrument_id = str(row.instrument_id)
        previous_source = previous_source_by_instrument.get(instrument_id)
        if previous_source is not None:
            output[instrument_id] = previous_source
        elif signal_source is not None:
            output[instrument_id] = signal_source
    return output


def _effective_gross_exposure_scale(
    configured_scale: float,
    cap: float | None,
) -> float:
    if cap is None:
        return configured_scale
    return min(configured_scale, cap)


def _drawdown_brake_scale(
    params: TreeScoreBacktestParams,
    *,
    current_equity: float,
    peak_equity: float,
) -> float | None:
    threshold = params.policy_drawdown_brake_threshold
    if threshold is None:
        return None
    if peak_equity <= 0:
        return params.policy_drawdown_brake_reduced_scale
    drawdown = current_equity / peak_equity - 1.0
    if drawdown <= threshold:
        return params.policy_drawdown_brake_reduced_scale
    return 1.0


def _cost_pressure_scale(
    params: TreeScoreBacktestParams,
    trade_metric_totals: dict[str, float],
) -> float | None:
    threshold_bps = params.policy_cost_pressure_threshold_bps
    if threshold_bps is None:
        return None
    realized_cost_bps = _realized_transaction_cost_bps(
        trade_metric_totals,
        params.initial_cash,
    )
    if realized_cost_bps >= threshold_bps:
        return params.policy_cost_pressure_reduced_scale
    return 1.0


def _cost_pressure_turnover_cap(
    params: TreeScoreBacktestParams,
    trade_metric_totals: dict[str, float],
) -> float | None:
    threshold_bps = params.policy_cost_pressure_threshold_bps
    cap = params.policy_cost_pressure_max_gross_turnover_per_rebalance
    if threshold_bps is None or cap is None:
        return None
    realized_cost_bps = _realized_transaction_cost_bps(
        trade_metric_totals,
        params.initial_cash,
    )
    if realized_cost_bps >= threshold_bps:
        return cap
    return None


def _realized_transaction_cost_bps(
    trade_metric_totals: dict[str, float],
    initial_cash: float,
) -> float:
    if initial_cash <= 0:
        return 0.0
    transaction_cost = (
        float(trade_metric_totals.get("total_commission", 0.0))
        + float(trade_metric_totals.get("total_stamp_tax", 0.0))
        + float(trade_metric_totals.get("total_slippage_cost", 0.0))
    )
    return transaction_cost / float(initial_cash) * 10_000.0


def _min_optional_scale(*scales: float | None) -> float | None:
    active_scales = [float(scale) for scale in scales if scale is not None]
    if not active_scales:
        return None
    return min(active_scales)


def _min_optional_turnover_cap(*caps: float | None) -> float | None:
    active_caps = [float(cap) for cap in caps if cap is not None]
    if not active_caps:
        return None
    return min(active_caps)


def _policy_for_params(
    params: TreeScoreBacktestParams,
    *,
    gross_exposure_scale: float | None = None,
    turnover_cap: float | None = None,
    partial_rebalance_rate: float | None = None,
    no_trade_weight_band: float | None = None,
) -> RankBufferDropPolicy | CostAwareOptimizerPolicy:
    if params.trade_policy == "rank_buffer_drop":
        return RankBufferDropPolicy(
            _rank_buffer_drop_config(
                params,
                gross_exposure_scale=gross_exposure_scale,
                turnover_cap=turnover_cap,
                partial_rebalance_rate=partial_rebalance_rate,
                no_trade_weight_band=no_trade_weight_band,
            )
        )
    if params.trade_policy == "cost_aware_optimizer":
        return CostAwareOptimizerPolicy(
            _cost_aware_optimizer_config(
                params,
                gross_exposure_scale=gross_exposure_scale,
                turnover_cap=turnover_cap,
                partial_rebalance_rate=partial_rebalance_rate,
                no_trade_weight_band=no_trade_weight_band,
            )
        )
    raise ValueError(f"unsupported trade policy: {params.trade_policy}")


def _rank_buffer_drop_config(
    params: TreeScoreBacktestParams,
    *,
    gross_exposure_scale: float | None = None,
    turnover_cap: float | None = None,
    partial_rebalance_rate: float | None = None,
    no_trade_weight_band: float | None = None,
) -> RankBufferDropConfig:
    return RankBufferDropConfig(
        target_count=params.top_n,
        entry_rank=params.policy_entry_rank or params.top_n,
        exit_rank=params.policy_exit_rank or max(params.top_n, params.hold_rank_buffer or params.top_n),
        max_entries_per_rebalance=params.policy_max_entries_per_rebalance,
        max_exits_per_rebalance=params.policy_max_exits_per_rebalance,
        min_hold_bars=params.policy_min_hold_bars,
        min_expected_edge_bps=params.policy_min_expected_edge_bps,
        estimated_cost_bps=params.policy_estimated_cost_bps,
        no_trade_weight_band=(
            params.policy_no_trade_weight_band
            if no_trade_weight_band is None
            else no_trade_weight_band
        ),
        partial_rebalance_rate=(
            params.policy_partial_rebalance_rate
            if partial_rebalance_rate is None
            else partial_rebalance_rate
        ),
        max_gross_turnover_per_rebalance=(
            params.policy_max_gross_turnover_per_rebalance
            if turnover_cap is None
            else turnover_cap
        ),
        gross_exposure_scale=(
            params.policy_gross_exposure_scale
            if gross_exposure_scale is None
            else gross_exposure_scale
        ),
    )


def _cost_aware_optimizer_config(
    params: TreeScoreBacktestParams,
    *,
    gross_exposure_scale: float | None = None,
    turnover_cap: float | None = None,
    partial_rebalance_rate: float | None = None,
    no_trade_weight_band: float | None = None,
) -> CostAwareOptimizerConfig:
    effective_turnover_cap = (
        params.policy_max_gross_turnover_per_rebalance
        if turnover_cap is None
        else turnover_cap
    )
    if params.policy_total_gross_turnover_budget is not None and turnover_cap is not None:
        # Let the optimizer spend the scheduled path budget for this timestamp;
        # a hard post-decision cap clips the realized gross delta if the native
        # sell/buy allocator would otherwise exceed it.
        effective_turnover_cap = turnover_cap
    return CostAwareOptimizerConfig(
        target_count=params.top_n,
        candidate_rank=params.optimizer_candidate_rank or params.policy_exit_rank or params.top_n * 3,
        min_hold_bars=params.policy_min_hold_bars,
        max_entries_per_rebalance=params.policy_max_entries_per_rebalance,
        max_exits_per_rebalance=params.policy_max_exits_per_rebalance,
        weighting=params.optimizer_weighting,  # type: ignore[arg-type]
        max_name_weight=params.optimizer_max_name_weight,
        score_to_edge_bps=params.optimizer_score_to_edge_bps,
        min_net_edge_bps=params.optimizer_min_net_edge_bps,
        estimated_cost_bps=params.policy_estimated_cost_bps,
        risk_penalty_multiplier=params.optimizer_risk_penalty_multiplier,
        target_cap_mode=params.optimizer_target_cap_mode,  # type: ignore[arg-type]
        no_trade_weight_band=(
            params.policy_no_trade_weight_band
            if no_trade_weight_band is None
            else no_trade_weight_band
        ),
        partial_rebalance_rate=(
            params.policy_partial_rebalance_rate
            if partial_rebalance_rate is None
            else partial_rebalance_rate
        ),
        max_gross_turnover_per_rebalance=effective_turnover_cap,
        max_gross_exposure_increase_per_rebalance=(
            params.optimizer_max_gross_exposure_increase_per_rebalance
        ),
        gross_exposure_scale=(
            params.policy_gross_exposure_scale
            if gross_exposure_scale is None
            else gross_exposure_scale
        ),
    )


def _policy_turnover_cap_for_signal(
    params: TreeScoreBacktestParams,
    *,
    remaining_turnover_budget: float | None,
    remaining_decision_count: int,
) -> float | None:
    del remaining_turnover_budget, remaining_decision_count
    return params.policy_max_gross_turnover_per_rebalance


def _initial_policy_budget_state(params: TreeScoreBacktestParams) -> PolicyBudgetState:
    return PolicyBudgetState(
        remaining_turnover_budget=params.policy_total_gross_turnover_budget,
        budget_period_key=None,
    )


def _policy_budget_state_for_signal(
    params: TreeScoreBacktestParams,
    *,
    signal_time: object,
    remaining_turnover_budget: float | None,
    budget_period_key: str | None,
) -> PolicyBudgetState:
    budget = params.policy_total_gross_turnover_budget
    if budget is None or params.policy_turnover_budget_period == "path":
        return PolicyBudgetState(remaining_turnover_budget, budget_period_key)
    current_key = _turnover_budget_period_key(signal_time, params.policy_turnover_budget_period)
    if current_key != budget_period_key:
        return PolicyBudgetState(budget, current_key)
    return PolicyBudgetState(remaining_turnover_budget, budget_period_key)


def _turnover_budget_period_key(value: object, period: str) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return str(value)
    if period == "month":
        return timestamp.strftime("%Y-%m")
    if period == "year":
        return timestamp.strftime("%Y")
    if period == "path":
        return "path"
    raise ValueError(f"unsupported turnover budget period: {period}")


def _path_turnover_cap_for_signal(
    params: TreeScoreBacktestParams,
    *,
    remaining_turnover_budget: float | None,
    remaining_decision_count: int,
) -> float | None:
    if remaining_turnover_budget is None:
        return params.policy_max_gross_turnover_per_rebalance
    path_cap = max(float(remaining_turnover_budget), 0.0)
    if params.policy_turnover_budget_pacing <= 0:
        return path_cap
    if remaining_decision_count <= 0:
        return 0.0
    paced_cap = (
        path_cap
        / remaining_decision_count
        * params.policy_turnover_budget_pacing
    )
    return min(path_cap, paced_cap)


def _enforce_path_turnover_cap(
    result: StrategyPolicyResult,
    cap: float | None,
) -> StrategyPolicyResult:
    if cap is None or cap < 0:
        return result
    decisions = result.trade_decisions.copy()
    if decisions.empty:
        return result
    gross_turnover = float(decisions["delta_weight"].astype(float).abs().sum())
    if gross_turnover <= cap + 1e-12 or gross_turnover <= 0:
        return result
    scale = cap / gross_turnover
    target_by_id: dict[str, float] = {}
    for row in decisions.itertuples(index=False):
        current = float(row.current_weight)
        target = current + float(row.delta_weight) * scale
        target_by_id[str(row.instrument_id)] = target
    return _replace_policy_targets(result, target_by_id, "path_turnover_budget_limited")


def _replace_policy_targets(
    result: StrategyPolicyResult,
    target_by_id: dict[str, float],
    flag: str,
) -> StrategyPolicyResult:
    if not target_by_id:
        return result
    target_series = pd.Series(target_by_id, dtype=float)
    intent = result.portfolio_intent.copy()
    decisions = result.trade_decisions.copy()
    orders = result.order_intents.copy()
    state = result.policy_state.copy()
    if not intent.empty:
        mapped_targets = intent["instrument_id"].astype(str).map(target_series)
        intent["policy_target_weight"] = mapped_targets.fillna(
            intent["policy_target_weight"].astype(float)
        )
        intent["aim_weight"] = intent["policy_target_weight"]
        intent["constraint_flags"] = intent["constraint_flags"].map(
            lambda value: _append_constraint_flag(value, flag)
        )
    if not decisions.empty:
        mapped_targets = decisions["instrument_id"].astype(str).map(target_series)
        decisions["target_weight"] = mapped_targets.fillna(
            decisions["target_weight"].astype(float)
        )
        decisions["aim_weight"] = decisions["target_weight"]
        decisions["delta_weight"] = (
            decisions["target_weight"].astype(float)
            - decisions["current_weight"].astype(float)
        )
        decisions["constraint_flags"] = decisions["constraint_flags"].map(
            lambda value: _append_constraint_flag(value, flag)
        )
        zero_delta = decisions["delta_weight"].astype(float).abs() <= 1e-12
        decisions.loc[zero_delta, "decision_reason"] = "turnover_budget_limited"
    if not orders.empty:
        orders = orders.loc[orders["instrument_id"].astype(str).isin(target_by_id)].copy()
        if not orders.empty:
            orders["target_weight"] = orders["instrument_id"].astype(str).map(target_by_id)
            current_by_id = decisions.assign(
                _instrument_id=decisions["instrument_id"].astype(str)
            ).set_index("_instrument_id")["current_weight"].astype(float)
            orders["_instrument_id"] = orders["instrument_id"].astype(str)
            orders["delta_weight"] = (
                orders["target_weight"].astype(float)
                - orders["_instrument_id"].map(current_by_id).astype(float)
            )
            orders = orders.drop(columns=["_instrument_id"])
            orders = orders.loc[orders["delta_weight"].astype(float).abs() > 1e-12].copy()
            orders["side"] = orders["delta_weight"].map(lambda value: "buy" if value > 0 else "sell")
    if not state.empty:
        mapped_targets = state["instrument_id"].astype(str).map(target_series)
        state = state.loc[mapped_targets.notna()].copy()
        state["current_weight"] = mapped_targets.loc[state.index].astype(float)
        state = state.loc[state["current_weight"].astype(float) > 1e-12].copy()
    diagnostics = result.diagnostics.copy()
    if not diagnostics.empty:
        planned = float(decisions["delta_weight"].astype(float).abs().sum())
        diagnostics.loc[:, "planned_gross_turnover"] = planned
        diagnostics.loc[:, "order_intent_count"] = int(
            decisions["delta_weight"].astype(float).abs().gt(1e-12).sum()
        )
        diagnostics.loc[:, "target_gross_exposure"] = float(
            decisions["target_weight"].astype(float).sum()
        )
        diagnostics.loc[:, "turnover_budget_limited_flag_count"] = int(
            decisions["constraint_flags"].astype(str).str.contains(flag).sum()
        )
    return StrategyPolicyResult(
        portfolio_intent=intent,
        trade_decisions=decisions,
        order_intents=orders,
        diagnostics=diagnostics,
        policy_state=state,
    )


def _append_constraint_flag(value: object, flag: str) -> str:
    flags = set(str(value or "").split(","))
    flags.discard("")
    flags.add(flag)
    return ",".join(sorted(flags))


def _load_policy_gross_exposure_schedule(
    params: TreeScoreBacktestParams,
) -> dict[str, float]:
    path = params.policy_gross_exposure_scale_path
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"policy gross exposure scale path not found: {path}")
    if path.suffix == ".parquet":
        schedule = pd.read_parquet(path)
    else:
        schedule = pd.read_csv(path)
    if "gross_exposure_scale" not in schedule.columns:
        if "policy_gross_exposure_scale" in schedule.columns:
            schedule = schedule.rename(
                columns={"policy_gross_exposure_scale": "gross_exposure_scale"}
            )
        else:
            raise ValueError("gross exposure schedule must contain gross_exposure_scale")
    if "timestamp" not in schedule.columns:
        raise ValueError("gross exposure schedule must contain timestamp")
    output: dict[str, float] = {}
    for row in schedule.itertuples(index=False):
        scale = float(getattr(row, "gross_exposure_scale"))
        if not 0 <= scale <= 1:
            raise ValueError("gross exposure schedule values must be in [0, 1]")
        key = _timestamp_key(getattr(row, "timestamp"))
        if key in output:
            raise ValueError(f"duplicate gross exposure schedule timestamp: {key}")
        output[key] = scale
    return output


def _timestamp_key(value: object) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(timestamp):
        return str(value)
    return timestamp.isoformat()


def _concat_policy_diagnostics(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def _summarize_policy_diagnostics(diagnostics: pd.DataFrame) -> dict[str, float | int]:
    if diagnostics.empty:
        return {
            "decision_timestamp_count": 0,
            "planned_gross_turnover": 0.0,
            "order_intent_count": 0,
            "entry_count": 0,
            "exit_count": 0,
            "resize_count": 0,
            "hold_count": 0,
            "no_trade_count": 0,
            "below_edge_count": 0,
            "below_weight_band_count": 0,
            "t1_sell_blocked_count": 0,
            "turnover_scaled_count": 0,
            "turnover_budget_limited_count": 0,
            "turnover_budget_limited_flag_count": 0,
            "risk_reduction_count": 0,
            "gross_exposure_scaled_count": 0,
            "average_target_gross_exposure": 0.0,
            "average_dynamic_turnover_cap": 0.0,
            "min_dynamic_turnover_cap": 0.0,
            "drawdown_brake_active_count": 0,
            "average_drawdown_brake_scale": 0.0,
            "min_drawdown_brake_scale": 0.0,
            "cost_pressure_active_count": 0,
            "average_cost_pressure_scale": 0.0,
            "min_cost_pressure_scale": 0.0,
            "cost_pressure_turnover_active_count": 0,
            "average_cost_pressure_turnover_cap": 0.0,
            "min_cost_pressure_turnover_cap": 0.0,
            "max_realized_cost_bps": 0.0,
            "turnover_budget_period_count": 0,
            "turnover_path_budget_remaining": 0.0,
            "source_change_reset_count": 0,
            "source_transition_forced_exit_count": 0,
        }
    target_gross = (
        diagnostics["target_gross_exposure"]
        if "target_gross_exposure" in diagnostics.columns
        else pd.Series(dtype=float)
    )
    dynamic_cap = (
        diagnostics["dynamic_turnover_cap"]
        if "dynamic_turnover_cap" in diagnostics.columns
        else pd.Series(dtype=float)
    )
    period_key = (
        diagnostics["turnover_budget_period_key"]
        if "turnover_budget_period_key" in diagnostics.columns
        else pd.Series(dtype=object)
    )
    path_budget_after = (
        diagnostics["turnover_path_budget_after"]
        if "turnover_path_budget_after" in diagnostics.columns
        else pd.Series(dtype=float)
    )
    drawdown_brake_scale = (
        diagnostics["drawdown_brake_scale"]
        if "drawdown_brake_scale" in diagnostics.columns
        else pd.Series(dtype=float)
    )
    drawdown_brake_active = (
        diagnostics["drawdown_brake_active"]
        if "drawdown_brake_active" in diagnostics.columns
        else pd.Series(dtype=bool)
    )
    cost_pressure_scale = (
        diagnostics["cost_pressure_scale"]
        if "cost_pressure_scale" in diagnostics.columns
        else pd.Series(dtype=float)
    )
    cost_pressure_active = (
        diagnostics["cost_pressure_active"]
        if "cost_pressure_active" in diagnostics.columns
        else pd.Series(dtype=bool)
    )
    cost_pressure_turnover_cap = (
        diagnostics["cost_pressure_turnover_cap"]
        if "cost_pressure_turnover_cap" in diagnostics.columns
        else pd.Series(dtype=float)
    )
    cost_pressure_turnover_active = (
        diagnostics["cost_pressure_turnover_active"]
        if "cost_pressure_turnover_active" in diagnostics.columns
        else pd.Series(dtype=bool)
    )
    realized_cost_bps = (
        diagnostics["realized_cost_bps"]
        if "realized_cost_bps" in diagnostics.columns
        else pd.Series(dtype=float)
    )
    return {
        "decision_timestamp_count": int(len(diagnostics)),
        "planned_gross_turnover": float(diagnostics["planned_gross_turnover"].sum()),
        "order_intent_count": int(diagnostics["order_intent_count"].sum()),
        "entry_count": int(diagnostics["entry_count"].sum()),
        "exit_count": int(diagnostics["exit_count"].sum()),
        "resize_count": int(diagnostics["resize_count"].sum()),
        "hold_count": int(diagnostics["hold_count"].sum()),
        "no_trade_count": int(diagnostics["no_trade_count"].sum()),
        "below_edge_count": int(diagnostics["below_edge_count"].sum()),
        "below_weight_band_count": int(diagnostics["below_weight_band_count"].sum()),
        "t1_sell_blocked_count": int(diagnostics["t1_sell_blocked_count"].sum()),
        "risk_reduction_count": int(
            diagnostics.get("risk_reduction_count", pd.Series(dtype=float)).sum()
        ),
        "gross_exposure_scaled_count": int(
            diagnostics.get("gross_exposure_scaled_count", pd.Series(dtype=float)).sum()
        ),
        "turnover_scaled_count": int(diagnostics["turnover_scaled_count"].sum()),
        "turnover_budget_limited_count": int(
            diagnostics.get("turnover_budget_limited_count", pd.Series(dtype=float)).sum()
        ),
        "turnover_budget_limited_flag_count": int(
            diagnostics.get(
                "turnover_budget_limited_flag_count", pd.Series(dtype=float)
            ).sum()
        ),
        "average_target_gross_exposure": (
            float(target_gross.mean()) if not target_gross.empty else 0.0
        ),
        "average_dynamic_turnover_cap": (
            float(dynamic_cap.mean()) if not dynamic_cap.empty else 0.0
        ),
        "min_dynamic_turnover_cap": float(dynamic_cap.min()) if not dynamic_cap.empty else 0.0,
        "drawdown_brake_active_count": (
            int(drawdown_brake_active.astype(bool).sum())
            if not drawdown_brake_active.empty
            else 0
        ),
        "average_drawdown_brake_scale": (
            float(drawdown_brake_scale.astype(float).mean())
            if not drawdown_brake_scale.empty
            else 0.0
        ),
        "min_drawdown_brake_scale": (
            float(drawdown_brake_scale.astype(float).min())
            if not drawdown_brake_scale.empty
            else 0.0
        ),
        "cost_pressure_active_count": (
            int(cost_pressure_active.astype(bool).sum())
            if not cost_pressure_active.empty
            else 0
        ),
        "average_cost_pressure_scale": (
            float(cost_pressure_scale.astype(float).mean())
            if not cost_pressure_scale.empty
            else 0.0
        ),
        "min_cost_pressure_scale": (
            float(cost_pressure_scale.astype(float).min())
            if not cost_pressure_scale.empty
            else 0.0
        ),
        "cost_pressure_turnover_active_count": (
            int(cost_pressure_turnover_active.astype(bool).sum())
            if not cost_pressure_turnover_active.empty
            else 0
        ),
        "average_cost_pressure_turnover_cap": (
            float(cost_pressure_turnover_cap.astype(float).mean())
            if not cost_pressure_turnover_cap.empty
            else 0.0
        ),
        "min_cost_pressure_turnover_cap": (
            float(cost_pressure_turnover_cap.astype(float).min())
            if not cost_pressure_turnover_cap.empty
            else 0.0
        ),
        "max_realized_cost_bps": (
            float(realized_cost_bps.astype(float).max())
            if not realized_cost_bps.empty
            else 0.0
        ),
        "turnover_budget_period_count": (
            int(period_key.dropna().nunique()) if not period_key.empty else 0
        ),
        "turnover_path_budget_remaining": (
            float(path_budget_after.iloc[-1]) if not path_budget_after.empty else 0.0
        ),
        "source_change_reset_count": int(
            diagnostics.get("source_change_reset", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "source_transition_forced_exit_count": int(
            diagnostics.get(
                "source_transition_forced_exit_count",
                pd.Series(dtype=float),
            ).sum()
        ),
    }


def _summary_payload(
    params: TreeScoreBacktestParams,
    *,
    bar_count: int,
    signal_count: int,
    execution_row_count: int,
    execution_constraint_counts: dict[str, int],
    instrument_count: int,
    policy_diagnostics: dict[str, float | int],
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "params": {
            "predictions_path": str(params.predictions_path),
            "catalog_path": str(params.catalog_path),
            "start": params.start,
            "end": params.end,
            "top_n": params.top_n,
            "initial_cash": params.initial_cash,
            "commission_bps": params.commission_bps,
            "slippage_bps": params.slippage_bps,
            "sell_stamp_tax_bps": params.sell_stamp_tax_bps,
            "min_commission": params.min_commission,
            "lot_size": params.lot_size,
            "trade_policy": params.trade_policy,
            "rebalance_every_n_bars": params.rebalance_every_n_bars,
            "hold_rank_buffer": params.hold_rank_buffer,
            "policy_entry_rank": params.policy_entry_rank,
            "policy_exit_rank": params.policy_exit_rank,
            "policy_max_entries_per_rebalance": params.policy_max_entries_per_rebalance,
            "policy_max_exits_per_rebalance": params.policy_max_exits_per_rebalance,
            "policy_min_hold_bars": params.policy_min_hold_bars,
            "policy_min_expected_edge_bps": params.policy_min_expected_edge_bps,
            "policy_estimated_cost_bps": params.policy_estimated_cost_bps,
            "policy_no_trade_weight_band": params.policy_no_trade_weight_band,
            "policy_partial_rebalance_rate": params.policy_partial_rebalance_rate,
            "policy_max_gross_turnover_per_rebalance": params.policy_max_gross_turnover_per_rebalance,
            "policy_total_gross_turnover_budget": params.policy_total_gross_turnover_budget,
            "policy_turnover_budget_period": params.policy_turnover_budget_period,
            "policy_turnover_budget_pacing": params.policy_turnover_budget_pacing,
            "policy_gross_exposure_scale": params.policy_gross_exposure_scale,
            "policy_gross_exposure_scale_path": (
                str(params.policy_gross_exposure_scale_path)
                if params.policy_gross_exposure_scale_path is not None
                else None
            ),
            "policy_drawdown_brake_threshold": params.policy_drawdown_brake_threshold,
            "policy_drawdown_brake_reduced_scale": (
                params.policy_drawdown_brake_reduced_scale
            ),
            "policy_cost_pressure_threshold_bps": params.policy_cost_pressure_threshold_bps,
            "policy_cost_pressure_reduced_scale": (
                params.policy_cost_pressure_reduced_scale
            ),
            "policy_cost_pressure_max_gross_turnover_per_rebalance": (
                params.policy_cost_pressure_max_gross_turnover_per_rebalance
            ),
            "policy_reset_on_source_change": params.policy_reset_on_source_change,
            "policy_force_source_transition_exits": (
                params.policy_force_source_transition_exits
            ),
            "policy_source_transition_exit_rate": params.policy_source_transition_exit_rate,
            "policy_source_transition_turnover_cap": (
                params.policy_source_transition_turnover_cap
            ),
            "policy_source_column": params.policy_source_column,
            "optimizer_candidate_rank": params.optimizer_candidate_rank,
            "optimizer_score_to_edge_bps": params.optimizer_score_to_edge_bps,
            "optimizer_min_net_edge_bps": params.optimizer_min_net_edge_bps,
            "optimizer_risk_penalty_multiplier": params.optimizer_risk_penalty_multiplier,
            "optimizer_target_cap_mode": params.optimizer_target_cap_mode,
            "optimizer_weighting": params.optimizer_weighting,
            "optimizer_max_name_weight": params.optimizer_max_name_weight,
            "optimizer_max_gross_exposure_increase_per_rebalance": (
                params.optimizer_max_gross_exposure_increase_per_rebalance
            ),
            "min_trade_weight": params.min_trade_weight,
            "exclude_st": params.exclude_st,
            "limit_up_bps": params.limit_up_bps,
            "limit_down_bps": params.limit_down_bps,
            "max_bar_turnover_participation": params.max_bar_turnover_participation,
            "allow_same_bar_capacity": params.allow_same_bar_capacity,
            "data_access_mode": params.data_access_mode,
            "streaming_chunk": params.streaming_chunk,
            "streaming_chunk_padding_days": params.streaming_chunk_padding_days,
        },
        "bar_count": int(bar_count),
        "signal_count": int(signal_count),
        "execution_row_count": int(execution_row_count),
        "execution_constraint_counts": execution_constraint_counts,
        "instrument_count": int(instrument_count),
        "policy_diagnostics": policy_diagnostics,
        "metrics": metrics,
    }


def _build_buffered_target_weights(
    ranked_signals: pd.DataFrame,
    params: TreeScoreBacktestParams,
) -> pd.DataFrame:
    if ranked_signals.empty:
        return _empty_target_weight_frame()
    rows: list[pd.DataFrame] = []
    previous_targets: list[str] = []
    buffer_rank = params.hold_rank_buffer
    for signal_time, group in ranked_signals.groupby("signal_time", sort=True):
        group = group.sort_values(["rank", "instrument_id"]).copy()
        rank_by_instrument = {
            str(row.instrument_id): int(row.rank)
            for row in group.itertuples(index=False)
        }
        selected: list[str] = []
        if buffer_rank is not None:
            selected.extend(
                instrument_id
                for instrument_id in previous_targets
                if rank_by_instrument.get(instrument_id, buffer_rank + 1)
                <= buffer_rank
            )
        for row in group.itertuples(index=False):
            instrument_id = str(row.instrument_id)
            if instrument_id not in selected:
                selected.append(instrument_id)
            if len(selected) >= params.top_n:
                break
        selected = selected[: params.top_n]
        if not selected:
            continue
        selected_frame = group.loc[group["instrument_id"].astype(str).isin(selected)].copy()
        selected_frame["_order"] = selected_frame["instrument_id"].astype(str).map(
            {instrument_id: index for index, instrument_id in enumerate(selected)}
        )
        selected_frame = selected_frame.sort_values("_order")
        selected_frame["bar_end_time"] = signal_time
        selected_frame["target_weight"] = 1.0 / len(selected_frame)
        rows.append(
            selected_frame.loc[
                :,
                [
                    "signal_time",
                    "bar_end_time",
                    "instrument_id",
                    "score",
                    "rank",
                    "target_weight",
                ],
            ]
        )
        previous_targets = selected
    if not rows:
        return _empty_target_weight_frame()
    return pd.concat(rows, ignore_index=True)


def _empty_target_weight_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_time",
            "bar_end_time",
            "instrument_id",
            "score",
            "rank",
            "target_weight",
        ]
    )


def _build_tree_score_executions(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    tracked_instruments: set[str] | None = None,
    sparse: bool = False,
) -> pd.DataFrame:
    signal_times = sorted(signals["signal_time"].unique().tolist())
    all_times = sorted(bars["bar_end_time"].unique().tolist())
    next_time_by_signal = {
        signal_time: all_times[index + 1]
        for index, signal_time in enumerate(all_times[:-1])
        if signal_time in signal_times
    }
    shifted = signals.copy()
    shifted["exec_time"] = shifted["signal_time"].map(next_time_by_signal)
    shifted = shifted.loc[shifted["exec_time"].notna()].copy()
    prices = bars.loc[
        :,
        [
            "bar_end_time",
            "instrument_id",
            "canonical_code",
            "open_price",
            "close_price",
            "turnover",
            "tradable_bar",
            "limit_up_open",
            "limit_down_open",
        ],
    ].rename(columns={"bar_end_time": "exec_time"})
    if sparse:
        relevant_instruments = {str(value) for value in tracked_instruments or set()}
        if not shifted.empty:
            relevant_instruments.update(shifted["instrument_id"].astype(str).unique())
        if not relevant_instruments:
            return pd.DataFrame(columns=[*prices.columns, "target_weight"])
        prices = prices.loc[
            _instrument_isin(prices["instrument_id"], relevant_instruments)
        ].copy()
    if shifted.empty:
        return prices.assign(target_weight=pd.NA)
    target_weights = shifted.loc[
        :,
        ["exec_time", "instrument_id", "target_weight"],
    ].copy()
    return prices.merge(target_weights, on=["exec_time", "instrument_id"], how="left")


def _bar_time_index(bars: pd.DataFrame) -> BarTimeIndex:
    bar_times = sorted(bars["bar_end_time"].unique().tolist())
    return BarTimeIndex(
        bar_times=bar_times,
        next_time_by_time={
            bar_time: bar_times[index + 1]
            for index, bar_time in enumerate(bar_times[:-1])
        },
        row_indices_by_time=dict(bars.groupby("bar_end_time", sort=False).indices),
    )


def _build_segment_tree_score_executions(
    bars: pd.DataFrame,
    bar_time_index: BarTimeIndex,
    signals: pd.DataFrame,
    *,
    tracked_instruments: set[str],
    start_exclusive: object,
    end_inclusive: object | None,
) -> pd.DataFrame:
    if end_inclusive is None:
        return _empty_tree_score_execution_frame()
    relevant_instruments = {str(value) for value in tracked_instruments}
    if not signals.empty:
        relevant_instruments.update(signals["instrument_id"].astype(str).unique())
    if not relevant_instruments:
        return _empty_tree_score_execution_frame()
    prices = _price_execution_rows_from_index(
        bars,
        bar_time_index,
        tracked_instruments=relevant_instruments,
        start_exclusive=start_exclusive,
        end_inclusive=end_inclusive,
    )
    if prices.empty or signals.empty:
        return prices
    shifted = signals.copy()
    shifted["exec_time"] = shifted["signal_time"].map(bar_time_index.next_time_by_time)
    shifted = shifted.loc[shifted["exec_time"].notna()]
    if shifted.empty:
        return prices
    target_weights = shifted.loc[
        :,
        ["exec_time", "instrument_id", "target_weight"],
    ].copy()
    return prices.drop(columns=["target_weight"]).merge(
        target_weights,
        on=["exec_time", "instrument_id"],
        how="left",
    )


def _next_time_by_signal(
    signals: pd.DataFrame,
    bar_times: list[object],
) -> dict[object, object]:
    signal_times = set(signals["signal_time"].unique().tolist())
    return {
        signal_time: bar_times[index + 1]
        for index, signal_time in enumerate(bar_times[:-1])
        if signal_time in signal_times
    }


def _price_execution_rows_from_index(
    bars: pd.DataFrame,
    bar_time_index: BarTimeIndex,
    *,
    tracked_instruments: set[str],
    start_exclusive: object | None,
    end_inclusive: object,
) -> pd.DataFrame:
    if not tracked_instruments:
        return _empty_tree_score_execution_frame()
    row_indices = _time_window_row_indices(
        bar_time_index,
        start_exclusive=start_exclusive,
        end_inclusive=end_inclusive,
    )
    if not row_indices:
        return _empty_tree_score_execution_frame()
    candidate = bars.take(row_indices)
    prices = candidate.loc[
        _instrument_isin(candidate["instrument_id"], tracked_instruments),
        [
            "bar_end_time",
            "instrument_id",
            "canonical_code",
            "open_price",
            "close_price",
            "turnover",
            "tradable_bar",
            "limit_up_open",
            "limit_down_open",
        ],
    ].rename(columns={"bar_end_time": "exec_time"})
    return prices.assign(target_weight=pd.NA)


def _instrument_isin(values: pd.Series, instruments: set[str]) -> pd.Series:
    if pd.api.types.is_string_dtype(values.dtype) and pd.api.types.infer_dtype(
        values,
        skipna=True,
    ) in {"string", "unicode", "empty"}:
        return values.isin(instruments)
    return values.astype(str).isin(instruments)


def _time_window_row_indices(
    bar_time_index: BarTimeIndex,
    *,
    start_exclusive: object | None,
    end_inclusive: object,
) -> list[int]:
    start_index = (
        0
        if start_exclusive is None
        else bisect_right(bar_time_index.bar_times, start_exclusive)
    )
    end_index = bisect_right(bar_time_index.bar_times, end_inclusive)
    row_indices: list[int] = []
    for bar_time in bar_time_index.bar_times[start_index:end_index]:
        row_indices.extend(bar_time_index.row_indices_by_time[bar_time])
    return row_indices


def _empty_tree_score_execution_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "exec_time",
            "instrument_id",
            "canonical_code",
            "open_price",
            "close_price",
            "turnover",
            "tradable_bar",
            "limit_up_open",
            "limit_down_open",
            "target_weight",
        ]
    )


def _next_segment_end(
    signal_time: object,
    *,
    next_signal_time: object | None,
    bar_times: list[object],
) -> object | None:
    if next_signal_time is not None:
        return next_signal_time
    for bar_time in bar_times:
        if bar_time > signal_time:
            return bar_time
    return None


def _write_outputs(result: dict[str, object], params: TreeScoreBacktestParams) -> None:
    summary = result["summary"]
    (params.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> TreeScoreBacktestParams:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument(
        "--catalog-path",
        default="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=0.0)
    parser.add_argument("--min-commission", type=float, default=0.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument(
        "--trade-policy",
        choices=("naive_top_n", "rank_buffer_drop", "cost_aware_optimizer"),
        default="naive_top_n",
    )
    parser.add_argument("--rebalance-every-n-bars", type=int, default=1)
    parser.add_argument("--hold-rank-buffer", type=int)
    parser.add_argument("--policy-entry-rank", type=int)
    parser.add_argument("--policy-exit-rank", type=int)
    parser.add_argument("--policy-max-entries-per-rebalance", type=int)
    parser.add_argument("--policy-max-exits-per-rebalance", type=int)
    parser.add_argument("--policy-min-hold-bars", type=int, default=0)
    parser.add_argument("--policy-min-expected-edge-bps", type=float)
    parser.add_argument("--policy-estimated-cost-bps", type=float)
    parser.add_argument("--policy-no-trade-weight-band", type=float, default=0.0)
    parser.add_argument("--policy-partial-rebalance-rate", type=float, default=1.0)
    parser.add_argument("--policy-max-gross-turnover-per-rebalance", type=float)
    parser.add_argument("--policy-total-gross-turnover-budget", type=float)
    parser.add_argument(
        "--policy-turnover-budget-period",
        choices=("path", "year", "month"),
        default="path",
    )
    parser.add_argument("--policy-turnover-budget-pacing", type=float, default=0.0)
    parser.add_argument("--policy-gross-exposure-scale", type=float, default=1.0)
    parser.add_argument("--policy-gross-exposure-scale-path")
    parser.add_argument("--policy-drawdown-brake-threshold", type=float)
    parser.add_argument("--policy-drawdown-brake-reduced-scale", type=float, default=0.5)
    parser.add_argument("--policy-cost-pressure-threshold-bps", type=float)
    parser.add_argument("--policy-cost-pressure-reduced-scale", type=float, default=0.7)
    parser.add_argument("--policy-cost-pressure-max-gross-turnover-per-rebalance", type=float)
    parser.add_argument(
        "--policy-reset-on-source-change",
        action="store_true",
        help="Reset policy-held target state when the per-timestamp signal source changes.",
    )
    parser.add_argument(
        "--policy-force-source-transition-exits",
        action="store_true",
        help=(
            "Force held names from a previous signal source to leave through "
            "normal capped exits when the current signal source changes."
        ),
    )
    parser.add_argument("--policy-source-transition-exit-rate", type=float, default=1.0)
    parser.add_argument("--policy-source-transition-turnover-cap", type=float)
    parser.add_argument("--policy-source-column", default="signal_source")
    parser.add_argument("--optimizer-candidate-rank", type=int)
    parser.add_argument("--optimizer-score-to-edge-bps", type=float, default=100.0)
    parser.add_argument("--optimizer-min-net-edge-bps", type=float, default=0.0)
    parser.add_argument("--optimizer-risk-penalty-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--optimizer-target-cap-mode",
        choices=("clip", "redistribute", "replace"),
        default="clip",
    )
    parser.add_argument(
        "--optimizer-weighting",
        choices=("equal", "utility"),
        default="utility",
    )
    parser.add_argument("--optimizer-max-name-weight", type=float)
    parser.add_argument("--optimizer-max-gross-exposure-increase-per-rebalance", type=float)
    parser.add_argument("--min-trade-weight", type=float, default=0.0)
    parser.add_argument("--exclude-st", action="store_true")
    parser.add_argument("--limit-up-bps", type=float)
    parser.add_argument("--limit-down-bps", type=float)
    parser.add_argument("--max-bar-turnover-participation", type=float)
    parser.add_argument(
        "--allow-same-bar-capacity",
        action="store_true",
        help=(
            "Explicitly allow turnover/volume from the execution bar to cap "
            "open-price fills. This is an optimistic bar-volume execution "
            "assumption and is disabled by default."
        ),
    )
    parser.add_argument(
        "--data-access-mode",
        choices=("data_portal", "fast_parquet"),
        default="data_portal",
    )
    parser.add_argument(
        "--streaming-chunk",
        choices=("year", "month", "week", "day"),
        default="month",
    )
    parser.add_argument("--streaming-chunk-padding-days", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.rebalance_every_n_bars <= 0:
        raise ValueError("--rebalance-every-n-bars must be positive")
    if args.hold_rank_buffer is not None and args.hold_rank_buffer < args.top_n:
        raise ValueError("--hold-rank-buffer must be greater than or equal to --top-n")
    if args.policy_entry_rank is not None and args.policy_entry_rank <= 0:
        raise ValueError("--policy-entry-rank must be positive")
    if args.policy_exit_rank is not None and args.policy_exit_rank <= 0:
        raise ValueError("--policy-exit-rank must be positive")
    entry_rank = args.policy_entry_rank or args.top_n
    exit_rank = args.policy_exit_rank or max(args.top_n, args.hold_rank_buffer or args.top_n)
    if exit_rank < entry_rank:
        raise ValueError("--policy-exit-rank must be greater than or equal to entry rank")
    if (
        args.policy_max_entries_per_rebalance is not None
        and args.policy_max_entries_per_rebalance < 0
    ):
        raise ValueError("--policy-max-entries-per-rebalance must be non-negative")
    if (
        args.policy_max_exits_per_rebalance is not None
        and args.policy_max_exits_per_rebalance < 0
    ):
        raise ValueError("--policy-max-exits-per-rebalance must be non-negative")
    if args.policy_min_hold_bars < 0:
        raise ValueError("--policy-min-hold-bars must be non-negative")
    if (
        args.policy_min_expected_edge_bps is not None
        and args.policy_min_expected_edge_bps < 0
    ):
        raise ValueError("--policy-min-expected-edge-bps must be non-negative")
    if args.policy_estimated_cost_bps is not None and args.policy_estimated_cost_bps < 0:
        raise ValueError("--policy-estimated-cost-bps must be non-negative")
    if args.policy_no_trade_weight_band < 0:
        raise ValueError("--policy-no-trade-weight-band must be non-negative")
    if not 0 < args.policy_partial_rebalance_rate <= 1:
        raise ValueError("--policy-partial-rebalance-rate must be in (0, 1]")
    if (
        args.policy_max_gross_turnover_per_rebalance is not None
        and args.policy_max_gross_turnover_per_rebalance < 0
    ):
        raise ValueError("--policy-max-gross-turnover-per-rebalance must be non-negative")
    if (
        args.policy_total_gross_turnover_budget is not None
        and args.policy_total_gross_turnover_budget < 0
    ):
        raise ValueError("--policy-total-gross-turnover-budget must be non-negative")
    if args.policy_turnover_budget_pacing < 0:
        raise ValueError("--policy-turnover-budget-pacing must be non-negative")
    if not 0 <= args.policy_gross_exposure_scale <= 1:
        raise ValueError("--policy-gross-exposure-scale must be in [0, 1]")
    if (
        args.policy_drawdown_brake_threshold is not None
        and not -1 < args.policy_drawdown_brake_threshold < 0
    ):
        raise ValueError("--policy-drawdown-brake-threshold must be in (-1, 0)")
    if not 0 <= args.policy_drawdown_brake_reduced_scale <= 1:
        raise ValueError("--policy-drawdown-brake-reduced-scale must be in [0, 1]")
    if (
        args.policy_cost_pressure_threshold_bps is not None
        and args.policy_cost_pressure_threshold_bps < 0
    ):
        raise ValueError("--policy-cost-pressure-threshold-bps must be non-negative")
    if not 0 <= args.policy_cost_pressure_reduced_scale <= 1:
        raise ValueError("--policy-cost-pressure-reduced-scale must be in [0, 1]")
    if (
        args.policy_cost_pressure_max_gross_turnover_per_rebalance is not None
        and args.policy_cost_pressure_max_gross_turnover_per_rebalance < 0
    ):
        raise ValueError(
            "--policy-cost-pressure-max-gross-turnover-per-rebalance must be non-negative"
        )
    if args.optimizer_candidate_rank is not None and args.optimizer_candidate_rank <= 0:
        raise ValueError("--optimizer-candidate-rank must be positive")
    if args.optimizer_score_to_edge_bps < 0:
        raise ValueError("--optimizer-score-to-edge-bps must be non-negative")
    if args.optimizer_min_net_edge_bps < 0:
        raise ValueError("--optimizer-min-net-edge-bps must be non-negative")
    if args.optimizer_risk_penalty_multiplier < 0:
        raise ValueError("--optimizer-risk-penalty-multiplier must be non-negative")
    if args.optimizer_max_name_weight is not None and not 0 < args.optimizer_max_name_weight <= 1:
        raise ValueError("--optimizer-max-name-weight must be in (0, 1]")
    if (
        args.optimizer_max_gross_exposure_increase_per_rebalance is not None
        and args.optimizer_max_gross_exposure_increase_per_rebalance < 0
    ):
        raise ValueError(
            "--optimizer-max-gross-exposure-increase-per-rebalance must be non-negative"
        )
    if args.commission_bps < 0:
        raise ValueError("--commission-bps must be non-negative")
    if args.slippage_bps < 0:
        raise ValueError("--slippage-bps must be non-negative")
    if args.sell_stamp_tax_bps < 0:
        raise ValueError("--sell-stamp-tax-bps must be non-negative")
    if args.min_commission < 0:
        raise ValueError("--min-commission must be non-negative")
    if not 0 <= args.min_trade_weight <= 1:
        raise ValueError("--min-trade-weight must be in [0, 1]")
    if args.limit_up_bps is not None and args.limit_up_bps <= 0:
        raise ValueError("--limit-up-bps must be positive")
    if args.limit_down_bps is not None and args.limit_down_bps <= 0:
        raise ValueError("--limit-down-bps must be positive")
    if (
        args.max_bar_turnover_participation is not None
        and not 0 < args.max_bar_turnover_participation <= 1
    ):
        raise ValueError("--max-bar-turnover-participation must be in (0, 1]")
    if (
        args.policy_reset_on_source_change or args.policy_force_source_transition_exits
    ) and not args.policy_source_column:
        raise ValueError(
            "--policy-source-column must be non-empty when source transition handling is enabled"
        )
    if not 0 < args.policy_source_transition_exit_rate <= 1:
        raise ValueError("--policy-source-transition-exit-rate must be in (0, 1]")
    if (
        args.policy_source_transition_turnover_cap is not None
        and args.policy_source_transition_turnover_cap < 0
    ):
        raise ValueError("--policy-source-transition-turnover-cap must be non-negative")
    if args.streaming_chunk_padding_days < 0:
        raise ValueError("--streaming-chunk-padding-days must be non-negative")
    return TreeScoreBacktestParams(
        predictions_path=Path(args.predictions_path),
        catalog_path=Path(args.catalog_path),
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        initial_cash=args.initial_cash,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        sell_stamp_tax_bps=args.sell_stamp_tax_bps,
        min_commission=args.min_commission,
        lot_size=args.lot_size,
        trade_policy=args.trade_policy,
        rebalance_every_n_bars=args.rebalance_every_n_bars,
        hold_rank_buffer=args.hold_rank_buffer,
        policy_entry_rank=args.policy_entry_rank,
        policy_exit_rank=args.policy_exit_rank,
        policy_max_entries_per_rebalance=args.policy_max_entries_per_rebalance,
        policy_max_exits_per_rebalance=args.policy_max_exits_per_rebalance,
        policy_min_hold_bars=args.policy_min_hold_bars,
        policy_min_expected_edge_bps=args.policy_min_expected_edge_bps,
        policy_estimated_cost_bps=_resolved_policy_estimated_cost_bps(args),
        policy_no_trade_weight_band=args.policy_no_trade_weight_band,
        policy_partial_rebalance_rate=args.policy_partial_rebalance_rate,
        policy_max_gross_turnover_per_rebalance=args.policy_max_gross_turnover_per_rebalance,
        policy_total_gross_turnover_budget=args.policy_total_gross_turnover_budget,
        policy_turnover_budget_period=args.policy_turnover_budget_period,
        policy_turnover_budget_pacing=args.policy_turnover_budget_pacing,
        policy_gross_exposure_scale=args.policy_gross_exposure_scale,
        policy_gross_exposure_scale_path=(
            Path(args.policy_gross_exposure_scale_path)
            if args.policy_gross_exposure_scale_path
            else None
        ),
        policy_drawdown_brake_threshold=args.policy_drawdown_brake_threshold,
        policy_drawdown_brake_reduced_scale=args.policy_drawdown_brake_reduced_scale,
        policy_cost_pressure_threshold_bps=args.policy_cost_pressure_threshold_bps,
        policy_cost_pressure_reduced_scale=args.policy_cost_pressure_reduced_scale,
        policy_cost_pressure_max_gross_turnover_per_rebalance=(
            args.policy_cost_pressure_max_gross_turnover_per_rebalance
        ),
        policy_reset_on_source_change=args.policy_reset_on_source_change,
        policy_force_source_transition_exits=args.policy_force_source_transition_exits,
        policy_source_transition_exit_rate=args.policy_source_transition_exit_rate,
        policy_source_transition_turnover_cap=args.policy_source_transition_turnover_cap,
        policy_source_column=args.policy_source_column,
        optimizer_candidate_rank=args.optimizer_candidate_rank,
        optimizer_score_to_edge_bps=args.optimizer_score_to_edge_bps,
        optimizer_min_net_edge_bps=args.optimizer_min_net_edge_bps,
        optimizer_risk_penalty_multiplier=args.optimizer_risk_penalty_multiplier,
        optimizer_target_cap_mode=args.optimizer_target_cap_mode,
        optimizer_weighting=args.optimizer_weighting,
        optimizer_max_name_weight=args.optimizer_max_name_weight,
        optimizer_max_gross_exposure_increase_per_rebalance=(
            args.optimizer_max_gross_exposure_increase_per_rebalance
        ),
        min_trade_weight=args.min_trade_weight,
        exclude_st=args.exclude_st,
        limit_up_bps=args.limit_up_bps,
        limit_down_bps=args.limit_down_bps,
        max_bar_turnover_participation=args.max_bar_turnover_participation,
        allow_same_bar_capacity=args.allow_same_bar_capacity,
        data_access_mode=args.data_access_mode,
        streaming_chunk=args.streaming_chunk,
        streaming_chunk_padding_days=args.streaming_chunk_padding_days,
        output_dir=Path(args.output_dir),
    )


def _resolved_policy_estimated_cost_bps(args: argparse.Namespace) -> float:
    if args.policy_estimated_cost_bps is not None:
        return float(args.policy_estimated_cost_bps)
    return _estimated_round_trip_cost_bps(
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        sell_stamp_tax_bps=args.sell_stamp_tax_bps,
    )


def _estimated_round_trip_cost_bps(
    *,
    commission_bps: float,
    slippage_bps: float,
    sell_stamp_tax_bps: float,
) -> float:
    return float(2.0 * commission_bps + 2.0 * slippage_bps + sell_stamp_tax_bps)


if __name__ == "__main__":
    main()
