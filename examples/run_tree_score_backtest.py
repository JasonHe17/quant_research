"""Backtest tree-model prediction scores with A-share T+1 execution."""

from __future__ import annotations

import argparse
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
    RankBufferDropConfig,
    RankBufferDropPolicy,
    empty_portfolio_state,
)

from run_baseline_a_real_backtest import (
    BacktestParams,
    SimulationState,
    _append_frame_csv,
    _empty_execution_constraint_counts,
    _empty_trade_metric_totals,
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
    min_trade_weight: float
    exclude_st: bool
    limit_up_bps: float | None
    limit_down_bps: float | None
    max_bar_turnover_participation: float | None
    data_access_mode: str
    streaming_chunk: str
    streaming_chunk_padding_days: int
    output_dir: Path


@dataclass(frozen=True, slots=True)
class TargetWeightBuildResult:
    """Target weights plus state and diagnostics from a strategy policy pass."""

    target_weights: pd.DataFrame
    policy_state: pd.DataFrame
    diagnostics: pd.DataFrame


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
    trades, equity_curve, _, state = _simulate(executions, backtest_params)
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
            "min_trade_weight": params.min_trade_weight,
            "exclude_st": params.exclude_st,
            "limit_up_bps": params.limit_up_bps,
            "limit_down_bps": params.limit_down_bps,
            "max_bar_turnover_participation": params.max_bar_turnover_participation,
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
        )
        signals = target_build.target_weights
        policy_state = target_build.policy_state
        if not target_build.diagnostics.empty:
            policy_diagnostics.append(target_build.diagnostics)
        total_signals += len(signals)
        executions = _build_tree_score_executions(bars, signals)
        if executions.empty:
            continue
        total_executions += len(executions)
        _merge_execution_constraint_counts(
            execution_constraint_counts,
            _execution_constraint_counts(executions),
        )
        period_trades, period_equity, _, state = _simulate(
            executions,
            backtest_params,
            state=state,
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
                    t.time_rank,
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
                rank
            FROM ranked
            WHERE rank <= ?
            ORDER BY signal_time, rank
        """
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
    return max(limits)


def _prediction_scan_target(path: Path) -> str:
    if path.is_dir():
        return str(path / "*.parquet")
    return str(path)


def _build_target_weights(
    ranked_signals: pd.DataFrame,
    params: TreeScoreBacktestParams,
    *,
    policy_state: pd.DataFrame | None = None,
) -> TargetWeightBuildResult:
    if params.trade_policy == "naive_top_n":
        return TargetWeightBuildResult(
            target_weights=_build_buffered_target_weights(ranked_signals, params),
            policy_state=policy_state
            if policy_state is not None
            else empty_portfolio_state(),
            diagnostics=pd.DataFrame(),
        )
    if params.trade_policy != "rank_buffer_drop":
        raise ValueError(f"unsupported trade policy: {params.trade_policy}")
    policy = RankBufferDropPolicy(_rank_buffer_drop_config(params))
    diagnostics: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    state = policy_state if policy_state is not None else empty_portfolio_state()
    for signal_time, group in ranked_signals.groupby("signal_time", sort=True):
        forecasts = group.rename(columns={"signal_time": "timestamp"}).loc[
            :, ["timestamp", "instrument_id", "score", "rank"]
        ]
        result = policy.decide(forecasts, state)
        state = result.policy_state
        diagnostics.append(result.diagnostics)
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
    if targets:
        target_weights = pd.concat(targets, ignore_index=True)
    else:
        target_weights = _empty_target_weight_frame()
    return TargetWeightBuildResult(
        target_weights=target_weights,
        policy_state=state,
        diagnostics=_concat_policy_diagnostics(diagnostics),
    )


def _rank_buffer_drop_config(params: TreeScoreBacktestParams) -> RankBufferDropConfig:
    return RankBufferDropConfig(
        target_count=params.top_n,
        entry_rank=params.policy_entry_rank or params.top_n,
        exit_rank=params.policy_exit_rank or max(params.top_n, params.hold_rank_buffer or params.top_n),
        max_entries_per_rebalance=params.policy_max_entries_per_rebalance,
        max_exits_per_rebalance=params.policy_max_exits_per_rebalance,
        min_hold_bars=params.policy_min_hold_bars,
        min_expected_edge_bps=params.policy_min_expected_edge_bps,
        estimated_cost_bps=params.policy_estimated_cost_bps,
        no_trade_weight_band=params.policy_no_trade_weight_band,
        partial_rebalance_rate=params.policy_partial_rebalance_rate,
        max_gross_turnover_per_rebalance=params.policy_max_gross_turnover_per_rebalance,
    )


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
        }
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
        "turnover_scaled_count": int(diagnostics["turnover_scaled_count"].sum()),
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
            "min_trade_weight": params.min_trade_weight,
            "exclude_st": params.exclude_st,
            "limit_up_bps": params.limit_up_bps,
            "limit_down_bps": params.limit_down_bps,
            "max_bar_turnover_participation": params.max_bar_turnover_participation,
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
    if shifted.empty:
        return prices.assign(target_weight=pd.NA)
    target_weights = shifted.loc[
        :,
        ["exec_time", "instrument_id", "target_weight"],
    ].copy()
    return prices.merge(target_weights, on=["exec_time", "instrument_id"], how="left")


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
        choices=("naive_top_n", "rank_buffer_drop"),
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
    parser.add_argument("--policy-estimated-cost-bps", type=float, default=0.0)
    parser.add_argument("--policy-no-trade-weight-band", type=float, default=0.0)
    parser.add_argument("--policy-partial-rebalance-rate", type=float, default=1.0)
    parser.add_argument("--policy-max-gross-turnover-per-rebalance", type=float)
    parser.add_argument("--min-trade-weight", type=float, default=0.0)
    parser.add_argument("--exclude-st", action="store_true")
    parser.add_argument("--limit-up-bps", type=float)
    parser.add_argument("--limit-down-bps", type=float)
    parser.add_argument("--max-bar-turnover-participation", type=float)
    parser.add_argument(
        "--data-access-mode",
        choices=("data_portal", "fast_parquet"),
        default="data_portal",
    )
    parser.add_argument(
        "--streaming-chunk",
        choices=("year", "month"),
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
    if args.policy_estimated_cost_bps < 0:
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
        policy_estimated_cost_bps=args.policy_estimated_cost_bps,
        policy_no_trade_weight_band=args.policy_no_trade_weight_band,
        policy_partial_rebalance_rate=args.policy_partial_rebalance_rate,
        policy_max_gross_turnover_per_rebalance=args.policy_max_gross_turnover_per_rebalance,
        min_trade_weight=args.min_trade_weight,
        exclude_st=args.exclude_st,
        limit_up_bps=args.limit_up_bps,
        limit_down_bps=args.limit_down_bps,
        max_bar_turnover_participation=args.max_bar_turnover_participation,
        data_access_mode=args.data_access_mode,
        streaming_chunk=args.streaming_chunk,
        streaming_chunk_padding_days=args.streaming_chunk_padding_days,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
