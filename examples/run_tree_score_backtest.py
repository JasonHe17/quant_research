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

from run_baseline_a_real_backtest import (
    BacktestParams,
    _append_frame_csv,
    _execution_constraint_counts,
    _final_positions,
    _load_bars,
    _simulate,
    _trade_metrics,
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
    rebalance_every_n_bars: int
    hold_rank_buffer: int | None
    min_trade_weight: float
    exclude_st: bool
    limit_up_bps: float | None
    limit_down_bps: float | None
    max_bar_turnover_participation: float | None
    output_dir: Path


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
    )
    params.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("trades.csv", "equity_curve.csv"):
        path = params.output_dir / filename
        if path.exists():
            path.unlink()

    ranked_signals = _load_ranked_score_signals(params)
    signals = _build_buffered_target_weights(ranked_signals, params)
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
            "rebalance_every_n_bars": params.rebalance_every_n_bars,
            "hold_rank_buffer": params.hold_rank_buffer,
            "min_trade_weight": params.min_trade_weight,
            "exclude_st": params.exclude_st,
            "limit_up_bps": params.limit_up_bps,
            "limit_down_bps": params.limit_down_bps,
            "max_bar_turnover_participation": params.max_bar_turnover_participation,
        },
        "bar_count": int(len(bars)),
        "signal_count": int(len(signals)),
        "execution_row_count": int(len(executions)),
        "execution_constraint_counts": execution_constraint_counts,
        "instrument_count": int(bars["instrument_id"].nunique()),
        "metrics": metrics,
    }
    return {
        "summary": summary,
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve,
        "final_positions": final_positions,
    }


def _load_ranked_score_signals(params: TreeScoreBacktestParams) -> pd.DataFrame:
    rank_limit = max(params.top_n, params.hold_rank_buffer or params.top_n)
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
                str(params.predictions_path),
                params.start,
                params.end,
                str(params.predictions_path),
                params.start,
                params.end,
                params.rebalance_every_n_bars,
                rank_limit,
            ],
        ).fetchdf()
    finally:
        connection.close()


def _build_buffered_target_weights(
    ranked_signals: pd.DataFrame,
    params: TreeScoreBacktestParams,
) -> pd.DataFrame:
    if ranked_signals.empty:
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
    return pd.concat(rows, ignore_index=True)


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
    parser.add_argument("--rebalance-every-n-bars", type=int, default=1)
    parser.add_argument("--hold-rank-buffer", type=int)
    parser.add_argument("--min-trade-weight", type=float, default=0.0)
    parser.add_argument("--exclude-st", action="store_true")
    parser.add_argument("--limit-up-bps", type=float)
    parser.add_argument("--limit-down-bps", type=float)
    parser.add_argument("--max-bar-turnover-participation", type=float)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.rebalance_every_n_bars <= 0:
        raise ValueError("--rebalance-every-n-bars must be positive")
    if args.hold_rank_buffer is not None and args.hold_rank_buffer < args.top_n:
        raise ValueError("--hold-rank-buffer must be greater than or equal to --top-n")
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
        rebalance_every_n_bars=args.rebalance_every_n_bars,
        hold_rank_buffer=args.hold_rank_buffer,
        min_trade_weight=args.min_trade_weight,
        exclude_st=args.exclude_st,
        limit_up_bps=args.limit_up_bps,
        limit_down_bps=args.limit_down_bps,
        max_bar_turnover_participation=args.max_bar_turnover_participation,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
