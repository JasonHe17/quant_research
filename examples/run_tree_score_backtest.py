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
    _build_next_bar_executions,
    _final_positions,
    _load_bars,
    _simulate,
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
    lot_size: int
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
    )
    params.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("trades.csv", "equity_curve.csv"):
        path = params.output_dir / filename
        if path.exists():
            path.unlink()

    signals = _load_top_score_signals(params)
    if signals.empty:
        raise ValueError("no score signals loaded for requested period")
    bars = _load_bars(backtest_params)
    if bars.empty:
        raise ValueError("no bars loaded for requested period")
    executions = _build_next_bar_executions(bars, signals)
    if executions.empty:
        raise ValueError("no executable tree score signals after next-bar shift")
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
            "lot_size": params.lot_size,
        },
        "bar_count": int(len(bars)),
        "signal_count": int(len(signals)),
        "execution_row_count": int(len(executions)),
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


def _load_top_score_signals(params: TreeScoreBacktestParams) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        query = """
            WITH ranked AS (
                SELECT
                    timestamp AS signal_time,
                    instrument_id,
                    score,
                    row_number() OVER (
                        PARTITION BY timestamp
                        ORDER BY score DESC, instrument_id ASC
                    ) AS rank
                FROM read_parquet(?)
                WHERE timestamp >= ?
                  AND timestamp <= ?
                  AND score IS NOT NULL
            )
            SELECT
                signal_time,
                signal_time AS bar_end_time,
                instrument_id,
                score,
                rank,
                1.0 / count(*) OVER (PARTITION BY signal_time) AS target_weight
            FROM ranked
            WHERE rank <= ?
            ORDER BY signal_time, rank
        """
        return connection.execute(
            query,
            [str(params.predictions_path), params.start, params.end, params.top_n],
        ).fetchdf()
    finally:
        connection.close()


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
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    return TreeScoreBacktestParams(
        predictions_path=Path(args.predictions_path),
        catalog_path=Path(args.catalog_path),
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        initial_cash=args.initial_cash,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        lot_size=args.lot_size,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
