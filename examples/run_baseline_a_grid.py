"""Run a small parameter grid for Baseline A real-data backtests."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.metrics.performance import total_return
from quant_research.metrics.risk import max_drawdown

from run_baseline_a_real_backtest import (
    BacktestParams,
    SimulationState,
    _append_frame_csv,
    _build_next_bar_executions,
    _final_positions,
    _load_bars_from_files,
    _minute_bar_files,
    _simulate,
)


@dataclass(slots=True)
class GridRun:
    name: str
    params: BacktestParams
    state: SimulationState
    equity_values: list[float] = field(default_factory=list)
    trade_count: int = 0
    signal_count: int = 0


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = _build_grid_runs(args)
    result_by_name = run_grid_shared_data(
        runs,
        write_run_artifacts=args.write_run_artifacts,
    )
    rows: list[dict[str, object]] = []
    for run in runs:
        result = result_by_name[run.name]
        metrics = result["metrics"]
        row = {
            "name": run.name,
            "lookback_bars": run.params.lookback_bars,
            "top_n": run.params.top_n,
            "min_avg_turnover": run.params.min_avg_turnover,
            "liquidity_window_bars": run.params.liquidity_window_bars,
            "bar_count": result["bar_count"],
            "signal_count": result["signal_count"],
            "instrument_count": result["instrument_count"],
            **metrics,
        }
        rows.append(row)
        run_dir = (
            Path(run.params.output_dir)
            if run.params.output_dir is not None
            else output_root / run.name
        )
        _write_summary(
            run_dir,
            row,
        )
        pd.DataFrame(rows).to_csv(output_root / "grid_summary.csv", index=False)
    summary = pd.DataFrame(rows).sort_values("total_return", ascending=False)
    print(summary.to_string(index=False))


def run_grid_shared_data(
    runs: list[GridRun], *, write_run_artifacts: bool
) -> dict[str, dict[str, object]]:
    if not runs:
        raise ValueError("empty parameter grid")
    base_params = runs[0].params
    files = _minute_bar_files(base_params)
    if not files:
        raise FileNotFoundError("no 5-minute CN equity parquet files found")
    for run in runs:
        run.equity_values = [run.params.initial_cash]
        if run.params.output_dir is None:
            continue
        run.params.output_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("trades.csv", "equity_curve.csv"):
            path = run.params.output_dir / filename
            if path.exists():
                path.unlink()
    total_bars = 0
    instruments: set[str] = set()
    for file_path in files:
        bars = _load_bars_from_files(base_params, [file_path])
        if bars.empty:
            continue
        total_bars += len(bars)
        instruments.update(str(value) for value in bars["instrument_id"].unique())
        instrument_count = bars["instrument_id"].nunique()
        print(
            f"loaded {file_path.name}: bars={len(bars)} instruments={instrument_count}",
            flush=True,
        )
        _prepare_grid_features(bars, runs)
        for index, run in enumerate(runs, start=1):
            print(f"running {index}/{len(runs)} {run.name}", flush=True)
            signals = _build_reversal_signals_from_features(bars, run.params)
            run.signal_count += len(signals)
            executions = _build_next_bar_executions(
                bars,
                signals,
                tracked_instruments=set(run.state.lots),
            )
            del signals
            if executions.empty:
                del executions
                continue
            period_trades, period_equity, _, run.state = _simulate(
                executions,
                run.params,
                state=run.state,
            )
            if not period_trades.empty:
                run.trade_count += len(period_trades)
                if write_run_artifacts and run.params.output_dir is not None:
                    _append_frame_csv(
                        period_trades,
                        run.params.output_dir / "trades.csv",
                    )
            if not period_equity.empty:
                run.equity_values.extend(
                    period_equity["equity"].astype(float).tolist()
                )
                if write_run_artifacts and run.params.output_dir is not None:
                    _append_frame_csv(
                        period_equity,
                        run.params.output_dir / "equity_curve.csv",
                    )
            del executions, period_trades, period_equity
            gc.collect()
        del bars
        gc.collect()
    results: dict[str, dict[str, object]] = {}
    for run in runs:
        if len(run.equity_values) == 1:
            raise ValueError(f"{run.name}: no executable signals after next-bar shift")
        final_positions = _final_positions(run.state.lots)
        if write_run_artifacts and run.params.output_dir is not None:
            final_positions.to_csv(
                run.params.output_dir / "final_positions.csv",
                index=False,
            )
        metrics = {
            "total_return": total_return(
                run.params.initial_cash,
                run.equity_values[-1],
            ),
            "max_drawdown": max_drawdown(run.equity_values),
            "trade_count": float(run.trade_count),
            "final_equity": float(run.equity_values[-1]),
        }
        results[run.name] = {
            "metrics": metrics,
            "final_positions": final_positions,
            "signal_count": run.signal_count,
            "bar_count": total_bars,
            "instrument_count": len(instruments),
        }
    return results


def _prepare_grid_features(bars: pd.DataFrame, runs: list[GridRun]) -> None:
    bars["close_price"] = bars["close_price"].astype(float)
    grouped = bars.groupby("instrument_id", sort=False)
    for lookback_bars in sorted({run.params.lookback_bars for run in runs}):
        factor_col = _factor_column(lookback_bars)
        lookback_return = grouped["close_price"].pct_change(periods=lookback_bars)
        bars[factor_col] = -lookback_return
    liquidity_windows = {
        run.params.liquidity_window_bars
        for run in runs
        if run.params.min_avg_turnover is not None
    }
    if not liquidity_windows:
        return
    bars["turnover"] = bars["turnover"].astype(float)
    for window in sorted(liquidity_windows):
        avg_col = _avg_turnover_column(window)
        bars[avg_col] = grouped["turnover"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )


def _build_reversal_signals_from_features(
    bars: pd.DataFrame, params: BacktestParams
) -> pd.DataFrame:
    factor_col = _factor_column(params.lookback_bars)
    mask = bars[factor_col].notna()
    if params.min_avg_turnover is not None:
        avg_col = _avg_turnover_column(params.liquidity_window_bars)
        mask = mask & (bars[avg_col] >= params.min_avg_turnover)
    frame = bars.loc[mask]
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "signal_time",
                "bar_end_time",
                "instrument_id",
                "canonical_code",
                "target_weight",
            ]
        )
    selected = (
        frame.sort_values(
            ["bar_end_time", factor_col, "canonical_code"],
            ascending=[True, False, True],
        )
        .groupby("bar_end_time", sort=False)
        .head(params.top_n)
        .loc[:, ["bar_end_time", "instrument_id", "canonical_code"]]
        .copy()
    )
    selected["signal_time"] = selected["bar_end_time"]
    selected["target_weight"] = 1.0 / selected.groupby(
        "bar_end_time", sort=False
    )["instrument_id"].transform("size")
    return selected.loc[
        :,
        [
            "signal_time",
            "bar_end_time",
            "instrument_id",
            "canonical_code",
            "target_weight",
        ],
    ].reset_index(drop=True)


def _factor_column(lookback_bars: int) -> str:
    return f"factor_lb_{lookback_bars}"


def _avg_turnover_column(window: int) -> str:
    return f"avg_turnover_w_{window}"


def _build_grid_runs(args: argparse.Namespace) -> list[GridRun]:
    output_root = Path(args.output_root)
    runs: list[GridRun] = []
    for lookback_bars, top_n, min_avg_turnover in itertools.product(
        args.lookback_bars,
        args.top_n,
        args.min_avg_turnover,
    ):
        name = _run_name(
            lookback_bars=lookback_bars,
            top_n=top_n,
            min_avg_turnover=min_avg_turnover,
        )
        run_dir = output_root / name
        params = BacktestParams(
            catalog_path=Path(args.catalog_path),
            start=args.start,
            end=args.end,
            top_n=top_n,
            initial_cash=args.initial_cash,
            lookback_bars=lookback_bars,
            min_avg_turnover=min_avg_turnover,
            liquidity_window_bars=args.liquidity_window_bars,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            lot_size=args.lot_size,
            max_symbols=args.max_symbols,
            output_dir=run_dir if args.write_run_artifacts else None,
            data_access_mode="fast_parquet",
        )
        state = SimulationState(
            cash=float(args.initial_cash),
            lots={},
            previous_date=None,
            last_prices={},
        )
        runs.append(GridRun(name=name, params=params, state=state))
    return runs


def _write_summary(
    run_dir: Path,
    row: dict[str, object],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(row, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_name(
    *,
    lookback_bars: int,
    top_n: int,
    min_avg_turnover: float | None,
) -> str:
    turnover = "none" if min_avg_turnover is None else f"{int(min_avg_turnover)}"
    return f"lb{lookback_bars}_top{top_n}_turnover{turnover}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-path",
        default="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lookback-bars", type=int, nargs="+", required=True)
    parser.add_argument("--top-n", type=int, nargs="+", required=True)
    parser.add_argument(
        "--min-avg-turnover",
        nargs="+",
        default=["none"],
        help="Use 'none' for no filter, otherwise numeric turnover thresholds.",
    )
    parser.add_argument("--liquidity-window-bars", type=int, default=3)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--write-run-artifacts", action="store_true")
    args = parser.parse_args()
    args.min_avg_turnover = [
        None if value.lower() == "none" else float(value)
        for value in args.min_avg_turnover
    ]
    return args


if __name__ == "__main__":
    main()
