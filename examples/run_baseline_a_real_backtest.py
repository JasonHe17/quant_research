"""Run Baseline A on real local 5-minute main-board data."""

from __future__ import annotations

import argparse
import gc
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


@dataclass(frozen=True, slots=True)
class BacktestParams:
    catalog_path: Path
    start: str
    end: str
    top_n: int
    initial_cash: float
    lookback_bars: int
    min_avg_turnover: float | None
    liquidity_window_bars: int
    commission_bps: float
    slippage_bps: float
    lot_size: int
    max_symbols: int | None
    output_dir: Path | None


@dataclass(slots=True)
class SimulationState:
    cash: float
    lots: dict[str, list[dict[str, object]]]
    previous_date: str | None
    last_prices: dict[str, float]


def main() -> None:
    params = _parse_args()
    result = run_backtest_streaming(params)
    if params.output_dir is not None and not result.get("artifacts_written", False):
        _write_outputs(result, params)
    elif params.output_dir is not None:
        _write_summary(result, params)
    _print_result(result)


def run_backtest_streaming(params: BacktestParams) -> dict[str, object]:
    files = _minute_bar_files(params)
    if not files:
        raise FileNotFoundError("no 5-minute CN equity parquet files found")
    state = SimulationState(
        cash=float(params.initial_cash),
        lots={},
        previous_date=None,
        last_prices={},
    )
    trades: list[pd.DataFrame] = []
    equity_curves: list[pd.DataFrame] = []
    equity_values = [params.initial_cash]
    trade_count = 0
    total_bars = 0
    total_signals = 0
    instruments: set[str] = set()
    if params.output_dir is not None:
        params.output_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("trades.csv", "equity_curve.csv"):
            path = params.output_dir / filename
            if path.exists():
                path.unlink()
    for file_path in files:
        bars = _load_bars_from_files(params, [file_path])
        if bars.empty:
            continue
        total_bars += len(bars)
        instruments.update(str(value) for value in bars["instrument_id"].unique())
        signals = _build_reversal_signals(bars, params)
        total_signals += len(signals)
        executions = _build_next_bar_executions(bars, signals)
        if executions.empty:
            continue
        period_trades, period_equity, _, state = _simulate(
            executions,
            params,
            state=state,
        )
        if not period_trades.empty:
            trade_count += len(period_trades)
            if params.output_dir is not None:
                _append_frame_csv(period_trades, params.output_dir / "trades.csv")
            else:
                trades.append(period_trades)
        if not period_equity.empty:
            equity_values.extend(period_equity["equity"].astype(float).tolist())
            if params.output_dir is not None:
                _append_frame_csv(period_equity, params.output_dir / "equity_curve.csv")
            else:
                equity_curves.append(period_equity)
        del bars, signals, executions, period_trades, period_equity
        gc.collect()
    if len(equity_values) == 1:
        raise ValueError("no executable signals after lookback and next-bar shift")
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    all_equity = pd.concat(equity_curves, ignore_index=True) if equity_curves else pd.DataFrame()
    final_positions = _final_positions(state.lots)
    if params.output_dir is not None:
        final_positions.to_csv(params.output_dir / "final_positions.csv", index=False)
    metrics = {
        "total_return": total_return(params.initial_cash, equity_values[-1]),
        "max_drawdown": max_drawdown(equity_values),
        "trade_count": float(trade_count if params.output_dir is not None else len(all_trades)),
        "final_equity": float(equity_values[-1]),
    }
    return {
        "metrics": metrics,
        "trades": all_trades,
        "equity_curve": all_equity,
        "final_positions": final_positions,
        "signal_count": total_signals,
        "bar_count": total_bars,
        "instrument_count": len(instruments),
        "artifacts_written": params.output_dir is not None,
    }


def run_backtest(bars: pd.DataFrame, params: BacktestParams) -> dict[str, object]:
    if bars.empty:
        raise ValueError("no bars loaded for requested period/universe")
    signals = _build_reversal_signals(bars, params)
    executions = _build_next_bar_executions(bars, signals)
    if executions.empty:
        raise ValueError("no executable signals after lookback and next-bar shift")
    trades, equity_curve, final_positions, _ = _simulate(executions, params)
    equity_values = [params.initial_cash] + equity_curve["equity"].astype(float).tolist()
    metrics = {
        "total_return": total_return(params.initial_cash, equity_values[-1]),
        "max_drawdown": max_drawdown(equity_values),
        "trade_count": float(len(trades)),
        "final_equity": float(equity_values[-1]),
    }
    return {
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve,
        "final_positions": final_positions,
        "signal_count": len(signals),
        "bar_count": len(bars),
        "instrument_count": bars["instrument_id"].nunique(),
    }


def _load_bars(params: BacktestParams) -> pd.DataFrame:
    files = _minute_bar_files(params)
    if not files:
        raise FileNotFoundError("no 5-minute CN equity parquet files found")
    return _load_bars_from_files(params, files)


def _load_bars_from_files(params: BacktestParams, files: list[Path]) -> pd.DataFrame:
    scan_target = ", ".join(f"'{path.as_posix()}'" for path in files)
    symbol_limit_clause = ""
    if params.max_symbols is not None:
        symbol_limit_clause = f"""
              AND canonical_code IN (
                  SELECT canonical_code
                  FROM (
                      SELECT DISTINCT canonical_code
                      FROM read_parquet([{scan_target}])
                      WHERE market = 'CN'
                        AND asset_type = 'equity'
                        AND frequency = '5m'
                        AND (
                            canonical_code LIKE '600%.SH'
                            OR canonical_code LIKE '601%.SH'
                            OR canonical_code LIKE '603%.SH'
                            OR canonical_code LIKE '605%.SH'
                            OR canonical_code LIKE '000%.SZ'
                            OR canonical_code LIKE '001%.SZ'
                            OR canonical_code LIKE '002%.SZ'
                            OR canonical_code LIKE '003%.SZ'
                        )
                      ORDER BY canonical_code
                      LIMIT {params.max_symbols}
                  )
              )
        """
    connection = duckdb.connect()
    try:
        query = f"""
            SELECT
                instrument_id,
                canonical_code,
                bar_end_time,
                open_price,
                close_price,
                volume,
                turnover
            FROM read_parquet([{scan_target}])
            WHERE market = 'CN'
              AND asset_type = 'equity'
              AND frequency = '5m'
              AND bar_end_time >= ?
              AND bar_end_time <= ?
              AND (
                  canonical_code LIKE '600%.SH'
                  OR canonical_code LIKE '601%.SH'
                  OR canonical_code LIKE '603%.SH'
                  OR canonical_code LIKE '605%.SH'
                  OR canonical_code LIKE '000%.SZ'
                  OR canonical_code LIKE '001%.SZ'
                  OR canonical_code LIKE '002%.SZ'
                  OR canonical_code LIKE '003%.SZ'
              )
              {symbol_limit_clause}
            ORDER BY bar_end_time, instrument_id
        """
        frame = connection.execute(query, [params.start, params.end]).fetchdf()
    finally:
        connection.close()
    if frame.empty:
        return frame
    return frame.sort_values(["instrument_id", "bar_end_time"]).reset_index(drop=True)


def _minute_bar_files(params: BacktestParams) -> list[Path]:
    canonical_root = params.catalog_path.parent.parent
    data_dir = canonical_root / "v1" / "market" / "records=minute_bar"
    start_year = int(params.start[:4])
    end_year = int(params.end[:4])
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        path = (
            data_dir
            / f"market_cn_equity_full__a股_分时数据_沪深__5分钟_按年汇总__{year}.parquet"
        )
        if path.exists():
            files.append(path)
    return files


def _build_reversal_signals(
    bars: pd.DataFrame, params: BacktestParams
) -> pd.DataFrame:
    frame = bars.copy()
    frame["close_price"] = frame["close_price"].astype(float)
    grouped = frame.groupby("instrument_id", sort=False)
    frame["lookback_return"] = grouped["close_price"].pct_change(
        periods=params.lookback_bars
    )
    frame["factor_value"] = -frame["lookback_return"]
    if params.min_avg_turnover is not None:
        frame["avg_turnover"] = grouped["turnover"].transform(
            lambda values: values.astype(float)
            .rolling(params.liquidity_window_bars, min_periods=params.liquidity_window_bars)
            .mean()
        )
        frame = frame.loc[frame["avg_turnover"] >= params.min_avg_turnover].copy()
    frame = frame.loc[frame["factor_value"].notna()].copy()
    selected: list[pd.DataFrame] = []
    for timestamp, group in frame.groupby("bar_end_time", sort=True):
        chosen = group.sort_values(
            ["factor_value", "canonical_code"],
            ascending=[False, True],
        ).head(params.top_n)
        if chosen.empty:
            continue
        output = chosen.loc[:, ["bar_end_time", "instrument_id", "canonical_code"]].copy()
        output["signal_time"] = timestamp
        output["target_weight"] = 1.0 / len(chosen)
        selected.append(output)
    if not selected:
        return pd.DataFrame(
            columns=[
                "signal_time",
                "bar_end_time",
                "instrument_id",
                "canonical_code",
                "target_weight",
            ]
        )
    return pd.concat(selected, ignore_index=True)


def _build_next_bar_executions(
    bars: pd.DataFrame, signals: pd.DataFrame
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
        :, ["bar_end_time", "instrument_id", "canonical_code", "open_price", "close_price"]
    ].rename(columns={"bar_end_time": "exec_time"})
    exec_times = shifted["exec_time"].drop_duplicates().tolist()
    prices = prices.loc[prices["exec_time"].isin(exec_times)].copy()
    targets = shifted.loc[
        :, ["exec_time", "instrument_id", "target_weight"]
    ].copy()
    merged = prices.merge(targets, on=["exec_time", "instrument_id"], how="left")
    merged["target_weight"] = merged["target_weight"].fillna(0.0)
    return merged


def _simulate(
    executions: pd.DataFrame,
    params: BacktestParams,
    *,
    state: SimulationState | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimulationState]:
    if state is None:
        state = SimulationState(
            cash=float(params.initial_cash),
            lots={},
            previous_date=None,
            last_prices={},
        )
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    for exec_time, group in executions.groupby("exec_time", sort=True):
        trade_date = str(exec_time)[:10]
        if state.previous_date is not None and trade_date != state.previous_date:
            for instrument_lots in state.lots.values():
                for lot in instrument_lots:
                    lot["sellable"] = True
        state.previous_date = trade_date
        price_by_instrument = {
            str(row.instrument_id): float(row.open_price)
            for row in group.itertuples(index=False)
        }
        close_by_instrument = {
            str(row.instrument_id): float(row.close_price)
            for row in group.itertuples(index=False)
        }
        state.last_prices.update(close_by_instrument)
        equity = state.cash + _positions_value(state.lots, state.last_prices)
        targets = {
            str(row.instrument_id): float(row.target_weight)
            for row in group.itertuples(index=False)
        }
        instruments = sorted(set(state.lots) | set(targets))
        for instrument_id in instruments:
            if instrument_id not in price_by_instrument:
                continue
            price = price_by_instrument[instrument_id]
            target_value = equity * targets.get(instrument_id, 0.0)
            current_value = _instrument_shares(state.lots, instrument_id) * price
            delta_value = target_value - current_value
            if delta_value > price * params.lot_size:
                shares = int(delta_value / price / params.lot_size) * params.lot_size
                cost_price = price * (1.0 + params.slippage_bps / 10_000.0)
                notional = shares * cost_price
                commission = notional * params.commission_bps / 10_000.0
                if notional + commission <= state.cash:
                    state.cash -= notional + commission
                    state.lots.setdefault(instrument_id, []).append(
                        {"shares": shares, "date": trade_date, "sellable": False}
                    )
                    trades.append(
                        _trade_row(exec_time, instrument_id, "buy", shares, cost_price, commission)
                    )
            elif delta_value < 0:
                shares_to_sell = int(
                    min(
                        -delta_value / price,
                        _sellable_shares(state.lots, instrument_id),
                    )
                )
                if shares_to_sell <= 0:
                    continue
                sell_price = price * (1.0 - params.slippage_bps / 10_000.0)
                sold = _remove_sellable_shares(
                    state.lots,
                    instrument_id,
                    shares_to_sell,
                )
                notional = sold * sell_price
                commission = notional * params.commission_bps / 10_000.0
                state.cash += notional - commission
                trades.append(
                    _trade_row(exec_time, instrument_id, "sell", sold, sell_price, commission)
                )
        equity_rows.append(
            {
                "timestamp": exec_time,
                "cash": state.cash,
                "positions_value": _positions_value(state.lots, state.last_prices),
                "equity": state.cash + _positions_value(state.lots, state.last_prices),
            }
        )
    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_rows),
        _final_positions(state.lots),
        state,
    )


def _positions_value(
    lots: dict[str, list[dict[str, object]]], prices: dict[str, float]
) -> float:
    return sum(_instrument_shares(lots, instrument_id) * prices.get(instrument_id, 0.0) for instrument_id in lots)


def _instrument_shares(lots: dict[str, list[dict[str, object]]], instrument_id: str) -> int:
    return sum(int(lot["shares"]) for lot in lots.get(instrument_id, []))


def _sellable_shares(lots: dict[str, list[dict[str, object]]], instrument_id: str) -> int:
    return sum(int(lot["shares"]) for lot in lots.get(instrument_id, []) if bool(lot["sellable"]))


def _remove_sellable_shares(
    lots: dict[str, list[dict[str, object]]], instrument_id: str, shares: int
) -> int:
    remaining = shares
    sold = 0
    retained: list[dict[str, object]] = []
    for lot in lots.get(instrument_id, []):
        lot_shares = int(lot["shares"])
        if not bool(lot["sellable"]) or remaining <= 0:
            retained.append(lot)
            continue
        sell = min(lot_shares, remaining)
        sold += sell
        remaining -= sell
        if sell < lot_shares:
            retained.append({**lot, "shares": lot_shares - sell})
    lots[instrument_id] = retained
    return sold


def _trade_row(
    timestamp: object,
    instrument_id: str,
    side: str,
    shares: int,
    price: float,
    commission: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "instrument_id": instrument_id,
        "side": side,
        "shares": shares,
        "price": price,
        "commission": commission,
        "notional": shares * price,
    }


def _final_positions(lots: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument_id": instrument_id,
                "shares": _instrument_shares(lots, instrument_id),
                "sellable_shares": _sellable_shares(lots, instrument_id),
            }
            for instrument_id in sorted(lots)
            if _instrument_shares(lots, instrument_id) > 0
        ]
    )


def _print_result(result: dict[str, object]) -> None:
    print("Baseline A real-data backtest")
    print(f"bars={result['bar_count']}")
    print(f"instruments={result['instrument_count']}")
    print(f"signals={result['signal_count']}")
    print("metrics:")
    for name, value in result["metrics"].items():
        print(f"  {name}: {value}")


def _write_outputs(result: dict[str, object], params: BacktestParams) -> None:
    assert params.output_dir is not None
    params.output_dir.mkdir(parents=True, exist_ok=True)
    trades = result["trades"]
    equity_curve = result["equity_curve"]
    final_positions = result["final_positions"]
    if isinstance(trades, pd.DataFrame):
        trades.to_csv(params.output_dir / "trades.csv", index=False)
    if isinstance(equity_curve, pd.DataFrame):
        equity_curve.to_csv(params.output_dir / "equity_curve.csv", index=False)
    if isinstance(final_positions, pd.DataFrame):
        final_positions.to_csv(params.output_dir / "final_positions.csv", index=False)
    _write_summary(result, params)


def _write_summary(result: dict[str, object], params: BacktestParams) -> None:
    assert params.output_dir is not None
    summary = {
        "params": {
            "start": params.start,
            "end": params.end,
            "top_n": params.top_n,
            "initial_cash": params.initial_cash,
            "lookback_bars": params.lookback_bars,
            "min_avg_turnover": params.min_avg_turnover,
            "liquidity_window_bars": params.liquidity_window_bars,
            "commission_bps": params.commission_bps,
            "slippage_bps": params.slippage_bps,
            "lot_size": params.lot_size,
            "max_symbols": params.max_symbols,
        },
        "bar_count": result["bar_count"],
        "instrument_count": result["instrument_count"],
        "signal_count": result["signal_count"],
        "metrics": result["metrics"],
    }
    (params.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_frame_csv(frame: pd.DataFrame, path: Path) -> None:
    header = not path.exists()
    frame.to_csv(path, index=False, mode="a", header=header)


def _parse_args() -> BacktestParams:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-path",
        default="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-n", type=int, required=True)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--lookback-bars", type=int, default=1)
    parser.add_argument("--min-avg-turnover", type=float)
    parser.add_argument("--liquidity-window-bars", type=int, default=1)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    return BacktestParams(
        catalog_path=Path(args.catalog_path),
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        initial_cash=args.initial_cash,
        lookback_bars=args.lookback_bars,
        min_avg_turnover=args.min_avg_turnover,
        liquidity_window_bars=args.liquidity_window_bars,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        lot_size=args.lot_size,
        max_symbols=args.max_symbols,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
