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
    sell_stamp_tax_bps: float = 0.0
    min_commission: float = 0.0
    min_trade_weight: float = 0.0
    exclude_st: bool = False
    limit_up_bps: float | None = None
    limit_down_bps: float | None = None
    max_bar_turnover_participation: float | None = None


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
    trade_metric_totals = _empty_trade_metric_totals()
    total_bars = 0
    total_signals = 0
    execution_constraint_counts = _empty_execution_constraint_counts()
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
        _merge_execution_constraint_counts(
            execution_constraint_counts,
            _execution_constraint_counts(executions),
        )
        period_trades, period_equity, _, state = _simulate(
            executions,
            params,
            state=state,
        )
        if not period_trades.empty:
            trade_count += len(period_trades)
            _merge_trade_metric_totals(
                trade_metric_totals,
                _trade_metric_totals(period_trades),
            )
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
    metrics.update(_trade_metrics_from_totals(trade_metric_totals, equity_values))
    return {
        "metrics": metrics,
        "trades": all_trades,
        "equity_curve": all_equity,
        "final_positions": final_positions,
        "signal_count": total_signals,
        "bar_count": total_bars,
        "instrument_count": len(instruments),
        "execution_constraint_counts": execution_constraint_counts,
        "artifacts_written": params.output_dir is not None,
    }


def run_backtest(bars: pd.DataFrame, params: BacktestParams) -> dict[str, object]:
    if bars.empty:
        raise ValueError("no bars loaded for requested period/universe")
    signals = _build_reversal_signals(bars, params)
    executions = _build_next_bar_executions(bars, signals)
    if executions.empty:
        raise ValueError("no executable signals after lookback and next-bar shift")
    execution_constraint_counts = _execution_constraint_counts(executions)
    trades, equity_curve, final_positions, _ = _simulate(executions, params)
    equity_values = [params.initial_cash] + equity_curve["equity"].astype(float).tolist()
    metrics = {
        "total_return": total_return(params.initial_cash, equity_values[-1]),
        "max_drawdown": max_drawdown(equity_values),
        "trade_count": float(len(trades)),
        "final_equity": float(equity_values[-1]),
    }
    metrics.update(_trade_metrics(trades, equity_values))
    return {
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_curve,
        "final_positions": final_positions,
        "signal_count": len(signals),
        "bar_count": len(bars),
        "instrument_count": bars["instrument_id"].nunique(),
        "execution_constraint_counts": execution_constraint_counts,
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
                turnover,
                raw_name
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
    frame = frame.sort_values(["instrument_id", "bar_end_time"]).reset_index(drop=True)
    return _add_execution_constraint_columns(frame, params)


def _add_execution_constraint_columns(
    frame: pd.DataFrame,
    params: BacktestParams,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["trade_date"] = frame["bar_end_time"].astype(str).str.slice(0, 10)
    daily_close = (
        frame.groupby(["instrument_id", "trade_date"], sort=False)["close_price"]
        .last()
        .rename("daily_close")
        .reset_index()
    )
    daily_close["previous_close"] = daily_close.groupby("instrument_id", sort=False)[
        "daily_close"
    ].shift(1)
    frame = frame.merge(
        daily_close.loc[:, ["instrument_id", "trade_date", "previous_close"]],
        on=["instrument_id", "trade_date"],
        how="left",
    )
    if params.exclude_st and "raw_name" in frame.columns:
        raw_name = frame["raw_name"].fillna("").astype(str).str.upper()
        frame["is_st"] = raw_name.str.contains("ST", regex=False)
    else:
        frame["is_st"] = False
    frame["tradable_bar"] = (
        frame["open_price"].astype(float).gt(0)
        & frame["close_price"].astype(float).gt(0)
        & frame["volume"].astype(float).gt(0)
        & frame["turnover"].astype(float).gt(0)
        & ~frame["is_st"].astype(bool)
    )
    previous_close = frame["previous_close"].astype(float)
    has_previous = previous_close.gt(0)
    if params.limit_up_bps is None:
        frame["limit_up_open"] = False
    else:
        limit_up_price = previous_close * (1.0 + params.limit_up_bps / 10_000.0)
        frame["limit_up_open"] = has_previous & (
            frame["open_price"].astype(float) >= limit_up_price
        )
    if params.limit_down_bps is None:
        frame["limit_down_open"] = False
    else:
        limit_down_price = previous_close * (1.0 - params.limit_down_bps / 10_000.0)
        frame["limit_down_open"] = has_previous & (
            frame["open_price"].astype(float) <= limit_down_price
        )
    return frame


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
        tradable_by_instrument = (
            {
                str(row.instrument_id): bool(row.tradable_bar)
                for row in group.itertuples(index=False)
            }
            if "tradable_bar" in group.columns
            else {}
        )
        limit_up_by_instrument = (
            {
                str(row.instrument_id): bool(row.limit_up_open)
                for row in group.itertuples(index=False)
            }
            if "limit_up_open" in group.columns
            else {}
        )
        limit_down_by_instrument = (
            {
                str(row.instrument_id): bool(row.limit_down_open)
                for row in group.itertuples(index=False)
            }
            if "limit_down_open" in group.columns
            else {}
        )
        turnover_by_instrument = (
            {
                str(row.instrument_id): float(row.turnover)
                for row in group.itertuples(index=False)
            }
            if "turnover" in group.columns
            else {}
        )
        state.last_prices.update(close_by_instrument)
        equity = state.cash + _positions_value(state.lots, state.last_prices)
        target_rows = group.loc[group["target_weight"].notna()]
        if not target_rows.empty:
            targets = {
                str(row.instrument_id): float(row.target_weight)
                for row in target_rows.itertuples(index=False)
            }
            instruments = sorted(set(state.lots) | set(targets))
            for instrument_id in instruments:
                if instrument_id not in price_by_instrument:
                    continue
                if not tradable_by_instrument.get(instrument_id, True):
                    continue
                price = price_by_instrument[instrument_id]
                target_value = equity * targets.get(instrument_id, 0.0)
                current_value = _instrument_shares(state.lots, instrument_id) * price
                delta_value = target_value - current_value
                if (
                    params.min_trade_weight > 0
                    and equity > 0
                    and abs(delta_value) / equity < params.min_trade_weight
                ):
                    continue
                if delta_value > price * params.lot_size:
                    if limit_up_by_instrument.get(instrument_id, False):
                        continue
                    shares = int(delta_value / price / params.lot_size) * params.lot_size
                    shares = _cap_trade_shares_by_turnover(
                        shares,
                        price=price,
                        turnover=turnover_by_instrument.get(instrument_id),
                        params=params,
                    )
                    if shares <= 0:
                        continue
                    cost_price = price * (1.0 + params.slippage_bps / 10_000.0)
                    notional = shares * cost_price
                    commission = _commission(notional, params)
                    slippage_cost = shares * (cost_price - price)
                    if notional + commission <= state.cash:
                        state.cash -= notional + commission
                        state.lots.setdefault(instrument_id, []).append(
                            {"shares": shares, "date": trade_date, "sellable": False}
                        )
                        trades.append(
                            _trade_row(
                                exec_time,
                                instrument_id,
                                "buy",
                                shares,
                                cost_price,
                                commission,
                                stamp_tax=0.0,
                                slippage_cost=slippage_cost,
                                reference_price=price,
                            )
                        )
                elif delta_value < 0:
                    if limit_down_by_instrument.get(instrument_id, False):
                        continue
                    shares_to_sell = int(
                        min(
                            -delta_value / price,
                            _sellable_shares(state.lots, instrument_id),
                        )
                    )
                    shares_to_sell = _cap_trade_shares_by_turnover(
                        shares_to_sell,
                        price=price,
                        turnover=turnover_by_instrument.get(instrument_id),
                        params=params,
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
                    commission = _commission(notional, params)
                    stamp_tax = notional * params.sell_stamp_tax_bps / 10_000.0
                    slippage_cost = sold * (price - sell_price)
                    state.cash += notional - commission - stamp_tax
                    trades.append(
                        _trade_row(
                            exec_time,
                            instrument_id,
                            "sell",
                            sold,
                            sell_price,
                            commission,
                            stamp_tax=stamp_tax,
                            slippage_cost=slippage_cost,
                            reference_price=price,
                        )
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
    *,
    stamp_tax: float,
    slippage_cost: float,
    reference_price: float,
) -> dict[str, object]:
    notional = shares * price
    reference_notional = shares * reference_price
    return {
        "timestamp": timestamp,
        "instrument_id": instrument_id,
        "side": side,
        "shares": shares,
        "price": price,
        "reference_price": reference_price,
        "commission": commission,
        "stamp_tax": stamp_tax,
        "slippage_cost": slippage_cost,
        "total_cost": commission + stamp_tax + slippage_cost,
        "notional": notional,
        "reference_notional": reference_notional,
    }


def _commission(notional: float, params: BacktestParams) -> float:
    commission = notional * params.commission_bps / 10_000.0
    if commission > 0 and params.min_commission > 0:
        return max(commission, params.min_commission)
    return commission


def _cap_trade_shares_by_turnover(
    shares: int,
    *,
    price: float,
    turnover: float | None,
    params: BacktestParams,
) -> int:
    if params.max_bar_turnover_participation is None:
        return shares
    if turnover is None or turnover <= 0 or price <= 0:
        return 0
    max_notional = turnover * params.max_bar_turnover_participation
    max_shares = int(max_notional / price / params.lot_size) * params.lot_size
    return max(0, min(shares, max_shares))


def _execution_constraint_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = _empty_execution_constraint_counts()
    counts["execution_row_count"] = int(len(frame))
    if frame.empty:
        return counts
    tradable = (
        frame["tradable_bar"].fillna(False).astype(bool)
        if "tradable_bar" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    limit_up = (
        frame["limit_up_open"].fillna(False).astype(bool)
        if "limit_up_open" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    limit_down = (
        frame["limit_down_open"].fillna(False).astype(bool)
        if "limit_down_open" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    target = (
        pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0).gt(0)
        if "target_weight" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    counts["non_tradable_row_count"] = int((~tradable).sum())
    counts["limit_up_open_row_count"] = int(limit_up.sum())
    counts["limit_down_open_row_count"] = int(limit_down.sum())
    counts["positive_target_row_count"] = int(target.sum())
    counts["positive_target_non_tradable_row_count"] = int((target & ~tradable).sum())
    counts["positive_target_limit_up_open_row_count"] = int((target & limit_up).sum())
    counts["positive_target_limit_down_open_row_count"] = int((target & limit_down).sum())
    return counts


def _empty_execution_constraint_counts() -> dict[str, int]:
    return {
        "execution_row_count": 0,
        "non_tradable_row_count": 0,
        "limit_up_open_row_count": 0,
        "limit_down_open_row_count": 0,
        "positive_target_row_count": 0,
        "positive_target_non_tradable_row_count": 0,
        "positive_target_limit_up_open_row_count": 0,
        "positive_target_limit_down_open_row_count": 0,
    }


def _merge_execution_constraint_counts(
    totals: dict[str, int],
    other: dict[str, int],
) -> None:
    for key, value in other.items():
        totals[key] += value


def _trade_metrics(trades: pd.DataFrame, equity_values: list[float]) -> dict[str, float]:
    if trades.empty:
        return _trade_metrics_from_totals(_empty_trade_metric_totals(), equity_values)
    return _trade_metrics_from_totals(_trade_metric_totals(trades), equity_values)


def _trade_metric_totals(trades: pd.DataFrame) -> dict[str, float]:
    buy_notional = float(trades.loc[trades["side"] == "buy", "notional"].sum())
    sell_notional = float(trades.loc[trades["side"] == "sell", "notional"].sum())
    return {
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "gross_traded_notional": float(trades["notional"].sum()),
        "total_commission": float(trades["commission"].sum()),
        "total_stamp_tax": float(trades["stamp_tax"].sum()),
        "total_slippage_cost": float(trades["slippage_cost"].sum()),
    }


def _empty_trade_metric_totals() -> dict[str, float]:
    return {
        "buy_notional": 0.0,
        "sell_notional": 0.0,
        "gross_traded_notional": 0.0,
        "total_commission": 0.0,
        "total_stamp_tax": 0.0,
        "total_slippage_cost": 0.0,
    }


def _merge_trade_metric_totals(
    totals: dict[str, float],
    other: dict[str, float],
) -> None:
    for key, value in other.items():
        totals[key] += value


def _trade_metrics_from_totals(
    totals: dict[str, float],
    equity_values: list[float],
) -> dict[str, float]:
    average_equity = sum(equity_values) / len(equity_values) if equity_values else 0.0
    total_transaction_cost = (
        totals["total_commission"]
        + totals["total_stamp_tax"]
        + totals["total_slippage_cost"]
    )
    return {
        **totals,
        "gross_turnover": totals["gross_traded_notional"] / average_equity
        if average_equity
        else 0.0,
        "total_transaction_cost": total_transaction_cost,
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
            "sell_stamp_tax_bps": params.sell_stamp_tax_bps,
            "min_commission": params.min_commission,
            "min_trade_weight": params.min_trade_weight,
            "exclude_st": params.exclude_st,
            "limit_up_bps": params.limit_up_bps,
            "limit_down_bps": params.limit_down_bps,
            "max_bar_turnover_participation": params.max_bar_turnover_participation,
            "lot_size": params.lot_size,
            "max_symbols": params.max_symbols,
        },
        "bar_count": result["bar_count"],
        "instrument_count": result["instrument_count"],
        "signal_count": result["signal_count"],
        "metrics": result["metrics"],
    }
    if "execution_constraint_counts" in result:
        summary["execution_constraint_counts"] = result["execution_constraint_counts"]
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
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=0.0)
    parser.add_argument("--min-commission", type=float, default=0.0)
    parser.add_argument("--min-trade-weight", type=float, default=0.0)
    parser.add_argument("--exclude-st", action="store_true")
    parser.add_argument("--limit-up-bps", type=float)
    parser.add_argument("--limit-down-bps", type=float)
    parser.add_argument("--max-bar-turnover-participation", type=float)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
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
        sell_stamp_tax_bps=args.sell_stamp_tax_bps,
        min_commission=args.min_commission,
        min_trade_weight=args.min_trade_weight,
        exclude_st=args.exclude_st,
        limit_up_bps=args.limit_up_bps,
        limit_down_bps=args.limit_down_bps,
        max_bar_turnover_participation=args.max_bar_turnover_participation,
    )


if __name__ == "__main__":
    main()
