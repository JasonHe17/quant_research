"""Run Baseline A on real local 5-minute main-board data."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime, timedelta
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
from quant_research.backtest import (
    CnEquityExecutionConstraintsConfig,
    TargetWeightExecutionConfig,
    TargetWeightSimulationState,
    add_cn_equity_execution_columns,
    empty_execution_constraint_counts as framework_empty_execution_constraint_counts,
    execution_constraint_counts as framework_execution_constraint_counts,
    final_positions as framework_final_positions,
    merge_execution_constraint_counts as framework_merge_execution_constraint_counts,
    simulate_target_weight_executions,
)
from quant_research.data import DataPortal
from quant_research.universe import is_cn_main_board_symbol


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
    allow_same_bar_capacity: bool = False
    data_access_mode: str = "data_portal"
    streaming_chunk: str = "year"
    streaming_chunk_padding_days: int = 10


@dataclass(frozen=True, slots=True)
class StreamingWorkUnit:
    files: list[Path]
    load_start: str
    load_end: str
    signal_start: str
    signal_end: str


SimulationState = TargetWeightSimulationState


def main() -> None:
    params = _parse_args()
    result = run_backtest_streaming(params)
    if params.output_dir is not None and not result.get("artifacts_written", False):
        _write_outputs(result, params)
    elif params.output_dir is not None:
        _write_summary(result, params)
    _print_result(result)


def run_backtest_streaming(params: BacktestParams) -> dict[str, object]:
    if params.data_access_mode == "data_portal":
        bars = _load_bars(params)
        return run_backtest(bars, params)
    work_units = _streaming_work_units(params)
    if not work_units:
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
    for work_unit in work_units:
        bars = _load_bars_from_files(
            params,
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
        signals = _build_reversal_signals(bars, params)
        signals = _filter_signals_to_work_unit(signals, work_unit)
        total_signals += len(signals)
        executions = _build_next_bar_executions(
            bars,
            signals,
            tracked_instruments=set(state.lots),
        )
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
    if params.data_access_mode == "data_portal":
        return _load_bars_via_data_portal(params)
    files = _minute_bar_files(params)
    if not files:
        raise FileNotFoundError("no 5-minute CN equity parquet files found")
    return _load_bars_from_files(params, files)


def _load_bars_via_data_portal(params: BacktestParams) -> pd.DataFrame:
    canonical_root = params.catalog_path.parent.parent
    data = DataPortal(
        canonical_root=canonical_root,
        catalog_path=params.catalog_path,
    )
    symbols = _main_board_symbols_from_data_portal(data, params)
    if not symbols:
        raise ValueError("no CN main-board equity symbols resolved from DataPortal")
    frame = data.get_bars(
        symbols,
        start=params.start,
        end=params.end,
        frequency="5m",
        adjustment="raw",
        market="CN",
        asset_type="equity",
        cache=False,
    )
    if frame.empty:
        return frame
    if "canonical_code" not in frame.columns and "query_symbol" in frame.columns:
        frame["canonical_code"] = frame["query_symbol"]
    if "raw_name" not in frame.columns:
        frame["raw_name"] = None
    columns = [
        "instrument_id",
        "canonical_code",
        "bar_end_time",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "turnover",
        "raw_name",
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"DataPortal bars missing required columns: {missing}")
    frame = frame.loc[:, columns].sort_values(
        ["instrument_id", "bar_end_time"]
    ).reset_index(drop=True)
    return _add_execution_constraint_columns(frame, params)


def _main_board_symbols_from_data_portal(
    data: DataPortal,
    params: BacktestParams,
) -> list[str]:
    instruments = data.list_instruments(
        market="CN",
        asset_type="equity",
        as_of=params.end[:10],
        cache=False,
    )
    if instruments.empty:
        return []
    code_column = "canonical_code" if "canonical_code" in instruments.columns else "symbol"
    if code_column not in instruments.columns:
        raise ValueError("DataPortal instruments missing canonical_code/symbol column")
    mask = instruments.apply(
        lambda row: is_cn_main_board_symbol(
            str(row[code_column]),
            market=str(row["market"]) if "market" in instruments.columns else "CN",
            asset_type=str(row["asset_type"])
            if "asset_type" in instruments.columns
            else "equity",
        ),
        axis=1,
    )
    symbols = sorted(instruments.loc[mask, code_column].astype(str).unique().tolist())
    if params.max_symbols is not None:
        symbols = symbols[: params.max_symbols]
    return symbols


def _load_bars_from_files(
    params: BacktestParams,
    files: list[Path],
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    scan_target = ", ".join(f"'{path.as_posix()}'" for path in files)
    query_start = start or params.start
    query_end = end or params.end
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
                high_price,
                low_price,
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
        frame = connection.execute(query, [query_start, query_end]).fetchdf()
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
    return add_cn_equity_execution_columns(
        frame,
        CnEquityExecutionConstraintsConfig(
            exclude_st=params.exclude_st,
            limit_up_bps=params.limit_up_bps,
            limit_down_bps=params.limit_down_bps,
        ),
    )


def _minute_bar_files(params: BacktestParams) -> list[Path]:
    return _minute_bar_files_for_range(params, start=params.start, end=params.end)


def _minute_bar_files_for_range(
    params: BacktestParams,
    *,
    start: str,
    end: str,
) -> list[Path]:
    canonical_root = params.catalog_path.parent.parent
    data_dir = canonical_root / "v1" / "market" / "records=minute_bar"
    start_year = int(start[:4])
    end_year = int(end[:4])
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        path = (
            data_dir
            / f"market_cn_equity_full__a股_分时数据_沪深__5分钟_按年汇总__{year}.parquet"
        )
        if path.exists():
            files.append(path)
    return files


def _streaming_work_units(params: BacktestParams) -> list[StreamingWorkUnit]:
    if params.streaming_chunk == "year":
        units: list[StreamingWorkUnit] = []
        for year in range(int(params.start[:4]), int(params.end[:4]) + 1):
            signal_start = max(params.start, f"{year}-01-01T00:00:00+08:00")
            signal_end = min(params.end, f"{year}-12-31T23:59:59+08:00")
            files = _minute_bar_files_for_range(
                params,
                start=signal_start,
                end=signal_end,
            )
            if files:
                units.append(
                    StreamingWorkUnit(
                        files=files,
                        load_start=signal_start,
                        load_end=signal_end,
                        signal_start=signal_start,
                        signal_end=signal_end,
                    )
                )
        return units
    if params.streaming_chunk != "month":
        raise ValueError(f"unsupported streaming chunk: {params.streaming_chunk}")

    start_dt = datetime.fromisoformat(params.start)
    end_dt = datetime.fromisoformat(params.end)
    month_start = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    units = []
    while month_start <= end_dt:
        next_month = _add_month(month_start)
        month_end = next_month - timedelta(seconds=1)
        signal_start_dt = max(start_dt, month_start)
        signal_end_dt = min(end_dt, month_end)
        load_start_dt = signal_start_dt - timedelta(
            days=max(0, params.streaming_chunk_padding_days)
        )
        load_end_dt = min(
            end_dt,
            signal_end_dt + timedelta(days=max(1, params.streaming_chunk_padding_days)),
        )
        load_start = _format_datetime(load_start_dt)
        load_end = _format_datetime(load_end_dt)
        files = _minute_bar_files_for_range(params, start=load_start, end=load_end)
        if files:
            units.append(
                StreamingWorkUnit(
                    files=files,
                    load_start=load_start,
                    load_end=load_end,
                    signal_start=_format_datetime(signal_start_dt),
                    signal_end=_format_datetime(signal_end_dt),
                )
            )
        month_start = next_month
    return units


def _filter_signals_to_work_unit(
    signals: pd.DataFrame,
    work_unit: StreamingWorkUnit,
) -> pd.DataFrame:
    if signals.empty:
        return signals
    return signals.loc[
        (signals["signal_time"] >= work_unit.signal_start)
        & (signals["signal_time"] <= work_unit.signal_end)
    ].copy()


def _add_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1)
    return value.replace(month=value.month + 1)


def _format_datetime(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


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
            ["bar_end_time", "factor_value", "canonical_code"],
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


def _build_next_bar_executions(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    tracked_instruments: set[str] | None = None,
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
    relevant_instruments = {str(value) for value in tracked_instruments or set()}
    if not shifted.empty:
        relevant_instruments.update(shifted["instrument_id"].astype(str).unique())
    if not relevant_instruments:
        return pd.DataFrame(columns=[*prices.columns, "target_weight"])
    prices = prices.loc[
        prices["instrument_id"].astype(str).isin(relevant_instruments)
    ].copy()
    targets = shifted.loc[
        :, ["exec_time", "instrument_id", "target_weight"]
    ].copy()
    return prices.merge(targets, on=["exec_time", "instrument_id"], how="left")


def _simulate(
    executions: pd.DataFrame,
    params: BacktestParams,
    *,
    state: SimulationState | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimulationState]:
    return simulate_target_weight_executions(
        executions,
        _execution_config_from_params(params),
        state=state,
    )


def _execution_config_from_params(params: BacktestParams) -> TargetWeightExecutionConfig:
    return TargetWeightExecutionConfig(
        initial_cash=params.initial_cash,
        commission_bps=params.commission_bps,
        slippage_bps=params.slippage_bps,
        sell_stamp_tax_bps=params.sell_stamp_tax_bps,
        min_commission=params.min_commission,
        min_trade_weight=params.min_trade_weight,
        lot_size=params.lot_size,
        max_bar_turnover_participation=params.max_bar_turnover_participation,
        allow_same_bar_capacity=params.allow_same_bar_capacity,
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
    return framework_execution_constraint_counts(frame)


def _empty_execution_constraint_counts() -> dict[str, int]:
    return framework_empty_execution_constraint_counts()


def _merge_execution_constraint_counts(
    totals: dict[str, int],
    other: dict[str, int],
) -> None:
    framework_merge_execution_constraint_counts(totals, other)


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
    return framework_final_positions(lots)


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
            "allow_same_bar_capacity": params.allow_same_bar_capacity,
            "data_access_mode": params.data_access_mode,
            "streaming_chunk": params.streaming_chunk,
            "streaming_chunk_padding_days": params.streaming_chunk_padding_days,
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
    parser.add_argument(
        "--allow-same-bar-capacity",
        action="store_true",
        help=(
            "Explicitly allow turnover/volume from the execution bar to cap "
            "open-price fills. Without this flag, open execution refuses "
            "same-bar capacity assumptions."
        ),
    )
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument(
        "--data-access-mode",
        choices=("data_portal", "fast_parquet"),
        default="data_portal",
        help=(
            "Use DataPortal by default. fast_parquet keeps the legacy optimized "
            "local scan path and should be treated as an explicit performance "
            "shortcut until quantdb exposes a batch bar SDK."
        ),
    )
    parser.add_argument(
        "--streaming-chunk",
        choices=("year", "month"),
        default="year",
        help="Chunk size used by fast_parquet streaming backtests.",
    )
    parser.add_argument(
        "--streaming-chunk-padding-days",
        type=int,
        default=10,
        help=(
            "Calendar-day padding loaded around each fast_parquet streaming "
            "chunk for lookback and next-bar execution continuity."
        ),
    )
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
    if args.streaming_chunk_padding_days < 0:
        raise ValueError("--streaming-chunk-padding-days must be non-negative")
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
        allow_same_bar_capacity=args.allow_same_bar_capacity,
        data_access_mode=args.data_access_mode,
        streaming_chunk=args.streaming_chunk,
        streaming_chunk_padding_days=args.streaming_chunk_padding_days,
    )


if __name__ == "__main__":
    main()
