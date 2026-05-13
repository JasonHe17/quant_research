"""Build Baseline A alpha features and forward-return labels from real 5m data."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from quant_research.datasets import (
    ForwardReturnLabelConfig,
    IntradayFeatureConfig,
    add_cross_sectional_label_rank,
    build_intraday_feature_matrix,
    build_forward_return_labels,
    join_alpha_features_and_labels,
)

from run_baseline_a_real_backtest import (
    BacktestParams,
    _load_bars_from_files,
    _minute_bar_files,
)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = BacktestParams(
        catalog_path=Path(args.catalog_path),
        start=args.start,
        end=args.end,
        top_n=1,
        initial_cash=1.0,
        lookback_bars=1,
        min_avg_turnover=None,
        liquidity_window_bars=1,
        commission_bps=0.0,
        slippage_bps=0.0,
        lot_size=100,
        max_symbols=args.max_symbols,
        output_dir=None,
        exclude_st=args.exclude_st,
        limit_up_bps=args.limit_up_bps,
        limit_down_bps=args.limit_down_bps,
    )
    files = _minute_bar_files(params)
    if not files:
        raise FileNotFoundError("no 5-minute CN equity parquet files found")
    rows: list[dict[str, object]] = []
    label_config = ForwardReturnLabelConfig(
        name=args.label_name,
        horizon_bars=args.horizon_bars,
        entry_lag_bars=args.entry_lag_bars,
    )
    chunks = _time_chunks(
        start=args.start,
        end=args.end,
        partition=args.partition,
    )
    if args.workers == 1:
        row_iterable = (
            _build_partition_dataset(
                partition_name=partition_name,
                core_start=core_start,
                core_end=core_end,
                params=params,
                files=files,
                args=args,
                label_config=label_config,
                output_dir=output_dir,
            )
            for partition_name, core_start, core_end in chunks
        )
        for row in row_iterable:
            if row is None:
                continue
            rows.append(row)
            rows.sort(key=lambda item: str(item["partition"]))
            _write_summary(output_dir, args, rows)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _build_partition_dataset,
                    partition_name=partition_name,
                    core_start=core_start,
                    core_end=core_end,
                    params=params,
                    files=files,
                    args=args,
                    label_config=label_config,
                    output_dir=output_dir,
                )
                for partition_name, core_start, core_end in chunks
            ]
            for future in as_completed(futures):
                row = future.result()
                if row is None:
                    continue
                rows.append(row)
                rows.sort(key=lambda item: str(item["partition"]))
                _write_summary(output_dir, args, rows)
    print(pd.DataFrame(rows).to_string(index=False))


def _build_partition_dataset(
    *,
    partition_name: str,
    core_start: pd.Timestamp,
    core_end: pd.Timestamp,
    params: BacktestParams,
    files: list[Path],
    args: argparse.Namespace,
    label_config: ForwardReturnLabelConfig,
    output_dir: Path,
) -> dict[str, object] | None:
    read_start = core_start - pd.Timedelta(days=args.padding_days)
    read_end = core_end + pd.Timedelta(days=args.padding_days)
    chunk_params = replace(
        params,
        start=_format_timestamp(read_start),
        end=_format_timestamp(read_end),
    )
    bars = _load_bars_from_files(chunk_params, files)
    if bars.empty:
        return None
    print(
        f"loaded {partition_name}: bars={len(bars)} "
        f"core={_format_timestamp(core_start)}..{_format_timestamp(core_end)}",
        flush=True,
    )
    features = build_intraday_feature_matrix(
        bars,
        IntradayFeatureConfig(
            factor_groups=tuple(args.factor_groups),
            reversal_lookback_bars=tuple(args.lookback_bars),
            momentum_lookback_bars=tuple(args.momentum_lookback_bars),
            volatility_windows=tuple(args.volatility_windows),
            volume_windows=tuple(args.volume_windows),
            turnover_windows=tuple(args.turnover_windows),
        ),
    )
    features = _filter_core_window(features, core_start=core_start, core_end=core_end)
    labels = build_forward_return_labels(bars, label_config)
    labels = _filter_core_window(labels, core_start=core_start, core_end=core_end)
    labels = _add_entry_execution_columns(labels, bars)
    label_row_count_before_entry_filter = len(labels)
    entry_filter_counts = _entry_execution_filter_counts(labels)
    labels = _filter_entry_execution_constraints(labels, args)
    labels = add_cross_sectional_label_rank(
        labels,
        label_column=args.label_name,
        rank_column=f"{args.label_name}_rank",
    )
    dataset = join_alpha_features_and_labels(features, labels)
    dataset_path = output_dir / f"dataset_{partition_name}.parquet"
    dataset.to_parquet(dataset_path, index=False)
    feature_path = None
    label_path = None
    if args.write_components:
        feature_path = output_dir / f"features_{partition_name}.parquet"
        label_path = output_dir / f"labels_{partition_name}.parquet"
        features.to_parquet(feature_path, index=False)
        labels.to_parquet(label_path, index=False)
    row = {
        "partition": partition_name,
        "bar_count": len(bars),
        "feature_row_count": len(features),
        "label_row_count_before_entry_filter": label_row_count_before_entry_filter,
        "label_row_count": len(labels),
        "dataset_row_count": len(dataset),
        "instrument_count": int(bars["instrument_id"].nunique()),
        **entry_filter_counts,
        "dataset_path": str(dataset_path),
        "features_path": str(feature_path) if feature_path is not None else None,
        "labels_path": str(label_path) if label_path is not None else None,
    }
    del bars, features, labels, dataset
    gc.collect()
    return row


def _time_chunks(
    *,
    start: str,
    end: str,
    partition: str,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("--start must be before --end")
    if partition == "yearly":
        frequency = "YS"
        name_format = "%Y"
    elif partition == "monthly":
        frequency = "MS"
        name_format = "%Y_%m"
    else:
        raise ValueError("--partition must be monthly or yearly")
    boundary_start = start_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if partition == "yearly":
        boundary_start = boundary_start.replace(month=1)
    starts = pd.date_range(boundary_start, end_ts, freq=frequency)
    chunks: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for index, period_start in enumerate(starts):
        next_start = (
            starts[index + 1]
            if index + 1 < len(starts)
            else _next_period_start(period_start, partition)
        )
        core_start = max(start_ts, period_start)
        core_end = min(end_ts, next_start - pd.Timedelta(microseconds=1))
        if core_start <= core_end:
            chunks.append((period_start.strftime(name_format), core_start, core_end))
    return chunks


def _filter_core_window(
    frame: pd.DataFrame,
    *,
    core_start: pd.Timestamp,
    core_end: pd.Timestamp,
) -> pd.DataFrame:
    return frame.loc[
        (frame["timestamp"] >= _format_timestamp(core_start))
        & (frame["timestamp"] <= _format_timestamp(core_end))
    ].reset_index(drop=True)


def _add_entry_execution_columns(labels: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return labels.assign(
            entry_tradable_bar=pd.Series(dtype=bool),
            entry_limit_up_open=pd.Series(dtype=bool),
            entry_limit_down_open=pd.Series(dtype=bool),
        )
    execution_columns = [
        "instrument_id",
        "bar_end_time",
        "tradable_bar",
        "limit_up_open",
        "limit_down_open",
    ]
    missing = [column for column in execution_columns if column not in bars.columns]
    if missing:
        raise ValueError(f"bars missing execution columns: {missing}")
    entry_execution = bars.loc[:, execution_columns].rename(
        columns={
            "bar_end_time": "entry_timestamp",
            "tradable_bar": "entry_tradable_bar",
            "limit_up_open": "entry_limit_up_open",
            "limit_down_open": "entry_limit_down_open",
        }
    )
    enriched = labels.merge(
        entry_execution,
        on=["instrument_id", "entry_timestamp"],
        how="left",
    )
    for column in (
        "entry_tradable_bar",
        "entry_limit_up_open",
        "entry_limit_down_open",
    ):
        enriched[column] = enriched[column].fillna(False).astype(bool)
    return enriched


def _entry_execution_filter_counts(labels: pd.DataFrame) -> dict[str, int]:
    if labels.empty:
        return {
            "entry_non_tradable_label_count": 0,
            "entry_limit_up_label_count": 0,
            "entry_limit_down_label_count": 0,
        }
    return {
        "entry_non_tradable_label_count": int(
            (~labels["entry_tradable_bar"].astype(bool)).sum()
        ),
        "entry_limit_up_label_count": int(labels["entry_limit_up_open"].astype(bool).sum()),
        "entry_limit_down_label_count": int(
            labels["entry_limit_down_open"].astype(bool).sum()
        ),
    }


def _filter_entry_execution_constraints(
    labels: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if labels.empty:
        return labels.reset_index(drop=True)
    mask = pd.Series(True, index=labels.index)
    if args.filter_entry_tradable:
        mask = mask & labels["entry_tradable_bar"].astype(bool)
    if args.filter_entry_limit_up:
        mask = mask & ~labels["entry_limit_up_open"].astype(bool)
    return labels.loc[mask].reset_index(drop=True)


def _next_period_start(period_start: pd.Timestamp, partition: str) -> pd.Timestamp:
    if partition == "yearly":
        return period_start + pd.DateOffset(years=1)
    return period_start + pd.DateOffset(months=1)


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat()


def _write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> None:
    payload = {
        "params": {
            "start": args.start,
            "end": args.end,
            "factor_groups": args.factor_groups,
            "lookback_bars": args.lookback_bars,
            "momentum_lookback_bars": args.momentum_lookback_bars,
            "volatility_windows": args.volatility_windows,
            "volume_windows": args.volume_windows,
            "turnover_windows": args.turnover_windows,
            "label_name": args.label_name,
            "horizon_bars": args.horizon_bars,
            "entry_lag_bars": args.entry_lag_bars,
            "exclude_st": args.exclude_st,
            "limit_up_bps": args.limit_up_bps,
            "limit_down_bps": args.limit_down_bps,
            "filter_entry_tradable": args.filter_entry_tradable,
            "filter_entry_limit_up": args.filter_entry_limit_up,
            "max_symbols": args.max_symbols,
            "write_components": args.write_components,
            "partition": args.partition,
            "padding_days": args.padding_days,
            "workers": args.workers,
        },
        "partitions": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-path",
        default="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--factor-groups",
        nargs="+",
        default=["reversal"],
        choices=(
            "all",
            "reversal",
            "momentum",
            "volatility",
            "volume",
            "turnover",
            "bar_return",
            "liquidity_impact",
        ),
    )
    parser.add_argument("--lookback-bars", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument(
        "--momentum-lookback-bars",
        type=int,
        nargs="+",
        default=[3, 6, 12],
    )
    parser.add_argument("--volatility-windows", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--volume-windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument("--turnover-windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument("--label-name", default="forward_return")
    parser.add_argument("--horizon-bars", type=int, default=48)
    parser.add_argument("--entry-lag-bars", type=int, default=1)
    parser.add_argument("--exclude-st", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-up-bps", type=float, default=980.0)
    parser.add_argument("--limit-down-bps", type=float, default=980.0)
    parser.add_argument(
        "--filter-entry-tradable",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--filter-entry-limit-up",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--partition", choices=("monthly", "yearly"), default="monthly")
    parser.add_argument("--padding-days", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--write-components", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in args.lookback_bars):
        raise ValueError("--lookback-bars values must be positive")
    if any(value <= 0 for value in args.momentum_lookback_bars):
        raise ValueError("--momentum-lookback-bars values must be positive")
    if any(value <= 0 for value in args.volatility_windows):
        raise ValueError("--volatility-windows values must be positive")
    if any(value <= 0 for value in args.volume_windows):
        raise ValueError("--volume-windows values must be positive")
    if any(value <= 0 for value in args.turnover_windows):
        raise ValueError("--turnover-windows values must be positive")
    if args.padding_days < 0:
        raise ValueError("--padding-days must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.limit_up_bps <= 0:
        raise ValueError("--limit-up-bps must be positive")
    if args.limit_down_bps <= 0:
        raise ValueError("--limit-down-bps must be positive")
    return args


if __name__ == "__main__":
    main()
