"""Build Baseline A alpha features and forward-return labels from real 5m data."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
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
    DatasetPartitionManifest,
    ForwardReturnLabelConfig,
    IntradayFeatureConfig,
    add_cross_sectional_label_rank,
    build_intraday_feature_matrix,
    build_multi_horizon_forward_return_labels,
    join_alpha_features_and_labels,
    write_dataset_manifest,
)
from quant_research.data.cache import catalog_reference_for_path
from quant_research.models import infer_feature_columns

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
        data_access_mode="fast_parquet",
    )
    files = _minute_bar_files(params)
    if not files:
        raise FileNotFoundError("no 5-minute CN equity parquet files found")
    rows: list[dict[str, object]] = []
    label_configs = _label_configs(args)
    chunks = _time_chunks(
        start=args.start,
        end=args.end,
        partition=args.partition,
    )
    effective_workers = _effective_workers(args)
    if effective_workers != args.workers:
        print(
            f"dataset workers reduced from {args.workers} to {effective_workers} "
            "by memory budget",
            flush=True,
        )
    if effective_workers == 1:
        row_iterable = (
            _build_partition_dataset(
                partition_name=partition_name,
                core_start=core_start,
                core_end=core_end,
                params=params,
                files=files,
                args=args,
                label_configs=label_configs,
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
        pending = list(chunks)
        running: dict[
            Future[dict[str, object] | None],
            tuple[str, pd.Timestamp, pd.Timestamp],
        ] = {}
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            while pending or running:
                while pending and len(running) < effective_workers:
                    partition_name, core_start, core_end = pending.pop(0)
                    future = executor.submit(
                        _build_partition_dataset,
                        partition_name=partition_name,
                        core_start=core_start,
                        core_end=core_end,
                        params=params,
                        files=files,
                        args=args,
                        label_configs=label_configs,
                        output_dir=output_dir,
                    )
                    running[future] = (partition_name, core_start, core_end)
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    running.pop(future)
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
    label_configs: tuple[ForwardReturnLabelConfig, ...],
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
            cross_sectional_reversal_lookback_bars=tuple(
                args.cross_sectional_reversal_lookback_bars
            ),
            conditional_reversal_lookback_bars=tuple(
                args.conditional_reversal_lookback_bars
            ),
            conditional_reversal_state_windows=tuple(
                args.conditional_reversal_state_windows
            ),
            conditional_reversal_low_vol_quantile=(
                args.conditional_reversal_low_vol_quantile
            ),
            conditional_reversal_min_volume_ratio=(
                args.conditional_reversal_min_volume_ratio
            ),
            eod_reversal_lookback_bars=tuple(args.eod_reversal_lookback_bars),
            eod_reversal_tail_bars=args.eod_reversal_tail_bars,
            eod_reversal_weight=args.eod_reversal_weight,
            momentum_lookback_bars=tuple(args.momentum_lookback_bars),
            volatility_windows=tuple(args.volatility_windows),
            volatility_state_change_specs=tuple(args.volatility_state_change_specs),
            price_position_windows=tuple(args.price_position_windows),
            range_volatility_windows=tuple(args.range_volatility_windows),
            efficiency_windows=tuple(args.efficiency_windows),
            volume_windows=tuple(args.volume_windows),
            volume_distribution_windows=tuple(args.volume_distribution_windows),
            turnover_windows=tuple(args.turnover_windows),
            vwap_deviation_windows=tuple(args.vwap_deviation_windows),
            downside_volatility_windows=tuple(args.downside_volatility_windows),
            return_skewness_windows=tuple(args.return_skewness_windows),
            lottery_max_windows=tuple(args.lottery_max_windows),
            money_flow_windows=tuple(args.money_flow_windows),
            signed_turnover_imbalance_windows=tuple(
                args.signed_turnover_imbalance_windows
            ),
            risk_adjusted_momentum_windows=tuple(args.risk_adjusted_momentum_windows),
            volume_confirmed_momentum_windows=tuple(
                args.volume_confirmed_momentum_windows
            ),
            turnover_stability_windows=tuple(args.turnover_stability_windows),
            liquidity_reliability_windows=tuple(args.liquidity_reliability_windows),
            liquidity_reliability_recovery_specs=tuple(
                args.liquidity_reliability_recovery_specs
            ),
            liquidity_reliability_recovery_balance_specs=tuple(
                args.liquidity_reliability_recovery_balance_specs
            ),
            return_turnover_correlation_windows=tuple(
                args.return_turnover_correlation_windows
            ),
            negative_return_persistence_windows=tuple(
                args.negative_return_persistence_windows
            ),
            sell_pressure_absorption_windows=tuple(
                args.sell_pressure_absorption_windows
            ),
            downside_turnover_decay_windows=tuple(
                args.downside_turnover_decay_windows
            ),
            sell_pressure_recovery_windows=tuple(
                args.sell_pressure_recovery_windows
            ),
            sell_pressure_exhaustion_windows=tuple(
                args.sell_pressure_exhaustion_windows
            ),
            sell_pressure_exhaustion_persistence_specs=tuple(
                args.sell_pressure_exhaustion_persistence_specs
            ),
            microstructure_recovery_windows=tuple(
                args.microstructure_recovery_windows
            ),
            microstructure_exhaustion_windows=tuple(
                args.microstructure_exhaustion_windows
            ),
            microstructure_recovery_acceleration_specs=tuple(
                args.microstructure_recovery_acceleration_specs
            ),
            same_slot_memory_windows=tuple(args.same_slot_memory_windows),
            weak_tape_gap_windows=tuple(args.weak_tape_gap_windows),
            sell_pressure_quality_windows=tuple(args.sell_pressure_quality_windows),
            event_shock_windows=tuple(args.event_shock_windows),
            daily_moving_average_windows=tuple(args.daily_moving_average_windows),
            daily_moving_average_pairs=tuple(args.daily_moving_average_pairs),
            market_downside_beta_windows=tuple(args.market_downside_beta_windows),
            market_state_windows=tuple(args.market_state_windows),
            breadth_resilience_windows=tuple(args.breadth_resilience_windows),
            breadth_shock_residual_resilience_windows=tuple(
                args.breadth_shock_residual_resilience_windows
            ),
            limit_pressure_resilience_windows=tuple(
                args.limit_pressure_resilience_windows
            ),
            regime_conditioned_lookback_bars=tuple(
                args.regime_conditioned_lookback_bars
            ),
            regime_conditioned_state_windows=tuple(
                args.regime_conditioned_state_windows
            ),
        ),
    )
    features = _filter_core_window(features, core_start=core_start, core_end=core_end)
    labels = build_multi_horizon_forward_return_labels(bars, label_configs)
    labels = _filter_core_window(labels, core_start=core_start, core_end=core_end)
    labels = _add_entry_execution_columns(labels, bars)
    label_row_count_before_entry_filter = len(labels)
    entry_filter_counts = _entry_execution_filter_counts(labels)
    labels = _filter_entry_execution_constraints(labels, args)
    for label_config in label_configs:
        labels = add_cross_sectional_label_rank(
            labels,
            label_column=label_config.name,
            rank_column=f"{label_config.name}_rank",
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
    label_columns = tuple(
        column
        for label_config in label_configs
        for column in (label_config.name, f"{label_config.name}_rank")
    )
    feature_columns = _feature_columns(dataset, label_configs)
    source_artifacts = {}
    if feature_path is not None:
        source_artifacts["features"] = str(feature_path)
    if label_path is not None:
        source_artifacts["labels"] = str(label_path)
    manifest = DatasetPartitionManifest.create(
        name="baseline_a_alpha_dataset",
        partition=partition_name,
        dataset_path=dataset_path,
        row_count=len(dataset),
        feature_columns=feature_columns,
        label_columns=label_columns,
        parameters=_manifest_parameters(args),
        data_snapshot=args.data_snapshot,
        catalog_reference=catalog_reference_for_path(args.catalog_path),
        source_artifacts=source_artifacts,
    )
    manifest_path = output_dir / f"dataset_{partition_name}.manifest.json"
    write_dataset_manifest(manifest, manifest_path)
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
        "manifest_path": str(manifest_path),
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
            "cross_sectional_reversal_lookback_bars": (
                args.cross_sectional_reversal_lookback_bars
            ),
            "conditional_reversal_lookback_bars": (
                args.conditional_reversal_lookback_bars
            ),
            "conditional_reversal_state_windows": (
                args.conditional_reversal_state_windows
            ),
            "conditional_reversal_low_vol_quantile": (
                args.conditional_reversal_low_vol_quantile
            ),
            "conditional_reversal_min_volume_ratio": (
                args.conditional_reversal_min_volume_ratio
            ),
            "eod_reversal_lookback_bars": args.eod_reversal_lookback_bars,
            "eod_reversal_tail_bars": args.eod_reversal_tail_bars,
            "eod_reversal_weight": args.eod_reversal_weight,
            "momentum_lookback_bars": args.momentum_lookback_bars,
            "volatility_windows": args.volatility_windows,
            "volatility_state_change_specs": [
                list(spec) for spec in args.volatility_state_change_specs
            ],
            "price_position_windows": args.price_position_windows,
            "range_volatility_windows": args.range_volatility_windows,
            "efficiency_windows": args.efficiency_windows,
            "volume_windows": args.volume_windows,
            "volume_distribution_windows": args.volume_distribution_windows,
            "turnover_windows": args.turnover_windows,
            "turnover_stability_windows": args.turnover_stability_windows,
            "liquidity_reliability_windows": args.liquidity_reliability_windows,
            "liquidity_reliability_recovery_specs": [
                list(spec) for spec in args.liquidity_reliability_recovery_specs
            ],
            "liquidity_reliability_recovery_balance_specs": [
                list(spec)
                for spec in args.liquidity_reliability_recovery_balance_specs
            ],
            "vwap_deviation_windows": args.vwap_deviation_windows,
            "downside_volatility_windows": args.downside_volatility_windows,
            "return_skewness_windows": args.return_skewness_windows,
            "lottery_max_windows": args.lottery_max_windows,
            "money_flow_windows": args.money_flow_windows,
            "signed_turnover_imbalance_windows": args.signed_turnover_imbalance_windows,
            "risk_adjusted_momentum_windows": args.risk_adjusted_momentum_windows,
            "volume_confirmed_momentum_windows": args.volume_confirmed_momentum_windows,
            "return_turnover_correlation_windows": args.return_turnover_correlation_windows,
            "negative_return_persistence_windows": (
                args.negative_return_persistence_windows
            ),
            "sell_pressure_absorption_windows": args.sell_pressure_absorption_windows,
            "downside_turnover_decay_windows": args.downside_turnover_decay_windows,
            "sell_pressure_recovery_windows": args.sell_pressure_recovery_windows,
            "sell_pressure_exhaustion_windows": args.sell_pressure_exhaustion_windows,
            "sell_pressure_exhaustion_persistence_specs": [
                list(spec) for spec in args.sell_pressure_exhaustion_persistence_specs
            ],
            "microstructure_recovery_windows": args.microstructure_recovery_windows,
            "microstructure_exhaustion_windows": (
                args.microstructure_exhaustion_windows
            ),
            "microstructure_recovery_acceleration_specs": [
                list(spec) for spec in args.microstructure_recovery_acceleration_specs
            ],
            "same_slot_memory_windows": args.same_slot_memory_windows,
            "weak_tape_gap_windows": args.weak_tape_gap_windows,
            "sell_pressure_quality_windows": args.sell_pressure_quality_windows,
            "event_shock_windows": args.event_shock_windows,
            "daily_moving_average_windows": args.daily_moving_average_windows,
            "daily_moving_average_pairs": [
                list(pair) for pair in args.daily_moving_average_pairs
            ],
            "market_downside_beta_windows": args.market_downside_beta_windows,
            "market_state_windows": args.market_state_windows,
            "breadth_resilience_windows": args.breadth_resilience_windows,
            "breadth_shock_residual_resilience_windows": (
                args.breadth_shock_residual_resilience_windows
            ),
            "limit_pressure_resilience_windows": args.limit_pressure_resilience_windows,
            "regime_conditioned_lookback_bars": args.regime_conditioned_lookback_bars,
            "regime_conditioned_state_windows": args.regime_conditioned_state_windows,
            "label_name": args.label_name,
            "horizon_bars": args.horizon_bars,
            "label_columns": _label_column_names(args),
            "entry_lag_bars": args.entry_lag_bars,
            "research_grid": _research_grid(args),
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
            "data_snapshot": args.data_snapshot,
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
            "cross_sectional_reversal",
            "conditional_reversal",
            "eod_reversal",
            "momentum",
            "volatility",
            "volatility_state_change",
            "price_position",
            "range_volatility",
            "efficiency",
            "volume",
            "volume_distribution_shape",
            "turnover",
            "turnover_stability",
            "liquidity_reliability",
            "liquidity_reliability_recovery",
            "liquidity_reliability_recovery_balance",
            "bar_return",
            "liquidity_impact",
            "vwap_deviation",
            "downside_volatility",
            "return_skewness",
            "lottery_max",
            "money_flow",
            "signed_turnover_imbalance",
            "risk_adjusted_momentum",
            "volume_confirmed_momentum",
            "intraday_gap",
            "market_state",
            "market_downside_beta",
            "breadth_resilience",
            "breadth_shock_residual_resilience",
            "limit_pressure_resilience",
            "regime_conditioned",
            "return_turnover_correlation",
            "negative_return_persistence",
            "sell_pressure_absorption",
            "downside_turnover_decay",
            "sell_pressure_recovery",
            "sell_pressure_exhaustion",
            "sell_pressure_exhaustion_persistence",
            "microstructure_exhaustion_alert",
            "microstructure_recovery_speed",
            "same_slot_intraday_memory",
            "overnight_intraday_tug_of_war",
            "weak_tape_overnight_gap",
            "sell_pressure_quality_state",
            "event_shock_proxy",
            "daily_moving_average",
        ),
    )
    parser.add_argument("--lookback-bars", type=int, nargs="+", default=[1, 6, 12, 24])
    parser.add_argument(
        "--cross-sectional-reversal-lookback-bars",
        type=int,
        nargs="+",
        default=[1, 6, 12, 24],
    )
    parser.add_argument(
        "--conditional-reversal-lookback-bars",
        type=int,
        nargs="+",
        default=[1, 6, 12, 24],
    )
    parser.add_argument(
        "--conditional-reversal-state-windows",
        type=int,
        nargs="+",
        default=[12],
    )
    parser.add_argument("--conditional-reversal-low-vol-quantile", type=float, default=0.5)
    parser.add_argument("--conditional-reversal-min-volume-ratio", type=float, default=0.0)
    parser.add_argument(
        "--eod-reversal-lookback-bars",
        type=int,
        nargs="+",
        default=[1, 6, 12, 24],
    )
    parser.add_argument("--eod-reversal-tail-bars", type=int, default=6)
    parser.add_argument("--eod-reversal-weight", type=float, default=1.5)
    parser.add_argument(
        "--momentum-lookback-bars",
        type=int,
        nargs="+",
        default=[3, 6, 12],
    )
    parser.add_argument("--volatility-windows", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument(
        "--volatility-state-change-specs",
        type=_two_window_spec,
        nargs="+",
        default=[(12, 48), (24, 96)],
        help="short:long volatility windows, for example 12:48",
    )
    parser.add_argument("--price-position-windows", type=int, nargs="+", default=[48])
    parser.add_argument("--range-volatility-windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument("--efficiency-windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument("--volume-windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument(
        "--volume-distribution-windows",
        type=int,
        nargs="+",
        default=[24, 48, 96],
    )
    parser.add_argument("--turnover-windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument("--turnover-stability-windows", type=int, nargs="+", default=[48])
    parser.add_argument(
        "--liquidity-reliability-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument(
        "--liquidity-reliability-recovery-specs",
        type=_liquidity_reliability_recovery_spec,
        nargs="+",
        default=[(48, 12, 24)],
        help=(
            "long:capacity:recovery windows, for example 48:12:24; emits "
            "low reliability premium gated by recent capacity and recovery"
        ),
    )
    parser.add_argument(
        "--liquidity-reliability-recovery-balance-specs",
        type=_liquidity_reliability_recovery_spec,
        nargs="+",
        default=[(48, 12, 24)],
        help=(
            "long:capacity:recovery windows, for example 48:12:24; emits "
            "low reliability recovery with smooth capacity and recovery balance"
        ),
    )
    parser.add_argument("--vwap-deviation-windows", type=int, nargs="+", default=[48])
    parser.add_argument(
        "--downside-volatility-windows",
        type=int,
        nargs="+",
        default=[12, 48],
    )
    parser.add_argument("--return-skewness-windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument("--lottery-max-windows", type=int, nargs="+", default=[24, 48, 96])
    parser.add_argument("--money-flow-windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument(
        "--signed-turnover-imbalance-windows",
        type=int,
        nargs="+",
        default=[12, 48],
    )
    parser.add_argument(
        "--risk-adjusted-momentum-windows",
        type=int,
        nargs="+",
        default=[12, 48],
    )
    parser.add_argument(
        "--volume-confirmed-momentum-windows",
        type=int,
        nargs="+",
        default=[12, 48],
    )
    parser.add_argument(
        "--return-turnover-correlation-windows",
        type=int,
        nargs="+",
        default=[12, 48],
    )
    parser.add_argument(
        "--negative-return-persistence-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument(
        "--sell-pressure-absorption-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument(
        "--downside-turnover-decay-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument(
        "--sell-pressure-recovery-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument(
        "--sell-pressure-exhaustion-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument(
        "--sell-pressure-exhaustion-persistence-specs",
        type=_sell_pressure_exhaustion_persistence_spec,
        nargs="+",
        default=[(96, 24, 48)],
        help=(
            "long:short:medium exhaustion windows, for example 96:24:48; "
            "emits long - 0.5 * (short + medium)"
        ),
    )
    parser.add_argument(
        "--microstructure-recovery-windows",
        type=int,
        nargs="+",
        default=[24, 48],
    )
    parser.add_argument(
        "--microstructure-exhaustion-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument(
        "--microstructure-recovery-acceleration-specs",
        type=_two_window_spec,
        nargs="+",
        default=[(12, 48), (24, 96)],
        help="short:long recovery-speed windows, for example 12:48",
    )
    parser.add_argument(
        "--same-slot-memory-windows",
        type=int,
        nargs="+",
        default=[5, 20],
    )
    parser.add_argument(
        "--sell-pressure-quality-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument("--event-shock-windows", type=int, nargs="+", default=[48])
    parser.add_argument(
        "--daily-moving-average-windows",
        type=int,
        nargs="+",
        default=[5, 10, 20],
    )
    parser.add_argument(
        "--daily-moving-average-pairs",
        type=_daily_moving_average_pair,
        nargs="+",
        default=[(5, 20), (10, 20)],
        help="short:long daily moving-average pairs, for example 5:20",
    )
    parser.add_argument("--market-downside-beta-windows", type=int, nargs="+", default=[48])
    parser.add_argument("--market-state-windows", type=int, nargs="+", default=[48])
    parser.add_argument("--weak-tape-gap-windows", type=int, nargs="+", default=[48])
    parser.add_argument("--breadth-resilience-windows", type=int, nargs="+", default=[48])
    parser.add_argument(
        "--breadth-shock-residual-resilience-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument(
        "--limit-pressure-resilience-windows", type=int, nargs="+", default=[48]
    )
    parser.add_argument(
        "--regime-conditioned-lookback-bars",
        type=int,
        nargs="+",
        default=[12, 24],
    )
    parser.add_argument(
        "--regime-conditioned-state-windows",
        type=int,
        nargs="+",
        default=[48],
    )
    parser.add_argument("--label-name", default="forward_return")
    parser.add_argument("--horizon-bars", type=int, nargs="+", default=[48])
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
    parser.add_argument(
        "--worker-memory-estimate-gb",
        type=float,
        default=10.0,
        help="estimated peak memory per dataset worker; used to cap process concurrency",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=0.0,
        help=(
            "memory budget for concurrent dataset workers; 0 auto-detects "
            "available memory and uses a conservative fraction"
        ),
    )
    parser.add_argument("--write-components", action="store_true")
    parser.add_argument("--data-snapshot")
    args = parser.parse_args()
    if any(value <= 0 for value in args.lookback_bars):
        raise ValueError("--lookback-bars values must be positive")
    if any(value <= 0 for value in args.cross_sectional_reversal_lookback_bars):
        raise ValueError(
            "--cross-sectional-reversal-lookback-bars values must be positive"
        )
    if any(value <= 0 for value in args.conditional_reversal_lookback_bars):
        raise ValueError("--conditional-reversal-lookback-bars values must be positive")
    if any(value <= 0 for value in args.conditional_reversal_state_windows):
        raise ValueError("--conditional-reversal-state-windows values must be positive")
    if not 0.0 < args.conditional_reversal_low_vol_quantile <= 1.0:
        raise ValueError("--conditional-reversal-low-vol-quantile must be in (0, 1]")
    if args.conditional_reversal_min_volume_ratio <= -1.0:
        raise ValueError("--conditional-reversal-min-volume-ratio must be greater than -1")
    if any(value <= 0 for value in args.eod_reversal_lookback_bars):
        raise ValueError("--eod-reversal-lookback-bars values must be positive")
    if args.eod_reversal_tail_bars <= 0:
        raise ValueError("--eod-reversal-tail-bars must be positive")
    if args.eod_reversal_weight <= 0.0:
        raise ValueError("--eod-reversal-weight must be positive")
    if any(value <= 0 for value in args.momentum_lookback_bars):
        raise ValueError("--momentum-lookback-bars values must be positive")
    if any(value <= 0 for value in args.volatility_windows):
        raise ValueError("--volatility-windows values must be positive")
    for short_window, long_window in args.volatility_state_change_specs:
        if short_window <= 1 or long_window <= 1:
            raise ValueError("--volatility-state-change-specs values must be at least 2")
        if short_window >= long_window:
            raise ValueError("--volatility-state-change-specs must be ordered short:long")
    if any(value <= 0 for value in args.price_position_windows):
        raise ValueError("--price-position-windows values must be positive")
    if any(value <= 0 for value in args.range_volatility_windows):
        raise ValueError("--range-volatility-windows values must be positive")
    if any(value <= 0 for value in args.efficiency_windows):
        raise ValueError("--efficiency-windows values must be positive")
    if any(value <= 0 for value in args.volume_windows):
        raise ValueError("--volume-windows values must be positive")
    if any(value <= 1 for value in args.volume_distribution_windows):
        raise ValueError("--volume-distribution-windows values must be at least 2")
    if any(value <= 0 for value in args.turnover_windows):
        raise ValueError("--turnover-windows values must be positive")
    if any(value <= 0 for value in args.turnover_stability_windows):
        raise ValueError("--turnover-stability-windows values must be positive")
    if any(value <= 0 for value in args.liquidity_reliability_windows):
        raise ValueError("--liquidity-reliability-windows values must be positive")
    for long_window, capacity_window, recovery_window in (
        args.liquidity_reliability_recovery_specs
    ):
        if long_window <= 1 or capacity_window <= 1 or recovery_window <= 1:
            raise ValueError(
                "--liquidity-reliability-recovery-specs values must be at least 2"
            )
        if long_window <= max(capacity_window, recovery_window):
            raise ValueError(
                "--liquidity-reliability-recovery-specs must be ordered "
                "long:capacity:recovery with long greater than capacity and recovery"
            )
    for long_window, capacity_window, recovery_window in (
        args.liquidity_reliability_recovery_balance_specs
    ):
        if long_window <= 1 or capacity_window <= 1 or recovery_window <= 1:
            raise ValueError(
                "--liquidity-reliability-recovery-balance-specs values must be at least 2"
            )
        if long_window <= max(capacity_window, recovery_window):
            raise ValueError(
                "--liquidity-reliability-recovery-balance-specs must be ordered "
                "long:capacity:recovery with long greater than capacity and recovery"
            )
    if any(value <= 0 for value in args.vwap_deviation_windows):
        raise ValueError("--vwap-deviation-windows values must be positive")
    if any(value <= 0 for value in args.downside_volatility_windows):
        raise ValueError("--downside-volatility-windows values must be positive")
    if any(value <= 0 for value in args.return_skewness_windows):
        raise ValueError("--return-skewness-windows values must be positive")
    if any(value <= 0 for value in args.lottery_max_windows):
        raise ValueError("--lottery-max-windows values must be positive")
    if any(value <= 0 for value in args.money_flow_windows):
        raise ValueError("--money-flow-windows values must be positive")
    if any(value <= 0 for value in args.signed_turnover_imbalance_windows):
        raise ValueError("--signed-turnover-imbalance-windows values must be positive")
    if any(value <= 0 for value in args.risk_adjusted_momentum_windows):
        raise ValueError("--risk-adjusted-momentum-windows values must be positive")
    if any(value <= 0 for value in args.volume_confirmed_momentum_windows):
        raise ValueError("--volume-confirmed-momentum-windows values must be positive")
    if any(value <= 0 for value in args.return_turnover_correlation_windows):
        raise ValueError("--return-turnover-correlation-windows values must be positive")
    if any(value <= 0 for value in args.negative_return_persistence_windows):
        raise ValueError("--negative-return-persistence-windows values must be positive")
    if any(value <= 0 for value in args.sell_pressure_absorption_windows):
        raise ValueError("--sell-pressure-absorption-windows values must be positive")
    if any(value <= 1 for value in args.downside_turnover_decay_windows):
        raise ValueError("--downside-turnover-decay-windows values must be at least 2")
    if any(value <= 0 for value in args.sell_pressure_recovery_windows):
        raise ValueError("--sell-pressure-recovery-windows values must be positive")
    if any(value <= 1 for value in args.sell_pressure_exhaustion_windows):
        raise ValueError("--sell-pressure-exhaustion-windows values must be at least 2")
    for long_window, short_window, medium_window in (
        args.sell_pressure_exhaustion_persistence_specs
    ):
        if long_window <= 1 or short_window <= 1 or medium_window <= 1:
            raise ValueError(
                "--sell-pressure-exhaustion-persistence-specs values must be at least 2"
            )
        if long_window <= max(short_window, medium_window):
            raise ValueError(
                "--sell-pressure-exhaustion-persistence-specs must be ordered "
                "long:short:medium with long greater than short and medium"
            )
    if any(value <= 1 for value in args.microstructure_recovery_windows):
        raise ValueError("--microstructure-recovery-windows values must be at least 2")
    if any(value <= 1 for value in args.microstructure_exhaustion_windows):
        raise ValueError("--microstructure-exhaustion-windows values must be at least 2")
    for short_window, long_window in args.microstructure_recovery_acceleration_specs:
        if short_window <= 1 or long_window <= 1:
            raise ValueError(
                "--microstructure-recovery-acceleration-specs values must be at least 2"
            )
        if short_window >= long_window:
            raise ValueError(
                "--microstructure-recovery-acceleration-specs must be ordered short:long"
            )
    if any(value <= 0 for value in args.same_slot_memory_windows):
        raise ValueError("--same-slot-memory-windows values must be positive")
    if any(value <= 0 for value in args.weak_tape_gap_windows):
        raise ValueError("--weak-tape-gap-windows values must be positive")
    if any(value <= 0 for value in args.sell_pressure_quality_windows):
        raise ValueError("--sell-pressure-quality-windows values must be positive")
    if any(value <= 0 for value in args.event_shock_windows):
        raise ValueError("--event-shock-windows values must be positive")
    if any(value <= 0 for value in args.daily_moving_average_windows):
        raise ValueError("--daily-moving-average-windows values must be positive")
    if any(short <= 0 or long <= 0 for short, long in args.daily_moving_average_pairs):
        raise ValueError("--daily-moving-average-pairs values must be positive")
    if any(short >= long for short, long in args.daily_moving_average_pairs):
        raise ValueError("--daily-moving-average-pairs must be ordered short:long")
    if any(value <= 0 for value in args.market_downside_beta_windows):
        raise ValueError("--market-downside-beta-windows values must be positive")
    if any(value <= 0 for value in args.market_state_windows):
        raise ValueError("--market-state-windows values must be positive")
    if any(value <= 0 for value in args.breadth_resilience_windows):
        raise ValueError("--breadth-resilience-windows values must be positive")
    if any(value <= 0 for value in args.breadth_shock_residual_resilience_windows):
        raise ValueError(
            "--breadth-shock-residual-resilience-windows values must be positive"
        )
    if any(value <= 0 for value in args.limit_pressure_resilience_windows):
        raise ValueError("--limit-pressure-resilience-windows values must be positive")
    if any(value <= 0 for value in args.regime_conditioned_lookback_bars):
        raise ValueError("--regime-conditioned-lookback-bars values must be positive")
    if any(value <= 0 for value in args.regime_conditioned_state_windows):
        raise ValueError("--regime-conditioned-state-windows values must be positive")
    if any(value <= 0 for value in args.horizon_bars):
        raise ValueError("--horizon-bars values must be positive")
    if args.padding_days < 0:
        raise ValueError("--padding-days must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.worker_memory_estimate_gb <= 0:
        raise ValueError("--worker-memory-estimate-gb must be positive")
    if args.memory_budget_gb < 0:
        raise ValueError("--memory-budget-gb must be non-negative")
    if args.limit_up_bps <= 0:
        raise ValueError("--limit-up-bps must be positive")
    if args.limit_down_bps <= 0:
        raise ValueError("--limit-down-bps must be positive")
    return args


def _manifest_parameters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "start": args.start,
        "end": args.end,
        "factor_groups": list(args.factor_groups),
        "lookback_bars": list(args.lookback_bars),
        "cross_sectional_reversal_lookback_bars": list(
            args.cross_sectional_reversal_lookback_bars
        ),
        "conditional_reversal_lookback_bars": list(
            args.conditional_reversal_lookback_bars
        ),
        "conditional_reversal_state_windows": list(
            args.conditional_reversal_state_windows
        ),
        "conditional_reversal_low_vol_quantile": (
            args.conditional_reversal_low_vol_quantile
        ),
        "conditional_reversal_min_volume_ratio": (
            args.conditional_reversal_min_volume_ratio
        ),
        "eod_reversal_lookback_bars": list(args.eod_reversal_lookback_bars),
        "eod_reversal_tail_bars": args.eod_reversal_tail_bars,
        "eod_reversal_weight": args.eod_reversal_weight,
        "momentum_lookback_bars": list(args.momentum_lookback_bars),
        "volatility_windows": list(args.volatility_windows),
        "volatility_state_change_specs": [
            list(spec) for spec in args.volatility_state_change_specs
        ],
        "price_position_windows": list(args.price_position_windows),
        "range_volatility_windows": list(args.range_volatility_windows),
        "efficiency_windows": list(args.efficiency_windows),
        "volume_windows": list(args.volume_windows),
        "volume_distribution_windows": list(args.volume_distribution_windows),
        "turnover_windows": list(args.turnover_windows),
        "turnover_stability_windows": list(args.turnover_stability_windows),
        "liquidity_reliability_windows": list(args.liquidity_reliability_windows),
        "liquidity_reliability_recovery_specs": [
            list(spec) for spec in args.liquidity_reliability_recovery_specs
        ],
        "liquidity_reliability_recovery_balance_specs": [
            list(spec) for spec in args.liquidity_reliability_recovery_balance_specs
        ],
        "vwap_deviation_windows": list(args.vwap_deviation_windows),
        "downside_volatility_windows": list(args.downside_volatility_windows),
        "return_skewness_windows": list(args.return_skewness_windows),
        "lottery_max_windows": list(args.lottery_max_windows),
        "money_flow_windows": list(args.money_flow_windows),
        "signed_turnover_imbalance_windows": list(
            args.signed_turnover_imbalance_windows
        ),
        "risk_adjusted_momentum_windows": list(args.risk_adjusted_momentum_windows),
        "volume_confirmed_momentum_windows": list(
            args.volume_confirmed_momentum_windows
        ),
        "return_turnover_correlation_windows": list(
            args.return_turnover_correlation_windows
        ),
        "negative_return_persistence_windows": list(
            args.negative_return_persistence_windows
        ),
        "sell_pressure_absorption_windows": list(args.sell_pressure_absorption_windows),
        "downside_turnover_decay_windows": list(args.downside_turnover_decay_windows),
        "sell_pressure_recovery_windows": list(args.sell_pressure_recovery_windows),
        "sell_pressure_exhaustion_windows": list(args.sell_pressure_exhaustion_windows),
        "sell_pressure_exhaustion_persistence_specs": [
            list(spec) for spec in args.sell_pressure_exhaustion_persistence_specs
        ],
        "microstructure_recovery_windows": list(args.microstructure_recovery_windows),
        "microstructure_exhaustion_windows": list(
            args.microstructure_exhaustion_windows
        ),
        "microstructure_recovery_acceleration_specs": [
            list(spec) for spec in args.microstructure_recovery_acceleration_specs
        ],
        "same_slot_memory_windows": list(args.same_slot_memory_windows),
        "weak_tape_gap_windows": list(args.weak_tape_gap_windows),
        "sell_pressure_quality_windows": list(args.sell_pressure_quality_windows),
        "event_shock_windows": list(args.event_shock_windows),
        "daily_moving_average_windows": list(args.daily_moving_average_windows),
        "daily_moving_average_pairs": [
            list(pair) for pair in args.daily_moving_average_pairs
        ],
        "market_downside_beta_windows": list(args.market_downside_beta_windows),
        "market_state_windows": list(args.market_state_windows),
        "breadth_resilience_windows": list(args.breadth_resilience_windows),
        "breadth_shock_residual_resilience_windows": list(
            args.breadth_shock_residual_resilience_windows
        ),
        "limit_pressure_resilience_windows": list(
            args.limit_pressure_resilience_windows
        ),
        "regime_conditioned_lookback_bars": list(
            args.regime_conditioned_lookback_bars
        ),
        "regime_conditioned_state_windows": list(
            args.regime_conditioned_state_windows
        ),
        "label_name": args.label_name,
        "horizon_bars": list(args.horizon_bars),
        "label_columns": _label_column_names(args),
        "entry_lag_bars": args.entry_lag_bars,
        "research_grid": _research_grid(args),
        "exclude_st": args.exclude_st,
        "limit_up_bps": args.limit_up_bps,
        "limit_down_bps": args.limit_down_bps,
        "filter_entry_tradable": args.filter_entry_tradable,
        "filter_entry_limit_up": args.filter_entry_limit_up,
        "max_symbols": args.max_symbols,
        "universe": {
            "market": "CN",
            "asset_type": "equity",
            "board": "main_board",
        },
        "partition": args.partition,
        "padding_days": args.padding_days,
        "workers": args.workers,
        "effective_workers": _effective_workers(args),
        "worker_memory_estimate_gb": args.worker_memory_estimate_gb,
        "memory_budget_gb": args.memory_budget_gb,
        "warmup_window": {
            "padding_days": args.padding_days,
            "max_lookback_bars": max(
                [
                    *args.lookback_bars,
                    *args.cross_sectional_reversal_lookback_bars,
                    *args.conditional_reversal_lookback_bars,
                    *args.conditional_reversal_state_windows,
                    *args.eod_reversal_lookback_bars,
                    *args.momentum_lookback_bars,
                    *args.volatility_windows,
                    *[
                        window
                        for spec in args.volatility_state_change_specs
                        for window in spec
                    ],
                    *args.price_position_windows,
                    *args.range_volatility_windows,
                    *args.efficiency_windows,
                    *args.volume_windows,
                    *args.volume_distribution_windows,
                    *args.turnover_windows,
                    *args.turnover_stability_windows,
                    *args.liquidity_reliability_windows,
                    *[
                        window
                        for spec in args.liquidity_reliability_recovery_specs
                        for window in spec
                    ],
                    *[
                        window
                        for spec in args.liquidity_reliability_recovery_balance_specs
                        for window in spec
                    ],
                    *args.vwap_deviation_windows,
                    *args.downside_volatility_windows,
                    *args.return_skewness_windows,
                    *args.lottery_max_windows,
                    *args.money_flow_windows,
                    *args.signed_turnover_imbalance_windows,
                    *args.risk_adjusted_momentum_windows,
                    *args.volume_confirmed_momentum_windows,
                    *args.return_turnover_correlation_windows,
                    *args.negative_return_persistence_windows,
                    *args.sell_pressure_absorption_windows,
                    *args.downside_turnover_decay_windows,
                    *args.sell_pressure_recovery_windows,
                    *args.sell_pressure_exhaustion_windows,
                    *[
                        window
                        for spec in args.sell_pressure_exhaustion_persistence_specs
                        for window in spec
                    ],
                    *args.microstructure_recovery_windows,
                    *[
                        window
                        for spec in args.microstructure_recovery_acceleration_specs
                        for window in spec
                    ],
                    *(window * 48 for window in args.same_slot_memory_windows),
                    *args.weak_tape_gap_windows,
                    *args.sell_pressure_quality_windows,
                    *(window * 2 for window in args.event_shock_windows),
                    *(window * 48 for window in args.daily_moving_average_windows),
                    *args.market_downside_beta_windows,
                    *args.market_state_windows,
                    *args.breadth_resilience_windows,
                    *args.breadth_shock_residual_resilience_windows,
                    *args.limit_pressure_resilience_windows,
                    *args.regime_conditioned_lookback_bars,
                    *args.regime_conditioned_state_windows,
                ]
            ),
        },
        "entry_exit_assumption": {
            "entry_lag_bars": args.entry_lag_bars,
            "horizon_bars": list(args.horizon_bars),
            "entry_price": "close_price",
            "exit_price": "close_price",
        },
    }


def _label_configs(args: argparse.Namespace) -> tuple[ForwardReturnLabelConfig, ...]:
    horizons = tuple(dict.fromkeys(int(value) for value in args.horizon_bars))
    if len(horizons) == 1:
        names = (args.label_name,)
    else:
        names = tuple(f"{args.label_name}_{horizon}b" for horizon in horizons)
    return tuple(
        ForwardReturnLabelConfig(
            name=name,
            horizon_bars=horizon,
            entry_lag_bars=args.entry_lag_bars,
        )
        for name, horizon in zip(names, horizons, strict=True)
    )


def _label_column_names(args: argparse.Namespace) -> list[str]:
    return [
        column
        for config in _label_configs(args)
        for column in (config.name, f"{config.name}_rank")
    ]


def _sell_pressure_exhaustion_persistence_spec(value: str) -> tuple[int, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected long:short:medium, for example 96:24:48"
        )
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sell-pressure exhaustion persistence spec values must be integers"
        ) from exc


def _liquidity_reliability_recovery_spec(value: str) -> tuple[int, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected long:capacity:recovery, for example 48:12:24"
        )
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "liquidity reliability recovery spec values must be integers"
        ) from exc


def _two_window_spec(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected short:long, for example 12:48")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "two-window spec values must be integers"
        ) from exc


def _daily_moving_average_pair(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected short:long, for example 5:20")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "daily moving-average pair values must be integers"
        ) from exc


def _feature_columns(
    dataset: pd.DataFrame,
    label_configs: tuple[ForwardReturnLabelConfig, ...],
) -> tuple[str, ...]:
    primary_label = label_configs[0].name
    exclude_columns = []
    for config in label_configs:
        exclude_columns.extend(
            [
                config.name,
                f"{config.name}_rank",
                f"{config.name}_exit_timestamp",
                f"{config.name}_exit_price",
            ]
        )
    return infer_feature_columns(
        dataset,
        label_column=primary_label,
        exclude_columns=tuple(exclude_columns),
    )


def _research_grid(args: argparse.Namespace) -> dict[str, object]:
    return {
        "observation_frequency": "5m",
        "execution_frequency": "5m",
        "feature_frequency": "5m",
        "label_frequency": "5m",
        "label_horizon_bars": list(
            dict.fromkeys(int(value) for value in args.horizon_bars)
        ),
        "entry_lag_bars": args.entry_lag_bars,
        "note": (
            "5m is the current observation and execution grid; labels may span "
            "multiple horizons so holding-period research is not fixed to one day."
        ),
    }


def _effective_workers(args: argparse.Namespace) -> int:
    if args.workers <= 1:
        return args.workers
    budget = _effective_memory_budget_gb(args)
    allowed = max(1, int(budget // args.worker_memory_estimate_gb))
    return min(args.workers, allowed)


def _effective_memory_budget_gb(args: argparse.Namespace) -> float:
    if args.memory_budget_gb > 0:
        return args.memory_budget_gb
    available = _available_memory_gb()
    if available is None:
        return args.worker_memory_estimate_gb
    return max(
        min(available * 0.60, available - 2.0),
        args.worker_memory_estimate_gb,
    )


def _available_memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0 / 1024.0
    return None


if __name__ == "__main__":
    main()
