"""Build Baseline A alpha features and forward-return labels from real 5m data."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.datasets import (
    ForwardReturnLabelConfig,
    add_cross_sectional_label_rank,
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
    for file_path in files:
        bars = _load_bars_from_files(params, [file_path])
        if bars.empty:
            continue
        year = file_path.stem.rsplit("__", maxsplit=1)[-1]
        print(f"loaded {year}: bars={len(bars)}", flush=True)
        features = _build_reversal_feature_matrix(bars, args.lookback_bars)
        labels = build_forward_return_labels(bars, label_config)
        labels = add_cross_sectional_label_rank(
            labels,
            label_column=args.label_name,
            rank_column=f"{args.label_name}_rank",
        )
        dataset = join_alpha_features_and_labels(features, labels)
        dataset_path = output_dir / f"dataset_{year}.parquet"
        dataset.to_parquet(dataset_path, index=False)
        feature_path = None
        label_path = None
        if args.write_components:
            feature_path = output_dir / f"features_{year}.parquet"
            label_path = output_dir / f"labels_{year}.parquet"
            features.to_parquet(feature_path, index=False)
            labels.to_parquet(label_path, index=False)
        row = {
            "year": year,
            "bar_count": len(bars),
            "feature_row_count": len(features),
            "label_row_count": len(labels),
            "dataset_row_count": len(dataset),
            "instrument_count": int(bars["instrument_id"].nunique()),
            "dataset_path": str(dataset_path),
            "features_path": str(feature_path) if feature_path is not None else None,
            "labels_path": str(label_path) if label_path is not None else None,
        }
        rows.append(row)
        _write_summary(output_dir, args, rows)
        del bars, features, labels, dataset
        gc.collect()
    print(pd.DataFrame(rows).to_string(index=False))


def _build_reversal_feature_matrix(
    bars: pd.DataFrame,
    lookback_bars_values: list[int],
) -> pd.DataFrame:
    frame = bars.sort_values(["instrument_id", "bar_end_time"]).copy()
    frame["close_price"] = frame["close_price"].astype(float)
    grouped = frame.groupby("instrument_id", sort=False)
    output = frame.loc[:, ["bar_end_time", "instrument_id"]].rename(
        columns={"bar_end_time": "timestamp"}
    )
    for lookback_bars in lookback_bars_values:
        factor_name = f"intraday_reversal_5m_lb{lookback_bars}"
        output[factor_name] = -grouped["close_price"].pct_change(
            periods=lookback_bars
        )
    feature_columns = [
        f"intraday_reversal_5m_lb{lookback_bars}"
        for lookback_bars in lookback_bars_values
    ]
    return output.loc[output[feature_columns].notna().any(axis=1)].reset_index(
        drop=True
    )


def _write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> None:
    payload = {
        "params": {
            "start": args.start,
            "end": args.end,
            "lookback_bars": args.lookback_bars,
            "label_name": args.label_name,
            "horizon_bars": args.horizon_bars,
            "entry_lag_bars": args.entry_lag_bars,
            "max_symbols": args.max_symbols,
            "write_components": args.write_components,
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
    parser.add_argument("--lookback-bars", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--label-name", default="forward_return")
    parser.add_argument("--horizon-bars", type=int, default=48)
    parser.add_argument("--entry-lag-bars", type=int, default=1)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--write-components", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in args.lookback_bars):
        raise ValueError("--lookback-bars values must be positive")
    return args


if __name__ == "__main__":
    main()
