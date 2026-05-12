"""Evaluate single alpha features in supervised alpha datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import (
    SingleFactorEvaluationConfig,
    evaluate_single_factors,
)
from quant_research.models import load_supervised_partitions


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = _dataset_paths(args)
    frame = load_supervised_partitions(dataset_paths)
    config = SingleFactorEvaluationConfig(
        label_column=args.label_column,
        feature_columns=tuple(args.feature_columns or ()),
        top_n=args.top_n,
        quantiles=args.quantiles,
        correlation_method=args.correlation_method,
    )
    result = evaluate_single_factors(frame, config)
    summary_path = output_dir / "single_factor_summary.csv"
    by_timestamp_path = output_dir / "single_factor_by_timestamp.csv"
    quantile_path = output_dir / "single_factor_quantiles.csv"
    correlation_path = output_dir / "feature_correlation.csv"
    result.summary.to_csv(summary_path, index=False)
    result.by_timestamp.to_csv(by_timestamp_path, index=False)
    result.quantile_returns.to_csv(quantile_path, index=False)
    result.feature_correlation.to_csv(correlation_path)
    payload = {
        "params": {
            "dataset_paths": [str(path) for path in dataset_paths],
            "label_column": args.label_column,
            "feature_columns": args.feature_columns,
            "top_n": args.top_n,
            "quantiles": args.quantiles,
            "correlation_method": args.correlation_method,
        },
        "artifacts": {
            "summary": str(summary_path),
            "by_timestamp": str(by_timestamp_path),
            "quantiles": str(quantile_path),
            "feature_correlation": str(correlation_path),
        },
        "summary": result.summary.to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result.summary.to_string(index=False))


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.dataset_paths:
        return [Path(path) for path in args.dataset_paths]
    dataset_dir = Path(args.dataset_dir)
    paths = sorted(dataset_dir.glob("dataset_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no dataset_*.parquet files found under {dataset_dir}")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir")
    parser.add_argument("--dataset-paths", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--feature-columns", nargs="+")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument(
        "--correlation-method",
        choices=("pearson", "spearman"),
        default="spearman",
    )
    args = parser.parse_args()
    if bool(args.dataset_dir) == bool(args.dataset_paths):
        raise ValueError("provide exactly one of --dataset-dir or --dataset-paths")
    return args


if __name__ == "__main__":
    main()
