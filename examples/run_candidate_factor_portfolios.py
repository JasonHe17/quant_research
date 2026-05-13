"""Build and optionally backtest candidate-factor portfolio scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.portfolio import (
    factor_combination_weights,
    load_candidate_factors,
    write_score_partitions,
)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = _dataset_paths(args)
    candidates = load_candidate_factors(
        Path(args.admission_report),
        statuses=tuple(args.statuses),
    )
    correlation = _load_correlation(Path(args.factor_correlation))
    weights_by_method = {
        method: factor_combination_weights(
            candidates,
            method=method,
            correlation=correlation,
            ridge=args.decorrelation_ridge,
        )
        for method in args.methods
    }
    scores_summary = write_score_partitions(
        dataset_paths,
        output_dir=output_dir / "scores",
        candidates=candidates,
        weights_by_method=weights_by_method,
    )
    summary = {
        "params": {
            "dataset_dir": args.dataset_dir,
            "admission_report": args.admission_report,
            "factor_correlation": args.factor_correlation,
            "methods": args.methods,
            "statuses": args.statuses,
            "max_partitions": args.max_partitions,
        },
        **scores_summary,
    }
    if args.run_backtests:
        summary["backtests"] = _run_backtests(args, scores_summary=scores_summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(Path(args.dataset_dir).glob("dataset_*.parquet"))
    if args.max_partitions is not None:
        paths = paths[: args.max_partitions]
    if not paths:
        raise FileNotFoundError(f"no dataset_*.parquet files found under {args.dataset_dir}")
    return paths


def _load_correlation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def _run_backtests(
    args: argparse.Namespace,
    *,
    scores_summary: dict[str, object],
) -> dict[str, object]:
    if not args.start or not args.end:
        raise ValueError("--start and --end are required with --run-backtests")
    rows = {}
    methods = scores_summary["methods"]
    if not isinstance(methods, dict):
        raise ValueError("invalid score summary methods")
    for method, payload in methods.items():
        if not isinstance(payload, dict):
            continue
        backtest_dir = Path(args.output_dir) / "backtests" / method
        command = [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "run_tree_score_backtest.py"),
            "--predictions-path",
            str(payload["path"]),
            "--catalog-path",
            args.catalog_path,
            "--start",
            args.start,
            "--end",
            args.end,
            "--top-n",
            str(args.top_n),
            "--initial-cash",
            str(args.initial_cash),
            "--commission-bps",
            str(args.commission_bps),
            "--slippage-bps",
            str(args.slippage_bps),
            "--sell-stamp-tax-bps",
            str(args.sell_stamp_tax_bps),
            "--min-commission",
            str(args.min_commission),
            "--lot-size",
            str(args.lot_size),
            "--rebalance-every-n-bars",
            str(args.rebalance_every_n_bars),
            "--min-trade-weight",
            str(args.min_trade_weight),
            "--limit-up-bps",
            str(args.limit_up_bps),
            "--limit-down-bps",
            str(args.limit_down_bps),
            "--data-access-mode",
            args.data_access_mode,
            "--streaming-chunk",
            args.streaming_chunk,
            "--streaming-chunk-padding-days",
            str(args.streaming_chunk_padding_days),
            "--output-dir",
            str(backtest_dir),
        ]
        if args.hold_rank_buffer is not None:
            command.extend(["--hold-rank-buffer", str(args.hold_rank_buffer)])
        if args.exclude_st:
            command.append("--exclude-st")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        rows[method] = json.loads(
            (backtest_dir / "summary.json").read_text(encoding="utf-8")
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="runs/framework_v1_acceptance/standard/alpha_dataset",
    )
    parser.add_argument(
        "--admission-report",
        default=(
            "runs/framework_v1_acceptance/standard/factor_admission/"
            "factor_admission_report.json"
        ),
    )
    parser.add_argument(
        "--factor-correlation",
        default="runs/framework_v1_acceptance/standard/factor_evaluation/feature_correlation.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/candidate_factor_portfolios",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("equal", "ic_weighted", "decorrelated"),
        default=["equal", "ic_weighted", "decorrelated"],
    )
    parser.add_argument("--statuses", nargs="+", default=["candidate"])
    parser.add_argument("--max-partitions", type=int)
    parser.add_argument("--decorrelation-ridge", type=float, default=0.05)
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument(
        "--catalog-path",
        default="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    )
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--rebalance-every-n-bars", type=int, default=1)
    parser.add_argument("--hold-rank-buffer", type=int)
    parser.add_argument("--min-trade-weight", type=float, default=0.0005)
    parser.add_argument("--exclude-st", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-up-bps", type=float, default=980.0)
    parser.add_argument("--limit-down-bps", type=float, default=980.0)
    parser.add_argument(
        "--data-access-mode",
        choices=("data_portal", "fast_parquet"),
        default="fast_parquet",
    )
    parser.add_argument(
        "--streaming-chunk",
        choices=("year", "month"),
        default="month",
    )
    parser.add_argument("--streaming-chunk-padding-days", type=int, default=10)
    args = parser.parse_args()
    if args.max_partitions is not None and args.max_partitions <= 0:
        raise ValueError("--max-partitions must be positive")
    if args.decorrelation_ridge < 0:
        raise ValueError("--decorrelation-ridge must be non-negative")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.rebalance_every_n_bars <= 0:
        raise ValueError("--rebalance-every-n-bars must be positive")
    if args.hold_rank_buffer is not None and args.hold_rank_buffer < args.top_n:
        raise ValueError("--hold-rank-buffer must be greater than or equal to --top-n")
    if args.streaming_chunk_padding_days < 0:
        raise ValueError("--streaming-chunk-padding-days must be non-negative")
    return args


if __name__ == "__main__":
    main()
