"""Run expanding-window tree-model training and T+1 score backtests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    name: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str


DEFAULT_FOLDS: tuple[WalkForwardFold, ...] = (
    WalkForwardFold(
        name="test_2024q4",
        train_end="2024-06-30T15:00:00+08:00",
        valid_start="2024-07-01T09:35:00+08:00",
        valid_end="2024-09-30T15:00:00+08:00",
        test_start="2024-10-01T09:35:00+08:00",
        test_end="2024-12-31T15:00:00+08:00",
    ),
    WalkForwardFold(
        name="test_2025q1",
        train_end="2024-09-30T15:00:00+08:00",
        valid_start="2024-10-01T09:35:00+08:00",
        valid_end="2024-12-31T15:00:00+08:00",
        test_start="2025-01-01T09:35:00+08:00",
        test_end="2025-03-31T15:00:00+08:00",
    ),
    WalkForwardFold(
        name="test_2025q2",
        train_end="2024-12-31T15:00:00+08:00",
        valid_start="2025-01-01T09:35:00+08:00",
        valid_end="2025-03-31T15:00:00+08:00",
        test_start="2025-04-01T09:35:00+08:00",
        test_end="2025-06-30T15:00:00+08:00",
    ),
    WalkForwardFold(
        name="test_2025q3",
        train_end="2025-03-31T15:00:00+08:00",
        valid_start="2025-04-01T09:35:00+08:00",
        valid_end="2025-06-30T15:00:00+08:00",
        test_start="2025-07-01T09:35:00+08:00",
        test_end="2025-09-30T15:00:00+08:00",
    ),
    WalkForwardFold(
        name="test_2025q4",
        train_end="2025-06-30T15:00:00+08:00",
        valid_start="2025-07-01T09:35:00+08:00",
        valid_end="2025-09-30T15:00:00+08:00",
        test_start="2025-10-01T09:35:00+08:00",
        test_end="2025-12-31T15:00:00+08:00",
    ),
)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    folds = DEFAULT_FOLDS[: args.max_folds] if args.max_folds else DEFAULT_FOLDS
    for fold in folds:
        rows.append(_run_fold(fold, args=args, output_dir=output_dir))
        _write_summary(rows, args=args, output_dir=output_dir)
    _write_summary(rows, args=args, output_dir=output_dir)
    print(pd.DataFrame(rows).to_string(index=False))


def _run_fold(
    fold: WalkForwardFold,
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    fold_dir = output_dir / fold.name
    model_dir = fold_dir / "model"
    backtest_dir = fold_dir / "backtest"
    if args.skip_existing and (backtest_dir / "summary.json").exists():
        return _fold_row(fold, model_dir=model_dir, backtest_dir=backtest_dir)
    if fold_dir.exists() and args.force:
        shutil.rmtree(fold_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    backtest_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "run_baseline_a_tree_model.py"),
            "--dataset-dir",
            args.dataset_dir,
            "--output-dir",
            str(model_dir),
            "--train-end",
            fold.train_end,
            "--valid-start",
            fold.valid_start,
            "--valid-end",
            fold.valid_end,
            "--test-start",
            fold.test_start,
            "--test-end",
            fold.test_end,
            "--top-n",
            str(args.top_n),
            "--num-threads",
            str(args.num_threads),
            "--num-boost-round",
            str(args.num_boost_round),
            "--early-stopping-rounds",
            str(args.early_stopping_rounds),
        ],
    )
    _run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "run_tree_score_backtest.py"),
            "--predictions-path",
            str(model_dir / "predictions.parquet"),
            "--catalog-path",
            args.catalog_path,
            "--start",
            fold.test_start,
            "--end",
            fold.test_end,
            "--top-n",
            str(args.top_n),
            "--initial-cash",
            str(args.initial_cash),
            "--commission-bps",
            str(args.commission_bps),
            "--sell-stamp-tax-bps",
            str(args.sell_stamp_tax_bps),
            "--min-commission",
            str(args.min_commission),
            "--slippage-bps",
            str(args.slippage_bps),
            "--rebalance-every-n-bars",
            str(args.rebalance_every_n_bars),
            "--min-trade-weight",
            str(args.min_trade_weight),
            "--output-dir",
            str(backtest_dir),
            *(
                ["--hold-rank-buffer", str(args.hold_rank_buffer)]
                if args.hold_rank_buffer is not None
                else []
            ),
        ],
    )
    return _fold_row(fold, model_dir=model_dir, backtest_dir=backtest_dir)


def _run_command(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _fold_row(
    fold: WalkForwardFold,
    *,
    model_dir: Path,
    backtest_dir: Path,
) -> dict[str, Any]:
    model = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    backtest = json.loads((backtest_dir / "summary.json").read_text(encoding="utf-8"))
    model_metrics = model["metrics"]
    backtest_metrics = backtest["metrics"]
    return {
        "fold": fold.name,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_rows": model["split_rows"]["train"],
        "valid_rows": model["split_rows"]["valid"],
        "test_rows": model["split_rows"]["test"],
        "best_iteration": model["params"]["best_iteration"],
        "model_rank_ic": model_metrics.get("spearman_rank_ic_mean"),
        "model_pearson_ic": model_metrics.get("pearson_ic_mean"),
        "model_top_bottom": model_metrics.get("top_minus_bottom_label"),
        "total_return": backtest_metrics.get("total_return"),
        "max_drawdown": backtest_metrics.get("max_drawdown"),
        "final_equity": backtest_metrics.get("final_equity"),
        "trade_count": backtest_metrics.get("trade_count"),
        "gross_turnover": backtest_metrics.get("gross_turnover"),
        "total_transaction_cost": backtest_metrics.get("total_transaction_cost"),
        "signal_count": backtest.get("signal_count"),
        "bar_count": backtest.get("bar_count"),
    }


def _write_summary(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "walk_forward_summary.csv", index=False)
    aggregate = _aggregate(frame)
    payload = {
        "params": {
            "dataset_dir": args.dataset_dir,
            "catalog_path": args.catalog_path,
            "top_n": args.top_n,
            "initial_cash": args.initial_cash,
            "commission_bps": args.commission_bps,
            "sell_stamp_tax_bps": args.sell_stamp_tax_bps,
            "min_commission": args.min_commission,
            "slippage_bps": args.slippage_bps,
            "rebalance_every_n_bars": args.rebalance_every_n_bars,
            "hold_rank_buffer": args.hold_rank_buffer,
            "min_trade_weight": args.min_trade_weight,
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "num_threads": args.num_threads,
        },
        "folds": rows,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _aggregate(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {}
    return {
        "fold_count": int(len(frame)),
        "positive_return_count": int((frame["total_return"] > 0).sum()),
        "mean_total_return": _nullable_float(frame["total_return"].mean()),
        "median_total_return": _nullable_float(frame["total_return"].median()),
        "min_total_return": _nullable_float(frame["total_return"].min()),
        "max_total_return": _nullable_float(frame["total_return"].max()),
        "mean_max_drawdown": _nullable_float(frame["max_drawdown"].mean()),
        "min_max_drawdown": _nullable_float(frame["max_drawdown"].min()),
        "mean_gross_turnover": _nullable_float(frame["gross_turnover"].mean()),
        "mean_model_rank_ic": _nullable_float(frame["model_rank_ic"].mean()),
        "mean_model_top_bottom": _nullable_float(frame["model_top_bottom"].mean()),
    }


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="research_store/intraday_alpha_dataset_v2_2024_2025_monthly",
    )
    parser.add_argument(
        "--catalog-path",
        default="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--rebalance-every-n-bars", type=int, default=6)
    parser.add_argument("--hold-rank-buffer", type=int, default=100)
    parser.add_argument("--min-trade-weight", type=float, default=0.002)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.rebalance_every_n_bars <= 0:
        raise ValueError("--rebalance-every-n-bars must be positive")
    if args.hold_rank_buffer is not None and args.hold_rank_buffer < args.top_n:
        raise ValueError("--hold-rank-buffer must be greater than or equal to --top-n")
    if args.max_folds is not None and args.max_folds <= 0:
        raise ValueError("--max-folds must be positive")
    return args


if __name__ == "__main__":
    main()
