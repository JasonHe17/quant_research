"""Build lagged rolling gross-exposure schedules for score policies."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.portfolio import (
    RollingRegimeGateConfig,
    build_rolling_regime_gate,
)


def main() -> None:
    args = _parse_args()
    summary = build_policy_regime_gate(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_policy_regime_gate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = _score_health_observations(args)
    config = RollingRegimeGateConfig(
        lookback_windows=args.lookback_windows,
        min_periods=args.min_periods,
        label_lag_windows=args.label_lag_windows,
        state_confirmation_windows=args.state_confirmation_windows,
        max_scale_change_per_window=args.max_scale_change_per_window,
        max_scale_increase_per_window=args.max_scale_increase_per_window,
        max_scale_decrease_per_window=args.max_scale_decrease_per_window,
        scale_change_deadband=args.scale_change_deadband,
        gate_mode=args.gate_mode,
        full_scale=args.full_scale,
        reduced_scale=args.reduced_scale,
        blocked_scale=args.blocked_scale,
        warmup_scale=args.warmup_scale,
        budget_min_scale=args.budget_min_scale,
        budget_max_scale=args.budget_max_scale,
        budget_top_return_floor=args.budget_top_return_floor,
        budget_top_return_ceiling=args.budget_top_return_ceiling,
        budget_spread_floor=args.budget_spread_floor,
        budget_spread_ceiling=args.budget_spread_ceiling,
        budget_rank_ic_floor=args.budget_rank_ic_floor,
        budget_rank_ic_ceiling=args.budget_rank_ic_ceiling,
        min_top_return=args.min_top_return,
        min_spread=args.min_spread,
        min_rank_ic=args.min_rank_ic,
        block_top_return=args.block_top_return,
        block_spread=args.block_spread,
        block_rank_ic=args.block_rank_ic,
    )
    schedule = build_rolling_regime_gate(observations, config)
    observations_path = output_dir / "regime_gate_observations.csv"
    schedule_path = output_dir / "gross_exposure_schedule.csv"
    summary_path = output_dir / "summary.json"
    observations.to_csv(observations_path, index=False)
    schedule.to_csv(schedule_path, index=False)
    summary = {
        "params": {
            "dataset_dir": args.dataset_dir,
            "scores_path": args.scores_path,
            "label_column": args.label_column,
            "top_n": args.top_n,
            "lookback_windows": args.lookback_windows,
            "min_periods": args.min_periods,
            "label_lag_windows": args.label_lag_windows,
            "state_confirmation_windows": args.state_confirmation_windows,
            "max_scale_change_per_window": args.max_scale_change_per_window,
            "max_scale_increase_per_window": args.max_scale_increase_per_window,
            "max_scale_decrease_per_window": args.max_scale_decrease_per_window,
            "scale_change_deadband": args.scale_change_deadband,
            "gate_mode": args.gate_mode,
            "full_scale": args.full_scale,
            "reduced_scale": args.reduced_scale,
            "blocked_scale": args.blocked_scale,
            "warmup_scale": args.warmup_scale,
            "budget_min_scale": args.budget_min_scale,
            "budget_max_scale": args.budget_max_scale,
            "budget_top_return_floor": args.budget_top_return_floor,
            "budget_top_return_ceiling": args.budget_top_return_ceiling,
            "budget_spread_floor": args.budget_spread_floor,
            "budget_spread_ceiling": args.budget_spread_ceiling,
            "budget_rank_ic_floor": args.budget_rank_ic_floor,
            "budget_rank_ic_ceiling": args.budget_rank_ic_ceiling,
            "min_top_return": args.min_top_return,
            "min_spread": args.min_spread,
            "min_rank_ic": args.min_rank_ic,
            "block_top_return": args.block_top_return,
            "block_spread": args.block_spread,
            "block_rank_ic": args.block_rank_ic,
            "partition_start": args.partition_start,
            "partition_end": args.partition_end,
            "max_partitions": args.max_partitions,
        },
        "observation_count": int(len(observations)),
        "schedule_count": int(len(schedule)),
        "scale_counts": _value_counts(schedule, "gross_exposure_scale"),
        "scale_summary": _scale_summary(schedule),
        "reason_counts": _value_counts(schedule, "gate_reason"),
        "raw_reason_counts": _value_counts(schedule, "raw_gate_reason"),
        "scale_step_limited_count": int(
            schedule["scale_step_limited"].sum()
        ) if "scale_step_limited" in schedule.columns else 0,
        "scale_deadband_held_count": int(
            schedule["scale_deadband_held"].sum()
        ) if "scale_deadband_held" in schedule.columns else 0,
        "artifacts": {
            "observations": str(observations_path),
            "schedule": str(schedule_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _score_health_observations(args: argparse.Namespace) -> pd.DataFrame:
    score_paths = _score_paths(args.scores_path)
    rows: list[dict[str, Any]] = []
    for dataset_path in _dataset_paths(args):
        partition = dataset_path.stem.removeprefix("dataset_")
        score_path = score_paths.get(partition)
        if score_path is None:
            raise FileNotFoundError(f"missing score partition for dataset_{partition}")
        rows.extend(
            _partition_score_health(
                dataset_path,
                score_path=score_path,
                top_n=args.top_n,
                label_column=args.label_column,
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "score_rank_ic",
                "score_top_n_mean_label",
                "score_bottom_n_mean_label",
                "score_top_minus_bottom_label",
                "market_mean_label",
                "sample_count",
            ]
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _partition_score_health(
    dataset_path: Path,
    *,
    score_path: Path,
    top_n: int,
    label_column: str,
) -> list[dict[str, Any]]:
    if not label_column:
        raise ValueError("label_column must be non-empty")
    dataset = pd.read_parquet(
        dataset_path,
        columns=["timestamp", "instrument_id", label_column],
    )
    scores = pd.read_parquet(score_path, columns=["timestamp", "instrument_id", "score"])
    frame = dataset.merge(scores, on=["timestamp", "instrument_id"], how="inner")
    rows = []
    for timestamp, group in frame.groupby("timestamp", sort=True):
        valid = group.dropna(subset=["score", label_column])
        n = min(top_n, len(valid))
        top = valid.nlargest(n, "score") if n else valid
        bottom = valid.nsmallest(n, "score") if n else valid
        rows.append(
            {
                "timestamp": timestamp,
                "score_rank_ic": _correlation(valid["score"], valid[label_column]),
                "score_top_n_mean_label": _mean(top[label_column]),
                "score_bottom_n_mean_label": _mean(bottom[label_column]),
                "score_top_minus_bottom_label": (
                    _mean(top[label_column]) - _mean(bottom[label_column])
                ),
                "market_mean_label": _mean(valid[label_column]),
                "sample_count": int(len(valid)),
            }
        )
    return rows


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(Path(args.dataset_dir).glob("dataset_*.parquet"))
    if args.partition_start:
        paths = [path for path in paths if _partition_name(path) >= args.partition_start]
    if args.partition_end:
        paths = [path for path in paths if _partition_name(path) <= args.partition_end]
    if args.max_partitions is not None:
        paths = paths[: args.max_partitions]
    if not paths:
        raise FileNotFoundError(f"no dataset_*.parquet files found under {args.dataset_dir}")
    return paths


def _score_paths(scores_path: str) -> dict[str, Path]:
    paths = [Path(path) for path in glob.glob(scores_path)]
    path = Path(scores_path)
    if path.is_dir():
        paths = sorted(path.glob("score_*.parquet"))
    elif path.is_file():
        paths = [path]
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"no score parquet files found for {scores_path}")
    output: dict[str, Path] = {}
    for score_path in paths:
        partition = score_path.stem.removeprefix("score_")
        output[partition] = score_path
    return output


def _partition_name(path: Path) -> str:
    return path.stem.removeprefix("dataset_")


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    series = frame[column]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and numeric.nunique(dropna=False) > 20:
        bins = pd.cut(
            numeric,
            bins=[-0.000001, 0.25, 0.50, 0.75, 1.0],
            labels=["[0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1.0]"],
            include_lowest=True,
        )
        counts = bins.value_counts(dropna=False).sort_index()
        return {str(key): int(value) for key, value in counts.items()}
    counts = series.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _scale_summary(schedule: pd.DataFrame) -> dict[str, float | int]:
    if schedule.empty or "gross_exposure_scale" not in schedule.columns:
        return {}
    scale = pd.to_numeric(schedule["gross_exposure_scale"], errors="coerce").dropna()
    if scale.empty:
        return {}
    return {
        "min": float(scale.min()),
        "mean": float(scale.mean()),
        "median": float(scale.median()),
        "max": float(scale.max()),
        "unique_count": int(scale.nunique()),
    }


def _mean(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").mean())


def _correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 2:
        return float("nan")
    return float(pair["left"].corr(pair["right"], method="spearman"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="runs/framework_v1_acceptance/standard/alpha_dataset",
    )
    parser.add_argument("--scores-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--lookback-windows", type=int, default=20)
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--label-lag-windows", type=int)
    parser.add_argument("--state-confirmation-windows", type=int, default=1)
    parser.add_argument("--max-scale-change-per-window", type=float)
    parser.add_argument("--max-scale-increase-per-window", type=float)
    parser.add_argument("--max-scale-decrease-per-window", type=float)
    parser.add_argument("--scale-change-deadband", type=float, default=0.0)
    parser.add_argument("--gate-mode", choices=("threshold", "budget"), default="threshold")
    parser.add_argument("--full-scale", type=float, default=1.0)
    parser.add_argument("--reduced-scale", type=float, default=0.5)
    parser.add_argument("--blocked-scale", type=float, default=0.0)
    parser.add_argument("--warmup-scale", type=float, default=1.0)
    parser.add_argument("--budget-min-scale", type=float, default=0.25)
    parser.add_argument("--budget-max-scale", type=float, default=1.0)
    parser.add_argument("--budget-top-return-floor", type=float, default=-0.001)
    parser.add_argument("--budget-top-return-ceiling", type=float, default=0.001)
    parser.add_argument("--budget-spread-floor", type=float, default=-0.001)
    parser.add_argument("--budget-spread-ceiling", type=float, default=0.001)
    parser.add_argument("--budget-rank-ic-floor", type=float, default=-0.05)
    parser.add_argument("--budget-rank-ic-ceiling", type=float, default=0.05)
    parser.add_argument("--min-top-return", type=float, default=0.0)
    parser.add_argument("--min-spread", type=float, default=0.0)
    parser.add_argument("--min-rank-ic", type=float, default=0.0)
    parser.add_argument("--block-top-return", type=float, default=-0.001)
    parser.add_argument("--block-spread", type=float, default=-0.001)
    parser.add_argument("--block-rank-ic", type=float, default=-0.05)
    parser.add_argument("--partition-start")
    parser.add_argument("--partition-end")
    parser.add_argument("--max-partitions", type=int)
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if not args.label_column:
        raise ValueError("--label-column must be non-empty")
    if args.max_partitions is not None and args.max_partitions <= 0:
        raise ValueError("--max-partitions must be positive")
    if args.partition_start and args.partition_end and args.partition_start > args.partition_end:
        raise ValueError("--partition-start must not be after --partition-end")
    if args.label_lag_windows is None:
        args.label_lag_windows = _default_label_lag_windows(args.label_column)
    RollingRegimeGateConfig(
        lookback_windows=args.lookback_windows,
        min_periods=args.min_periods,
        label_lag_windows=args.label_lag_windows,
        state_confirmation_windows=args.state_confirmation_windows,
        max_scale_change_per_window=args.max_scale_change_per_window,
        max_scale_increase_per_window=args.max_scale_increase_per_window,
        max_scale_decrease_per_window=args.max_scale_decrease_per_window,
        scale_change_deadband=args.scale_change_deadband,
        gate_mode=args.gate_mode,
        full_scale=args.full_scale,
        reduced_scale=args.reduced_scale,
        blocked_scale=args.blocked_scale,
        warmup_scale=args.warmup_scale,
        budget_min_scale=args.budget_min_scale,
        budget_max_scale=args.budget_max_scale,
        budget_top_return_floor=args.budget_top_return_floor,
        budget_top_return_ceiling=args.budget_top_return_ceiling,
        budget_spread_floor=args.budget_spread_floor,
        budget_spread_ceiling=args.budget_spread_ceiling,
        budget_rank_ic_floor=args.budget_rank_ic_floor,
        budget_rank_ic_ceiling=args.budget_rank_ic_ceiling,
        min_top_return=args.min_top_return,
        min_spread=args.min_spread,
        min_rank_ic=args.min_rank_ic,
        block_top_return=args.block_top_return,
        block_spread=args.block_spread,
        block_rank_ic=args.block_rank_ic,
    )
    return args


def _default_label_lag_windows(label_column: str) -> int:
    suffix = label_column.rsplit("_", 1)[-1]
    if suffix.endswith("b") and suffix[:-1].isdigit():
        return int(suffix[:-1])
    return 48


if __name__ == "__main__":
    main()
