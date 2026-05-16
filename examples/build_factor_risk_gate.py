"""Build a gross-exposure schedule from a market-wide factor risk proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    args = _parse_args()
    summary = build_factor_risk_gate(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_factor_risk_gate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = _factor_risk_observations(args)
    schedule = _factor_risk_schedule(observations, args)
    if args.base_schedule:
        schedule = _combine_with_base_schedule(schedule, Path(args.base_schedule))
    observations_path = output_dir / "factor_risk_observations.csv"
    schedule_path = output_dir / "gross_exposure_schedule.csv"
    summary_path = output_dir / "summary.json"
    observations.to_csv(observations_path, index=False)
    schedule.to_csv(schedule_path, index=False)
    summary = {
        "params": {
            "dataset_dir": args.dataset_dir,
            "feature": args.feature,
            "aggregate": args.aggregate,
            "lookback_windows": args.lookback_windows,
            "min_periods": args.min_periods,
            "high_quantile": args.high_quantile,
            "extreme_quantile": args.extreme_quantile,
            "full_scale": args.full_scale,
            "reduced_scale": args.reduced_scale,
            "blocked_scale": args.blocked_scale,
            "warmup_scale": args.warmup_scale,
            "base_schedule": args.base_schedule,
            "combine_mode": args.combine_mode,
            "partition_start": args.partition_start,
            "partition_end": args.partition_end,
            "max_partitions": args.max_partitions,
        },
        "observation_count": int(len(observations)),
        "schedule_count": int(len(schedule)),
        "scale_counts": _value_counts(schedule, "gross_exposure_scale", max_values=20),
        "risk_state_counts": _value_counts(schedule, "risk_state"),
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


def _factor_risk_observations(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_path in _dataset_paths(args):
        frame = pd.read_parquet(
            dataset_path,
            columns=["timestamp", args.feature],
        )
        grouped = frame.dropna(subset=[args.feature]).groupby("timestamp", sort=True)[
            args.feature
        ]
        if args.aggregate == "mean":
            values = grouped.mean()
        elif args.aggregate == "median":
            values = grouped.median()
        else:
            values = grouped.quantile(args.aggregate_quantile)
        counts = grouped.count()
        rows.extend(
            {
                "timestamp": timestamp,
                "risk_value": float(value),
                "sample_count": int(counts.loc[timestamp]),
            }
            for timestamp, value in values.items()
        )
        del frame
    if not rows:
        return pd.DataFrame(columns=["timestamp", "risk_value", "sample_count"])
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _factor_risk_schedule(
    observations: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "gross_exposure_scale",
                "risk_state",
                "risk_value",
                "rolling_high_threshold",
                "rolling_extreme_threshold",
                "rolling_observation_count",
            ]
        )
    output = observations.copy()
    risk = pd.to_numeric(output["risk_value"], errors="coerce")
    historical = risk.shift(1)
    rolling = historical.rolling(
        window=args.lookback_windows,
        min_periods=args.min_periods,
    )
    high_threshold = rolling.quantile(args.high_quantile)
    extreme_threshold = rolling.quantile(args.extreme_quantile)
    observation_count = (
        historical.rolling(window=args.lookback_windows, min_periods=1).count().astype(int)
    )
    output["rolling_high_threshold"] = high_threshold
    output["rolling_extreme_threshold"] = extreme_threshold
    output["rolling_observation_count"] = observation_count
    output["risk_state"] = "warmup"
    output["gross_exposure_scale"] = args.warmup_scale
    ready = observation_count >= args.min_periods
    full = ready & (risk <= high_threshold)
    reduced = ready & (risk > high_threshold) & (risk <= extreme_threshold)
    blocked = ready & (risk > extreme_threshold)
    output.loc[full, "risk_state"] = "full"
    output.loc[full, "gross_exposure_scale"] = args.full_scale
    output.loc[reduced, "risk_state"] = "reduced"
    output.loc[reduced, "gross_exposure_scale"] = args.reduced_scale
    output.loc[blocked, "risk_state"] = "blocked"
    output.loc[blocked, "gross_exposure_scale"] = args.blocked_scale
    return output


def _combine_with_base_schedule(schedule: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    if base_path.suffix == ".parquet":
        base = pd.read_parquet(base_path)
    else:
        base = pd.read_csv(base_path)
    if "gross_exposure_scale" not in base.columns:
        if "policy_gross_exposure_scale" in base.columns:
            base = base.rename(
                columns={"policy_gross_exposure_scale": "base_gross_exposure_scale"}
            )
        else:
            raise ValueError("base schedule must contain gross_exposure_scale")
    else:
        base = base.rename(columns={"gross_exposure_scale": "base_gross_exposure_scale"})
    if "timestamp" not in base.columns:
        raise ValueError("base schedule must contain timestamp")
    joined = schedule.merge(
        base.loc[:, ["timestamp", "base_gross_exposure_scale"]],
        on="timestamp",
        how="left",
    )
    joined["base_gross_exposure_scale"] = joined["base_gross_exposure_scale"].fillna(1.0)
    joined["factor_gross_exposure_scale"] = joined["gross_exposure_scale"]
    joined["gross_exposure_scale"] = joined[
        ["factor_gross_exposure_scale", "base_gross_exposure_scale"]
    ].min(axis=1)
    return joined


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


def _partition_name(path: Path) -> str:
    return path.stem.removeprefix("dataset_")


def _value_counts(
    frame: pd.DataFrame,
    column: str,
    *,
    max_values: int | None = None,
) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].value_counts(dropna=False)
    if max_values is not None and len(counts) > max_values:
        top_counts = counts.head(max_values)
        result = {str(key): int(value) for key, value in top_counts.items()}
        result["__other__"] = int(counts.iloc[max_values:].sum())
        return result
    return {str(key): int(value) for key, value in counts.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--aggregate",
        choices=("mean", "median", "quantile"),
        default="mean",
    )
    parser.add_argument("--aggregate-quantile", type=float, default=0.75)
    parser.add_argument("--lookback-windows", type=int, default=240)
    parser.add_argument("--min-periods", type=int, default=48)
    parser.add_argument("--high-quantile", type=float, default=0.80)
    parser.add_argument("--extreme-quantile", type=float, default=0.95)
    parser.add_argument("--full-scale", type=float, default=1.0)
    parser.add_argument("--reduced-scale", type=float, default=0.5)
    parser.add_argument("--blocked-scale", type=float, default=0.0)
    parser.add_argument("--warmup-scale", type=float, default=1.0)
    parser.add_argument("--base-schedule")
    parser.add_argument(
        "--combine-mode",
        choices=("min",),
        default="min",
        help="how to combine factor risk scale with base schedule",
    )
    parser.add_argument("--partition-start")
    parser.add_argument("--partition-end")
    parser.add_argument("--max-partitions", type=int)
    args = parser.parse_args()
    if args.max_partitions is not None and args.max_partitions <= 0:
        raise ValueError("--max-partitions must be positive")
    if args.partition_start and args.partition_end and args.partition_start > args.partition_end:
        raise ValueError("--partition-start must not be after --partition-end")
    if not 0 < args.aggregate_quantile < 1:
        raise ValueError("--aggregate-quantile must be in (0, 1)")
    if args.lookback_windows <= 0:
        raise ValueError("--lookback-windows must be positive")
    if args.min_periods <= 0:
        raise ValueError("--min-periods must be positive")
    if args.min_periods > args.lookback_windows:
        raise ValueError("--min-periods must be <= --lookback-windows")
    if not 0 < args.high_quantile < args.extreme_quantile < 1:
        raise ValueError("--high-quantile and --extreme-quantile must satisfy 0 < high < extreme < 1")
    for name in ("full_scale", "reduced_scale", "blocked_scale", "warmup_scale"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


if __name__ == "__main__":
    main()
