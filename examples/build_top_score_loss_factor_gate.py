"""Build target-factor weight scales from lagged top-score basket losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    args = _parse_args()
    summary = build_top_score_loss_factor_gate(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_top_score_loss_factor_gate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = _load_diagnostics(Path(args.diagnostics_dir), args)
    states = _gate_states(diagnostics, args)
    schedule = _expand_schedule(states, tuple(args.target_features))

    diagnostics_path = output_dir / "top_score_loss_gate_observations.csv"
    schedule_path = output_dir / "factor_weight_scale_schedule.csv"
    summary_path = output_dir / "summary.json"
    states.to_csv(diagnostics_path, index=False)
    schedule.to_csv(schedule_path, index=False)
    summary = {
        "params": {
            "diagnostics_dir": args.diagnostics_dir,
            "target_features": args.target_features,
            "partition_start": args.partition_start,
            "partition_end": args.partition_end,
            "label_lag_windows": args.label_lag_windows,
            "lookback_windows": args.lookback_windows,
            "min_periods": args.min_periods,
            "loss_threshold": args.loss_threshold,
            "extreme_loss_threshold": args.extreme_loss_threshold,
            "full_scale": args.full_scale,
            "reduced_scale": args.reduced_scale,
            "blocked_scale": args.blocked_scale,
            "warmup_scale": args.warmup_scale,
        },
        "observation_count": int(len(states)),
        "schedule_count": int(len(schedule)),
        "scale_counts": _value_counts(schedule, "weight_scale"),
        "state_counts": _value_counts(states, "gate_state"),
        "feature_summary": (
            schedule.groupby("feature")["weight_scale"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
            .to_dict("records")
        ),
        "artifacts": {
            "observations": str(diagnostics_path),
            "schedule": str(schedule_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_diagnostics(diagnostics_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    paths = sorted(diagnostics_dir.glob("factor_contribution_*.csv"))
    if args.partition_start:
        paths = [path for path in paths if _partition(path) >= args.partition_start]
    if args.partition_end:
        paths = [path for path in paths if _partition(path) <= args.partition_end]
    if not paths:
        raise FileNotFoundError(f"no factor contribution diagnostics under {diagnostics_dir}")
    frames = [
        pd.read_csv(path, usecols=["timestamp", "top_score_mean_label"])
        for path in paths
    ]
    output = pd.concat(frames, ignore_index=True)
    output["timestamp"] = output["timestamp"].astype(str)
    output["top_score_mean_label"] = pd.to_numeric(
        output["top_score_mean_label"],
        errors="coerce",
    )
    output = output.dropna(subset=["top_score_mean_label"])
    return output.sort_values("timestamp").reset_index(drop=True)


def _partition(path: Path) -> str:
    return path.stem.removeprefix("factor_contribution_")


def _gate_states(diagnostics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    output = diagnostics.copy()
    labels = output["top_score_mean_label"]
    matured = labels.shift(args.label_lag_windows)
    rolling = matured.rolling(
        window=args.lookback_windows,
        min_periods=args.min_periods,
    )
    rolling_mean = rolling.mean()
    rolling_negative_rate = (matured < 0).rolling(
        window=args.lookback_windows,
        min_periods=args.min_periods,
    ).mean()
    observation_count = matured.rolling(
        window=args.lookback_windows,
        min_periods=1,
    ).count()

    output["matured_top_score_mean_label"] = matured
    output["rolling_top_score_mean_label"] = rolling_mean
    output["rolling_top_score_negative_rate"] = rolling_negative_rate
    output["rolling_observation_count"] = observation_count.astype(int)
    output["gate_state"] = "warmup"
    output["weight_scale"] = float(args.warmup_scale)
    ready = observation_count >= args.min_periods
    full = ready & (rolling_mean >= args.loss_threshold)
    reduced = (
        ready
        & (rolling_mean < args.loss_threshold)
        & (rolling_mean > args.extreme_loss_threshold)
    )
    blocked = ready & (rolling_mean <= args.extreme_loss_threshold)
    output.loc[full, "gate_state"] = "full"
    output.loc[full, "weight_scale"] = float(args.full_scale)
    output.loc[reduced, "gate_state"] = "reduced"
    output.loc[reduced, "weight_scale"] = float(args.reduced_scale)
    output.loc[blocked, "gate_state"] = "blocked"
    output.loc[blocked, "weight_scale"] = float(args.blocked_scale)
    return output


def _expand_schedule(states: pd.DataFrame, target_features: tuple[str, ...]) -> pd.DataFrame:
    if not target_features:
        raise ValueError("at least one target feature is required")
    schedules = []
    for feature in target_features:
        frame = states.loc[:, ["timestamp", "weight_scale", "gate_state"]].copy()
        frame["feature"] = feature
        frame["shrink_reason"] = "lagged_top_score_loss_gate:" + frame["gate_state"]
        schedules.append(frame)
    output = pd.concat(schedules, ignore_index=True)
    return output.loc[:, ["timestamp", "feature", "weight_scale", "shrink_reason"]]


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts = frame[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics-dir",
        default=(
            "runs/candidate_factor_portfolios/"
            "fixed_framework_alpha_rank_v66_nohealth_2026_05_31_standard/"
            "full_base/scores/decorrelated/diagnostics"
        ),
    )
    parser.add_argument(
        "--target-features",
        nargs="+",
        default=[
            "intraday_overnight_gap_5m",
            "intraday_weak_tape_gap_up_risk_5m_w48",
        ],
    )
    parser.add_argument("--partition-start")
    parser.add_argument("--partition-end")
    parser.add_argument("--label-lag-windows", type=int, default=49)
    parser.add_argument("--lookback-windows", type=int, default=48)
    parser.add_argument("--min-periods", type=int, default=24)
    parser.add_argument("--loss-threshold", type=float, default=0.0)
    parser.add_argument("--extreme-loss-threshold", type=float, default=-0.003)
    parser.add_argument("--full-scale", type=float, default=1.0)
    parser.add_argument("--reduced-scale", type=float, default=0.5)
    parser.add_argument("--blocked-scale", type=float, default=0.25)
    parser.add_argument("--warmup-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=(
            "runs/candidate_factor_portfolios/"
            "fixed_framework_alpha_rank_v66_top_score_loss_gate_2024_screen/"
            "schedules"
        ),
    )
    args = parser.parse_args()
    if args.label_lag_windows < 0:
        raise ValueError("--label-lag-windows must be non-negative")
    if args.lookback_windows <= 0:
        raise ValueError("--lookback-windows must be positive")
    if args.min_periods <= 0:
        raise ValueError("--min-periods must be positive")
    if args.min_periods > args.lookback_windows:
        raise ValueError("--min-periods must be <= --lookback-windows")
    if args.extreme_loss_threshold > args.loss_threshold:
        raise ValueError("--extreme-loss-threshold must be <= --loss-threshold")
    for name in ("full_scale", "reduced_scale", "blocked_scale", "warmup_scale"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


if __name__ == "__main__":
    main()
