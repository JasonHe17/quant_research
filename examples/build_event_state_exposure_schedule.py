"""Build a lagged gross-exposure schedule from event-state diagnostics."""

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
    summary = build_event_state_exposure_schedule(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_event_state_exposure_schedule(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states = _load_event_states(Path(args.event_states_path))
    schedule = _state_schedule(states, args)
    if args.base_schedule:
        schedule = _combine_with_base_schedule(schedule, Path(args.base_schedule))

    schedule_path = output_dir / "gross_exposure_schedule.csv"
    summary_path = output_dir / "summary.json"
    state_counts_path = output_dir / "event_state_scale_summary.csv"
    schedule.to_csv(schedule_path, index=False)
    state_summary = _state_scale_summary(schedule)
    state_summary.to_csv(state_counts_path, index=False)
    summary = {
        "params": {
            "event_states_path": args.event_states_path,
            "state_column": args.state_column,
            "lag_windows": args.lag_windows,
            "full_scale": args.full_scale,
            "reduced_scale": args.reduced_scale,
            "blocked_scale": args.blocked_scale,
            "warmup_scale": args.warmup_scale,
            "reduced_states": args.reduced_states,
            "blocked_states": args.blocked_states,
            "warmup_states": args.warmup_states,
            "base_schedule": args.base_schedule,
            "combine_mode": args.combine_mode,
        },
        "schedule_count": int(len(schedule)),
        "scale_counts": _value_counts(schedule, "gross_exposure_scale"),
        "raw_scale_counts": _value_counts(schedule, "event_state_gross_exposure_scale"),
        "state_counts": _value_counts(schedule, "effective_event_state"),
        "source_state_counts": _value_counts(schedule, "source_event_state"),
        "lagged_missing_count": int(schedule["effective_event_state"].isna().sum()),
        "artifacts": {
            "schedule": str(schedule_path),
            "state_scale_summary": str(state_counts_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_event_states(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"event states path not found: {path}")
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError("event states table must contain timestamp")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if frame["timestamp"].duplicated().any():
        duplicates = frame.loc[frame["timestamp"].duplicated(), "timestamp"].head(5)
        raise ValueError(f"duplicate event-state timestamps: {duplicates.tolist()}")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _state_schedule(states: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    state_column = args.state_column
    if state_column not in states.columns:
        raise ValueError(f"event states table must contain {state_column}")
    output = states.loc[:, ["timestamp", state_column]].copy()
    output = output.rename(columns={state_column: "source_event_state"})
    if args.lag_windows:
        output["effective_event_state"] = output["source_event_state"].shift(args.lag_windows)
    else:
        output["effective_event_state"] = output["source_event_state"]
    output["event_state_gross_exposure_scale"] = output["effective_event_state"].map(
        lambda state: _scale_for_state(state, args)
    )
    output["gross_exposure_scale"] = output["event_state_gross_exposure_scale"]
    output["event_state_gate_reason"] = output["effective_event_state"].map(
        lambda state: _reason_for_state(state, args)
    )
    return output


def _scale_for_state(state: Any, args: argparse.Namespace) -> float:
    if pd.isna(state):
        return float(args.warmup_scale)
    state_value = str(state)
    if state_value in set(args.blocked_states):
        return float(args.blocked_scale)
    if state_value in set(args.reduced_states):
        return float(args.reduced_scale)
    if state_value in set(args.warmup_states):
        return float(args.warmup_scale)
    return float(args.full_scale)


def _reason_for_state(state: Any, args: argparse.Namespace) -> str:
    if pd.isna(state):
        return "lagged_state_missing"
    state_value = str(state)
    if state_value in set(args.blocked_states):
        return "blocked_event_state"
    if state_value in set(args.reduced_states):
        return "reduced_event_state"
    if state_value in set(args.warmup_states):
        return "warmup_event_state"
    return "full_event_state"


def _combine_with_base_schedule(schedule: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    if base_path.suffix == ".parquet":
        base = pd.read_parquet(base_path)
    else:
        base = pd.read_csv(base_path)
    if "timestamp" not in base.columns:
        raise ValueError("base schedule must contain timestamp")
    base = base.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    if "gross_exposure_scale" not in base.columns:
        if "policy_gross_exposure_scale" in base.columns:
            base = base.rename(
                columns={"policy_gross_exposure_scale": "base_gross_exposure_scale"}
            )
        else:
            raise ValueError("base schedule must contain gross_exposure_scale")
    else:
        base = base.rename(columns={"gross_exposure_scale": "base_gross_exposure_scale"})
    joined = schedule.merge(
        base.loc[:, ["timestamp", "base_gross_exposure_scale"]],
        on="timestamp",
        how="left",
    )
    joined["base_gross_exposure_scale"] = joined["base_gross_exposure_scale"].fillna(1.0)
    joined["gross_exposure_scale"] = joined[
        ["event_state_gross_exposure_scale", "base_gross_exposure_scale"]
    ].min(axis=1)
    return joined


def _state_scale_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    grouped = schedule.groupby(
        ["effective_event_state", "gross_exposure_scale"],
        dropna=False,
        sort=True,
    )
    summary = grouped.agg(timestamp_count=("timestamp", "size")).reset_index()
    total = max(int(summary["timestamp_count"].sum()), 1)
    summary["timestamp_share"] = summary["timestamp_count"] / total
    return summary


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-states-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-column", default="event_state")
    parser.add_argument(
        "--lag-windows",
        type=int,
        default=1,
        help="number of timestamp rows to lag states before applying exposure control",
    )
    parser.add_argument("--full-scale", type=float, default=1.0)
    parser.add_argument("--reduced-scale", type=float, default=0.5)
    parser.add_argument("--blocked-scale", type=float, default=0.0)
    parser.add_argument("--warmup-scale", type=float, default=1.0)
    parser.add_argument(
        "--reduced-states",
        nargs="*",
        default=["limit_diffusion"],
    )
    parser.add_argument(
        "--blocked-states",
        nargs="*",
        default=["limit_diffusion_extreme"],
    )
    parser.add_argument("--warmup-states", nargs="*", default=["warmup"])
    parser.add_argument("--base-schedule")
    parser.add_argument(
        "--combine-mode",
        choices=("min",),
        default="min",
        help="how to combine event-state scale with a base schedule",
    )
    args = parser.parse_args()
    if args.lag_windows < 0:
        raise ValueError("--lag-windows must be non-negative")
    for name in ("full_scale", "reduced_scale", "blocked_scale", "warmup_scale"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


if __name__ == "__main__":
    main()
