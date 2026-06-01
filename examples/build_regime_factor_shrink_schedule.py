"""Build a low-degree regime-based factor shrink schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_regime_factor_shrink_schedule(
    regime_schedule: pd.DataFrame,
    *,
    features: tuple[str, ...],
    stress_states: tuple[str, ...] = ("stress",),
    stress_scale: float = 0.5,
    normal_scale: float = 1.0,
) -> pd.DataFrame:
    """Map a timestamp-level regime schedule to per-factor weight scales."""
    if not features:
        raise ValueError("at least one feature is required")
    if not 0 <= stress_scale <= 1:
        raise ValueError("stress_scale must be in [0, 1]")
    if not 0 <= normal_scale <= 1:
        raise ValueError("normal_scale must be in [0, 1]")
    missing = {"timestamp", "regime_state"} - set(regime_schedule.columns)
    if missing:
        raise ValueError(f"regime schedule missing columns: {sorted(missing)}")
    states = (
        regime_schedule.loc[:, ["timestamp", "regime_state"]]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if states.empty:
        raise ValueError("regime schedule is empty")
    stress_state_set = set(stress_states)
    rows: list[pd.DataFrame] = []
    for feature in features:
        current = states.copy()
        current["feature"] = feature
        stress_mask = current["regime_state"].astype(str).isin(stress_state_set)
        current["weight_scale"] = normal_scale
        current.loc[stress_mask, "weight_scale"] = stress_scale
        current["shrink_reason"] = "regime_normal"
        current.loc[stress_mask, "shrink_reason"] = "regime_stress_shrink"
        rows.append(current)
    output = pd.concat(rows, ignore_index=True)
    return output.loc[
        :,
        ["timestamp", "feature", "weight_scale", "regime_state", "shrink_reason"],
    ].sort_values(["timestamp", "feature"]).reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    regime_schedule = _read_frame(Path(args.regime_weight_schedule))
    schedule = build_regime_factor_shrink_schedule(
        regime_schedule,
        features=tuple(args.features),
        stress_states=tuple(args.stress_states),
        stress_scale=args.stress_scale,
        normal_scale=args.normal_scale,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        schedule.to_parquet(output_path, index=False)
    else:
        schedule.to_csv(output_path, index=False)
    summary = {
        "path": str(output_path),
        "source_regime_weight_schedule": args.regime_weight_schedule,
        "features": list(args.features),
        "stress_states": list(args.stress_states),
        "stress_scale": args.stress_scale,
        "normal_scale": args.normal_scale,
        "row_count": int(len(schedule)),
        "timestamp_count": int(schedule["timestamp"].nunique()),
        "stress_timestamp_count": int(
            schedule.loc[
                schedule["regime_state"].astype(str).isin(set(args.stress_states)),
                "timestamp",
            ].nunique()
        ),
    }
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime-weight-schedule",
        required=True,
        help=(
            "CSV/parquet schedule with timestamp and regime_state columns; "
            "only the lagged observable state is used"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--stress-states", nargs="+", default=["stress"])
    parser.add_argument("--stress-scale", type=float, default=0.5)
    parser.add_argument("--normal-scale", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
