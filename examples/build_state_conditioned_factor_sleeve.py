"""Build a lagged broad-tape factor sleeve schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    args = _parse_args()
    summary = build_state_conditioned_factor_sleeve(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_state_conditioned_factor_sleeve(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _load_lagged_broad_tape_state(
        Path(args.dataset_dir),
        return_column=args.return_column,
        breadth_column=args.breadth_column,
        partition_start=args.partition_start,
        partition_end=args.partition_end,
    )
    sleeve = _sleeve_schedule(
        state,
        target_features=tuple(args.target_features),
        rule=args.rule,
        full_scale=args.full_scale,
        reduced_scale=args.reduced_scale,
        blocked_scale=args.blocked_scale,
        warmup_scale=args.warmup_scale,
    )
    base = _load_base_schedule(args.base_schedule)
    schedule = _combine_with_base_schedule(
        sleeve,
        base,
        mode=args.combine_mode,
    )

    state_path = output_dir / "broad_tape_state.csv"
    sleeve_path = output_dir / "factor_sleeve_schedule.csv"
    schedule_path = output_dir / "factor_weight_scale_schedule.csv"
    summary_path = output_dir / "summary.json"
    state.to_csv(state_path, index=False)
    sleeve.to_csv(sleeve_path, index=False)
    schedule.to_csv(schedule_path, index=False)
    summary = {
        "params": {
            "dataset_dir": args.dataset_dir,
            "target_features": args.target_features,
            "return_column": args.return_column,
            "breadth_column": args.breadth_column,
            "rule": args.rule,
            "full_scale": args.full_scale,
            "reduced_scale": args.reduced_scale,
            "blocked_scale": args.blocked_scale,
            "warmup_scale": args.warmup_scale,
            "base_schedule": args.base_schedule,
            "combine_mode": args.combine_mode,
            "partition_start": args.partition_start,
            "partition_end": args.partition_end,
        },
        "state_count": int(len(state)),
        "schedule_count": int(len(schedule)),
        "sleeve_scale_counts": _value_counts(sleeve, "weight_scale"),
        "sleeve_state_counts": _value_counts(sleeve, "sleeve_state"),
        "feature_summary": (
            schedule.groupby("feature")["weight_scale"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
            .to_dict("records")
            if not schedule.empty
            else []
        ),
        "artifacts": {
            "state": str(state_path),
            "sleeve_schedule": str(sleeve_path),
            "schedule": str(schedule_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_lagged_broad_tape_state(
    dataset_dir: Path,
    *,
    return_column: str,
    breadth_column: str,
    partition_start: str | None,
    partition_end: str | None,
) -> pd.DataFrame:
    paths = _dataset_paths(
        dataset_dir,
        partition_start=partition_start,
        partition_end=partition_end,
    )
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["timestamp", return_column, breadth_column],
        )
        grouped = (
            frame.groupby("timestamp", as_index=False, sort=True)
            .agg({return_column: "mean", breadth_column: "mean"})
            .rename(
                columns={
                    return_column: "market_return",
                    breadth_column: "market_breadth",
                }
            )
        )
        frames.append(grouped)
    if not frames:
        raise FileNotFoundError(f"no dataset partitions found under {dataset_dir}")
    output = (
        pd.concat(frames, ignore_index=True)
        .groupby("timestamp", as_index=False, sort=True)
        .agg({"market_return": "mean", "market_breadth": "mean"})
    )
    output["lagged_market_return"] = output["market_return"].shift(1)
    output["lagged_market_breadth"] = output["market_breadth"].shift(1)
    output["timestamp"] = _timestamp_strings(output["timestamp"])
    return output


def _sleeve_schedule(
    state: pd.DataFrame,
    *,
    target_features: tuple[str, ...],
    rule: str,
    full_scale: float,
    reduced_scale: float,
    blocked_scale: float,
    warmup_scale: float,
) -> pd.DataFrame:
    if not target_features:
        raise ValueError("at least one target feature is required")
    if rule not in {
        "return_positive",
        "breadth_majority",
        "return_and_breadth",
        "monotone_breadth_return",
    }:
        raise ValueError(f"unsupported sleeve rule: {rule}")

    rows = state.loc[
        :,
        [
            "timestamp",
            "lagged_market_return",
            "lagged_market_breadth",
        ],
    ].copy()
    rows["sleeve_state"] = "blocked"
    rows["weight_scale"] = float(blocked_scale)
    warmup = rows["lagged_market_return"].isna() | rows["lagged_market_breadth"].isna()
    return_positive = rows["lagged_market_return"] > 0.0
    breadth_majority = rows["lagged_market_breadth"] >= 0.5

    if rule == "return_positive":
        active = return_positive
        rows.loc[active, "sleeve_state"] = "full"
        rows.loc[active, "weight_scale"] = float(full_scale)
    elif rule == "breadth_majority":
        active = breadth_majority
        rows.loc[active, "sleeve_state"] = "full"
        rows.loc[active, "weight_scale"] = float(full_scale)
    elif rule == "return_and_breadth":
        active = return_positive & breadth_majority
        rows.loc[active, "sleeve_state"] = "full"
        rows.loc[active, "weight_scale"] = float(full_scale)
    else:
        full = return_positive & breadth_majority
        reduced = (return_positive | breadth_majority) & ~full
        rows.loc[reduced, "sleeve_state"] = "reduced"
        rows.loc[reduced, "weight_scale"] = float(reduced_scale)
        rows.loc[full, "sleeve_state"] = "full"
        rows.loc[full, "weight_scale"] = float(full_scale)

    rows.loc[warmup, "sleeve_state"] = "warmup"
    rows.loc[warmup, "weight_scale"] = float(warmup_scale)
    schedules = []
    for feature in target_features:
        frame = rows.copy()
        frame["feature"] = feature
        frame["shrink_reason"] = "lagged_broad_tape_sleeve:" + frame["sleeve_state"]
        schedules.append(frame)
    output = pd.concat(schedules, ignore_index=True)
    return output.loc[
        :,
        [
            "timestamp",
            "feature",
            "weight_scale",
            "shrink_reason",
            "sleeve_state",
            "lagged_market_return",
            "lagged_market_breadth",
        ],
    ]


def _combine_with_base_schedule(
    sleeve: pd.DataFrame,
    base: pd.DataFrame | None,
    *,
    mode: str,
) -> pd.DataFrame:
    if base is None or base.empty:
        return sleeve.loc[:, ["timestamp", "feature", "weight_scale", "shrink_reason"]]
    if mode not in {"min", "multiply", "override"}:
        raise ValueError("combine mode must be min, multiply, or override")
    base_frame = base.loc[:, ["timestamp", "feature", "weight_scale", "shrink_reason"]]
    sleeve_frame = sleeve.loc[:, ["timestamp", "feature", "weight_scale", "shrink_reason"]]
    joined = base_frame.rename(
        columns={
            "weight_scale": "base_weight_scale",
            "shrink_reason": "base_shrink_reason",
        }
    ).merge(
        sleeve_frame.rename(
            columns={
                "weight_scale": "sleeve_weight_scale",
                "shrink_reason": "sleeve_shrink_reason",
            }
        ),
        on=["timestamp", "feature"],
        how="outer",
        sort=False,
    )
    base_scale = pd.to_numeric(joined["base_weight_scale"], errors="coerce")
    sleeve_scale = pd.to_numeric(joined["sleeve_weight_scale"], errors="coerce")
    if mode == "min":
        weight_scale = pd.concat(
            [base_scale.fillna(1.0), sleeve_scale.fillna(1.0)],
            axis=1,
        ).min(axis=1)
    elif mode == "multiply":
        weight_scale = base_scale.fillna(1.0) * sleeve_scale.fillna(1.0)
    else:
        weight_scale = sleeve_scale.fillna(base_scale.fillna(1.0))
    joined["weight_scale"] = weight_scale.clip(0.0, 1.0)
    joined["shrink_reason"] = [
        _join_reasons(base_reason, sleeve_reason)
        for base_reason, sleeve_reason in zip(
            joined["base_shrink_reason"],
            joined["sleeve_shrink_reason"],
            strict=False,
        )
    ]
    return (
        joined.loc[:, ["timestamp", "feature", "weight_scale", "shrink_reason"]]
        .sort_values(["timestamp", "feature"])
        .reset_index(drop=True)
    )


def _load_base_schedule(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    schedule_path = Path(path)
    if not schedule_path.exists():
        raise FileNotFoundError(f"base schedule not found: {path}")
    frame = pd.read_csv(schedule_path)
    missing = {"timestamp", "feature", "weight_scale"} - set(frame.columns)
    if missing:
        raise ValueError(f"base schedule missing columns: {sorted(missing)}")
    output = frame.copy()
    output["timestamp"] = _timestamp_strings(output["timestamp"])
    output["feature"] = output["feature"].astype(str)
    output["weight_scale"] = pd.to_numeric(output["weight_scale"], errors="coerce")
    if output["weight_scale"].isna().any():
        raise ValueError("base schedule contains non-numeric weight_scale")
    if not output["weight_scale"].between(0.0, 1.0).all():
        raise ValueError("base schedule weight_scale values must be in [0, 1]")
    if "shrink_reason" not in output.columns:
        output["shrink_reason"] = "base_weight_scale"
    return output.loc[:, ["timestamp", "feature", "weight_scale", "shrink_reason"]]


def _dataset_paths(
    dataset_dir: Path,
    *,
    partition_start: str | None,
    partition_end: str | None,
) -> list[Path]:
    paths = sorted(dataset_dir.glob("dataset_*.parquet"))
    if partition_start:
        paths = [path for path in paths if _partition(path) >= partition_start]
    if partition_end:
        paths = [path for path in paths if _partition(path) <= partition_end]
    return paths


def _partition(path: Path) -> str:
    return path.stem.removeprefix("dataset_")


def _timestamp_strings(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        return values.astype(str)
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace(
        r"(\+|\-)(\d{2})(\d{2})$",
        r"\1\2:\3",
        regex=True,
    )


def _join_reasons(base_reason: object, sleeve_reason: object) -> str:
    parts = []
    for value in (base_reason, sleeve_reason):
        if pd.isna(value):
            continue
        text = str(value)
        if text:
            parts.extend(part for part in text.split(",") if part)
    return ",".join(dict.fromkeys(parts))


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts = frame[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="runs/framework_v1_acceptance/standard/alpha_dataset",
    )
    parser.add_argument(
        "--target-features",
        nargs="+",
        default=["intraday_daily_ma_deviation_5m_d10"],
    )
    parser.add_argument("--return-column", default="market_state_return_5m")
    parser.add_argument("--breadth-column", default="market_state_breadth_5m")
    parser.add_argument(
        "--rule",
        choices=(
            "return_positive",
            "breadth_majority",
            "return_and_breadth",
            "monotone_breadth_return",
        ),
        default="return_and_breadth",
    )
    parser.add_argument("--full-scale", type=float, default=1.0)
    parser.add_argument("--reduced-scale", type=float, default=0.5)
    parser.add_argument("--blocked-scale", type=float, default=0.0)
    parser.add_argument("--warmup-scale", type=float, default=0.0)
    parser.add_argument("--base-schedule")
    parser.add_argument("--combine-mode", choices=("min", "multiply", "override"), default="min")
    parser.add_argument("--partition-start")
    parser.add_argument("--partition-end")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for name in ("full_scale", "reduced_scale", "blocked_scale", "warmup_scale"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


if __name__ == "__main__":
    main()
