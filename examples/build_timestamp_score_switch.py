"""Switch baseline and challenger score partitions by timestamp schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    args = _parse_args()
    summary = build_timestamp_score_switch(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_timestamp_score_switch(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    score_dir = output_dir / "scores" / args.method_name
    score_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_existing:
        for old_path in score_dir.glob("score_*.parquet"):
            old_path.unlink()
    schedule = _load_active_timestamps(
        Path(args.schedule),
        feature=args.schedule_feature,
        active_threshold=args.active_threshold,
    )
    rows_by_partition = _write_switched_scores(
        baseline_score_dir=Path(args.baseline_score_dir),
        challenger_score_dir=Path(args.challenger_score_dir),
        score_dir=score_dir,
        active_timestamps=schedule,
        partition_start=args.partition_start,
        partition_end=args.partition_end,
        resume_existing=args.resume_existing,
    )
    summary = {
        "status": "completed",
        "params": {
            "baseline_score_dir": args.baseline_score_dir,
            "challenger_score_dir": args.challenger_score_dir,
            "schedule": args.schedule,
            "schedule_feature": args.schedule_feature,
            "active_threshold": args.active_threshold,
            "output_dir": args.output_dir,
            "method_name": args.method_name,
            "partition_start": args.partition_start,
            "partition_end": args.partition_end,
        },
        "schedule": {
            "active_timestamp_count": len(schedule),
        },
        "scores": {
            "path": str(score_dir / "score_*.parquet"),
            "partition_count": len(rows_by_partition),
            "row_count": int(sum(rows_by_partition.values())),
            "rows_by_partition": rows_by_partition,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_active_timestamps(
    schedule_path: Path,
    *,
    feature: str,
    active_threshold: float,
) -> set[str]:
    if not schedule_path.exists():
        raise FileNotFoundError(f"schedule not found: {schedule_path}")
    schedule = pd.read_csv(schedule_path, usecols=["timestamp", "feature", "weight_scale"])
    selected = schedule.loc[schedule["feature"].astype(str) == feature].copy()
    if selected.empty:
        raise ValueError(f"schedule has no rows for feature: {feature}")
    selected["weight_scale"] = pd.to_numeric(selected["weight_scale"], errors="coerce")
    if selected["weight_scale"].isna().any():
        raise ValueError("schedule contains non-numeric weight_scale")
    active = selected.loc[selected["weight_scale"] > active_threshold, "timestamp"]
    return set(_timestamp_strings(active))


def _write_switched_scores(
    *,
    baseline_score_dir: Path,
    challenger_score_dir: Path,
    score_dir: Path,
    active_timestamps: set[str],
    partition_start: str | None,
    partition_end: str | None,
    resume_existing: bool,
) -> dict[str, int]:
    rows_by_partition: dict[str, int] = {}
    for baseline_path in sorted(baseline_score_dir.glob("score_*.parquet")):
        partition = baseline_path.stem.removeprefix("score_")
        if partition_start and partition < partition_start:
            continue
        if partition_end and partition > partition_end:
            continue
        output_path = score_dir / f"score_{partition}.parquet"
        if resume_existing and output_path.exists():
            rows_by_partition[partition] = int(
                len(pd.read_parquet(output_path, columns=["score"]))
            )
            continue
        challenger_path = challenger_score_dir / f"score_{partition}.parquet"
        if not challenger_path.exists():
            raise FileNotFoundError(f"challenger score partition not found: {challenger_path}")
        baseline = pd.read_parquet(
            baseline_path,
            columns=["timestamp", "instrument_id", "score"],
        )
        challenger = pd.read_parquet(
            challenger_path,
            columns=["timestamp", "instrument_id", "score"],
        )
        baseline_timestamps = _timestamp_strings(baseline["timestamp"])
        active_mask = baseline_timestamps.isin(active_timestamps)
        inactive = baseline.loc[~active_mask, ["timestamp", "instrument_id", "score"]].copy()
        inactive["signal_source"] = "baseline"
        active_timestamp_values = set(baseline.loc[active_mask, "timestamp"].astype(str))
        challenger_mask = challenger["timestamp"].astype(str).isin(active_timestamp_values)
        active = challenger.loc[
            challenger_mask,
            ["timestamp", "instrument_id", "score"],
        ].copy()
        active["signal_source"] = "challenger"
        output = pd.concat([inactive, active], ignore_index=True)
        output = output.sort_values(
            ["timestamp", "score", "instrument_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        output.to_parquet(output_path, index=False)
        rows_by_partition[partition] = int(len(output))
    return rows_by_partition


def _timestamp_strings(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        return values.astype(str)
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace(
        r"(\+|\-)(\d{2})(\d{2})$",
        r"\1\2:\3",
        regex=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-score-dir", required=True)
    parser.add_argument("--challenger-score-dir", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument(
        "--schedule-feature",
        default="intraday_daily_ma_deviation_5m_d10",
    )
    parser.add_argument("--active-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="decorrelated")
    parser.add_argument("--partition-start")
    parser.add_argument("--partition-end")
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
