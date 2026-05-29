"""Build score streams that switch between baseline and challenger by state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    args = _parse_args()
    summary = build_state_conditioned_score_switch(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_state_conditioned_score_switch(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    score_dir = output_dir / "scores" / args.method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_existing:
        for old_path in score_dir.glob("score_*.parquet"):
            old_path.unlink()
    schedule = _daily_state_schedule(
        Path(args.dataset_dir),
        state_column=args.state_column,
        activation_quantile=args.activation_quantile,
        min_history_days=args.min_history_days,
        active_when=args.active_when,
        history_start=args.schedule_history_start or args.start,
        start=args.start,
        end=args.end,
    )
    schedule_path = output_dir / "state_switch_schedule.csv"
    schedule.to_csv(schedule_path, index=False)
    partition_rows = _write_switched_scores(
        baseline_score_dir=Path(args.baseline_score_dir),
        challenger_score_dir=Path(args.challenger_score_dir),
        score_dir=score_dir,
        schedule=schedule,
        start=args.start,
        end=args.end,
        resume_existing=args.resume_existing,
    )
    summary = {
        "status": "completed",
        "params": {
            "baseline_score_dir": args.baseline_score_dir,
            "challenger_score_dir": args.challenger_score_dir,
            "dataset_dir": args.dataset_dir,
            "output_dir": args.output_dir,
            "method_name": args.method_name,
            "state_column": args.state_column,
            "activation_quantile": args.activation_quantile,
            "min_history_days": args.min_history_days,
            "active_when": args.active_when,
            "schedule_history_start": args.schedule_history_start or args.start,
            "start": args.start,
            "end": args.end,
        },
        "schedule": {
            "path": str(schedule_path),
            "day_count": int(len(schedule)),
            "active_day_count": int(schedule["active"].sum()) if not schedule.empty else 0,
            "active_day_share": (
                float(schedule["active"].mean()) if not schedule.empty else 0.0
            ),
        },
        "scores": {
            "path": str(score_dir / "score_*.parquet"),
            "partition_count": len(partition_rows),
            "row_count": int(sum(partition_rows.values())),
            "rows_by_partition": partition_rows,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _daily_state_schedule(
    dataset_dir: Path,
    *,
    state_column: str,
    activation_quantile: float,
    min_history_days: int,
    active_when: str,
    history_start: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    history_start_at = _timestamp(history_start)
    start_at = _timestamp(start)
    end_at = _timestamp(end)
    if history_start_at > start_at:
        raise ValueError("schedule history start must be on or before start")
    frames: list[pd.DataFrame] = []
    for path in _dataset_paths(dataset_dir, start=history_start_at, end=end_at):
        frame = pd.read_parquet(path, columns=["timestamp", state_column])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.loc[
            (frame["timestamp"] >= history_start_at) & (frame["timestamp"] <= end_at)
        ].copy()
        if frame.empty:
            continue
        frame["trade_date"] = frame["timestamp"].dt.tz_convert("Asia/Shanghai").dt.date
        by_timestamp = frame.groupby("timestamp", sort=True)[state_column].mean()
        daily = by_timestamp.groupby(
            by_timestamp.index.tz_convert("Asia/Shanghai").date,
            sort=True,
        ).mean()
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": [str(value) for value in daily.index],
                    "state_value": daily.to_numpy(dtype=float),
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "state_value",
                "lagged_state_value",
                "lagged_quantile_threshold",
                "active",
            ]
        )
    schedule = (
        pd.concat(frames, ignore_index=True)
        .groupby("trade_date", as_index=False, sort=True)["state_value"]
        .mean()
    )
    lagged = schedule["state_value"].shift(1)
    threshold = lagged.expanding(min_periods=min_history_days).quantile(
        activation_quantile,
    )
    if active_when == "gte":
        active = lagged >= threshold
    elif active_when == "lte":
        active = lagged <= threshold
    else:
        raise ValueError("active_when must be gte or lte")
    schedule["lagged_state_value"] = lagged
    schedule["lagged_quantile_threshold"] = threshold
    schedule["active"] = active.fillna(False).astype(bool)
    start_date = str(start_at.tz_convert("Asia/Shanghai").date())
    end_date = str(end_at.tz_convert("Asia/Shanghai").date())
    return schedule.loc[
        (schedule["trade_date"] >= start_date) & (schedule["trade_date"] <= end_date)
    ].reset_index(drop=True)


def _write_switched_scores(
    *,
    baseline_score_dir: Path,
    challenger_score_dir: Path,
    score_dir: Path,
    schedule: pd.DataFrame,
    start: str,
    end: str,
    resume_existing: bool,
) -> dict[str, int]:
    active_dates = set(
        schedule.loc[schedule["active"].fillna(False).astype(bool), "trade_date"].astype(str)
    )
    start_partition = _timestamp(start).tz_convert("Asia/Shanghai").strftime("%Y_%m")
    end_partition = _timestamp(end).tz_convert("Asia/Shanghai").strftime("%Y_%m")
    rows_by_partition: dict[str, int] = {}
    for baseline_path in sorted(baseline_score_dir.glob("score_*.parquet")):
        partition = baseline_path.stem.removeprefix("score_")
        if partition < start_partition or partition > end_partition:
            continue
        output_path = score_dir / f"score_{partition}.parquet"
        if resume_existing and output_path.exists():
            rows_by_partition[partition] = int(
                len(pd.read_parquet(output_path, columns=["score"]))
            )
            continue
        challenger_path = challenger_score_dir / f"score_{partition}.parquet"
        baseline = pd.read_parquet(
            baseline_path,
            columns=["timestamp", "instrument_id", "score"],
        )
        baseline["trade_date"] = _score_trade_dates(baseline["timestamp"])
        if active_dates and challenger_path.exists():
            challenger = pd.read_parquet(
                challenger_path,
                columns=["timestamp", "instrument_id", "score"],
            )
            challenger["trade_date"] = _score_trade_dates(challenger["timestamp"])
            active_timestamps = set(
                baseline.loc[
                    baseline["trade_date"].isin(active_dates),
                    "timestamp",
                ].astype(str)
            )
            inactive = baseline.loc[
                ~baseline["timestamp"].astype(str).isin(active_timestamps),
                ["timestamp", "instrument_id", "score"],
            ].copy()
            inactive["signal_source"] = "baseline"
            active = challenger.loc[
                challenger["timestamp"].astype(str).isin(active_timestamps),
                ["timestamp", "instrument_id", "score"],
            ].copy()
            active["signal_source"] = "challenger"
            output = pd.concat([inactive, active], ignore_index=True)
        else:
            output = baseline.loc[:, ["timestamp", "instrument_id", "score"]].copy()
            output["signal_source"] = "baseline"
        output = output.sort_values(
            ["timestamp", "score", "instrument_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        output.to_parquet(output_path, index=False)
        rows_by_partition[partition] = int(len(output))
    return rows_by_partition


def _dataset_paths(
    dataset_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    start_partition = start.tz_convert("Asia/Shanghai").strftime("%Y_%m")
    end_partition = end.tz_convert("Asia/Shanghai").strftime("%Y_%m")
    return [
        path
        for path in sorted(dataset_dir.glob("dataset_*.parquet"))
        if start_partition <= path.stem.removeprefix("dataset_") <= end_partition
    ]


def _score_trade_dates(timestamp: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(timestamp, utc=True, errors="coerce")
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%d")
    )


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-score-dir", required=True)
    parser.add_argument("--challenger-score-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="state_switch")
    parser.add_argument("--state-column", default="market_state_downside_mean_5m_w48")
    parser.add_argument("--activation-quantile", type=float, default=0.33)
    parser.add_argument("--min-history-days", type=int, default=20)
    parser.add_argument("--active-when", choices=("gte", "lte"), default="gte")
    parser.add_argument(
        "--schedule-history-start",
        help=(
            "optional earlier dataset start used only for lagged expanding state "
            "thresholds; switched score output still starts at --start"
        ),
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("baseline_score_dir", "challenger_score_dir", "dataset_dir"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not 0 < args.activation_quantile < 1:
        raise ValueError("--activation-quantile must be in (0, 1)")
    if args.min_history_days <= 0:
        raise ValueError("--min-history-days must be positive")


if __name__ == "__main__":
    main()
