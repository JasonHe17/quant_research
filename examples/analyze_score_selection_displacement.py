"""Analyze top-N selection displacement between two score streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def analyze_score_selection_displacement(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_rows = selection_displacement_by_timestamp(
        baseline_score_dir=Path(args.baseline_score_dir),
        challenger_score_dir=Path(args.challenger_score_dir),
        dataset_dir=Path(args.dataset_dir),
        regime_schedule=Path(args.regime_schedule) if args.regime_schedule else None,
        top_n=args.top_n,
        label_column=args.label_column,
        partition_start=args.partition_start,
        partition_end=args.partition_end,
    )
    monthly = _aggregate_displacement(timestamp_rows, ["month"])
    state = _aggregate_displacement(timestamp_rows, ["regime_state"])
    monthly_state = _aggregate_displacement(timestamp_rows, ["month", "regime_state"])

    timestamp_rows.to_csv(output_dir / "selection_displacement_by_timestamp.csv", index=False)
    monthly.to_csv(output_dir / "monthly_selection_displacement.csv", index=False)
    state.to_csv(output_dir / "state_selection_displacement.csv", index=False)
    monthly_state.to_csv(output_dir / "monthly_state_selection_displacement.csv", index=False)
    summary = {
        "status": "completed",
        "params": {
            "baseline_score_dir": args.baseline_score_dir,
            "challenger_score_dir": args.challenger_score_dir,
            "dataset_dir": args.dataset_dir,
            "regime_schedule": args.regime_schedule,
            "label_column": args.label_column,
            "top_n": args.top_n,
            "partition_start": args.partition_start,
            "partition_end": args.partition_end,
        },
        "overall": _aggregate_displacement(timestamp_rows, []).to_dict("records")[0]
        if not timestamp_rows.empty
        else {},
        "state": state.to_dict("records"),
        "monthly_path": str(output_dir / "monthly_selection_displacement.csv"),
        "state_path": str(output_dir / "state_selection_displacement.csv"),
        "timestamp_path": str(output_dir / "selection_displacement_by_timestamp.csv"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def selection_displacement_by_timestamp(
    *,
    baseline_score_dir: Path,
    challenger_score_dir: Path,
    dataset_dir: Path,
    regime_schedule: Path | None,
    top_n: int,
    label_column: str,
    partition_start: str | None = None,
    partition_end: str | None = None,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    states = _load_regime_states(regime_schedule)
    rows: list[dict[str, Any]] = []
    for dataset_path in _dataset_paths(dataset_dir, partition_start, partition_end):
        partition = dataset_path.stem.removeprefix("dataset_")
        baseline_path = baseline_score_dir / f"score_{partition}.parquet"
        challenger_path = challenger_score_dir / f"score_{partition}.parquet"
        if not baseline_path.exists() or not challenger_path.exists():
            continue
        labels = pd.read_parquet(
            dataset_path,
            columns=["timestamp", "instrument_id", label_column],
        )
        labels["timestamp"] = _timestamp_strings(labels["timestamp"])
        labels = labels.loc[
            labels[label_column].notna(),
            ["timestamp", "instrument_id", label_column],
        ]
        baseline_top = _top_n_scores(baseline_path, top_n=top_n).rename(
            columns={"score": "baseline_score"}
        )
        challenger_top = _top_n_scores(challenger_path, top_n=top_n).rename(
            columns={"score": "challenger_score"}
        )
        joined = baseline_top.merge(
            challenger_top,
            on=["timestamp", "instrument_id"],
            how="outer",
            indicator=True,
        ).merge(labels, on=["timestamp", "instrument_id"], how="left")
        for timestamp, group in joined.groupby("timestamp", sort=True):
            rows.append(_timestamp_displacement_row(timestamp, group, label_column, states))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _timestamp_displacement_row(
    timestamp: str,
    group: pd.DataFrame,
    label_column: str,
    states: dict[str, str],
) -> dict[str, Any]:
    both = group["_merge"] == "both"
    removed = group["_merge"] == "left_only"
    added = group["_merge"] == "right_only"
    baseline_mask = both | removed
    challenger_mask = both | added
    row: dict[str, Any] = {
        "timestamp": timestamp,
        "month": pd.to_datetime(timestamp, utc=True).strftime("%Y-%m"),
        "regime_state": states.get(timestamp, "unknown"),
        "baseline_top_count": int(baseline_mask.sum()),
        "challenger_top_count": int(challenger_mask.sum()),
        "union_count": int(len(group)),
        "overlap_count": int(both.sum()),
        "added_count": int(added.sum()),
        "removed_count": int(removed.sum()),
    }
    denominator = min(row["baseline_top_count"], row["challenger_top_count"])
    row["overlap_share"] = (
        float(row["overlap_count"] / denominator) if denominator else 0.0
    )
    for prefix, mask in (
        ("baseline_top", baseline_mask),
        ("challenger_top", challenger_mask),
        ("added", added),
        ("removed", removed),
    ):
        labels = pd.to_numeric(group.loc[mask, label_column], errors="coerce")
        row[f"{prefix}_label_sum"] = float(labels.sum(skipna=True))
        row[f"{prefix}_label_count"] = int(labels.notna().sum())
        row[f"{prefix}_label_mean"] = _safe_mean(
            row[f"{prefix}_label_sum"],
            row[f"{prefix}_label_count"],
        )
    row["top_label_delta"] = _safe_delta(
        row["challenger_top_label_mean"],
        row["baseline_top_label_mean"],
    )
    row["replacement_label_delta"] = _safe_delta(
        row["added_label_mean"],
        row["removed_label_mean"],
    )
    return row


def _aggregate_displacement(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [
        *keys,
        "timestamp_count",
        "baseline_top_count",
        "challenger_top_count",
        "overlap_count",
        "overlap_share",
        "added_count",
        "removed_count",
        "baseline_top_label_mean",
        "challenger_top_label_mean",
        "top_label_delta",
        "added_label_mean",
        "removed_label_mean",
        "replacement_label_delta",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if keys:
        grouped = frame.groupby(keys, sort=True, dropna=False)
        base = grouped.agg(
            timestamp_count=("timestamp", "nunique"),
            baseline_top_count=("baseline_top_count", "sum"),
            challenger_top_count=("challenger_top_count", "sum"),
            overlap_count=("overlap_count", "sum"),
            added_count=("added_count", "sum"),
            removed_count=("removed_count", "sum"),
            baseline_top_label_sum=("baseline_top_label_sum", "sum"),
            baseline_top_label_count=("baseline_top_label_count", "sum"),
            challenger_top_label_sum=("challenger_top_label_sum", "sum"),
            challenger_top_label_count=("challenger_top_label_count", "sum"),
            added_label_sum=("added_label_sum", "sum"),
            added_label_count=("added_label_count", "sum"),
            removed_label_sum=("removed_label_sum", "sum"),
            removed_label_count=("removed_label_count", "sum"),
        ).reset_index()
    else:
        base = pd.DataFrame(
            [
                {
                    "timestamp_count": int(frame["timestamp"].nunique()),
                    "baseline_top_count": int(frame["baseline_top_count"].sum()),
                    "challenger_top_count": int(frame["challenger_top_count"].sum()),
                    "overlap_count": int(frame["overlap_count"].sum()),
                    "added_count": int(frame["added_count"].sum()),
                    "removed_count": int(frame["removed_count"].sum()),
                    "baseline_top_label_sum": float(frame["baseline_top_label_sum"].sum()),
                    "baseline_top_label_count": int(frame["baseline_top_label_count"].sum()),
                    "challenger_top_label_sum": float(
                        frame["challenger_top_label_sum"].sum()
                    ),
                    "challenger_top_label_count": int(
                        frame["challenger_top_label_count"].sum()
                    ),
                    "added_label_sum": float(frame["added_label_sum"].sum()),
                    "added_label_count": int(frame["added_label_count"].sum()),
                    "removed_label_sum": float(frame["removed_label_sum"].sum()),
                    "removed_label_count": int(frame["removed_label_count"].sum()),
                }
            ]
        )
    denominator = base[["baseline_top_count", "challenger_top_count"]].min(axis=1)
    base["overlap_share"] = base["overlap_count"] / denominator.replace(0, pd.NA)
    for prefix in ("baseline_top", "challenger_top", "added", "removed"):
        base[f"{prefix}_label_mean"] = base.apply(
            lambda row: _safe_mean(
                row[f"{prefix}_label_sum"],
                row[f"{prefix}_label_count"],
            ),
            axis=1,
        )
    base["top_label_delta"] = (
        base["challenger_top_label_mean"] - base["baseline_top_label_mean"]
    )
    base["replacement_label_delta"] = (
        base["added_label_mean"] - base["removed_label_mean"]
    )
    return base.loc[:, columns]


def _top_n_scores(path: Path, *, top_n: int) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["timestamp", "instrument_id", "score"])
    frame["timestamp"] = _timestamp_strings(frame["timestamp"])
    frame = frame.loc[frame["score"].notna()].sort_values(
        ["timestamp", "score", "instrument_id"],
        ascending=[True, False, True],
    )
    frame["rank"] = frame.groupby("timestamp", sort=False).cumcount() + 1
    return frame.loc[
        frame["rank"] <= top_n,
        ["timestamp", "instrument_id", "score"],
    ]


def _load_regime_states(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=["timestamp", "regime_state"])
    else:
        frame = pd.read_csv(path, usecols=["timestamp", "regime_state"])
    frame["timestamp"] = _timestamp_strings(frame["timestamp"])
    frame = frame.drop_duplicates("timestamp")
    return dict(zip(frame["timestamp"], frame["regime_state"].astype(str), strict=False))


def _dataset_paths(
    dataset_dir: Path,
    partition_start: str | None,
    partition_end: str | None,
) -> list[Path]:
    paths = sorted(dataset_dir.glob("dataset_*.parquet"))
    if partition_start:
        paths = [p for p in paths if p.stem.removeprefix("dataset_") >= partition_start]
    if partition_end:
        paths = [p for p in paths if p.stem.removeprefix("dataset_") <= partition_end]
    if not paths:
        raise FileNotFoundError(f"no dataset partitions found in {dataset_dir}")
    return paths


def _timestamp_strings(values: pd.Series) -> pd.Series:
    return values.astype(str)


def _safe_mean(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(total / count)


def _safe_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def main() -> None:
    args = _parse_args()
    summary = analyze_score_selection_displacement(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-score-dir", required=True)
    parser.add_argument("--challenger-score-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--regime-schedule")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--partition-start")
    parser.add_argument("--partition-end")
    return parser.parse_args()


if __name__ == "__main__":
    main()
