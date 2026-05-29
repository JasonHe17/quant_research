"""Build a no-leak adaptive state-switch score stream.

The selector chooses state-gate parameters from historical labelled score
performance only.  For each output period, it evaluates candidate quantiles on
history ending before the period start minus an embargo, then applies the best
candidate to the future period.  If no candidate beats the historical baseline
proxy, the period stays on the baseline score stream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_state_conditioned_score_switch import (  # noqa: E402
    _score_trade_dates,
    _timestamp,
    _write_switched_scores,
)


def main() -> None:
    args = _parse_args()
    summary = build_adaptive_state_conditioned_score_switch(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_adaptive_state_conditioned_score_switch(
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    score_dir = output_dir / "scores" / args.method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_existing:
        for old_path in score_dir.glob("score_*.parquet"):
            old_path.unlink()

    history_start = args.history_start or args.start
    state = _daily_state_table(
        Path(args.dataset_dir),
        state_column=args.state_column,
        history_start=history_start,
        end=args.end,
    )
    candidate_schedules = _candidate_schedules(
        state,
        candidate_quantiles=tuple(args.candidate_quantiles),
        min_history_days=args.min_history_days,
        active_when=args.active_when,
    )
    daily_performance = _daily_source_performance(
        dataset_dir=Path(args.dataset_dir),
        baseline_score_dir=Path(args.baseline_score_dir),
        challenger_score_dir=Path(args.challenger_score_dir),
        label_column=args.label_column,
        top_n=args.top_n,
        start=history_start,
        end=args.end,
    )
    selector = _selection_schedule(
        state,
        daily_performance,
        candidate_schedules=candidate_schedules,
        candidate_quantiles=tuple(args.candidate_quantiles),
        start=args.start,
        end=args.end,
        selection_frequency=args.selection_frequency,
        lookback_days=args.selection_lookback_days,
        min_observations=args.selection_min_observations,
        embargo_days=args.selection_embargo_days,
        min_edge=args.selection_min_edge,
        min_edge_t_stat=args.selection_min_edge_t_stat,
        fallback_to_baseline=not args.no_baseline_fallback,
    )
    schedule = _adaptive_daily_schedule(
        state,
        selector,
        candidate_schedules=candidate_schedules,
        start=args.start,
        end=args.end,
    )
    schedule_path = output_dir / "adaptive_state_switch_schedule.csv"
    selector_path = output_dir / "adaptive_state_switch_selector.csv"
    performance_path = output_dir / "daily_source_performance.csv"
    schedule.to_csv(schedule_path, index=False)
    selector.to_csv(selector_path, index=False)
    daily_performance.to_csv(performance_path, index=False)

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
            "label_column": args.label_column,
            "candidate_quantiles": args.candidate_quantiles,
            "active_when": args.active_when,
            "min_history_days": args.min_history_days,
            "history_start": history_start,
            "start": args.start,
            "end": args.end,
            "selection_frequency": args.selection_frequency,
            "selection_lookback_days": args.selection_lookback_days,
            "selection_min_observations": args.selection_min_observations,
            "selection_embargo_days": args.selection_embargo_days,
            "selection_min_edge": args.selection_min_edge,
            "selection_min_edge_t_stat": args.selection_min_edge_t_stat,
            "baseline_fallback": not args.no_baseline_fallback,
            "top_n": args.top_n,
        },
        "selection": {
            "path": str(selector_path),
            "period_count": int(len(selector)),
            "baseline_period_count": int(selector["baseline_fallback"].sum())
            if not selector.empty
            else 0,
        },
        "schedule": {
            "path": str(schedule_path),
            "day_count": int(len(schedule)),
            "active_day_count": int(schedule["active"].sum())
            if not schedule.empty
            else 0,
            "active_day_share": float(schedule["active"].mean())
            if not schedule.empty
            else 0.0,
        },
        "daily_performance": {
            "path": str(performance_path),
            "day_count": int(len(daily_performance)),
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


def _daily_state_table(
    dataset_dir: Path,
    *,
    state_column: str,
    history_start: str,
    end: str,
) -> pd.DataFrame:
    start_at = _timestamp(history_start)
    end_at = _timestamp(end)
    frames: list[pd.DataFrame] = []
    for path in _dataset_paths(dataset_dir, start=start_at, end=end_at):
        frame = pd.read_parquet(path, columns=["timestamp", state_column])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.loc[
            (frame["timestamp"] >= start_at) & (frame["timestamp"] <= end_at)
        ].copy()
        if frame.empty:
            continue
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
        return pd.DataFrame(columns=["trade_date", "state_value", "lagged_state_value"])
    output = (
        pd.concat(frames, ignore_index=True)
        .groupby("trade_date", as_index=False, sort=True)["state_value"]
        .mean()
    )
    output["lagged_state_value"] = output["state_value"].shift(1)
    return output


def _candidate_schedules(
    state: pd.DataFrame,
    *,
    candidate_quantiles: tuple[float, ...],
    min_history_days: int,
    active_when: str,
) -> dict[float, pd.DataFrame]:
    schedules: dict[float, pd.DataFrame] = {}
    lagged = state["lagged_state_value"]
    for quantile in candidate_quantiles:
        threshold = lagged.expanding(min_periods=min_history_days).quantile(quantile)
        if active_when == "gte":
            active = lagged >= threshold
        elif active_when == "lte":
            active = lagged <= threshold
        else:
            raise ValueError("active_when must be gte or lte")
        schedule = state.loc[:, ["trade_date", "state_value", "lagged_state_value"]].copy()
        schedule["candidate_quantile"] = quantile
        schedule["lagged_quantile_threshold"] = threshold
        schedule["candidate_active"] = active.fillna(False).astype(bool)
        schedules[quantile] = schedule
    return schedules


def _daily_source_performance(
    *,
    dataset_dir: Path,
    baseline_score_dir: Path,
    challenger_score_dir: Path,
    label_column: str,
    top_n: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_at = _timestamp(start)
    end_at = _timestamp(end)
    rows: list[pd.DataFrame] = []
    for dataset_path in _dataset_paths(dataset_dir, start=start_at, end=end_at):
        partition = dataset_path.stem.removeprefix("dataset_")
        label = pd.read_parquet(
            dataset_path,
            columns=["timestamp", "instrument_id", label_column],
        )
        label = label.loc[label[label_column].notna()].copy()
        if label.empty:
            continue
        label["timestamp"] = _timestamp_strings(label["timestamp"])
        for source, score_dir in (
            ("baseline", baseline_score_dir),
            ("challenger", challenger_score_dir),
        ):
            score_path = score_dir / f"score_{partition}.parquet"
            if not score_path.exists():
                continue
            score = pd.read_parquet(
                score_path,
                columns=["timestamp", "instrument_id", "score"],
            )
            if score.empty:
                continue
            timestamp = pd.to_datetime(score["timestamp"], utc=True, errors="coerce")
            score = score.loc[(timestamp >= start_at) & (timestamp <= end_at)].copy()
            if score.empty:
                continue
            top = (
                score.sort_values(
                    ["timestamp", "score", "instrument_id"],
                    ascending=[True, False, True],
                )
                .groupby("timestamp", sort=False)
                .head(top_n)
                .loc[:, ["timestamp", "instrument_id"]]
            )
            merged = top.merge(label, on=["timestamp", "instrument_id"], how="inner")
            if merged.empty:
                continue
            by_timestamp = (
                merged.groupby("timestamp", as_index=False, sort=True)[label_column]
                .agg(["mean", "count"])
                .reset_index()
                .rename(
                    columns={
                        "mean": f"{source}_top_n_mean_label",
                        "count": f"{source}_sample_count",
                    }
                )
            )
            by_timestamp["trade_date"] = _score_trade_dates(by_timestamp["timestamp"])
            daily = by_timestamp.groupby("trade_date", as_index=False).agg(
                **{
                    f"{source}_top_n_mean_label": (
                        f"{source}_top_n_mean_label",
                        "mean",
                    ),
                    f"{source}_sample_count": (f"{source}_sample_count", "sum"),
                }
            )
            rows.append(daily)
    if not rows:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "baseline_top_n_mean_label",
                "baseline_sample_count",
                "challenger_top_n_mean_label",
                "challenger_sample_count",
                "challenger_minus_baseline",
            ]
        )
    output = pd.concat(rows, ignore_index=True, sort=False)
    output = output.groupby("trade_date", as_index=False, sort=True).mean(numeric_only=True)
    output["challenger_minus_baseline"] = (
        output["challenger_top_n_mean_label"] - output["baseline_top_n_mean_label"]
    )
    return output


def _selection_schedule(
    state: pd.DataFrame,
    daily_performance: pd.DataFrame,
    *,
    candidate_schedules: dict[float, pd.DataFrame],
    candidate_quantiles: tuple[float, ...],
    start: str,
    end: str,
    selection_frequency: str,
    lookback_days: int,
    min_observations: int,
    embargo_days: int,
    min_edge: float,
    min_edge_t_stat: float,
    fallback_to_baseline: bool,
) -> pd.DataFrame:
    periods = _selection_periods(start=start, end=end, frequency=selection_frequency)
    performance = daily_performance.copy()
    performance["date"] = pd.to_datetime(performance["trade_date"]).dt.date
    state_by_quantile = {
        quantile: schedule.loc[:, ["trade_date", "candidate_active"]].rename(
            columns={"candidate_active": f"active_{_quantile_label(quantile)}"}
        )
        for quantile, schedule in candidate_schedules.items()
    }
    for frame in state_by_quantile.values():
        performance = performance.merge(frame, on="trade_date", how="left")
    rows: list[dict[str, Any]] = []
    for period_start, period_end in periods:
        cutoff = (period_start - pd.Timedelta(days=embargo_days)).date()
        lookback_start = (
            period_start - pd.Timedelta(days=embargo_days + lookback_days)
        ).date()
        history = performance.loc[
            (performance["date"] >= lookback_start) & (performance["date"] < cutoff)
        ].copy()
        baseline = history["baseline_top_n_mean_label"].dropna()
        baseline_mean = float(baseline.mean()) if not baseline.empty else None
        best_quantile: float | None = None
        best_mean: float | None = None
        best_edge: float | None = None
        best_edge_t_stat: float | None = None
        best_observations = 0
        for quantile in candidate_quantiles:
            active_column = f"active_{_quantile_label(quantile)}"
            candidate = history.copy()
            active = candidate[active_column].fillna(False).astype(bool)
            candidate["candidate_top_n_mean_label"] = candidate[
                "baseline_top_n_mean_label"
            ]
            candidate.loc[active, "candidate_top_n_mean_label"] = candidate.loc[
                active, "challenger_top_n_mean_label"
            ]
            values = candidate["candidate_top_n_mean_label"].dropna()
            if len(values) < min_observations:
                continue
            mean_value = float(values.mean())
            edge_values = (
                candidate["candidate_top_n_mean_label"]
                - candidate["baseline_top_n_mean_label"]
            ).dropna()
            edge = float(edge_values.mean()) if not edge_values.empty else None
            edge_t_stat = _t_stat(edge_values)
            eligible = True
            if fallback_to_baseline:
                eligible = (
                    edge is not None
                    and edge > min_edge
                    and (
                        min_edge_t_stat <= 0
                        or (
                            edge_t_stat is not None
                            and edge_t_stat >= min_edge_t_stat
                        )
                    )
                )
            if eligible and (best_mean is None or mean_value > best_mean):
                best_quantile = quantile
                best_mean = mean_value
                best_edge = edge
                best_edge_t_stat = edge_t_stat
                best_observations = int(len(values))
        use_baseline = best_quantile is None
        rows.append(
            {
                "period_start": str(period_start.date()),
                "period_end": str(period_end.date()),
                "selection_cutoff": str(cutoff),
                "lookback_start": str(lookback_start),
                "selected_quantile": None if use_baseline else best_quantile,
                "selected_candidate_mean_label": best_mean,
                "baseline_mean_label": baseline_mean,
                "selected_edge": best_edge,
                "selected_edge_t_stat": best_edge_t_stat,
                "observation_count": best_observations,
                "baseline_fallback": bool(use_baseline),
            }
        )
    return pd.DataFrame(rows)


def _adaptive_daily_schedule(
    state: pd.DataFrame,
    selector: pd.DataFrame,
    *,
    candidate_schedules: dict[float, pd.DataFrame],
    start: str,
    end: str,
) -> pd.DataFrame:
    start_date = _timestamp(start).tz_convert("Asia/Shanghai").date()
    end_date = _timestamp(end).tz_convert("Asia/Shanghai").date()
    output = state.copy()
    output["date"] = pd.to_datetime(output["trade_date"]).dt.date
    output = output.loc[(output["date"] >= start_date) & (output["date"] <= end_date)].copy()
    output["selected_quantile"] = pd.NA
    output["lagged_quantile_threshold"] = pd.NA
    output["active"] = False
    output["baseline_fallback"] = True
    for row in selector.itertuples(index=False):
        period_start = pd.Timestamp(row.period_start).date()
        period_end = pd.Timestamp(row.period_end).date()
        mask = (output["date"] >= period_start) & (output["date"] <= period_end)
        selected_quantile = row.selected_quantile
        output.loc[mask, "baseline_fallback"] = bool(row.baseline_fallback)
        if pd.isna(selected_quantile):
            continue
        quantile = float(selected_quantile)
        schedule = candidate_schedules[quantile].set_index("trade_date")
        dates = output.loc[mask, "trade_date"]
        output.loc[mask, "selected_quantile"] = quantile
        output.loc[mask, "lagged_quantile_threshold"] = dates.map(
            schedule["lagged_quantile_threshold"]
        )
        output.loc[mask, "active"] = dates.map(schedule["candidate_active"]).fillna(False)
    return output.drop(columns=["date"]).reset_index(drop=True)


def _selection_periods(
    *,
    start: str,
    end: str,
    frequency: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_at = _timestamp(start).tz_convert("Asia/Shanghai").normalize()
    end_at = _timestamp(end).tz_convert("Asia/Shanghai").normalize()
    if frequency == "monthly":
        starts = pd.date_range(start_at.replace(day=1), end_at, freq="MS")
    elif frequency == "weekly":
        starts = pd.date_range(start_at, end_at, freq="W-MON")
        if len(starts) == 0 or starts[0] > start_at:
            starts = pd.DatetimeIndex([start_at, *starts])
    else:
        raise ValueError("--selection-frequency must be monthly or weekly")
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for index, raw_start in enumerate(starts):
        period_start = max(raw_start, start_at)
        next_start = starts[index + 1] if index + 1 < len(starts) else end_at + pd.Timedelta(days=1)
        period_end = min(next_start - pd.Timedelta(days=1), end_at)
        if period_start <= period_end:
            periods.append((period_start, period_end))
    return periods


def _t_stat(values: pd.Series) -> float | None:
    values = values.dropna().astype(float)
    if len(values) < 2:
        return None
    std = float(values.std(ddof=1))
    if std <= 0:
        return None
    return float(values.mean() / (std / (len(values) ** 0.5)))


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


def _timestamp_strings(timestamp: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(timestamp, utc=True, errors="coerce")
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
        + "+08:00"
    )


def _quantile_label(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):03d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-score-dir", required=True)
    parser.add_argument("--challenger-score-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="adaptive_state_switch")
    parser.add_argument("--state-column", default="market_state_downside_mean_5m_w48")
    parser.add_argument("--label-column", default="forward_return_48b")
    parser.add_argument(
        "--candidate-quantiles",
        nargs="+",
        type=float,
        default=[0.33, 0.50, 0.67],
    )
    parser.add_argument("--active-when", choices=("gte", "lte"), default="gte")
    parser.add_argument("--min-history-days", type=int, default=20)
    parser.add_argument("--history-start")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--selection-frequency",
        choices=("monthly", "weekly"),
        default="monthly",
    )
    parser.add_argument("--selection-lookback-days", type=int, default=252)
    parser.add_argument("--selection-min-observations", type=int, default=40)
    parser.add_argument("--selection-embargo-days", type=int, default=3)
    parser.add_argument("--selection-min-edge", type=float, default=0.0)
    parser.add_argument(
        "--selection-min-edge-t-stat",
        type=float,
        default=2.0,
        help=(
            "minimum historical t-stat of candidate daily edge over baseline; "
            "set <=0 to disable"
        ),
    )
    parser.add_argument("--no-baseline-fallback", action="store_true")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("baseline_score_dir", "challenger_score_dir", "dataset_dir"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not args.candidate_quantiles:
        raise ValueError("--candidate-quantiles must be non-empty")
    for quantile in args.candidate_quantiles:
        if not 0 < quantile < 1:
            raise ValueError("--candidate-quantiles values must be in (0, 1)")
    if args.min_history_days <= 0:
        raise ValueError("--min-history-days must be positive")
    if args.selection_lookback_days <= 0:
        raise ValueError("--selection-lookback-days must be positive")
    if args.selection_min_observations <= 0:
        raise ValueError("--selection-min-observations must be positive")
    if args.selection_embargo_days < 0:
        raise ValueError("--selection-embargo-days must be non-negative")
    if args.selection_min_edge_t_stat < 0:
        raise ValueError("--selection-min-edge-t-stat must be non-negative")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")


if __name__ == "__main__":
    main()
