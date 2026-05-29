"""Build an adaptive score stream selected by historical backtest performance.

This selector is intended for no-leak parameter adaptation.  Candidate score
streams and their continuous historical backtests can be generated after the
fact, but the selector only reads equity/trade rows strictly before each
period's selection cutoff.  Future rows never enter the per-period decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_state_conditioned_score_switch import _score_trade_dates, _timestamp  # noqa: E402


def main() -> None:
    args = _parse_args()
    summary = build_backtest_adaptive_state_switch(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_backtest_adaptive_state_switch(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    score_dir = output_dir / "scores" / args.method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_existing:
        for old_path in score_dir.glob("score_*.parquet"):
            old_path.unlink()

    candidate_score_dirs = _parse_named_paths(args.candidate_score)
    candidate_backtest_dirs = _parse_named_paths(args.candidate_backtest)
    missing_backtests = sorted(set(candidate_score_dirs) - set(candidate_backtest_dirs))
    if missing_backtests:
        raise ValueError(f"missing candidate backtests for: {missing_backtests}")
    candidates = tuple(candidate_score_dirs.keys())
    backtests = {
        "baseline": _load_backtest(Path(args.baseline_backtest_dir)),
        **{
            name: _load_backtest(Path(candidate_backtest_dirs[name]))
            for name in candidates
        },
    }
    selector = _selection_schedule(
        backtests,
        candidates=candidates,
        start=args.start,
        end=args.end,
        frequency=args.selection_frequency,
        lookback_days=args.selection_lookback_days,
        embargo_days=args.selection_embargo_days,
        min_equity_points=args.selection_min_equity_points,
        min_objective_edge=args.selection_min_objective_edge,
        return_weight=args.return_weight,
        drawdown_penalty=args.drawdown_penalty,
        turnover_penalty=args.turnover_penalty,
        cost_penalty=args.cost_penalty,
        fallback_to_baseline=not args.no_baseline_fallback,
    )
    schedule = _daily_schedule(selector, start=args.start, end=args.end)
    selector_path = output_dir / "backtest_adaptive_selector.csv"
    schedule_path = output_dir / "backtest_adaptive_schedule.csv"
    selector.to_csv(selector_path, index=False)
    schedule.to_csv(schedule_path, index=False)
    partition_rows = _write_selected_scores(
        baseline_score_dir=Path(args.baseline_score_dir),
        candidate_score_dirs={
            name: Path(path) for name, path in candidate_score_dirs.items()
        },
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
            "baseline_backtest_dir": args.baseline_backtest_dir,
            "candidate_score": args.candidate_score,
            "candidate_backtest": args.candidate_backtest,
            "output_dir": args.output_dir,
            "method_name": args.method_name,
            "start": args.start,
            "end": args.end,
            "selection_frequency": args.selection_frequency,
            "selection_lookback_days": args.selection_lookback_days,
            "selection_embargo_days": args.selection_embargo_days,
            "selection_min_equity_points": args.selection_min_equity_points,
            "selection_min_objective_edge": args.selection_min_objective_edge,
            "return_weight": args.return_weight,
            "drawdown_penalty": args.drawdown_penalty,
            "turnover_penalty": args.turnover_penalty,
            "cost_penalty": args.cost_penalty,
            "baseline_fallback": not args.no_baseline_fallback,
        },
        "selection": {
            "path": str(selector_path),
            "period_count": int(len(selector)),
            "baseline_period_count": int(selector["baseline_fallback"].sum())
            if not selector.empty
            else 0,
            "selected_methods": selector["selected_method"].value_counts().to_dict()
            if not selector.empty
            else {},
        },
        "schedule": {
            "path": str(schedule_path),
            "day_count": int(len(schedule)),
            "selected_methods": schedule["selected_method"].value_counts().to_dict()
            if not schedule.empty
            else {},
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


def _load_backtest(backtest_dir: Path) -> dict[str, pd.DataFrame]:
    if not backtest_dir.exists():
        raise FileNotFoundError(f"backtest dir not found: {backtest_dir}")
    equity_path = backtest_dir / "equity_curve.csv"
    trades_path = backtest_dir / "trades.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"equity curve not found: {equity_path}")
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"]).sort_values("timestamp")
    equity["trade_date"] = _local_dates(equity["timestamp"])
    if trades_path.exists():
        trades = pd.read_csv(trades_path, parse_dates=["timestamp"])
        trades["trade_date"] = _local_dates(trades["timestamp"])
    else:
        trades = pd.DataFrame(columns=["timestamp", "trade_date", "notional", "total_cost"])
    return {"equity": equity, "trades": trades}


def _selection_schedule(
    backtests: dict[str, dict[str, pd.DataFrame]],
    *,
    candidates: tuple[str, ...],
    start: str,
    end: str,
    frequency: str,
    lookback_days: int,
    embargo_days: int,
    min_equity_points: int,
    min_objective_edge: float,
    return_weight: float,
    drawdown_penalty: float,
    turnover_penalty: float,
    cost_penalty: float,
    fallback_to_baseline: bool,
) -> pd.DataFrame:
    periods = _selection_periods(start=start, end=end, frequency=frequency)
    rows: list[dict[str, Any]] = []
    for period_start, period_end in periods:
        cutoff = (period_start - pd.Timedelta(days=embargo_days)).date()
        lookback_start = (
            period_start - pd.Timedelta(days=embargo_days + lookback_days)
        ).date()
        metrics = {
            name: _window_metrics(
                backtest,
                lookback_start=lookback_start,
                cutoff=cutoff,
                min_equity_points=min_equity_points,
                return_weight=return_weight,
                drawdown_penalty=drawdown_penalty,
                turnover_penalty=turnover_penalty,
                cost_penalty=cost_penalty,
            )
            for name, backtest in backtests.items()
        }
        baseline = metrics["baseline"]
        selected_method = "baseline"
        selected_metrics = baseline
        for name in candidates:
            candidate_metrics = metrics[name]
            if candidate_metrics["valid"] is not True:
                continue
            if fallback_to_baseline and baseline["valid"] is True:
                edge = candidate_metrics["objective"] - baseline["objective"]
                if edge <= min_objective_edge:
                    continue
            if (
                selected_metrics["valid"] is not True
                or candidate_metrics["objective"] > selected_metrics["objective"]
            ):
                selected_method = name
                selected_metrics = candidate_metrics
        baseline_objective = baseline.get("objective")
        selected_objective = selected_metrics.get("objective")
        row = {
            "period_start": str(period_start.date()),
            "period_end": str(period_end.date()),
            "selection_cutoff": str(cutoff),
            "lookback_start": str(lookback_start),
            "selected_method": selected_method,
            "baseline_fallback": selected_method == "baseline",
            "baseline_objective": baseline_objective,
            "selected_objective": selected_objective,
            "selected_objective_edge": (
                selected_objective - baseline_objective
                if selected_objective is not None and baseline_objective is not None
                else None
            ),
        }
        for name, payload in metrics.items():
            prefix = _method_column_prefix(name)
            for key, value in payload.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _window_metrics(
    backtest: dict[str, pd.DataFrame],
    *,
    lookback_start: object,
    cutoff: object,
    min_equity_points: int,
    return_weight: float,
    drawdown_penalty: float,
    turnover_penalty: float,
    cost_penalty: float,
) -> dict[str, Any]:
    equity = backtest["equity"]
    trades = backtest["trades"]
    equity_window = equity.loc[
        (equity["trade_date"] >= lookback_start) & (equity["trade_date"] < cutoff)
    ].copy()
    if len(equity_window) < min_equity_points:
        return {
            "valid": False,
            "equity_point_count": int(len(equity_window)),
            "total_return": None,
            "max_drawdown": None,
            "gross_turnover": None,
            "cost_return": None,
            "trade_count": None,
            "objective": None,
        }
    start_equity = float(equity_window["equity"].iloc[0])
    end_equity = float(equity_window["equity"].iloc[-1])
    curve = equity_window["equity"].astype(float)
    max_drawdown = float((curve / curve.cummax() - 1.0).min())
    trade_window = trades.loc[
        (trades["trade_date"] >= lookback_start) & (trades["trade_date"] < cutoff)
    ].copy()
    gross_notional = (
        float(trade_window["notional"].abs().sum())
        if "notional" in trade_window
        else 0.0
    )
    total_cost = (
        float(trade_window["total_cost"].sum())
        if "total_cost" in trade_window
        else 0.0
    )
    total_return = end_equity / start_equity - 1.0 if start_equity > 0 else 0.0
    gross_turnover = gross_notional / start_equity if start_equity > 0 else 0.0
    cost_return = total_cost / start_equity if start_equity > 0 else 0.0
    objective = (
        return_weight * total_return
        + drawdown_penalty * max_drawdown
        - turnover_penalty * gross_turnover
        - cost_penalty * cost_return
    )
    return {
        "valid": True,
        "equity_point_count": int(len(equity_window)),
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "gross_turnover": gross_turnover,
        "cost_return": cost_return,
        "trade_count": int(len(trade_window)),
        "objective": objective,
    }


def _daily_schedule(
    selector: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_date = _timestamp(start).tz_convert("Asia/Shanghai").date()
    end_date = _timestamp(end).tz_convert("Asia/Shanghai").date()
    rows: list[dict[str, Any]] = []
    for row in selector.itertuples(index=False):
        period_start = max(pd.Timestamp(row.period_start).date(), start_date)
        period_end = min(pd.Timestamp(row.period_end).date(), end_date)
        for day in pd.date_range(period_start, period_end, freq="D"):
            rows.append(
                {
                    "trade_date": str(day.date()),
                    "selected_method": row.selected_method,
                    "baseline_fallback": bool(row.baseline_fallback),
                    "selected_objective": row.selected_objective,
                    "baseline_objective": row.baseline_objective,
                    "selected_objective_edge": row.selected_objective_edge,
                }
            )
    return pd.DataFrame(rows)


def _write_selected_scores(
    *,
    baseline_score_dir: Path,
    candidate_score_dirs: dict[str, Path],
    score_dir: Path,
    schedule: pd.DataFrame,
    start: str,
    end: str,
    resume_existing: bool,
) -> dict[str, int]:
    start_partition = _timestamp(start).tz_convert("Asia/Shanghai").strftime("%Y_%m")
    end_partition = _timestamp(end).tz_convert("Asia/Shanghai").strftime("%Y_%m")
    method_by_date = dict(zip(schedule["trade_date"], schedule["selected_method"], strict=False))
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
        baseline = pd.read_parquet(
            baseline_path,
            columns=["timestamp", "instrument_id", "score"],
        )
        baseline["trade_date"] = _score_trade_dates(baseline["timestamp"])
        frames: list[pd.DataFrame] = []
        partition_dates = sorted(set(baseline["trade_date"].astype(str)))
        for method in sorted(set(method_by_date.get(date, "baseline") for date in partition_dates)):
            selected_dates = {
                date
                for date in partition_dates
                if method_by_date.get(date, "baseline") == method
            }
            if not selected_dates:
                continue
            if method == "baseline":
                frame = baseline.loc[
                    baseline["trade_date"].isin(selected_dates),
                    ["timestamp", "instrument_id", "score"],
                ].copy()
                frame["signal_source"] = "baseline"
            else:
                candidate_path = candidate_score_dirs[method] / f"score_{partition}.parquet"
                if not candidate_path.exists():
                    frame = baseline.loc[
                        baseline["trade_date"].isin(selected_dates),
                        ["timestamp", "instrument_id", "score"],
                    ].copy()
                    frame["signal_source"] = "baseline"
                else:
                    frame = pd.read_parquet(candidate_path)
                    frame["trade_date"] = _score_trade_dates(frame["timestamp"])
                    frame = frame.loc[frame["trade_date"].isin(selected_dates)].copy()
                    if "signal_source" not in frame.columns:
                        frame["signal_source"] = method
                    frame = frame.loc[
                        :, ["timestamp", "instrument_id", "score", "signal_source"]
                    ]
            frames.append(frame)
        output = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["timestamp", "instrument_id", "score", "signal_source"])
        )
        output = output.sort_values(
            ["timestamp", "score", "instrument_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
        output.to_parquet(output_path, index=False)
        rows_by_partition[partition] = int(len(output))
    return rows_by_partition


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
        next_start = (
            starts[index + 1]
            if index + 1 < len(starts)
            else end_at + pd.Timedelta(days=1)
        )
        period_end = min(next_start - pd.Timedelta(days=1), end_at)
        if period_start <= period_end:
            periods.append((period_start, period_end))
    return periods


def _local_dates(timestamp: pd.Series) -> pd.Series:
    return pd.to_datetime(timestamp, utc=True).dt.tz_convert("Asia/Shanghai").dt.date


def _parse_named_paths(values: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise ValueError("named paths must be name=path")
        if name in output:
            raise ValueError(f"duplicate named path: {name}")
        output[name] = path
    return output


def _method_column_prefix(name: str) -> str:
    return (
        name.replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-score-dir", required=True)
    parser.add_argument("--baseline-backtest-dir", required=True)
    parser.add_argument(
        "--candidate-score",
        action="append",
        default=[],
        help="candidate score directory as name=path; repeatable",
    )
    parser.add_argument(
        "--candidate-backtest",
        action="append",
        default=[],
        help="candidate backtest directory as name=path; repeatable",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="backtest_adaptive_state_switch")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--selection-frequency",
        choices=("monthly", "weekly"),
        default="monthly",
    )
    parser.add_argument("--selection-lookback-days", type=int, default=252)
    parser.add_argument("--selection-embargo-days", type=int, default=3)
    parser.add_argument("--selection-min-equity-points", type=int, default=40)
    parser.add_argument("--selection-min-objective-edge", type=float, default=0.0)
    parser.add_argument("--return-weight", type=float, default=1.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.5)
    parser.add_argument("--turnover-penalty", type=float, default=0.001)
    parser.add_argument("--cost-penalty", type=float, default=0.0)
    parser.add_argument("--no-baseline-fallback", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("baseline_score_dir", "baseline_backtest_dir"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not args.candidate_score:
        raise ValueError("--candidate-score must be provided")
    if not args.candidate_backtest:
        raise ValueError("--candidate-backtest must be provided")
    for item in args.candidate_score:
        _, _, path = item.partition("=")
        if not Path(path).exists():
            raise FileNotFoundError(f"candidate score dir not found: {path}")
    for item in args.candidate_backtest:
        _, _, path = item.partition("=")
        if not Path(path).exists():
            raise FileNotFoundError(f"candidate backtest dir not found: {path}")
    if args.selection_lookback_days <= 0:
        raise ValueError("--selection-lookback-days must be positive")
    if args.selection_embargo_days < 0:
        raise ValueError("--selection-embargo-days must be non-negative")
    if args.selection_min_equity_points <= 0:
        raise ValueError("--selection-min-equity-points must be positive")
    if args.drawdown_penalty < 0:
        raise ValueError("--drawdown-penalty must be non-negative")
    if args.turnover_penalty < 0:
        raise ValueError("--turnover-penalty must be non-negative")
    if args.cost_penalty < 0:
        raise ValueError("--cost-penalty must be non-negative")


if __name__ == "__main__":
    main()
