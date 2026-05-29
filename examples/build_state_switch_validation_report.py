"""Build a validation report for a frozen state-switch challenger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    args = _parse_args()
    report = build_state_switch_validation_report(args)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


def build_state_switch_validation_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _scenario_rows(args)
    monthly = _period_performance(
        Path(args.baseline_backtest_dir),
        Path(args.challenger_backtest_dir),
        schedule_path=Path(args.schedule_path),
        freq="M",
        initial_cash=args.initial_cash,
    )
    quarterly = _period_performance(
        Path(args.baseline_backtest_dir),
        Path(args.challenger_backtest_dir),
        schedule_path=Path(args.schedule_path),
        freq="Q",
        initial_cash=args.initial_cash,
    )
    trade_profile = _trade_profile(
        Path(args.baseline_backtest_dir),
        Path(args.challenger_backtest_dir),
    )
    scenarios.to_csv(output_dir / "scenario_summary.csv", index=False)
    monthly.to_csv(output_dir / "monthly_performance.csv", index=False)
    quarterly.to_csv(output_dir / "quarterly_performance.csv", index=False)
    trade_profile.to_csv(output_dir / "trade_profile.csv", index=False)
    report = {
        "status": "completed",
        "params": {
            "baseline_backtest_dir": args.baseline_backtest_dir,
            "challenger_backtest_dir": args.challenger_backtest_dir,
            "schedule_path": args.schedule_path,
            "output_dir": args.output_dir,
            "scenario_summaries": args.scenario_summary,
        },
        "headline": _headline(scenarios, monthly, quarterly),
        "outputs": {
            "scenario_summary": str(output_dir / "scenario_summary.csv"),
            "monthly_performance": str(output_dir / "monthly_performance.csv"),
            "quarterly_performance": str(output_dir / "quarterly_performance.csv"),
            "trade_profile": str(output_dir / "trade_profile.csv"),
            "markdown": str(output_dir / "validation_report.md"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_report.md").write_text(
        _markdown_report(report, scenarios, monthly, quarterly, trade_profile),
        encoding="utf-8",
    )
    return report


def _scenario_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in args.scenario_summary:
        name, path_text = _split_named_path(item)
        summary = json.loads(Path(path_text).read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        diagnostics = summary.get("policy_diagnostics", {})
        execution = summary.get("execution_constraint_counts", {})
        rows.append(
            {
                "scenario": name,
                "total_return": float(metrics["total_return"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "gross_turnover": float(metrics.get("gross_turnover", 0.0)),
                "trade_count": float(metrics.get("trade_count", 0.0)),
                "total_transaction_cost": float(
                    metrics.get("total_transaction_cost", 0.0)
                ),
                "source_transition_forced_exit_count": int(
                    diagnostics.get("source_transition_forced_exit_count", 0)
                ),
                "turnover_scaled_count": int(
                    diagnostics.get("turnover_scaled_count", 0)
                ),
                "capacity_limited_event_count": int(
                    execution.get("capacity_limited_event_count", 0)
                ),
                "capacity_unfilled_notional": float(
                    execution.get("capacity_unfilled_notional", 0.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def _period_performance(
    baseline_dir: Path,
    challenger_dir: Path,
    *,
    schedule_path: Path,
    freq: str,
    initial_cash: float,
) -> pd.DataFrame:
    schedule_dates = _schedule_date_bounds(schedule_path)
    baseline = _single_period_performance(
        baseline_dir,
        method="baseline",
        freq=freq,
        initial_cash=initial_cash,
        schedule_dates=schedule_dates,
    )
    challenger = _single_period_performance(
        challenger_dir,
        method="challenger",
        freq=freq,
        initial_cash=initial_cash,
        schedule_dates=schedule_dates,
    )
    frame = baseline.merge(challenger, on="period", how="outer", sort=True)
    frame["return_delta"] = frame["challenger_return"] - frame["baseline_return"]
    active = _active_days_by_period(schedule_path, freq=freq)
    return frame.merge(active, on="period", how="left").fillna({"active_days": 0})


def _single_period_performance(
    backtest_dir: Path,
    *,
    method: str,
    freq: str,
    initial_cash: float,
    schedule_dates: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    equity = pd.read_csv(backtest_dir / "equity_curve.csv", parse_dates=["timestamp"])
    trades = pd.read_csv(backtest_dir / "trades.csv", parse_dates=["timestamp"])
    equity = equity.sort_values("timestamp")
    start_date, end_date = schedule_dates
    equity_dates = equity["timestamp"].dt.tz_convert("Asia/Shanghai").dt.normalize()
    trade_dates = trades["timestamp"].dt.tz_convert("Asia/Shanghai").dt.normalize()
    equity = equity.loc[(equity_dates >= start_date) & (equity_dates <= end_date)].copy()
    trades = trades.loc[(trade_dates >= start_date) & (trade_dates <= end_date)].copy()
    equity["period"] = _period_labels(equity["timestamp"], freq)
    trades["period"] = _period_labels(trades["timestamp"], freq)
    previous_equity = float(initial_cash)
    rows: list[dict[str, Any]] = []
    for period, group in equity.groupby("period", sort=True):
        end_equity = float(group["equity"].iloc[-1])
        curve = pd.concat(
            [pd.Series([previous_equity]), group["equity"].astype(float)],
            ignore_index=True,
        )
        period_trades = trades.loc[trades["period"] == period]
        rows.append(
            {
                "period": period,
                f"{method}_return": end_equity / previous_equity - 1.0,
                f"{method}_max_drawdown": float((curve / curve.cummax() - 1.0).min()),
                f"{method}_trade_count": int(len(period_trades)),
                f"{method}_cost": float(period_trades["total_cost"].sum())
                if not period_trades.empty
                else 0.0,
                f"{method}_gross_notional": float(period_trades["notional"].abs().sum())
                if not period_trades.empty
                else 0.0,
            }
        )
        previous_equity = end_equity
    return pd.DataFrame(rows)


def _active_days_by_period(schedule_path: Path, *, freq: str) -> pd.DataFrame:
    schedule = pd.read_csv(schedule_path, parse_dates=["trade_date"])
    schedule["period"] = _period_labels(schedule["trade_date"], freq)
    active = (
        schedule.groupby("period", as_index=False)["active"]
        .sum()
        .rename(columns={"active": "active_days"})
    )
    active["active_days"] = active["active_days"].astype(int)
    return active


def _schedule_date_bounds(schedule_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    schedule = pd.read_csv(schedule_path, parse_dates=["trade_date"])
    dates = pd.to_datetime(schedule["trade_date"]).dt.tz_localize("Asia/Shanghai")
    dates = dates.dt.normalize()
    return dates.min(), dates.max()


def _period_labels(values: pd.Series, freq: str) -> pd.Series:
    timestamps = pd.to_datetime(values)
    if freq == "M":
        return timestamps.dt.strftime("%Y-%m")
    if freq == "Q":
        quarter = ((timestamps.dt.month - 1) // 3 + 1).astype(str)
        return timestamps.dt.year.astype(str) + "Q" + quarter
    raise ValueError(f"unsupported frequency: {freq}")


def _trade_profile(baseline_dir: Path, challenger_dir: Path) -> pd.DataFrame:
    rows = []
    for method, path in (("baseline", baseline_dir), ("challenger", challenger_dir)):
        trades = pd.read_csv(path / "trades.csv", parse_dates=["timestamp"])
        by_day = trades.assign(day=trades["timestamp"].dt.strftime("%Y-%m-%d"))
        daily_notional = by_day.groupby("day")["notional"].apply(lambda s: s.abs().sum())
        rows.append(
            {
                "method": method,
                "trade_count": int(len(trades)),
                "instrument_count": int(trades["instrument_id"].nunique()),
                "gross_notional": float(trades["notional"].abs().sum()),
                "total_cost": float(trades["total_cost"].sum()),
                "avg_daily_gross_notional": float(daily_notional.mean()),
                "p95_daily_gross_notional": float(daily_notional.quantile(0.95)),
                "max_daily_gross_notional": float(daily_notional.max()),
            }
        )
    return pd.DataFrame(rows)


def _headline(
    scenarios: pd.DataFrame,
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> dict[str, Any]:
    standard = scenarios.loc[scenarios["scenario"].eq("standard")]
    if standard.empty:
        standard = scenarios.head(1)
    row = standard.iloc[0]
    return {
        "standard_total_return": float(row["total_return"]),
        "standard_max_drawdown": float(row["max_drawdown"]),
        "standard_gross_turnover": float(row["gross_turnover"]),
        "positive_month_delta_count": int((monthly["return_delta"] > 0).sum()),
        "month_count": int(len(monthly)),
        "positive_quarter_delta_count": int((quarterly["return_delta"] > 0).sum()),
        "quarter_count": int(len(quarterly)),
    }


def _markdown_report(
    report: dict[str, Any],
    scenarios: pd.DataFrame,
    monthly: pd.DataFrame,
    quarterly: pd.DataFrame,
    trade_profile: pd.DataFrame,
) -> str:
    lines = [
        "# State Switch Validation Report",
        "",
        "## Headline",
        "",
        json.dumps(report["headline"], ensure_ascii=False, indent=2),
        "",
        "## Scenarios",
        "",
        _markdown_table(scenarios),
        "",
        "## Quarterly Performance",
        "",
        _markdown_table(quarterly),
        "",
        "## Trade Profile",
        "",
        _markdown_table(trade_profile),
        "",
        "## Monthly Performance",
        "",
        _markdown_table(monthly),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = [_format_markdown_value(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _split_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, value
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"invalid scenario summary: {value}")
    return name, path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-backtest-dir", required=True)
    parser.add_argument("--challenger-backtest-dir", required=True)
    parser.add_argument("--schedule-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument(
        "--scenario-summary",
        action="append",
        required=True,
        help="Named summary path, e.g. standard=path/to/summary.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
