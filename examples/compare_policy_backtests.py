"""Compare two score-policy backtest outputs overall and by month."""

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

from quant_research.metrics.risk import max_drawdown


def main() -> None:
    args = _parse_args()
    summary = compare_policy_backtests(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def compare_policy_backtests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    overall = pd.DataFrame(
        [
            _overall_row(args.baseline_name, baseline_dir),
            _overall_row(args.candidate_name, candidate_dir),
        ]
    )
    monthly = _monthly_comparison(
        baseline_dir,
        candidate_dir,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        initial_cash=args.initial_cash,
        start=args.start,
        end=args.end,
    )
    overall_path = output_dir / "overall_comparison.csv"
    monthly_path = output_dir / "monthly_comparison.csv"
    summary_path = output_dir / "summary.json"
    overall.to_csv(overall_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    summary = {
        "params": {
            "baseline_dir": str(baseline_dir),
            "candidate_dir": str(candidate_dir),
            "baseline_name": args.baseline_name,
            "candidate_name": args.candidate_name,
            "initial_cash": args.initial_cash,
            "start": args.start,
            "end": args.end,
        },
        "artifacts": {
            "overall_comparison": str(overall_path),
            "monthly_comparison": str(monthly_path),
            "summary": str(summary_path),
        },
        "overall": overall.to_dict("records"),
        "monthly": monthly.to_dict("records"),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _overall_row(name: str, backtest_dir: Path) -> dict[str, Any]:
    payload = json.loads((backtest_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    diagnostics = payload.get("policy_diagnostics", {})
    return {
        "variant": name,
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "gross_turnover": metrics.get("gross_turnover"),
        "trade_count": metrics.get("trade_count"),
        "total_transaction_cost": metrics.get("total_transaction_cost"),
        "final_equity": metrics.get("final_equity"),
        "average_target_gross_exposure": diagnostics.get("average_target_gross_exposure"),
        "planned_gross_turnover": diagnostics.get("planned_gross_turnover"),
        "risk_reduction_count": diagnostics.get("risk_reduction_count"),
    }


def _monthly_comparison(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    baseline_name: str,
    candidate_name: str,
    initial_cash: float,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    baseline = _monthly_rows(
        baseline_dir,
        variant=baseline_name,
        initial_cash=initial_cash,
        start=start,
        end=end,
    )
    candidate = _monthly_rows(
        candidate_dir,
        variant=candidate_name,
        initial_cash=initial_cash,
        start=start,
        end=end,
    )
    merged = baseline.merge(candidate, on="month", suffixes=("_baseline", "_candidate"))
    merged["return_delta"] = merged["return_candidate"] - merged["return_baseline"]
    merged["max_drawdown_delta"] = (
        merged["max_drawdown_candidate"] - merged["max_drawdown_baseline"]
    )
    merged["trade_count_delta"] = (
        merged["trade_count_candidate"] - merged["trade_count_baseline"]
    )
    merged["total_transaction_cost_delta"] = (
        merged["total_transaction_cost_candidate"]
        - merged["total_transaction_cost_baseline"]
    )
    return merged


def _monthly_rows(
    backtest_dir: Path,
    *,
    variant: str,
    initial_cash: float,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    equity_path = backtest_dir / "equity_curve.csv"
    trades_path = backtest_dir / "trades.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"missing equity curve: {equity_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"missing trades file: {trades_path}")
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"])
    trades = pd.read_csv(trades_path, parse_dates=["timestamp"])
    if start is not None:
        start_ts = pd.Timestamp(start)
        equity = equity[equity["timestamp"] >= start_ts].copy()
        trades = trades[trades["timestamp"] >= start_ts].copy()
    if end is not None:
        end_ts = pd.Timestamp(end)
        equity = equity[equity["timestamp"] <= end_ts].copy()
        trades = trades[trades["timestamp"] <= end_ts].copy()
    if equity.empty:
        return pd.DataFrame()
    equity["month"] = equity["timestamp"].dt.strftime("%Y-%m")
    if trades.empty:
        trades["month"] = pd.Series(dtype="object")
    else:
        trades["month"] = trades["timestamp"].dt.strftime("%Y-%m")
    previous_equity = float(initial_cash)
    rows: list[dict[str, Any]] = []
    for month, month_end_row in equity.groupby("month").tail(1).set_index("month").iterrows():
        month_equity = equity[equity["month"] == month]
        end_equity = float(month_end_row["equity"])
        curve = [previous_equity, *month_equity["equity"].astype(float).tolist()]
        month_trades = trades[trades["month"] == month]
        rows.append(
            {
                "variant": variant,
                "month": str(month),
                "return": end_equity / previous_equity - 1.0,
                "end_equity": end_equity,
                "max_drawdown": max_drawdown(curve),
                "trade_count": int(len(month_trades)),
                "total_transaction_cost": (
                    float(month_trades["total_cost"].sum())
                    if not month_trades.empty
                    else 0.0
                ),
                "gross_traded_notional": (
                    float(month_trades["notional"].abs().sum())
                    if not month_trades.empty
                    else 0.0
                ),
            }
        )
        previous_equity = end_equity
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    if args.initial_cash <= 0:
        raise ValueError("--initial-cash must be positive")
    return args


if __name__ == "__main__":
    main()
