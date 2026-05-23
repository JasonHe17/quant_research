"""Review a frozen score-overlay replay against a baseline score portfolio."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class PortfolioSpec:
    name: str
    summary_path: Path
    method: str
    policy: str


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate = PortfolioSpec(
        name=args.candidate_name,
        summary_path=Path(args.candidate_summary),
        method=args.candidate_method,
        policy=args.policy,
    )
    fallback = PortfolioSpec(
        name=args.fallback_name,
        summary_path=Path(args.candidate_summary),
        method=args.fallback_method,
        policy=args.policy,
    )
    baseline = PortfolioSpec(
        name=args.baseline_name,
        summary_path=Path(args.baseline_summary),
        method=args.baseline_method,
        policy=args.policy,
    )
    active = PortfolioSpec(
        name=args.active_name,
        summary_path=Path(args.active_summary),
        method=args.active_method,
        policy=args.policy,
    )

    summaries = {
        "candidate": _portfolio_summary(candidate),
        "fallback": _portfolio_summary(fallback),
        "baseline": _portfolio_summary(baseline),
        "active": _portfolio_summary(active),
    }
    monthly = _monthly_checks(
        candidate=candidate,
        baseline=baseline,
        start_month=args.start_month,
        end_month=args.end_month,
        output_dir=output_dir,
    )
    checks = _checks(
        summaries=summaries,
        monthly=monthly,
        true_unseen_available=args.true_unseen_available,
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_type": "score_overlay_frozen_promotion_review",
        "subject": args.subject,
        "candidate": _spec_json(candidate),
        "fallback": _spec_json(fallback),
        "baseline": _spec_json(baseline),
        "active": _spec_json(active),
        "true_unseen_available": args.true_unseen_available,
        "default_strategy_change": False,
        "decision": _decision(checks),
        "summaries": summaries,
        "monthly_concentration": monthly,
        "checks": checks,
        "artifacts": {
            "review_json": str(output_dir / "promotion_review.json"),
            "review_markdown": str(output_dir / "promotion_review.md"),
            "monthly_full_base_csv": str(output_dir / "monthly_full_base.csv"),
            "monthly_full_high_cost_csv": str(output_dir / "monthly_full_high_cost.csv"),
        },
    }
    (output_dir / "promotion_review.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "promotion_review.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "decision": report["decision"]}, ensure_ascii=True))


def _portfolio_summary(spec: PortfolioSpec) -> dict[str, Any]:
    payload = json.loads(spec.summary_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("results", [])
        if row.get("method") == spec.method and row.get("policy") == spec.policy
    ]
    by_scenario = {str(row["scenario"]): row for row in rows}
    leaderboard = [
        row
        for row in payload.get("policy_leaderboard", [])
        if row.get("method") == spec.method and row.get("policy") == spec.policy
    ]
    return {
        "name": spec.name,
        "summary_path": str(spec.summary_path),
        "method": spec.method,
        "policy": spec.policy,
        "validation": payload.get("validation", {}),
        "leaderboard": leaderboard[0] if leaderboard else {},
        "scenarios": {
            scenario: {
                "total_return": _num(row.get("total_return")),
                "full_base_return": _num(row.get("full_base_return")),
                "final_equity": _num(row.get("final_equity")),
                "gross_turnover": _num(row.get("gross_turnover")),
                "max_drawdown": _num(row.get("max_drawdown")),
                "total_transaction_cost": _num(row.get("total_transaction_cost")),
                "trade_count": _num(row.get("trade_count")),
            }
            for scenario, row in sorted(by_scenario.items())
        },
    }


def _monthly_checks(
    *,
    candidate: PortfolioSpec,
    baseline: PortfolioSpec,
    start_month: str,
    end_month: str,
    output_dir: Path,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for scenario, filename in (
        ("full_base", "monthly_full_base.csv"),
        ("full_high_cost", "monthly_full_high_cost.csv"),
    ):
        candidate_returns = _monthly_returns(
            _equity_curve_path(candidate, scenario),
            start_month=start_month,
            end_month=end_month,
        )
        baseline_returns = _monthly_returns(
            _equity_curve_path(baseline, scenario),
            start_month=start_month,
            end_month=end_month,
        )
        frame = pd.DataFrame(
            {
                "candidate_return": candidate_returns,
                "baseline_return": baseline_returns,
            }
        ).dropna()
        frame["delta_return"] = frame["candidate_return"] - frame["baseline_return"]
        frame.index = frame.index.astype(str)
        frame.index.name = "month"
        frame.to_csv(output_dir / filename)
        checks[scenario] = {
            "candidate": _concentration(frame["candidate_return"]),
            "baseline": _concentration(frame["baseline_return"]),
            "delta_candidate_vs_baseline": _concentration(frame["delta_return"]),
            "candidate_positive_months": int((frame["candidate_return"] > 0).sum()),
            "baseline_positive_months": int((frame["baseline_return"] > 0).sum()),
            "month_count": int(len(frame)),
            "csv": str(output_dir / filename),
        }
    return checks


def _monthly_returns(path: Path, *, start_month: str, end_month: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp", "equity"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Asia/Shanghai")
    frame = frame.sort_values("timestamp")
    frame["month"] = frame["timestamp"].dt.tz_localize(None).dt.to_period("M")
    month_end = frame.groupby("month", sort=True)["equity"].last()
    returns = month_end.pct_change().dropna()
    return returns.loc[start_month:end_month]


def _equity_curve_path(spec: PortfolioSpec, scenario: str) -> Path:
    path = (
        spec.summary_path.parent
        / scenario
        / "backtests"
        / spec.method
        / spec.policy
        / "equity_curve.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _concentration(values: pd.Series) -> dict[str, Any]:
    abs_values = values.abs().sort_values(ascending=False)
    abs_sum = float(abs_values.sum())
    return {
        "sum_return": float(values.sum()),
        "positive_months": int((values > 0).sum()),
        "worst_month_return": float(values.min()),
        "best_month_return": float(values.max()),
        "top3_abs_share": float(abs_values.head(3).sum() / abs_sum) if abs_sum else None,
        "top5_abs_share": float(abs_values.head(5).sum() / abs_sum) if abs_sum else None,
    }


def _checks(
    *,
    summaries: dict[str, Any],
    monthly: dict[str, Any],
    true_unseen_available: bool,
) -> list[dict[str, Any]]:
    candidate = summaries["candidate"]
    fallback = summaries["fallback"]
    baseline = summaries["baseline"]
    checks = [
        _check(
            "frozen_replay_completed",
            "pass",
            "Fixed 5% candidate and 2.5% fallback replay completed.",
        ),
        _gt_check(
            "candidate_beats_baseline_full_base",
            _scenario(candidate, "full_base", "total_return"),
            _scenario(baseline, "full_base", "total_return"),
        ),
        _gt_check(
            "candidate_beats_baseline_full_high_cost",
            _scenario(candidate, "full_high_cost", "total_return"),
            _scenario(baseline, "full_high_cost", "total_return"),
        ),
        _gt_check(
            "candidate_drawdown_improves_baseline",
            _scenario(candidate, "full_base", "max_drawdown"),
            _scenario(baseline, "full_base", "max_drawdown"),
        ),
        _lt_check(
            "candidate_turnover_not_higher_than_baseline",
            _scenario(candidate, "full_base", "gross_turnover"),
            _scenario(baseline, "full_base", "gross_turnover"),
        ),
        _check(
            "candidate_2023_positive",
            "pass" if _scenario(candidate, "year_2023_base", "total_return") > 0 else "warn",
            "Promotion to default still waits on the negative 2023 annual slice.",
            {
                "candidate_2023_return": _scenario(candidate, "year_2023_base", "total_return"),
            },
        ),
        _check(
            "fallback_repairs_2023_vs_candidate_and_baseline",
            "pass"
            if _scenario(fallback, "year_2023_base", "total_return")
            > max(
                _scenario(candidate, "year_2023_base", "total_return"),
                _scenario(baseline, "year_2023_base", "total_return"),
            )
            else "warn",
            "The 2.5% overlay is the stability fallback if annual-slice stability is prioritized.",
            {
                "fallback_2023_return": _scenario(fallback, "year_2023_base", "total_return"),
                "candidate_2023_return": _scenario(candidate, "year_2023_base", "total_return"),
                "baseline_2023_return": _scenario(baseline, "year_2023_base", "total_return"),
            },
        ),
        _monthly_check("monthly_concentration_full_base", monthly["full_base"]),
        _monthly_check("monthly_concentration_full_high_cost", monthly["full_high_cost"]),
        _check(
            "true_unseen_data_available",
            "pass" if true_unseen_available else "warn",
            "No post-2025 overlay score partitions were available in this replay.",
        ),
    ]
    return checks


def _monthly_check(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    candidate_top3 = payload["candidate"]["top3_abs_share"]
    delta_top3 = payload["delta_candidate_vs_baseline"]["top3_abs_share"]
    status = (
        "pass"
        if candidate_top3 is not None
        and delta_top3 is not None
        and candidate_top3 <= 0.40
        and delta_top3 <= 0.45
        else "warn"
    )
    return _check(
        name,
        status,
        "Pass if absolute monthly returns and deltas are not dominated by the top three months.",
        {
            "candidate_top3_abs_share": candidate_top3,
            "delta_top3_abs_share": delta_top3,
            "month_count": payload["month_count"],
        },
    )


def _gt_check(name: str, current: float, baseline: float) -> dict[str, Any]:
    return _check(
        name,
        "pass" if current > baseline else "fail",
        "Pass if current is greater than baseline.",
        {"current": current, "baseline": baseline, "delta": current - baseline},
    )


def _lt_check(name: str, current: float, baseline: float) -> dict[str, Any]:
    return _check(
        name,
        "pass" if current <= baseline else "warn",
        "Pass if current is less than or equal to baseline.",
        {"current": current, "baseline": baseline, "delta": current - baseline},
    )


def _check(
    name: str,
    status: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "details": details or {},
    }


def _decision(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "fail" for check in checks):
        return "rejected_by_frozen_promotion_review"
    if any(check["status"] == "warn" for check in checks):
        return "accepted_as_frozen_challenger_no_default_change"
    return "accepted_for_default_change_review"


def _scenario(summary: dict[str, Any], scenario: str, metric: str) -> float:
    return _num(summary.get("scenarios", {}).get(scenario, {}).get(metric))


def _spec_json(spec: PortfolioSpec) -> dict[str, str]:
    return {
        "name": spec.name,
        "summary_path": str(spec.summary_path),
        "method": spec.method,
        "policy": spec.policy,
    }


def _markdown(report: dict[str, Any]) -> str:
    candidate = report["summaries"]["candidate"]
    fallback = report["summaries"]["fallback"]
    baseline = report["summaries"]["baseline"]
    active = report["summaries"]["active"]
    lines = [
        "# Score Overlay Frozen Promotion Review",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Decision",
        "",
        f"Decision: `{report['decision']}`.",
        "",
        "Default strategy change remains `false`. This is a frozen replay, not a true unseen-data replay, because no post-2025 overlay score partitions were available.",
        "",
        "## Replay Metrics",
        "",
        "| Portfolio | Full return | High-cost return | Zero-cost return | 2023 | 2024 | 2025 | Full max DD | Full turnover |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _metric_row(candidate),
        _metric_row(fallback),
        _metric_row(baseline),
        _metric_row(active),
        "",
        "## Gate Results",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for check in report["checks"]:
        lines.append(f"| `{check['name']}` | {check['status'].upper()} | {check['reason']} |")
    lines.extend(
        [
            "",
            "## Monthly Concentration",
            "",
            "| Scenario | Candidate positive months | Baseline positive months | Candidate top-3 abs share | Delta top-3 abs share | Candidate worst month |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario, payload in report["monthly_concentration"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{scenario}`",
                    str(payload["candidate_positive_months"]),
                    str(payload["baseline_positive_months"]),
                    _pct(payload["candidate"]["top3_abs_share"]),
                    _pct(payload["delta_candidate_vs_baseline"]["top3_abs_share"]),
                    _pct(payload["candidate"]["worst_month_return"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "- Keep the 5% overlay as the preferred frozen challenger.",
            "- Keep the 2.5% overlay as the stability fallback if annual-slice repair is prioritized.",
            "- Do not switch the default strategy until a true unseen-data replay is available and the 2023-style annual weakness is resolved or explicitly accepted.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, path in report["artifacts"].items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def _metric_row(summary: dict[str, Any]) -> str:
    return (
        f"| {summary['name']} | "
        f"{_pct(_scenario(summary, 'full_base', 'total_return'))} | "
        f"{_pct(_scenario(summary, 'full_high_cost', 'total_return'))} | "
        f"{_pct(_scenario(summary, 'full_zero_cost', 'total_return'))} | "
        f"{_pct(_scenario(summary, 'year_2023_base', 'total_return'))} | "
        f"{_pct(_scenario(summary, 'year_2024_base', 'total_return'))} | "
        f"{_pct(_scenario(summary, 'year_2025_base', 'total_return'))} | "
        f"{_pct(_scenario(summary, 'full_base', 'max_drawdown'))} | "
        f"{_num(summary.get('scenarios', {}).get('full_base', {}).get('gross_turnover')):.3f} |"
    )


def _pct(value: Any) -> str:
    number = _num(value)
    if pd.isna(number):
        return ""
    return f"{number:.2%}"


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--candidate-name", default="fixed 5% overlay")
    parser.add_argument("--fallback-method", required=True)
    parser.add_argument("--fallback-name", default="fixed 2.5% overlay")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--baseline-method", default="equal")
    parser.add_argument("--baseline-name", default="daily-MA frontier")
    parser.add_argument("--active-summary", required=True)
    parser.add_argument("--active-method", default="equal")
    parser.add_argument("--active-name", default="active/default")
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-month", default="2023-01")
    parser.add_argument("--end-month", default="2025-12")
    parser.add_argument("--true-unseen-available", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
