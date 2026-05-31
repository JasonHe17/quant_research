"""Summarize candidate policy monthly attribution from score diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    args = _parse_args()
    summary = summarize_candidate_policy_attribution(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def summarize_candidate_policy_attribution(args: argparse.Namespace) -> dict[str, Any]:
    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pnl = _load_monthly_pnl(
        validation_dir,
        scenario=args.scenario,
        method=args.method,
        policy=args.policy,
        year=args.year,
    )
    diagnostics_dir = (
        validation_dir
        / args.score_scenario
        / "scores"
        / args.method
        / "diagnostics"
    )
    monthly_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for month in pnl["month"].astype(str):
        diagnostics = _load_month_diagnostics(diagnostics_dir, month)
        monthly_rows.append(_summarize_month(month, diagnostics))
        feature_rows.extend(_summarize_dominant_features(month, diagnostics))

    monthly = pnl.merge(pd.DataFrame(monthly_rows), on="month", how="left")
    features = pd.DataFrame(feature_rows)
    monthly_path = output_dir / "monthly_score_contribution_attribution.csv"
    features_path = output_dir / "dominant_feature_attribution_by_month.csv"
    report_path = output_dir / "attribution_report.md"
    monthly.to_csv(monthly_path, index=False)
    features.to_csv(features_path, index=False)
    report_path.write_text(
        _build_report(args, monthly=monthly, features=features),
        encoding="utf-8",
    )
    return {
        "artifacts": {
            "monthly": str(monthly_path),
            "dominant_features": str(features_path),
            "report": str(report_path),
        },
        "worst_months": monthly.sort_values("return")
        .head(args.report_months)
        .to_dict("records"),
    }


def _load_monthly_pnl(
    validation_dir: Path,
    *,
    scenario: str,
    method: str,
    policy: str,
    year: int,
) -> pd.DataFrame:
    path = validation_dir / "validation_monthly_summary.csv"
    frame = pd.read_csv(path)
    frame = frame[
        (frame["scenario"] == scenario)
        & (frame["method"] == method)
        & (frame["policy"] == policy)
        & frame["month"].astype(str).str.startswith(f"{year}-")
    ].copy()
    if frame.empty:
        raise ValueError(f"no monthly PnL rows found for {scenario}/{method}/{policy}")
    return frame.sort_values("month").reset_index(drop=True)


def _load_month_diagnostics(diagnostics_dir: Path, month: str) -> pd.DataFrame:
    partition = month.replace("-", "_")
    path = diagnostics_dir / f"factor_contribution_{partition}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing factor contribution diagnostics: {path}")
    return pd.read_csv(path)


def _summarize_month(month: str, diagnostics: pd.DataFrame) -> dict[str, Any]:
    dominant = diagnostics["largest_contribution_feature"].value_counts()
    top_feature = str(dominant.index[0]) if not dominant.empty else ""
    second_feature = str(dominant.index[1]) if len(dominant) > 1 else ""
    return {
        "month": month,
        "diagnostic_observation_count": int(len(diagnostics)),
        "top_score_mean_label": float(diagnostics["top_score_mean_label"].mean()),
        "top_score_median_label": float(diagnostics["top_score_mean_label"].median()),
        "top_score_positive_rate": float(
            (diagnostics["top_score_mean_label"] > 0).mean()
        ),
        "mean_largest_abs_contribution_share": float(
            diagnostics["largest_abs_contribution_share"].mean()
        ),
        "mean_top_two_abs_contribution_share": float(
            diagnostics["top_two_abs_contribution_share"].mean()
        ),
        "dominant_feature": top_feature,
        "dominant_feature_observation_share": (
            float(dominant.iloc[0] / len(diagnostics)) if len(diagnostics) else 0.0
        ),
        "second_feature": second_feature,
        "second_feature_observation_share": (
            float(dominant.iloc[1] / len(diagnostics))
            if len(dominant) > 1 and len(diagnostics)
            else 0.0
        ),
    }


def _summarize_dominant_features(
    month: str,
    diagnostics: pd.DataFrame,
) -> list[dict[str, Any]]:
    grouped = diagnostics.groupby("largest_contribution_feature").agg(
        observation_count=("timestamp", "count"),
        mean_top_score_label=("top_score_mean_label", "mean"),
        median_top_score_label=("top_score_mean_label", "median"),
        mean_largest_abs_contribution_share=(
            "largest_abs_contribution_share",
            "mean",
        ),
        mean_top_two_abs_contribution_share=(
            "top_two_abs_contribution_share",
            "mean",
        ),
        mean_total_abs_contribution=("total_abs_contribution", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["month"] = month
    grouped["observation_share"] = grouped["observation_count"] / len(diagnostics)
    return grouped.to_dict("records")


def _build_report(
    args: argparse.Namespace,
    *,
    monthly: pd.DataFrame,
    features: pd.DataFrame,
) -> str:
    worst = monthly.sort_values("return").head(args.report_months)
    lines = [
        "# Candidate Policy Attribution",
        "",
        f"- Scenario: `{args.scenario}`",
        f"- Score scenario: `{args.score_scenario}`",
        f"- Method: `{args.method}`",
        f"- Policy: `{args.policy}`",
        f"- Year: `{args.year}`",
        "",
        "## Worst Months",
        "",
        "| Month | Return | Max DD | Cost | Top-score label | Label positive rate | Largest share | Top-2 share | Dominant feature |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in worst.to_dict("records"):
        lines.append(
            "| {month} | {ret} | {dd} | {cost} | {label} | {pos} | {largest} | {top2} | `{feature}` |".format(
                month=row["month"],
                ret=_format_pct(row["return"]),
                dd=_format_pct(row["max_drawdown"]),
                cost=_format_number(row["total_transaction_cost"], digits=0),
                label=_format_pct(row["top_score_mean_label"]),
                pos=_format_pct(row["top_score_positive_rate"]),
                largest=_format_pct(row["mean_largest_abs_contribution_share"]),
                top2=_format_pct(row["mean_top_two_abs_contribution_share"]),
                feature=row["dominant_feature"],
            )
        )

    lines.extend(
        [
            "",
            "## Dominant Features In Worst Months",
            "",
            "| Month | Feature | Obs share | Top-score label | Largest share | Top-2 share |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    focused = features[features["month"].astype(str).isin(set(worst["month"].astype(str)))]
    focused = focused.sort_values(["month", "observation_count"], ascending=[True, False])
    for row in focused.to_dict("records"):
        lines.append(
            "| {month} | `{feature}` | {obs} | {label} | {largest} | {top2} |".format(
                month=row["month"],
                feature=row["largest_contribution_feature"],
                obs=_format_pct(row["observation_share"]),
                label=_format_pct(row["mean_top_score_label"]),
                largest=_format_pct(row["mean_largest_abs_contribution_share"]),
                top2=_format_pct(row["mean_top_two_abs_contribution_share"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _format_number(value: Any, *, digits: int = 4) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-dir",
        default="runs/candidate_factor_portfolios/partial_rebalance_validation_standard",
    )
    parser.add_argument("--scenario", default="year_2024_base")
    parser.add_argument("--score-scenario", default="full_base")
    parser.add_argument("--method", default="decorrelated")
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--output-dir",
        default="runs/candidate_factor_portfolios/partial_rebalance_validation_standard/attribution",
    )
    parser.add_argument("--report-months", type=int, default=5)
    args = parser.parse_args()
    if args.report_months <= 0:
        raise ValueError("--report-months must be positive")
    return args


if __name__ == "__main__":
    main()
