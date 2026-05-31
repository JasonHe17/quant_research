"""Analyze monthly regime failures for candidate policy validation runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.portfolio import CandidateFactor, load_candidate_factors


@dataclass(frozen=True, slots=True)
class MonthlyDiagnostics:
    """Diagnostics for one month of candidate factor portfolio scores."""

    composite: dict[str, Any]
    factor_rows: list[dict[str, Any]]
    exposure_rows: list[dict[str, Any]]


def main() -> None:
    args = _parse_args()
    summary = analyze_candidate_policy_regime(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def analyze_candidate_policy_regime(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_scenario = args.score_scenario or args.scenario
    weights_scenario = args.weights_scenario or args.scenario
    candidates = load_candidate_factors(
        Path(args.admission_report),
        include_features=tuple(args.include_features),
    )
    weights = _load_method_weights(
        Path(args.validation_dir),
        scenario=weights_scenario,
        method=args.method,
    )
    pnl_rows = _load_monthly_pnl(
        Path(args.validation_dir),
        scenario=args.scenario,
        method=args.method,
        policy=args.policy,
    )
    diagnostics = [
        _diagnose_month(
            dataset_path,
            score_path=_score_path(args, dataset_path, score_scenario=score_scenario),
            candidates=candidates,
            weights=weights,
            top_n=args.top_n,
            label_column=args.label_column,
        )
        for dataset_path in _dataset_paths(args)
    ]
    composite = pd.DataFrame([item.composite for item in diagnostics])
    factor_rows = pd.DataFrame(
        [row for item in diagnostics for row in item.factor_rows]
    )
    exposure_rows = pd.DataFrame(
        [row for item in diagnostics for row in item.exposure_rows]
    )
    composite = composite.merge(
        pnl_rows,
        on="month",
        how="left",
        suffixes=("", "_pnl"),
    )
    composite_path = output_dir / "composite_monthly.csv"
    factors_path = output_dir / "factor_legs_monthly.csv"
    exposure_path = output_dir / "top_score_exposure_monthly.csv"
    report_path = output_dir / "regime_failure_report.md"
    composite.to_csv(composite_path, index=False)
    factor_rows.to_csv(factors_path, index=False)
    exposure_rows.to_csv(exposure_path, index=False)
    report = _build_report(
        args,
        composite=composite,
        factor_rows=factor_rows,
        exposure_rows=exposure_rows,
    )
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "params": {
            "dataset_dir": args.dataset_dir,
            "validation_dir": args.validation_dir,
            "scenario": args.scenario,
            "score_scenario": score_scenario,
            "weights_scenario": weights_scenario,
            "method": args.method,
            "policy": args.policy,
            "year": args.year,
            "label_column": args.label_column,
            "top_n": args.top_n,
            "include_features": args.include_features,
        },
        "artifacts": {
            "composite_monthly": str(composite_path),
            "factor_legs_monthly": str(factors_path),
            "top_score_exposure_monthly": str(exposure_path),
            "report": str(report_path),
        },
        "worst_months": composite.sort_values("portfolio_return")
        .head(args.report_months)
        .to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    dataset_dir = Path(args.dataset_dir)
    if args.months:
        months = args.months
    else:
        months = [f"{args.year}_{month:02d}" for month in range(1, 13)]
    paths = [dataset_dir / f"dataset_{month}.parquet" for month in months]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dataset partitions: {missing}")
    return paths


def _score_path(
    args: argparse.Namespace,
    dataset_path: Path,
    *,
    score_scenario: str,
) -> Path:
    partition = dataset_path.stem.removeprefix("dataset_")
    path = (
        Path(args.validation_dir)
        / score_scenario
        / "scores"
        / args.method
        / f"score_{partition}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing score partition: {path}")
    return path


def _diagnose_month(
    dataset_path: Path,
    *,
    score_path: Path,
    candidates: tuple[CandidateFactor, ...],
    weights: dict[str, float],
    top_n: int,
    label_column: str,
) -> MonthlyDiagnostics:
    if not label_column:
        raise ValueError("label_column must be non-empty")
    feature_columns = [candidate.feature for candidate in candidates]
    dataset = pd.read_parquet(
        dataset_path,
        columns=[
            "timestamp",
            "instrument_id",
            label_column,
            "entry_tradable_bar",
            "entry_limit_up_open",
            "entry_limit_down_open",
            *feature_columns,
        ],
    )
    scores = pd.read_parquet(score_path)
    frame = dataset.merge(scores, on=["timestamp", "instrument_id"], how="inner")
    month = dataset_path.stem.removeprefix("dataset_").replace("_", "-")
    composite_rows: list[dict[str, Any]] = []
    factor_rows_by_feature: dict[str, list[dict[str, Any]]] = {
        candidate.feature: [] for candidate in candidates
    }
    exposure_rows_by_feature: dict[str, list[dict[str, Any]]] = {
        candidate.feature: [] for candidate in candidates
    }
    for _, group in frame.groupby("timestamp", sort=True):
        timestamp_diag = _timestamp_diagnostics(
            group,
            candidates=candidates,
            weights=weights,
            top_n=top_n,
            label_column=label_column,
        )
        composite_rows.append(timestamp_diag["composite"])
        for row in timestamp_diag["factor_rows"]:
            factor_rows_by_feature[row["feature"]].append(row)
        for row in timestamp_diag["exposure_rows"]:
            exposure_rows_by_feature[row["feature"]].append(row)
    composite = _summarize_composite(month, composite_rows, frame, label_column)
    factor_rows = [
        _summarize_factor(month, feature, rows)
        for feature, rows in factor_rows_by_feature.items()
        if rows
    ]
    exposure_rows = [
        _summarize_exposure(month, feature, rows)
        for feature, rows in exposure_rows_by_feature.items()
        if rows
    ]
    return MonthlyDiagnostics(
        composite=composite,
        factor_rows=factor_rows,
        exposure_rows=exposure_rows,
    )


def _timestamp_diagnostics(
    group: pd.DataFrame,
    *,
    candidates: tuple[CandidateFactor, ...],
    weights: dict[str, float],
    top_n: int,
    label_column: str = "forward_return",
) -> dict[str, Any]:
    if not label_column:
        raise ValueError("label_column must be non-empty")
    valid_score = group.dropna(subset=["score", label_column])
    n = min(top_n, len(valid_score))
    top_score = valid_score.nlargest(n, "score") if n else valid_score
    bottom_score = valid_score.nsmallest(n, "score") if n else valid_score
    composite = {
        "sample_count": len(valid_score),
        "score_rank_ic": _correlation(valid_score["score"], valid_score[label_column]),
        "score_top_n_mean_label": _mean(top_score[label_column]),
        "score_bottom_n_mean_label": _mean(bottom_score[label_column]),
        "score_top_minus_bottom_label": (
            _mean(top_score[label_column])
            - _mean(bottom_score[label_column])
        ),
    }
    factor_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    contribution_by_feature: dict[str, pd.Series] = {}
    oriented_rank_by_feature: dict[str, pd.Series] = {}
    for candidate in candidates:
        valid = group.dropna(subset=[candidate.feature, label_column])
        if valid.empty:
            continue
        raw_rank = valid[candidate.feature].rank(method="average", pct=True)
        oriented_rank = raw_rank if candidate.direction > 0 else 1.0 - raw_rank
        factor_top = valid.loc[oriented_rank.nlargest(min(top_n, len(valid))).index]
        factor_bottom = valid.loc[oriented_rank.nsmallest(min(top_n, len(valid))).index]
        factor_rows.append(
            {
                "feature": candidate.feature,
                "sample_count": len(valid),
                "direction": candidate.direction,
                "directional_rank_ic": _correlation(
                    oriented_rank,
                    valid[label_column],
                ),
                "top_n_mean_label": _mean(factor_top[label_column]),
                "bottom_n_mean_label": _mean(factor_bottom[label_column]),
                "top_minus_bottom_label": (
                    _mean(factor_top[label_column])
                    - _mean(factor_bottom[label_column])
                ),
            }
        )
        oriented_rank_by_feature[candidate.feature] = oriented_rank
        contribution_by_feature[candidate.feature] = (
            (oriented_rank - 0.5) * float(weights.get(candidate.feature, 0.0))
        )
    if not top_score.empty:
        top_index = top_score.index
        absolute_total = sum(
            float(contribution.reindex(top_index).abs().mean())
            for contribution in contribution_by_feature.values()
        )
        for feature, oriented_rank in oriented_rank_by_feature.items():
            contribution = contribution_by_feature[feature]
            abs_contribution = float(contribution.reindex(top_index).abs().mean())
            exposure_rows.append(
                {
                    "feature": feature,
                    "top_score_oriented_rank_mean": float(
                        oriented_rank.reindex(top_index).mean()
                    ),
                    "universe_oriented_rank_mean": float(oriented_rank.mean()),
                    "top_score_abs_contribution": abs_contribution,
                    "top_score_abs_contribution_share": (
                        abs_contribution / absolute_total if absolute_total > 0 else 0.0
                    ),
                }
            )
    return {
        "composite": composite,
        "factor_rows": factor_rows,
        "exposure_rows": exposure_rows,
    }


def _summarize_composite(
    month: str,
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    label_column: str = "forward_return",
) -> dict[str, Any]:
    table = pd.DataFrame(rows)
    return {
        "month": month,
        "timestamp_count": len(table),
        "sample_count": int(table["sample_count"].sum()) if not table.empty else 0,
        "score_rank_ic_mean": _mean(table["score_rank_ic"]),
        "score_rank_ic_positive_rate": _positive_rate(table["score_rank_ic"]),
        "score_top_n_mean_label": _mean(table["score_top_n_mean_label"]),
        "score_bottom_n_mean_label": _mean(table["score_bottom_n_mean_label"]),
        "score_top_minus_bottom_label": _mean(table["score_top_minus_bottom_label"]),
        "label_column": label_column,
        "market_mean_label": _mean(frame[label_column]),
        "market_median_label": float(frame[label_column].median()),
        "market_label_std": float(frame[label_column].std(ddof=1)),
        "entry_tradable_rate": _mean(frame["entry_tradable_bar"].astype(float)),
        "entry_limit_up_open_rate": _mean(frame["entry_limit_up_open"].astype(float)),
        "entry_limit_down_open_rate": _mean(
            frame["entry_limit_down_open"].astype(float)
        ),
    }


def _summarize_factor(
    month: str,
    feature: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    table = pd.DataFrame(rows)
    return {
        "month": month,
        "feature": feature,
        "direction": int(table["direction"].iloc[0]),
        "timestamp_count": len(table),
        "sample_count": int(table["sample_count"].sum()),
        "directional_rank_ic_mean": _mean(table["directional_rank_ic"]),
        "directional_rank_ic_positive_rate": _positive_rate(
            table["directional_rank_ic"]
        ),
        "top_n_mean_label": _mean(table["top_n_mean_label"]),
        "bottom_n_mean_label": _mean(table["bottom_n_mean_label"]),
        "top_minus_bottom_label": _mean(table["top_minus_bottom_label"]),
    }


def _summarize_exposure(
    month: str,
    feature: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    table = pd.DataFrame(rows)
    return {
        "month": month,
        "feature": feature,
        "top_score_oriented_rank_mean": _mean(table["top_score_oriented_rank_mean"]),
        "universe_oriented_rank_mean": _mean(table["universe_oriented_rank_mean"]),
        "top_score_abs_contribution": _mean(table["top_score_abs_contribution"]),
        "top_score_abs_contribution_share": _mean(
            table["top_score_abs_contribution_share"]
        ),
    }


def _load_method_weights(
    validation_dir: Path,
    *,
    scenario: str,
    method: str,
) -> dict[str, float]:
    summary_path = validation_dir / scenario / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    methods = payload.get("methods", {})
    if method not in methods:
        raise KeyError(f"method {method!r} not found in {summary_path}")
    weights = methods[method].get("weights", {})
    return {str(key): float(value) for key, value in weights.items()}


def _load_monthly_pnl(
    validation_dir: Path,
    *,
    scenario: str,
    method: str,
    policy: str,
) -> pd.DataFrame:
    path = validation_dir / "validation_monthly_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["month", "portfolio_return"])
    frame = pd.read_csv(path)
    frame = frame[
        (frame["scenario"] == scenario)
        & (frame["method"] == method)
        & (frame["policy"] == policy)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=["month", "portfolio_return"])
    return frame.rename(
        columns={
            "return": "portfolio_return",
            "max_drawdown": "portfolio_max_drawdown",
            "trade_count": "portfolio_trade_count",
            "total_transaction_cost": "portfolio_transaction_cost",
            "gross_traded_notional": "portfolio_gross_traded_notional",
        }
    )[
        [
            "month",
            "portfolio_return",
            "portfolio_max_drawdown",
            "portfolio_trade_count",
            "portfolio_transaction_cost",
            "portfolio_gross_traded_notional",
        ]
    ]


def _build_report(
    args: argparse.Namespace,
    *,
    composite: pd.DataFrame,
    factor_rows: pd.DataFrame,
    exposure_rows: pd.DataFrame,
) -> str:
    worst = composite.sort_values("portfolio_return").head(args.report_months)
    lines = [
        "# Candidate Policy Regime Diagnostic",
        "",
        f"- Scenario: `{args.scenario}`",
        f"- Score scenario: `{args.score_scenario or args.scenario}`",
        f"- Weights scenario: `{args.weights_scenario or args.scenario}`",
        f"- Method: `{args.method}`",
        f"- Policy: `{args.policy}`",
        f"- Year: `{args.year}`",
        "",
        "## Worst Months",
        "",
        "| Month | Portfolio return | Score IC | Score spread | Market label | Tradable rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in worst.to_dict("records"):
        lines.append(
            "| {month} | {ret} | {ic} | {spread} | {market} | {tradable} |".format(
                month=row["month"],
                ret=_format_pct(row.get("portfolio_return")),
                ic=_format_number(row.get("score_rank_ic_mean")),
                spread=_format_pct(row.get("score_top_minus_bottom_label")),
                market=_format_pct(row.get("market_mean_label")),
                tradable=_format_pct(row.get("entry_tradable_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Factor Legs In Worst Months",
            "",
            "| Month | Feature | Directional IC | Top-bottom spread | Top mean label |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    worst_months = set(worst["month"].astype(str))
    focused_factors = factor_rows[factor_rows["month"].astype(str).isin(worst_months)]
    focused_factors = focused_factors.sort_values(["month", "top_minus_bottom_label"])
    for row in focused_factors.to_dict("records"):
        lines.append(
            "| {month} | `{feature}` | {ic} | {spread} | {top} |".format(
                month=row["month"],
                feature=row["feature"],
                ic=_format_number(row.get("directional_rank_ic_mean")),
                spread=_format_pct(row.get("top_minus_bottom_label")),
                top=_format_pct(row.get("top_n_mean_label")),
            )
        )
    lines.extend(
        [
            "",
            "## Top-Score Exposure In Worst Months",
            "",
            "| Month | Feature | Top oriented rank | Abs contribution share |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    focused_exposure = exposure_rows[
        exposure_rows["month"].astype(str).isin(worst_months)
    ].sort_values(["month", "top_score_abs_contribution_share"], ascending=[True, False])
    for row in focused_exposure.to_dict("records"):
        lines.append(
            "| {month} | `{feature}` | {rank} | {share} |".format(
                month=row["month"],
                feature=row["feature"],
                rank=_format_number(row.get("top_score_oriented_rank_mean")),
                share=_format_pct(row.get("top_score_abs_contribution_share")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _mean(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").mean())


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float((numeric > 0).mean())


def _correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 2:
        return float("nan")
    return float(pair["left"].corr(pair["right"], method="spearman"))


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="runs/framework_v1_acceptance/standard/alpha_dataset",
    )
    parser.add_argument(
        "--admission-report",
        default=(
            "runs/framework_v1_acceptance/standard/factor_admission/"
            "factor_admission_report.json"
        ),
    )
    parser.add_argument(
        "--validation-dir",
        default="runs/candidate_factor_portfolios/partial_rebalance_validation_standard",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/candidate_factor_portfolios/partial_rebalance_validation_standard/regime_diagnostics_2024",
    )
    parser.add_argument("--scenario", default="year_2024_base")
    parser.add_argument(
        "--score-scenario",
        help="scenario directory that contains score partitions; defaults to --scenario",
    )
    parser.add_argument(
        "--weights-scenario",
        help="scenario summary that contains method weights; defaults to --scenario",
    )
    parser.add_argument("--method", default="decorrelated")
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--months", nargs="+")
    parser.add_argument(
        "--include-features",
        nargs="+",
        default=[],
        help="optional factor feature allowlist for shared admission reports",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--report-months", type=int, default=5)
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if not args.label_column:
        raise ValueError("--label-column must be non-empty")
    if args.report_months <= 0:
        raise ValueError("--report-months must be positive")
    return args


if __name__ == "__main__":
    main()
