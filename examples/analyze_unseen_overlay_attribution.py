"""Attribute a true-unseen score-overlay replay without rerunning backtests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method: str
    label: str
    score_dir: Path


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.validation_summary).read_text(encoding="utf-8"))
    methods = _method_specs(summary)
    scenarios = ("full_base", "full_high_cost", "full_zero_cost")
    rows = _summary_rows(summary, policy=args.policy)

    portfolio_summary = _portfolio_summary(
        rows=rows,
        methods=methods,
        scenarios=scenarios,
    )
    portfolio_monthly = _portfolio_monthly_returns(
        backtest_root=Path(args.backtest_root),
        methods=methods,
        scenarios=scenarios,
        policy=args.policy,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    portfolio_monthly.to_csv(output_dir / "portfolio_monthly_returns.csv", index=False)

    risk_schedule = _risk_schedule(Path(args.risk_schedule))
    risk_state_returns = _risk_state_returns(
        backtest_root=Path(args.backtest_root),
        methods=methods,
        policy=args.policy,
        risk_schedule=risk_schedule,
        start_timestamp=args.start_timestamp,
    )
    risk_state_returns.to_csv(output_dir / "risk_state_equity_returns.csv", index=False)

    score_label_monthly, score_label_state, top_overlap = _score_label_attribution(
        methods=methods,
        alpha_dataset_glob=args.alpha_dataset_glob,
        risk_schedule=risk_schedule,
        top_n=args.top_n,
    )
    score_label_monthly.to_csv(output_dir / "score_label_by_month.csv", index=False)
    score_label_state.to_csv(output_dir / "score_label_by_risk_state.csv", index=False)
    top_overlap.to_csv(output_dir / "top_overlap_vs_primary.csv", index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_type": "unseen_overlay_attribution",
        "subject": args.subject,
        "inputs": {
            "validation_summary": args.validation_summary,
            "backtest_root": args.backtest_root,
            "risk_schedule": args.risk_schedule,
            "alpha_dataset_glob": args.alpha_dataset_glob,
            "policy": args.policy,
            "top_n": args.top_n,
            "start_month": args.start_month,
            "end_month": args.end_month,
        },
        "methods": {spec.method: {"label": spec.label, "score_dir": str(spec.score_dir)} for spec in methods},
        "portfolio_summary": portfolio_summary,
        "key_findings": _key_findings(portfolio_summary, portfolio_monthly, score_label_monthly),
        "artifacts": {
            "report_json": str(output_dir / "attribution_summary.json"),
            "report_markdown": str(output_dir / "attribution_report.md"),
            "portfolio_monthly_returns": str(output_dir / "portfolio_monthly_returns.csv"),
            "risk_state_equity_returns": str(output_dir / "risk_state_equity_returns.csv"),
            "score_label_by_month": str(output_dir / "score_label_by_month.csv"),
            "score_label_by_risk_state": str(output_dir / "score_label_by_risk_state.csv"),
            "top_overlap_vs_primary": str(output_dir / "top_overlap_vs_primary.csv"),
        },
    }
    (output_dir / "attribution_summary.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "attribution_report.md").write_text(
        _markdown(report, portfolio_monthly, risk_state_returns, score_label_monthly, score_label_state, top_overlap),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "report": str(output_dir / "attribution_report.md"),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def _method_specs(summary: dict[str, Any]) -> list[MethodSpec]:
    labels = {
        "overlay_unseen_grid_w00": "primary_only_0pct",
        "overlay_unseen_grid_w025": "fallback_2p5pct",
        "overlay_unseen_grid_w05": "frozen_challenger_5pct",
    }
    specs: list[MethodSpec] = []
    for method, payload in sorted(summary.get("methods", {}).items()):
        path = str(payload.get("path", ""))
        score_dir = Path(path.replace("/*.parquet", "").replace("*", ""))
        specs.append(MethodSpec(method=method, label=labels.get(method, method), score_dir=score_dir))
    if not specs:
        raise ValueError("validation summary does not contain score methods")
    return specs


def _summary_rows(summary: dict[str, Any], *, policy: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("method")), str(row.get("scenario"))): row
        for row in summary.get("results", [])
        if row.get("policy") == policy
    }


def _portfolio_summary(
    *,
    rows: dict[tuple[str, str], dict[str, Any]],
    methods: list[MethodSpec],
    scenarios: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for spec in methods:
        method_payload: dict[str, Any] = {}
        for scenario in scenarios:
            row = rows.get((spec.method, scenario), {})
            method_payload[scenario] = {
                "total_return": _num(row.get("total_return")),
                "max_drawdown": _num(row.get("max_drawdown")),
                "gross_turnover": _num(row.get("gross_turnover")),
                "trade_count": _num(row.get("trade_count")),
                "total_transaction_cost": _num(row.get("total_transaction_cost")),
            }
        zero = method_payload.get("full_zero_cost", {}).get("total_return")
        base = method_payload.get("full_base", {}).get("total_return")
        high = method_payload.get("full_high_cost", {}).get("total_return")
        method_payload["cost_drag"] = {
            "zero_to_base_return_drag": _diff(zero, base),
            "base_to_high_cost_return_drag": _diff(base, high),
            "zero_to_high_cost_return_drag": _diff(zero, high),
        }
        payload[spec.label] = method_payload
    return payload


def _portfolio_monthly_returns(
    *,
    backtest_root: Path,
    methods: list[MethodSpec],
    scenarios: tuple[str, ...],
    policy: str,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in methods:
        for scenario in scenarios:
            returns = _monthly_returns(
                _equity_curve_path(backtest_root, scenario, spec.method, policy),
                start_month=start_month,
                end_month=end_month,
            )
            for month, value in returns.items():
                rows.append(
                    {
                        "method": spec.method,
                        "label": spec.label,
                        "scenario": scenario,
                        "month": str(month),
                        "return": float(value),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    base = frame.loc[frame["scenario"] == "full_base", ["method", "month", "return"]].rename(
        columns={"return": "full_base_return"}
    )
    high = frame.loc[frame["scenario"] == "full_high_cost", ["method", "month", "return"]].rename(
        columns={"return": "full_high_cost_return"}
    )
    zero = frame.loc[frame["scenario"] == "full_zero_cost", ["method", "month", "return"]].rename(
        columns={"return": "full_zero_cost_return"}
    )
    drag = base.merge(high, on=["method", "month"], how="left").merge(zero, on=["method", "month"], how="left")
    drag["zero_to_base_drag"] = drag["full_zero_cost_return"] - drag["full_base_return"]
    drag["base_to_high_cost_drag"] = drag["full_base_return"] - drag["full_high_cost_return"]
    frame = frame.merge(
        drag.loc[:, ["method", "month", "zero_to_base_drag", "base_to_high_cost_drag"]],
        on=["method", "month"],
        how="left",
    )
    return frame.sort_values(["scenario", "method", "month"]).reset_index(drop=True)


def _monthly_returns(path: Path, *, start_month: str, end_month: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp", "equity"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Asia/Shanghai")
    frame = frame.sort_values("timestamp")
    frame["month"] = frame["timestamp"].dt.tz_localize(None).dt.to_period("M")
    month_end = frame.groupby("month", sort=True)["equity"].last()
    returns = month_end.pct_change().dropna()
    return returns.loc[start_month:end_month]


def _equity_curve_path(backtest_root: Path, scenario: str, method: str, policy: str) -> Path:
    path = backtest_root / scenario / "backtests" / method / policy / "equity_curve.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _risk_schedule(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["timestamp", "risk_state", "gross_exposure_scale", "risk_value"],
    )
    frame["timestamp"] = frame["timestamp"].astype(str)
    return frame


def _risk_state_returns(
    *,
    backtest_root: Path,
    methods: list[MethodSpec],
    policy: str,
    risk_schedule: pd.DataFrame,
    start_timestamp: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp(start_timestamp)
    for spec in methods:
        path = _equity_curve_path(backtest_root, "full_base", spec.method, policy)
        frame = pd.read_csv(path, usecols=["timestamp", "equity"])
        frame["timestamp_dt"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Asia/Shanghai")
        frame = frame.sort_values("timestamp_dt")
        frame["bar_return"] = frame["equity"].pct_change().fillna(0.0)
        frame = frame.loc[frame["timestamp_dt"] >= start].copy()
        frame = frame.merge(risk_schedule, on="timestamp", how="left")
        frame["risk_state"] = frame["risk_state"].fillna("unknown")
        for state, group in frame.groupby("risk_state", sort=True):
            returns = group["bar_return"].astype(float)
            rows.append(
                {
                    "method": spec.method,
                    "label": spec.label,
                    "risk_state": state,
                    "bar_count": int(len(group)),
                    "compound_return": float((1.0 + returns).prod() - 1.0),
                    "mean_bar_return": float(returns.mean()),
                    "positive_bar_rate": float((returns > 0).mean()),
                    "avg_gross_exposure_scale": float(group["gross_exposure_scale"].mean()),
                    "avg_risk_value": float(group["risk_value"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "risk_state"]).reset_index(drop=True)


def _score_label_attribution(
    *,
    methods: list[MethodSpec],
    alpha_dataset_glob: str,
    risk_schedule: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    state_lookup = risk_schedule.loc[:, ["timestamp", "risk_state"]]
    alpha_paths = sorted(Path().glob(alpha_dataset_glob))
    if not alpha_paths:
        raise FileNotFoundError(alpha_dataset_glob)

    all_state_frames: list[pd.DataFrame] = []
    for alpha_path in alpha_paths:
        month = alpha_path.stem.removeprefix("dataset_").replace("_", "-")
        labels = pd.read_parquet(
            alpha_path,
            columns=["timestamp", "instrument_id", "forward_return", "entry_tradable_bar"],
        )
        labels = labels.loc[labels["entry_tradable_bar"].astype(bool)].copy()
        labels["timestamp"] = labels["timestamp"].astype(str)
        universe_ts = labels.groupby("timestamp", sort=False)["forward_return"].mean().rename("universe_mean")

        tops: dict[str, pd.DataFrame] = {}
        top_ts_frames: dict[str, pd.DataFrame] = {}
        for spec in methods:
            score_path = spec.score_dir / f"score_{month.replace('-', '_')}.parquet"
            if not score_path.exists():
                raise FileNotFoundError(score_path)
            scores = pd.read_parquet(score_path, columns=["timestamp", "instrument_id", "score"])
            scores["timestamp"] = scores["timestamp"].astype(str)
            top = scores.groupby("timestamp", sort=False).head(top_n).copy()
            top["top_rank"] = top.groupby("timestamp", sort=False).cumcount() + 1
            merged = top.merge(labels, on=["timestamp", "instrument_id"], how="inner")
            tops[spec.method] = merged.loc[:, ["timestamp", "instrument_id"]].copy()
            top_ts = (
                merged.groupby("timestamp", sort=False)
                .agg(
                    top_forward_return=("forward_return", "mean"),
                    top_hit_rate=("forward_return", lambda values: float((values > 0).mean())),
                    top_count=("forward_return", "size"),
                )
                .join(universe_ts, how="left")
                .reset_index()
            )
            top_ts["top_minus_universe"] = top_ts["top_forward_return"] - top_ts["universe_mean"]
            top_ts["method"] = spec.method
            top_ts["label"] = spec.label
            top_ts["month"] = month
            top_ts_frames[spec.method] = top_ts
            all_state_frames.append(top_ts.merge(state_lookup, on="timestamp", how="left"))
            monthly_rows.append(
                {
                    "method": spec.method,
                    "label": spec.label,
                    "month": month,
                    "timestamp_count": int(top_ts["timestamp"].nunique()),
                    "avg_top_forward_return": float(top_ts["top_forward_return"].mean()),
                    "avg_top_hit_rate": float(top_ts["top_hit_rate"].mean()),
                    "avg_universe_forward_return": float(top_ts["universe_mean"].mean()),
                    "avg_top_minus_universe": float(top_ts["top_minus_universe"].mean()),
                    "avg_top_count": float(top_ts["top_count"].mean()),
                }
            )

        primary_method = methods[0].method
        primary_top = tops[primary_method].assign(primary_member=1)
        for spec in methods:
            candidate = tops[spec.method].assign(candidate_member=1)
            overlap = candidate.merge(primary_top, on=["timestamp", "instrument_id"], how="inner")
            by_timestamp = overlap.groupby("timestamp", sort=False).size() / float(top_n)
            overlap_rows.append(
                {
                    "method": spec.method,
                    "label": spec.label,
                    "month": month,
                    "avg_top_overlap_with_primary": float(by_timestamp.mean()) if not by_timestamp.empty else 0.0,
                    "min_top_overlap_with_primary": float(by_timestamp.min()) if not by_timestamp.empty else 0.0,
                    "timestamp_count": int(len(by_timestamp)),
                }
            )

    state_frame = pd.concat(all_state_frames, ignore_index=True)
    state_frame["risk_state"] = state_frame["risk_state"].fillna("unknown")
    for (method, label, state), group in state_frame.groupby(["method", "label", "risk_state"], sort=True):
        state_rows.append(
            {
                "method": method,
                "label": label,
                "risk_state": state,
                "timestamp_count": int(group["timestamp"].nunique()),
                "avg_top_forward_return": float(group["top_forward_return"].mean()),
                "avg_top_hit_rate": float(group["top_hit_rate"].mean()),
                "avg_universe_forward_return": float(group["universe_mean"].mean()),
                "avg_top_minus_universe": float(group["top_minus_universe"].mean()),
                "avg_top_count": float(group["top_count"].mean()),
            }
        )

    return (
        pd.DataFrame(monthly_rows).sort_values(["method", "month"]).reset_index(drop=True),
        pd.DataFrame(state_rows).sort_values(["method", "risk_state"]).reset_index(drop=True),
        pd.DataFrame(overlap_rows).sort_values(["method", "month"]).reset_index(drop=True),
    )


def _key_findings(
    portfolio_summary: dict[str, Any],
    portfolio_monthly: pd.DataFrame,
    score_label_monthly: pd.DataFrame,
) -> dict[str, Any]:
    primary = portfolio_summary.get("primary_only_0pct", {})
    fallback = portfolio_summary.get("fallback_2p5pct", {})
    challenger = portfolio_summary.get("frozen_challenger_5pct", {})
    finding = {
        "default_change": "reject_5pct_default_change",
        "fallback_status": "watchlist_only",
        "primary_full_base_return": _nested(primary, "full_base", "total_return"),
        "fallback_full_base_return": _nested(fallback, "full_base", "total_return"),
        "challenger_full_base_return": _nested(challenger, "full_base", "total_return"),
        "primary_full_high_cost_return": _nested(primary, "full_high_cost", "total_return"),
        "fallback_full_high_cost_return": _nested(fallback, "full_high_cost", "total_return"),
        "challenger_full_high_cost_return": _nested(challenger, "full_high_cost", "total_return"),
    }
    if not portfolio_monthly.empty:
        full_base = portfolio_monthly.loc[portfolio_monthly["scenario"] == "full_base"]
        worst = full_base.sort_values("return").groupby("label", sort=False).head(1)
        finding["worst_full_base_months"] = {
            str(row["label"]): {"month": str(row["month"]), "return": float(row["return"])}
            for _, row in worst.iterrows()
        }
    if not score_label_monthly.empty:
        avg = score_label_monthly.groupby("label", sort=True)["avg_top_minus_universe"].mean()
        finding["avg_top_minus_universe_by_method"] = {str(k): float(v) for k, v in avg.items()}
    return finding


def _nested(payload: dict[str, Any], key: str, metric: str) -> float | None:
    value = payload.get(key, {}).get(metric)
    return _num(value)


def _markdown(
    report: dict[str, Any],
    portfolio_monthly: pd.DataFrame,
    risk_state_returns: pd.DataFrame,
    score_label_monthly: pd.DataFrame,
    score_label_state: pd.DataFrame,
    top_overlap: pd.DataFrame,
) -> str:
    key_findings = report["key_findings"]
    lines = [
        "# 2026-YTD Unseen Overlay Attribution",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "## Decision Context",
        "",
        _decision_context(key_findings),
        "",
        "The 2026 sample is treated as a high-instability tape. This report does not use "
        "external event labels; it attributes the realized replay by month, risk state, "
        "cost drag, and top-score forward-return quality.",
        "",
        "## Portfolio Summary",
        "",
        _portfolio_table(report["portfolio_summary"]),
        "",
        "## Monthly Full-Base Returns",
        "",
        _pivot_table(
            portfolio_monthly.loc[portfolio_monthly["scenario"] == "full_base"],
            index="month",
            columns="label",
            values="return",
            percent=True,
        ),
        "",
        "## Cost Drag",
        "",
        _cost_table(report["portfolio_summary"]),
        "",
        "## Risk-State Equity Attribution",
        "",
        _table(
            risk_state_returns,
            columns=[
                "label",
                "risk_state",
                "bar_count",
                "compound_return",
                "positive_bar_rate",
                "avg_gross_exposure_scale",
            ],
            percent_columns={"compound_return", "positive_bar_rate"},
        ),
        "",
        "## Top-50 Label Quality By Month",
        "",
        _table(
            score_label_monthly,
            columns=[
                "label",
                "month",
                "avg_top_forward_return",
                "avg_universe_forward_return",
                "avg_top_minus_universe",
                "avg_top_hit_rate",
            ],
            percent_columns={
                "avg_top_forward_return",
                "avg_universe_forward_return",
                "avg_top_minus_universe",
                "avg_top_hit_rate",
            },
        ),
        "",
        "## Top-50 Label Quality By Risk State",
        "",
        _table(
            score_label_state,
            columns=[
                "label",
                "risk_state",
                "timestamp_count",
                "avg_top_forward_return",
                "avg_universe_forward_return",
                "avg_top_minus_universe",
                "avg_top_hit_rate",
            ],
            percent_columns={
                "avg_top_forward_return",
                "avg_universe_forward_return",
                "avg_top_minus_universe",
                "avg_top_hit_rate",
            },
        ),
        "",
        "## Top-50 Overlap Versus Primary",
        "",
        _table(
            top_overlap,
            columns=["label", "month", "avg_top_overlap_with_primary", "min_top_overlap_with_primary"],
            percent_columns={"avg_top_overlap_with_primary", "min_top_overlap_with_primary"},
        ),
        "",
        "## Interpretation",
        "",
        *_interpretation_lines(key_findings),
        "",
    ]
    return "\n".join(lines)


def _decision_context(key_findings: dict[str, Any]) -> str:
    primary_base = key_findings.get("primary_full_base_return")
    fallback_base = key_findings.get("fallback_full_base_return")
    fallback_high = key_findings.get("fallback_full_high_cost_return")
    challenger_base = key_findings.get("challenger_full_base_return")
    challenger_high = key_findings.get("challenger_full_high_cost_return")
    if challenger_base is not None and challenger_high is not None:
        return (
            "The true post-2025 replay rejects the frozen 5% overlay as a default change. "
            "The 2.5% overlay remains watchlist-only: it improved production-cost "
            "full-base return versus primary-only, but high-cost return stayed negative."
        )
    if fallback_base is not None and fallback_high is not None:
        if primary_base is not None and fallback_base > primary_base and fallback_high <= 0:
            return (
                "This constrained 2.5% retry improves full-base return versus primary-only, "
                "but it remains watchlist-only because high-cost return is still negative."
            )
        return (
            "This constrained 2.5% retry does not pass the true-unseen gate. "
            "It fails the full-base or high-cost requirement versus primary-only."
        )
    return "This report attributes the true-unseen overlay replay."


def _interpretation_lines(key_findings: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    primary_base = key_findings.get("primary_full_base_return")
    primary_high = key_findings.get("primary_full_high_cost_return")
    fallback_base = key_findings.get("fallback_full_base_return")
    fallback_high = key_findings.get("fallback_full_high_cost_return")
    challenger_base = key_findings.get("challenger_full_base_return")
    challenger_high = key_findings.get("challenger_full_high_cost_return")
    if challenger_base is not None and challenger_high is not None:
        lines.append(
            "- The 5% overlay is not a promotion candidate because the true-unseen "
            "full-base and high-cost returns are both worse than primary-only."
        )
    if fallback_base is not None and fallback_high is not None:
        if (
            primary_base is not None
            and primary_high is not None
            and fallback_base > primary_base
            and fallback_high <= 0
        ):
            lines.append(
                "- The 2.5% overlay has a small positive full-base edge in 2026-YTD, "
                "but its high-cost line remains negative."
            )
        elif primary_base is not None and fallback_base <= primary_base:
            lines.append(
                "- The constrained 2.5% retry fails because full-base return does not "
                "improve over primary-only."
            )
        elif primary_high is not None and fallback_high <= primary_high:
            lines.append(
                "- The constrained 2.5% retry fails because high-cost return does not "
                "improve over primary-only."
            )
        else:
            lines.append(
                "- The constrained 2.5% retry improves the tested headline returns, "
                "but still requires a separate promotion review before any default change."
            )
    lines.append(
        "- Do not reopen a broad overlay-weight search. Future work should require a "
        "new mechanism or new unseen data rather than another local overlay patch."
    )
    return lines


def _portfolio_table(summary: dict[str, Any]) -> str:
    rows = []
    for label, payload in summary.items():
        rows.append(
            {
                "label": label,
                "full_base_return": _nested(payload, "full_base", "total_return"),
                "full_high_cost_return": _nested(payload, "full_high_cost", "total_return"),
                "full_zero_cost_return": _nested(payload, "full_zero_cost", "total_return"),
                "full_base_max_drawdown": _nested(payload, "full_base", "max_drawdown"),
                "full_base_turnover": _nested(payload, "full_base", "gross_turnover"),
            }
        )
    return _table(pd.DataFrame(rows), rows[0].keys() if rows else [], percent_columns={"full_base_return", "full_high_cost_return", "full_zero_cost_return", "full_base_max_drawdown"})


def _cost_table(summary: dict[str, Any]) -> str:
    rows = []
    for label, payload in summary.items():
        cost = payload.get("cost_drag", {})
        rows.append(
            {
                "label": label,
                "zero_to_base_return_drag": cost.get("zero_to_base_return_drag"),
                "base_to_high_cost_return_drag": cost.get("base_to_high_cost_return_drag"),
                "zero_to_high_cost_return_drag": cost.get("zero_to_high_cost_return_drag"),
                "base_transaction_cost": _nested(payload, "full_base", "total_transaction_cost"),
                "high_cost_transaction_cost": _nested(payload, "full_high_cost", "total_transaction_cost"),
            }
        )
    return _table(
        pd.DataFrame(rows),
        rows[0].keys() if rows else [],
        percent_columns={
            "zero_to_base_return_drag",
            "base_to_high_cost_return_drag",
            "zero_to_high_cost_return_drag",
        },
    )


def _pivot_table(
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str,
    percent: bool = False,
) -> str:
    if frame.empty:
        return "(empty)"
    pivot = frame.pivot_table(index=index, columns=columns, values=values, aggfunc="first").reset_index()
    return _table(pivot, pivot.columns, percent_columns=set(pivot.columns) - {index} if percent else set())


def _table(
    frame: pd.DataFrame,
    columns: Any,
    *,
    percent_columns: set[str] | None = None,
) -> str:
    if frame.empty:
        return "(empty)"
    percent_columns = percent_columns or set()
    cols = list(columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.loc[:, cols].iterrows():
        values = [_format(row[col], percent=col in percent_columns) for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.6f}"
    return str(value)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="daily_ma_frontier_gap_weak_tape_overlay_unseen_2026")
    parser.add_argument(
        "--validation-summary",
        default="runs/candidate_factor_portfolios/unseen_2026_overlay_control_grid_2026_05_23/validation_summary.json",
    )
    parser.add_argument(
        "--backtest-root",
        default="runs/candidate_factor_portfolios/unseen_2026_overlay_control_grid_2026_05_23",
    )
    parser.add_argument(
        "--risk-schedule",
        default="runs/candidate_factor_portfolios/unseen_2026_overlay_5pct_2026_05_23/factor_risk_gate/ribbon_dispersion/gross_exposure_schedule.csv",
    )
    parser.add_argument(
        "--alpha-dataset-glob",
        default="runs/factor_research/unified_daily_ma_gap_weak_tape_unseen_2026_05_23/alpha_dataset/dataset_2026_*.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/candidate_factor_portfolios/unseen_2026_overlay_control_grid_2026_05_23/attribution_v1",
    )
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--start-month", default="2026-01")
    parser.add_argument("--end-month", default="2026-05")
    parser.add_argument("--start-timestamp", default="2026-01-01T00:00:00+08:00")
    return parser.parse_args()


if __name__ == "__main__":
    main()
