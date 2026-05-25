"""Analyze residual monthly risk for joined-selection event-shock validations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_VALIDATION_DIR = (
    "runs/candidate_factor_portfolios/"
    "event_limit_diffusion_2026_05_24_joined_selection_block_standard"
)
DEFAULT_EVENT_STATE_DIR = (
    "runs/factor_research/event_shock_proxy_2026_05_24/"
    "event_state_regime_diagnostics"
)
DEFAULT_EXPOSURE_SCHEDULE = (
    "runs/candidate_factor_portfolios/"
    "event_limit_diffusion_2026_05_24_event_state_block_limit_standard/"
    "event_state_exposure_gate/gross_exposure_schedule.csv"
)
DEFAULT_OUTPUT_DIR = (
    "runs/factor_research/event_shock_proxy_2026_05_24/"
    "joined_selection_residual_risk"
)
TOXIC_EVENT_STATES = ("limit_diffusion", "limit_diffusion_extreme")
POST_SHOCK_STATES = ("post_shock_stabilization", "shock_elevated", "shock_extreme")


def main() -> None:
    args = _parse_args()
    summary = analyze_joined_selection_residual_risk(args)
    print(json.dumps(_json_safe(summary), ensure_ascii=True, indent=2, sort_keys=True))


def analyze_joined_selection_residual_risk(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = Path(args.validation_dir)
    monthly = _load_monthly_portfolio(
        validation_dir,
        scenario=args.scenario,
        method=args.method,
        policy=args.policy,
    )
    contribution_monthly, feature_dominance = _load_factor_contribution_monthly(
        validation_dir,
        scenario=args.scenario,
        method=args.method,
    )
    factor_health, factor_health_monthly = _load_factor_health_by_month(
        _factor_health_path(args, validation_dir)
    )
    event_state_summary = _load_event_state_summary(Path(args.event_state_summary))
    event_state_performance = _load_event_state_performance(
        Path(args.event_state_performance)
    )
    gate_monthly = _load_event_state_gate_by_month(Path(args.exposure_schedule))
    residual = _build_residual_table(
        monthly,
        contribution_monthly=contribution_monthly,
        factor_health_monthly=factor_health_monthly,
        event_state_summary=event_state_summary,
        event_state_performance=event_state_performance,
        gate_monthly=gate_monthly,
        loss_threshold=args.loss_threshold,
        drawdown_threshold=args.drawdown_threshold,
    )

    monthly_path = output_dir / "monthly_residual_risk.csv"
    dominance_path = output_dir / "feature_dominance_by_month.csv"
    health_path = output_dir / "factor_health_by_month.csv"
    gate_path = output_dir / "event_state_gate_by_month.csv"
    event_perf_path = output_dir / "event_state_performance_by_month.csv"
    report_path = output_dir / "residual_risk_report.md"

    residual.to_csv(monthly_path, index=False)
    feature_dominance.to_csv(dominance_path, index=False)
    factor_health.to_csv(health_path, index=False)
    gate_monthly.to_csv(gate_path, index=False)
    event_state_performance.to_csv(event_perf_path, index=False)
    report_path.write_text(
        _render_report(
            args,
            residual=residual,
            feature_dominance=feature_dominance,
        ),
        encoding="utf-8",
    )

    summary = {
        "params": {
            "validation_dir": args.validation_dir,
            "scenario": args.scenario,
            "method": args.method,
            "policy": args.policy,
            "event_state_summary": args.event_state_summary,
            "event_state_performance": args.event_state_performance,
            "exposure_schedule": args.exposure_schedule,
            "factor_health_schedule": str(_factor_health_path(args, validation_dir)),
            "loss_threshold": args.loss_threshold,
            "drawdown_threshold": args.drawdown_threshold,
        },
        "artifacts": {
            "monthly_residual_risk": str(monthly_path),
            "feature_dominance_by_month": str(dominance_path),
            "factor_health_by_month": str(health_path),
            "event_state_gate_by_month": str(gate_path),
            "event_state_performance_by_month": str(event_perf_path),
            "report": str(report_path),
            "summary": str(output_dir / "summary.json"),
        },
        "month_count": int(len(residual)),
        "loss_month_count": int(residual["loss_month"].sum()),
        "drawdown_month_count": int(residual["drawdown_month"].sum()),
        "worst_months": _json_records(
            residual.sort_values("portfolio_return").head(args.report_months)
        ),
        "loss_vs_gain": _loss_vs_gain(residual),
        "return_correlations": _return_correlations(residual),
        "dominant_feature_counts": _dominant_feature_counts(feature_dominance),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return summary


def _load_monthly_portfolio(
    validation_dir: Path,
    *,
    scenario: str,
    method: str,
    policy: str,
) -> pd.DataFrame:
    path = validation_dir / "validation_monthly_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"monthly validation summary not found: {path}")
    frame = pd.read_csv(path)
    required = {"scenario", "method", "policy", "month", "return", "max_drawdown"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame[
        (frame["scenario"] == scenario)
        & (frame["method"] == method)
        & (frame["policy"] == policy)
    ].copy()
    if frame.empty:
        raise ValueError(
            "no monthly rows found for "
            f"scenario={scenario!r}, method={method!r}, policy={policy!r}"
        )
    frame = frame.rename(
        columns={
            "return": "portfolio_return",
            "end_equity": "portfolio_end_equity",
            "max_drawdown": "portfolio_max_drawdown",
            "trade_count": "portfolio_trade_count",
            "total_transaction_cost": "portfolio_transaction_cost",
            "gross_traded_notional": "portfolio_gross_traded_notional",
        }
    )
    keep = [
        "month",
        "portfolio_return",
        "portfolio_end_equity",
        "portfolio_max_drawdown",
        "portfolio_trade_count",
        "portfolio_transaction_cost",
        "portfolio_gross_traded_notional",
    ]
    return frame[[column for column in keep if column in frame.columns]].sort_values(
        "month"
    )


def _load_factor_contribution_monthly(
    validation_dir: Path,
    *,
    scenario: str,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics_dir = validation_dir / scenario / "scores" / method / "diagnostics"
    paths = sorted(diagnostics_dir.glob("factor_contribution_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no factor_contribution_*.csv files found under {diagnostics_dir}"
        )
    monthly_rows: list[dict[str, Any]] = []
    dominance_rows: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path)
        month = _month_from_contribution_path(path)
        required = {
            "top_score_mean_label",
            "largest_contribution_feature",
            "largest_abs_contribution_share",
            "top_two_abs_contribution_share",
            "total_abs_contribution",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        frame = frame.copy()
        for column in (
            "top_score_mean_label",
            "largest_abs_contribution_share",
            "top_two_abs_contribution_share",
            "total_abs_contribution",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        counts = frame["largest_contribution_feature"].value_counts(dropna=False)
        dominant_feature = str(counts.index[0]) if not counts.empty else None
        dominant_count = int(counts.iloc[0]) if not counts.empty else 0
        observation_count = len(frame)
        monthly_rows.append(
            {
                "month": month,
                "contribution_observation_count": observation_count,
                "contribution_top_score_mean_label": _mean(
                    frame["top_score_mean_label"]
                ),
                "contribution_negative_top_label_share": _share_below(
                    frame["top_score_mean_label"], 0.0
                ),
                "contribution_avg_largest_abs_contribution_share": _mean(
                    frame["largest_abs_contribution_share"]
                ),
                "contribution_max_largest_abs_contribution_share": _max(
                    frame["largest_abs_contribution_share"]
                ),
                "contribution_avg_top_two_abs_contribution_share": _mean(
                    frame["top_two_abs_contribution_share"]
                ),
                "contribution_avg_total_abs_contribution": _mean(
                    frame["total_abs_contribution"]
                ),
                "contribution_dominant_feature": dominant_feature,
                "contribution_dominant_feature_share": (
                    dominant_count / observation_count if observation_count else np.nan
                ),
            }
        )
        grouped = frame.groupby("largest_contribution_feature", dropna=False, sort=True)
        for feature, group in grouped:
            dominance_rows.append(
                {
                    "month": month,
                    "feature": str(feature),
                    "largest_contribution_count": int(len(group)),
                    "largest_contribution_share": (
                        len(group) / observation_count if observation_count else np.nan
                    ),
                    "average_largest_abs_contribution_share_when_largest": _mean(
                        group["largest_abs_contribution_share"]
                    ),
                    "average_top_two_abs_contribution_share_when_largest": _mean(
                        group["top_two_abs_contribution_share"]
                    ),
                    "average_top_score_mean_label_when_largest": _mean(
                        group["top_score_mean_label"]
                    ),
                }
            )
    return (
        pd.DataFrame(monthly_rows).sort_values("month").reset_index(drop=True),
        pd.DataFrame(dominance_rows).sort_values(["month", "feature"]).reset_index(
            drop=True
        ),
    )


def _load_factor_health_by_month(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"factor health schedule not found: {path}")
    frame = pd.read_csv(path)
    required = {"timestamp", "feature", "health_state", "health_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    numeric_columns = [
        "health_score",
        "recommended_weight_scale",
        "weight_scale",
        "rolling_rank_ic",
        "rolling_top_label",
        "rolling_bottom_label",
        "rolling_top_minus_bottom_label",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (month, feature), group in frame.groupby(["month", "feature"], sort=True):
        state_counts = group["health_state"].value_counts()
        observation_count = len(group)
        rows.append(
            {
                "month": month,
                "feature": feature,
                "health_observation_count": observation_count,
                "average_health_score": _mean(group["health_score"]),
                "min_health_score": _min(group["health_score"]),
                "average_recommended_weight_scale": _mean_optional(
                    group, "recommended_weight_scale"
                ),
                "average_weight_scale": _mean_optional(group, "weight_scale"),
                "average_rolling_rank_ic": _mean_optional(group, "rolling_rank_ic"),
                "average_rolling_top_label": _mean_optional(group, "rolling_top_label"),
                "average_rolling_bottom_label": _mean_optional(
                    group, "rolling_bottom_label"
                ),
                "average_rolling_top_minus_bottom_label": _mean_optional(
                    group, "rolling_top_minus_bottom_label"
                ),
                "rolling_rank_ic_inversion_share": _share_below_optional(
                    group, "rolling_rank_ic", 0.0
                ),
                "rolling_top_label_negative_share": _share_below_optional(
                    group, "rolling_top_label", 0.0
                ),
                "rolling_spread_inversion_share": _share_below_optional(
                    group, "rolling_top_minus_bottom_label", 0.0
                ),
                "healthy_count": int(state_counts.get("healthy", 0)),
                "watch_count": int(state_counts.get("watch", 0)),
                "impaired_count": int(state_counts.get("impaired", 0)),
                "warmup_count": int(state_counts.get("warmup", 0)),
                "healthy_share": _state_share(state_counts, "healthy", observation_count),
                "watch_share": _state_share(state_counts, "watch", observation_count),
                "impaired_share": _state_share(
                    state_counts, "impaired", observation_count
                ),
                "warmup_share": _state_share(state_counts, "warmup", observation_count),
            }
        )
    factor_health = pd.DataFrame(rows)
    monthly = _aggregate_factor_health(factor_health)
    return factor_health, monthly


def _aggregate_factor_health(factor_health: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, group in factor_health.groupby("month", sort=True):
        observation_count = group["health_observation_count"].sum()
        rows.append(
            {
                "month": month,
                "health_feature_count": int(group["feature"].nunique()),
                "health_observation_count": int(observation_count),
                "health_average_score": _weighted_mean(
                    group,
                    "average_health_score",
                    "health_observation_count",
                ),
                "health_min_score": _min(group["min_health_score"]),
                "health_average_recommended_weight_scale": _weighted_mean(
                    group,
                    "average_recommended_weight_scale",
                    "health_observation_count",
                ),
                "health_average_weight_scale": _weighted_mean(
                    group,
                    "average_weight_scale",
                    "health_observation_count",
                ),
                "health_rolling_rank_ic_inversion_share": _weighted_mean(
                    group,
                    "rolling_rank_ic_inversion_share",
                    "health_observation_count",
                ),
                "health_rolling_top_label_negative_share": _weighted_mean(
                    group,
                    "rolling_top_label_negative_share",
                    "health_observation_count",
                ),
                "health_rolling_spread_inversion_share": _weighted_mean(
                    group,
                    "rolling_spread_inversion_share",
                    "health_observation_count",
                ),
                "health_healthy_share": (
                    int(group["healthy_count"].sum()) / observation_count
                    if observation_count
                    else np.nan
                ),
                "health_watch_share": (
                    int(group["watch_count"].sum()) / observation_count
                    if observation_count
                    else np.nan
                ),
                "health_impaired_share": (
                    int(group["impaired_count"].sum()) / observation_count
                    if observation_count
                    else np.nan
                ),
                "health_warmup_share": (
                    int(group["warmup_count"].sum()) / observation_count
                    if observation_count
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _load_event_state_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"event-state summary not found: {path}")
    frame = pd.read_csv(path)
    if "month" not in frame.columns:
        raise ValueError(f"{path} must contain month")
    frame = frame.copy()
    numeric = [
        column
        for column in frame.columns
        if column != "month" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for state in (*TOXIC_EVENT_STATES, *POST_SHOCK_STATES):
        column = f"state_share_{state}"
        if column not in frame.columns:
            frame[column] = 0.0
    frame["event_state_toxic_share"] = sum(
        frame[f"state_share_{state}"] for state in TOXIC_EVENT_STATES
    )
    frame["event_state_post_or_shock_share"] = sum(
        frame[f"state_share_{state}"] for state in POST_SHOCK_STATES
    )
    rename = {
        "timestamp_count": "event_state_timestamp_count",
        "market_mean_label": "event_state_market_mean_label",
        "score_rank_ic_mean": "event_state_score_rank_ic_mean",
        "score_top_n_mean_label": "event_state_score_top_n_mean_label",
        "score_top_minus_universe_label": "event_state_score_top_minus_universe_label",
        "score_top_minus_bottom_label": "event_state_score_top_minus_bottom_label",
        "event_intensity_mean": "event_state_intensity_mean",
        "limit_pressure_rate_mean": "event_state_limit_pressure_rate_mean",
        "portfolio_return": "event_state_raw_portfolio_return",
        "portfolio_max_drawdown": "event_state_raw_portfolio_max_drawdown",
        "portfolio_trade_count": "event_state_raw_portfolio_trade_count",
        "portfolio_transaction_cost": "event_state_raw_portfolio_transaction_cost",
    }
    return frame.rename(columns=rename).sort_values("month").reset_index(drop=True)


def _load_event_state_performance(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"event-state performance not found: {path}")
    frame = pd.read_csv(path)
    required = {"month", "event_state", "timestamp_count", "score_top_n_mean_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame.copy()
    for column in (
        "timestamp_count",
        "score_top_n_mean_label",
        "score_top_minus_universe_label",
        "score_top_minus_bottom_label",
        "event_intensity_mean",
        "limit_pressure_rate_mean",
        "month_state_share",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        toxic = group[group["event_state"].isin(TOXIC_EVENT_STATES)]
        allowed = group[~group["event_state"].isin(TOXIC_EVENT_STATES)]
        worst = group.sort_values("score_top_n_mean_label").head(1)
        row = {
            "month": month,
            "event_perf_toxic_state_top_n_label": _weighted_mean(
                toxic, "score_top_n_mean_label", "timestamp_count"
            ),
            "event_perf_allowed_state_top_n_label": _weighted_mean(
                allowed, "score_top_n_mean_label", "timestamp_count"
            ),
            "event_perf_worst_state": (
                str(worst["event_state"].iloc[0]) if not worst.empty else None
            ),
            "event_perf_worst_state_top_n_label": (
                float(worst["score_top_n_mean_label"].iloc[0])
                if not worst.empty
                else np.nan
            ),
        }
        for state in (*TOXIC_EVENT_STATES, *POST_SHOCK_STATES):
            state_rows = group[group["event_state"] == state]
            row[f"event_perf_{state}_share"] = _sum_optional(
                state_rows, "month_state_share"
            )
            row[f"event_perf_{state}_top_n_label"] = _weighted_mean(
                state_rows, "score_top_n_mean_label", "timestamp_count"
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _load_event_state_gate_by_month(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"event-state exposure schedule not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "timestamp",
        "source_event_state",
        "effective_event_state",
        "gross_exposure_scale",
        "event_state_gate_reason",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    frame["gross_exposure_scale"] = pd.to_numeric(
        frame["gross_exposure_scale"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("month", sort=True):
        count = len(group)
        blocked = group["gross_exposure_scale"] <= 0
        reduced = (group["gross_exposure_scale"] > 0) & (
            group["gross_exposure_scale"] < 1
        )
        full = group["gross_exposure_scale"] >= 1
        effective = group["effective_event_state"].astype("string")
        source = group["source_event_state"].astype("string")
        rows.append(
            {
                "month": month,
                "gate_observation_count": count,
                "gate_average_gross_exposure_scale": _mean(
                    group["gross_exposure_scale"]
                ),
                "gate_blocked_scale_share": float(blocked.mean()) if count else np.nan,
                "gate_reduced_scale_share": float(reduced.mean()) if count else np.nan,
                "gate_full_scale_share": float(full.mean()) if count else np.nan,
                "gate_effective_toxic_state_share": _categorical_share(
                    effective, TOXIC_EVENT_STATES
                ),
                "gate_source_toxic_state_share": _categorical_share(
                    source, TOXIC_EVENT_STATES
                ),
                "gate_blocked_reason_share": _categorical_share(
                    group["event_state_gate_reason"].astype("string"),
                    ("blocked_event_state",),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _build_residual_table(
    monthly: pd.DataFrame,
    *,
    contribution_monthly: pd.DataFrame,
    factor_health_monthly: pd.DataFrame,
    event_state_summary: pd.DataFrame,
    event_state_performance: pd.DataFrame,
    gate_monthly: pd.DataFrame,
    loss_threshold: float,
    drawdown_threshold: float,
) -> pd.DataFrame:
    residual = monthly.copy()
    for table in (
        contribution_monthly,
        event_state_summary,
        event_state_performance,
        gate_monthly,
        factor_health_monthly,
    ):
        residual = residual.merge(table, on="month", how="left")
    residual["loss_month"] = residual["portfolio_return"] < loss_threshold
    residual["drawdown_month"] = residual["portfolio_max_drawdown"] <= drawdown_threshold
    residual["year"] = residual["month"].astype(str).str.slice(0, 4)
    return residual.sort_values("month").reset_index(drop=True)


def _loss_vs_gain(residual: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    metrics = [
        "portfolio_return",
        "portfolio_max_drawdown",
        "event_state_toxic_share",
        "event_state_post_or_shock_share",
        "gate_blocked_scale_share",
        "gate_average_gross_exposure_scale",
        "contribution_top_score_mean_label",
        "contribution_negative_top_label_share",
        "contribution_avg_largest_abs_contribution_share",
        "contribution_dominant_feature_share",
        "health_average_score",
        "health_impaired_share",
        "health_rolling_top_label_negative_share",
    ]
    loss = residual[residual["loss_month"]]
    gain = residual[~residual["loss_month"]]
    output: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        if metric not in residual.columns:
            continue
        output[metric] = {
            "loss_month_mean": _nullable_float(_mean(loss[metric])),
            "non_loss_month_mean": _nullable_float(_mean(gain[metric])),
            "difference": _nullable_float(_mean(loss[metric]) - _mean(gain[metric])),
        }
    return output


def _return_correlations(residual: pd.DataFrame) -> dict[str, float | None]:
    exclude = {
        "portfolio_return",
        "portfolio_end_equity",
        "portfolio_trade_count",
        "portfolio_transaction_cost",
        "portfolio_gross_traded_notional",
        "loss_month",
        "drawdown_month",
    }
    correlations: dict[str, float | None] = {}
    for column in residual.columns:
        if column in exclude or column == "month" or column == "year":
            continue
        if not pd.api.types.is_numeric_dtype(residual[column]):
            continue
        value = _corr(residual["portfolio_return"], residual[column])
        if value is not None:
            correlations[column] = value
    return dict(
        sorted(
            correlations.items(),
            key=lambda item: abs(item[1]) if item[1] is not None else -1,
            reverse=True,
        )
    )


def _dominant_feature_counts(feature_dominance: pd.DataFrame) -> dict[str, int]:
    if feature_dominance.empty:
        return {}
    idx = feature_dominance.groupby("month")["largest_contribution_share"].idxmax()
    counts = feature_dominance.loc[idx, "feature"].value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _render_report(
    args: argparse.Namespace,
    *,
    residual: pd.DataFrame,
    feature_dominance: pd.DataFrame,
) -> str:
    worst = residual.sort_values("portfolio_return").head(args.report_months)
    loss_stats = _loss_vs_gain(residual)
    lines = [
        "# Joined Selection Residual Risk Diagnostic",
        "",
        f"- Validation dir: `{args.validation_dir}`",
        f"- Scenario: `{args.scenario}`",
        f"- Method: `{args.method}`",
        f"- Policy: `{args.policy}`",
        "",
        "This is an attribution diagnostic only. It does not fit thresholds, "
        "change weights, or use any specific month to define a trading rule.",
        "",
        "## Worst Months",
        "",
        "| Month | Return | Drawdown | Event toxic share | Gate blocked share | "
        "Top label | Negative top-label share | Dominant feature | "
        "Health impaired share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in worst.to_dict("records"):
        lines.append(
            "| {month} | {ret} | {dd} | {toxic} | {gate} | {top} | "
            "{neg} | `{feature}` | {health} |".format(
                month=row["month"],
                ret=_format_pct(row.get("portfolio_return")),
                dd=_format_pct(row.get("portfolio_max_drawdown")),
                toxic=_format_pct(row.get("event_state_toxic_share")),
                gate=_format_pct(row.get("gate_blocked_scale_share")),
                top=_format_pct(row.get("contribution_top_score_mean_label")),
                neg=_format_pct(row.get("contribution_negative_top_label_share")),
                feature=row.get("contribution_dominant_feature") or "-",
                health=_format_pct(row.get("health_impaired_share")),
            )
        )
    lines.extend(
        [
            "",
            "## Loss Versus Non-Loss Months",
            "",
            "| Metric | Loss mean | Non-loss mean | Difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in (
        "event_state_toxic_share",
        "gate_blocked_scale_share",
        "contribution_top_score_mean_label",
        "contribution_negative_top_label_share",
        "contribution_avg_largest_abs_contribution_share",
        "health_average_score",
        "health_impaired_share",
    ):
        item = loss_stats.get(metric, {})
        lines.append(
            "| `{metric}` | {loss} | {gain} | {diff} |".format(
                metric=metric,
                loss=_format_mixed(item.get("loss_month_mean")),
                gain=_format_mixed(item.get("non_loss_month_mean")),
                diff=_format_mixed(item.get("difference")),
            )
        )
    lines.extend(
        [
            "",
            "## Feature Dominance In Worst Months",
            "",
            "| Month | Feature | Largest-count share | Top label when largest |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    worst_months = set(worst["month"].astype(str))
    focused = feature_dominance[
        feature_dominance["month"].astype(str).isin(worst_months)
    ].sort_values(["month", "largest_contribution_share"], ascending=[True, False])
    for row in focused.to_dict("records"):
        lines.append(
            "| {month} | `{feature}` | {share} | {top} |".format(
                month=row["month"],
                feature=row["feature"],
                share=_format_pct(row.get("largest_contribution_share")),
                top=_format_pct(row.get("average_top_score_mean_label_when_largest")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _factor_health_path(args: argparse.Namespace, validation_dir: Path) -> Path:
    if args.factor_health_schedule:
        return Path(args.factor_health_schedule)
    return validation_dir / args.scenario / "factor_health" / "factor_health_schedule.csv"


def _month_from_contribution_path(path: Path) -> str:
    suffix = path.stem.removeprefix("factor_contribution_")
    return suffix.replace("_", "-")


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.mean())


def _mean_optional(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    return _mean(frame[column])


def _sum_optional(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").sum())


def _min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.min())


def _max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.max())


def _share_below(values: pd.Series, threshold: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float((numeric < threshold).mean())


def _share_below_optional(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
) -> float:
    if column not in frame.columns:
        return float("nan")
    return _share_below(frame[column], threshold)


def _state_share(counts: pd.Series, state: str, observation_count: int) -> float:
    if not observation_count:
        return float("nan")
    return float(counts.get(state, 0) / observation_count)


def _weighted_mean(frame: pd.DataFrame, value_column: str, weight_column: str) -> float:
    if frame.empty or value_column not in frame.columns or weight_column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _categorical_share(values: pd.Series, targets: tuple[str, ...]) -> float:
    valid = values.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.astype(str).isin(set(targets)).mean())


def _corr(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(pair) < 3:
        return None
    if pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return None
    return float(pair["left"].corr(pair["right"]))


def _nullable_float(value: float) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in frame.to_dict("records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _format_pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _format_mixed(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if abs(number) <= 1:
        return f"{number * 100:.2f}%"
    return f"{number:.4f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--scenario", default="full_base")
    parser.add_argument("--method", default="decorrelated")
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument(
        "--event-state-summary",
        default=f"{DEFAULT_EVENT_STATE_DIR}/monthly_event_state_summary.csv",
    )
    parser.add_argument(
        "--event-state-performance",
        default=f"{DEFAULT_EVENT_STATE_DIR}/monthly_event_state_performance.csv",
    )
    parser.add_argument("--exposure-schedule", default=DEFAULT_EXPOSURE_SCHEDULE)
    parser.add_argument("--factor-health-schedule")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--loss-threshold", type=float, default=0.0)
    parser.add_argument("--drawdown-threshold", type=float, default=-0.10)
    parser.add_argument("--report-months", type=int, default=8)
    args = parser.parse_args()
    if args.report_months <= 0:
        raise ValueError("--report-months must be positive")
    if args.drawdown_threshold > 0:
        raise ValueError("--drawdown-threshold should be zero or negative")
    return args


if __name__ == "__main__":
    main()
