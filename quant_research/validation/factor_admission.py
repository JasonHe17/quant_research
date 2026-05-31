"""Factor admission reporting for Framework v1 acceptance runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


FACTOR_ADMISSION_ROLES = frozenset(
    {
        "alpha_rank",
        "risk_penalty",
        "entry_filter",
        "state_allocator",
        "event_overlay",
    }
)


@dataclass(frozen=True, slots=True)
class FactorAdmissionThresholds:
    """Thresholds used to classify factors after standard acceptance."""

    min_coverage: float = 0.95
    min_timestamp_count: int = 1_000
    min_abs_rank_ic_mean: float = 0.001
    min_abs_rank_ic_t_stat: float = 2.0
    min_directional_ic_hit_rate: float = 0.52
    min_stable_years: int = 2
    min_years_observed: int = 3
    min_cost_adjusted_spread: float = 0.0
    max_top_n_turnover: float = 0.95
    cost_bps: float = 13.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be in [0, 1]")
        if self.min_timestamp_count <= 0:
            raise ValueError("min_timestamp_count must be positive")
        if self.min_abs_rank_ic_mean < 0:
            raise ValueError("min_abs_rank_ic_mean must be non-negative")
        if self.min_abs_rank_ic_t_stat < 0:
            raise ValueError("min_abs_rank_ic_t_stat must be non-negative")
        if not 0 <= self.min_directional_ic_hit_rate <= 1:
            raise ValueError("min_directional_ic_hit_rate must be in [0, 1]")
        if self.min_stable_years <= 0:
            raise ValueError("min_stable_years must be positive")
        if self.min_years_observed <= 0:
            raise ValueError("min_years_observed must be positive")
        if self.max_top_n_turnover < 0:
            raise ValueError("max_top_n_turnover must be non-negative")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")


def build_factor_admission_report(
    *,
    benchmark_summary: dict[str, Any],
    factor_summary: pd.DataFrame,
    by_timestamp: pd.DataFrame,
    thresholds: FactorAdmissionThresholds | None = None,
    feature_roles: dict[str, str] | None = None,
    feature_expected_directions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a machine-readable factor admission report."""

    thresholds = thresholds or FactorAdmissionThresholds()
    feature_roles = _normalize_feature_roles(feature_roles)
    feature_expected_directions = _normalize_feature_expected_directions(
        feature_expected_directions
    )
    _require_columns(factor_summary, ("feature",))
    _require_columns(
        by_timestamp,
        (
            "feature",
            "timestamp",
            "sample_count",
            "spearman_rank_ic",
            "top_minus_bottom_label",
            "top_n_turnover",
        ),
    )
    factor_summary = factor_summary.copy()
    by_timestamp = by_timestamp.copy()
    by_timestamp["year"] = pd.to_datetime(
        by_timestamp["timestamp"],
        errors="coerce",
        utc=True,
    ).dt.year
    summary_by_feature = factor_summary.set_index("feature").to_dict("index")
    rows = [
        _factor_admission_row(
            feature=str(feature),
            group=group,
            summary=summary_by_feature.get(str(feature), {}),
            thresholds=thresholds,
            evaluation_role=feature_roles.get(str(feature), "alpha_rank"),
            expected_direction=feature_expected_directions.get(str(feature)),
        )
        for feature, group in by_timestamp.groupby("feature", sort=True)
    ]
    rows.sort(
        key=lambda row: (
            _status_rank(row["admission_status"]),
            -abs(_number(row["spearman_rank_ic_mean"]) or 0.0),
            row["feature"],
        )
    )
    status_counts = {
        status: sum(1 for row in rows if row["admission_status"] == status)
        for status in ("candidate", "watchlist", "reject")
    }
    role_counts = _role_status_counts(rows)
    acceptance = benchmark_summary.get("acceptance", {})
    backtests = benchmark_summary.get("backtests", {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "benchmark_status": benchmark_summary.get("status"),
            "acceptance_overall_status": acceptance.get("overall_status"),
            "acceptance_failed_count": acceptance.get("failed_count"),
            "acceptance_warning_count": acceptance.get("warning_count"),
        },
        "thresholds": asdict(thresholds),
        "summary": {
            "factor_count": len(rows),
            "candidate_count": status_counts["candidate"],
            "watchlist_count": status_counts["watchlist"],
            "reject_count": status_counts["reject"],
            "role_counts": role_counts,
        },
        "backtest_snapshot": _backtest_snapshot(backtests),
        "factors": rows,
    }


def write_factor_admission_outputs(
    report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Write JSON, CSV, and Markdown factor admission outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "factor_admission_report.json"
    csv_path = output_dir / "factor_admission_table.csv"
    markdown_path = output_dir / "factor_admission_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    table = pd.DataFrame(report.get("factors", []))
    table.to_csv(csv_path, index=False)
    markdown_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }


def _factor_admission_row(
    *,
    feature: str,
    group: pd.DataFrame,
    summary: dict[str, Any],
    thresholds: FactorAdmissionThresholds,
    evaluation_role: str,
    expected_direction: str | None,
) -> dict[str, Any]:
    rank_ic = pd.to_numeric(group["spearman_rank_ic"], errors="coerce").dropna()
    rank_ic_mean = _mean(rank_ic)
    prior_direction = _expected_direction_sign(expected_direction)
    if prior_direction is None:
        direction = 1.0 if (rank_ic_mean or 0.0) >= 0 else -1.0
        direction_source = "sample_rank_ic_sign"
    else:
        direction = prior_direction
        direction_source = "registry_expected_direction"
    direction_label = "long" if direction > 0 else "invert"
    rank_ic_std = _std(rank_ic)
    rank_ic_se = rank_ic_std / math.sqrt(len(rank_ic)) if rank_ic_std and len(rank_ic) else None
    rank_ic_t_stat = rank_ic_mean / rank_ic_se if rank_ic_se else None
    positive_rate = _mean((rank_ic > 0).astype(float))
    directional_hit_rate = (
        positive_rate if direction > 0 else 1.0 - positive_rate
        if positive_rate is not None
        else None
    )
    spread = _mean(pd.to_numeric(group["top_minus_bottom_label"], errors="coerce"))
    turnover = _mean(pd.to_numeric(group["top_n_turnover"], errors="coerce"))
    directional_spread = direction * spread if spread is not None else None
    cost_adjusted_spread = (
        directional_spread - turnover * thresholds.cost_bps / 10_000.0
        if directional_spread is not None and turnover is not None
        else None
    )
    yearly_ic = _yearly_ic(group)
    stable_years = sum(
        1
        for value in yearly_ic.values()
        if value is not None and value != 0 and math.copysign(1.0, value) == direction
    )
    checks = _factor_checks(
        coverage=_coverage(summary, group),
        timestamp_count=int(summary.get("timestamp_count") or group["timestamp"].nunique()),
        abs_rank_ic_mean=abs(rank_ic_mean or 0.0),
        abs_rank_ic_t_stat=abs(rank_ic_t_stat or 0.0),
        directional_hit_rate=directional_hit_rate,
        yearly_ic=yearly_ic,
        stable_years=stable_years,
        cost_adjusted_spread=cost_adjusted_spread,
        turnover=turnover,
        thresholds=thresholds,
        evaluation_role=evaluation_role,
    )
    hard_failures = [
        check
        for check in checks
        if check["severity"] == "hard" and check["status"] == "fail"
    ]
    soft_failures = [
        check
        for check in checks
        if check["severity"] == "soft" and check["status"] == "fail"
    ]
    admission_status = (
        "reject"
        if hard_failures
        else "watchlist"
        if soft_failures
        else "candidate"
    )
    return {
        "feature": feature,
        "evaluation_role": evaluation_role,
        "admission_status": admission_status,
        "direction": direction_label,
        "expected_direction": expected_direction,
        "direction_source": direction_source,
        "sample_count": int(summary.get("sample_count") or group["sample_count"].sum()),
        "coverage": _coverage(summary, group),
        "timestamp_count": int(summary.get("timestamp_count") or group["timestamp"].nunique()),
        "spearman_rank_ic_mean": rank_ic_mean,
        "spearman_rank_ic_std": rank_ic_std,
        "spearman_rank_ic_t_stat": rank_ic_t_stat,
        "directional_ic_hit_rate": directional_hit_rate,
        "top_minus_bottom_label": spread,
        "directional_top_minus_bottom_label": directional_spread,
        "top_n_turnover": turnover,
        "cost_adjusted_top_minus_bottom_label": cost_adjusted_spread,
        "year_count": len(yearly_ic),
        "stable_year_count": stable_years,
        "yearly_spearman_rank_ic_mean": yearly_ic,
        "failed_checks": [
            check["name"]
            for check in checks
            if check["status"] == "fail" and check["severity"] != "info"
        ],
        "informational_failed_checks": [
            check["name"]
            for check in checks
            if check["status"] == "fail" and check["severity"] == "info"
        ],
        "checks": checks,
    }


def _factor_checks(
    *,
    coverage: float,
    timestamp_count: int,
    abs_rank_ic_mean: float,
    abs_rank_ic_t_stat: float,
    directional_hit_rate: float | None,
    yearly_ic: dict[str, float | None],
    stable_years: int,
    cost_adjusted_spread: float | None,
    turnover: float | None,
    thresholds: FactorAdmissionThresholds,
    evaluation_role: str,
) -> list[dict[str, Any]]:
    severity = _role_check_severity(evaluation_role)
    return [
        _check(
            "coverage",
            coverage >= thresholds.min_coverage,
            severity["coverage"],
            actual=coverage,
            threshold=thresholds.min_coverage,
        ),
        _check(
            "timestamp_count",
            timestamp_count >= thresholds.min_timestamp_count,
            "hard",
            actual=timestamp_count,
            threshold=thresholds.min_timestamp_count,
        ),
        _check(
            "abs_rank_ic_mean",
            abs_rank_ic_mean >= thresholds.min_abs_rank_ic_mean,
            severity["rank_ic"],
            actual=abs_rank_ic_mean,
            threshold=thresholds.min_abs_rank_ic_mean,
        ),
        _check(
            "abs_rank_ic_t_stat",
            abs_rank_ic_t_stat >= thresholds.min_abs_rank_ic_t_stat,
            severity["rank_ic"],
            actual=abs_rank_ic_t_stat,
            threshold=thresholds.min_abs_rank_ic_t_stat,
        ),
        _check(
            "directional_ic_hit_rate",
            directional_hit_rate is not None
            and directional_hit_rate >= thresholds.min_directional_ic_hit_rate,
            severity["rank_ic"],
            actual=directional_hit_rate,
            threshold=thresholds.min_directional_ic_hit_rate,
        ),
        _check(
            "years_observed",
            len(yearly_ic) >= thresholds.min_years_observed,
            severity["stability"],
            actual=len(yearly_ic),
            threshold=thresholds.min_years_observed,
        ),
        _check(
            "stable_year_count",
            stable_years >= thresholds.min_stable_years,
            severity["stability"],
            actual=stable_years,
            threshold=thresholds.min_stable_years,
        ),
        _check(
            "cost_adjusted_spread",
            cost_adjusted_spread is not None
            and cost_adjusted_spread > thresholds.min_cost_adjusted_spread,
            severity["implementation"],
            actual=cost_adjusted_spread,
            threshold=thresholds.min_cost_adjusted_spread,
        ),
        _check(
            "top_n_turnover",
            turnover is not None and turnover <= thresholds.max_top_n_turnover,
            severity["implementation"],
            actual=turnover,
            threshold=thresholds.max_top_n_turnover,
        ),
    ]


def _check(
    name: str,
    passed: bool,
    severity: str,
    *,
    actual: object,
    threshold: object,
) -> dict[str, object]:
    return {
        "name": name,
        "severity": severity,
        "status": "pass" if passed else "fail",
        "actual": actual,
        "threshold": threshold,
    }


def _normalize_feature_roles(feature_roles: dict[str, str] | None) -> dict[str, str]:
    if not feature_roles:
        return {}
    normalized = {}
    for feature, role in feature_roles.items():
        if role not in FACTOR_ADMISSION_ROLES:
            raise ValueError(
                f"unknown evaluation role for {feature}: {role}; "
                f"expected one of {sorted(FACTOR_ADMISSION_ROLES)}"
            )
        normalized[str(feature)] = str(role)
    return normalized


def _normalize_feature_expected_directions(
    feature_expected_directions: dict[str, str] | None,
) -> dict[str, str]:
    if not feature_expected_directions:
        return {}
    allowed = {"long", "invert", "neutral", "mixed"}
    normalized = {}
    for feature, direction in feature_expected_directions.items():
        direction = str(direction)
        if direction not in allowed:
            raise ValueError(
                f"unknown expected direction for {feature}: {direction}; "
                f"expected one of {sorted(allowed)}"
            )
        normalized[str(feature)] = direction
    return normalized


def _expected_direction_sign(expected_direction: str | None) -> float | None:
    if expected_direction == "long":
        return 1.0
    if expected_direction == "invert":
        return -1.0
    return None


def _role_check_severity(evaluation_role: str) -> dict[str, str]:
    if evaluation_role == "alpha_rank":
        return {
            "coverage": "hard",
            "rank_ic": "hard",
            "stability": "soft",
            "implementation": "soft",
        }
    if evaluation_role in {"risk_penalty", "entry_filter"}:
        return {
            "coverage": "hard",
            "rank_ic": "hard",
            "stability": "soft",
            "implementation": "info",
        }
    if evaluation_role == "event_overlay":
        return {
            "coverage": "info",
            "rank_ic": "hard",
            "stability": "soft",
            "implementation": "info",
        }
    if evaluation_role == "state_allocator":
        return {
            "coverage": "hard",
            "rank_ic": "info",
            "stability": "info",
            "implementation": "info",
        }
    raise ValueError(
        f"unknown evaluation role: {evaluation_role}; "
        f"expected one of {sorted(FACTOR_ADMISSION_ROLES)}"
    )


def _role_status_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for row in rows:
        role = str(row.get("evaluation_role", "alpha_rank"))
        status = str(row.get("admission_status", "reject"))
        output.setdefault(role, {"candidate": 0, "watchlist": 0, "reject": 0})
        output[role][status] = output[role].get(status, 0) + 1
    return dict(sorted(output.items()))


def _yearly_ic(group: pd.DataFrame) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    valid = group.loc[group["year"].notna()].copy()
    for year, year_group in valid.groupby("year", sort=True):
        values[str(int(year))] = _mean(
            pd.to_numeric(year_group["spearman_rank_ic"], errors="coerce")
        )
    return values


def _coverage(summary: dict[str, Any], group: pd.DataFrame) -> float:
    if summary.get("coverage") is not None:
        return float(summary["coverage"])
    sample_count = float(group["sample_count"].sum())
    return 1.0 if sample_count > 0 else 0.0


def _backtest_snapshot(backtests: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("full_base", "full_high_cost"):
        metrics = backtests.get(name, {}).get("metrics", {})
        output[name] = {
            key: metrics.get(key)
            for key in (
                "total_return",
                "max_drawdown",
                "final_equity",
                "trade_count",
                "total_transaction_cost",
            )
        }
    return output


def _render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    source = report["source"]
    lines = [
        "# Factor Admission Report",
        "",
        "## Acceptance Source",
        "",
        f"- Benchmark status: `{source.get('benchmark_status')}`",
        f"- Acceptance status: `{source.get('acceptance_overall_status')}`",
        f"- Failed gates: `{source.get('acceptance_failed_count')}`",
        f"- Warning gates: `{source.get('acceptance_warning_count')}`",
        "",
        "## Admission Summary",
        "",
        f"- Factors evaluated: `{summary['factor_count']}`",
        f"- Candidates: `{summary['candidate_count']}`",
        f"- Watchlist: `{summary['watchlist_count']}`",
        f"- Rejected: `{summary['reject_count']}`",
        "",
        "### Role Counts",
        "",
        "| role | candidate | watchlist | reject |",
        "| --- | ---: | ---: | ---: |",
    ]
    for role, counts in summary.get("role_counts", {}).items():
        lines.append(
            "| {role} | {candidate} | {watchlist} | {reject} |".format(
                role=role,
                candidate=counts.get("candidate", 0),
                watchlist=counts.get("watchlist", 0),
                reject=counts.get("reject", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Factor Table",
            "",
            "| feature | role | status | direction | rank_ic | t_stat | hit_rate | cost_adj_spread | stable_years | failed_checks |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report.get("factors", []):
        lines.append(
            "| {feature} | {role} | {status} | {direction} | {rank_ic} | {t_stat} | "
            "{hit_rate} | {spread} | {stable_years} | {failed} |".format(
                feature=row["feature"],
                role=row.get("evaluation_role", "alpha_rank"),
                status=row["admission_status"],
                direction=row["direction"],
                rank_ic=_format_number(row["spearman_rank_ic_mean"]),
                t_stat=_format_number(row["spearman_rank_ic_t_stat"]),
                hit_rate=_format_number(row["directional_ic_hit_rate"]),
                spread=_format_number(row["cost_adjusted_top_minus_bottom_label"]),
                stable_years=row["stable_year_count"],
                failed=", ".join(row["failed_checks"]) or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _status_rank(status: str) -> int:
    return {"candidate": 0, "watchlist": 1, "reject": 2}.get(status, 3)


def _mean(values: pd.Series) -> float | None:
    value = values.dropna().mean()
    return None if pd.isna(value) else float(value)


def _std(values: pd.Series) -> float | None:
    value = values.dropna().std(ddof=1)
    return None if pd.isna(value) else float(value)


def _number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _format_number(value: object) -> str:
    number = _number(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
