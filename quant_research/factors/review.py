"""Unified candidate-factor review report generation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from quant_research.factors.registry import (
    FactorRegistry,
    FactorRegistryIssue,
    validate_factor_registry,
)


def build_factor_candidate_review(
    registry: FactorRegistry,
    *,
    factor_id: str,
    admission_report: dict[str, Any] | None = None,
    portfolio_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one unified review report for a registered factor."""

    entry = registry.get(factor_id)
    registry_report = validate_factor_registry(registry)
    entry_issues = tuple(
        issue for issue in registry_report.issues if issue.factor_id == factor_id
    )
    admission_rows = _admission_rows(entry.feature_columns, admission_report)
    checks = _review_checks(
        entry=entry.to_dict(),
        entry_issues=entry_issues,
        admission_rows=admission_rows,
        portfolio_validation=portfolio_validation,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "factor_id": factor_id,
        "status": _review_status(checks, admission_rows, entry_issues),
        "factor": entry.to_dict(),
        "registry_validation": {
            "status": "fail"
            if any(issue.severity == "error" for issue in entry_issues)
            else "warn"
            if entry_issues
            else "pass",
            "issues": [issue.to_dict() for issue in entry_issues],
        },
        "single_factor_admission": {
            "source": _admission_source(admission_report),
            "rows": admission_rows,
        },
        "portfolio_validation": _portfolio_validation_summary(portfolio_validation),
        "checks": checks,
    }


def write_factor_candidate_review(
    report: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write candidate-factor review JSON and Markdown outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "factor_candidate_review.json"
    markdown_path = output / "factor_candidate_review.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_factor_candidate_review_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_factor_candidate_review_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable candidate-factor review."""

    factor = report["factor"]
    lines = [
        "# Factor Candidate Review",
        "",
        f"- Factor: `{report['factor_id']}`",
        f"- Status: `{report['status']}`",
        f"- Family: `{factor.get('family')}`",
        f"- Registry status: `{factor.get('status')}`",
        f"- Expected direction: `{factor.get('expected_direction')}`",
        f"- Features: `{', '.join(factor.get('feature_columns', []))}`",
        "",
        "## Hypothesis",
        "",
        str(factor.get("hypothesis", "")) or "-",
        "",
        "## Checklist",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    for check in report.get("checks", []):
        lines.append(
            "| {name} | {status} | {details} |".format(
                name=check["name"],
                status=check["status"],
                details=str(check.get("details", "")).replace("|", "\\|"),
            )
        )
    lines.extend(["", "## Single-Factor Admission", ""])
    rows = report.get("single_factor_admission", {}).get("rows", [])
    if not rows:
        lines.append("No admission row found.")
    else:
        lines.extend(
            [
                "| feature | status | direction | rank_ic | t_stat | cost_adj_spread |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| {feature} | {status} | {direction} | {rank_ic} | {t_stat} | {spread} |".format(
                    feature=row.get("feature", ""),
                    status=row.get("admission_status", ""),
                    direction=row.get("direction", ""),
                    rank_ic=_format_number(row.get("spearman_rank_ic_mean")),
                    t_stat=_format_number(row.get("spearman_rank_ic_t_stat")),
                    spread=_format_number(
                        row.get("cost_adjusted_top_minus_bottom_label")
                    ),
                )
            )
    lines.extend(["", "## Portfolio Validation", ""])
    portfolio = report.get("portfolio_validation", {})
    if portfolio.get("status") == "not_provided":
        lines.append("No portfolio validation summary was provided.")
    else:
        lines.append(f"- Status: `{portfolio.get('status')}`")
        lines.append(f"- Overall validation: `{portfolio.get('overall_status')}`")
        if portfolio.get("summary_type"):
            lines.append(f"- Summary type: `{portfolio.get('summary_type')}`")
        if portfolio.get("result_count") is not None:
            lines.append(f"- Result count: `{portfolio.get('result_count')}`")
        primary = portfolio.get("primary_result")
        if isinstance(primary, dict) and primary:
            lines.append(
                "- Primary result: method `{method}`, total_return `{total_return}`, "
                "max_drawdown `{max_drawdown}`, gross_turnover `{gross_turnover}`".format(
                    method=primary.get("method"),
                    total_return=_format_number(primary.get("total_return")),
                    max_drawdown=_format_number(primary.get("max_drawdown")),
                    gross_turnover=_format_number(primary.get("gross_turnover")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    """Load an optional JSON file."""

    if not path:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {json_path}")
    return payload


def _admission_rows(
    feature_columns: tuple[str, ...],
    admission_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not admission_report:
        return []
    feature_set = set(feature_columns)
    return [
        row
        for row in admission_report.get("factors", [])
        if isinstance(row, dict) and row.get("feature") in feature_set
    ]


def _admission_source(admission_report: dict[str, Any] | None) -> dict[str, Any]:
    if not admission_report:
        return {"status": "not_provided"}
    return {
        "status": "provided",
        "generated_at": admission_report.get("generated_at"),
        "summary": admission_report.get("summary", {}),
        "thresholds": admission_report.get("thresholds", {}),
    }


def _review_checks(
    *,
    entry: dict[str, Any],
    entry_issues: tuple[FactorRegistryIssue, ...],
    admission_rows: list[dict[str, Any]],
    portfolio_validation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    error_issues = [issue for issue in entry_issues if issue.severity == "error"]
    admission_statuses = {
        str(row.get("admission_status")) for row in admission_rows
    }
    return [
        _check(
            "registry_no_errors",
            "pass" if not error_issues else "fail",
            {"error_count": len(error_issues)},
        ),
        _check(
            "point_in_time_safe",
            "pass" if entry.get("point_in_time_safe") else "fail",
            {"point_in_time_safe": entry.get("point_in_time_safe")},
        ),
        _check(
            "live_available",
            "pass" if entry.get("live_available") else "fail",
            {"live_available": entry.get("live_available")},
        ),
        _check(
            "single_factor_admission_present",
            "pass" if admission_rows else "warn",
            {"row_count": len(admission_rows)},
        ),
        _check(
            "single_factor_not_rejected",
            "pass"
            if admission_rows and "reject" not in admission_statuses
            else "warn"
            if not admission_rows
            else "fail",
            {"admission_statuses": sorted(admission_statuses)},
        ),
        _check(
            "portfolio_validation_present",
            "pass" if portfolio_validation else "pending",
            {
                "provided": bool(portfolio_validation),
                "overall_status": (portfolio_validation or {})
                .get("validation", {})
                .get("overall_status"),
            },
        ),
    ]


def _review_status(
    checks: list[dict[str, Any]],
    admission_rows: list[dict[str, Any]],
    entry_issues: tuple[FactorRegistryIssue, ...],
) -> str:
    if any(issue.severity == "error" for issue in entry_issues):
        return "blocked"
    if any(check["status"] == "fail" for check in checks):
        return "blocked"
    statuses = {str(row.get("admission_status")) for row in admission_rows}
    if "candidate" in statuses:
        return "ready_for_portfolio_review"
    if "watchlist" in statuses:
        return "watchlist"
    return "pending_single_factor_review"


def _portfolio_validation_summary(
    portfolio_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    if not portfolio_validation:
        return {"status": "not_provided"}
    if "validation" in portfolio_validation:
        return _policy_validation_summary(portfolio_validation)
    if "backtest_summary" in portfolio_validation or "methods" in portfolio_validation:
        return _candidate_portfolio_summary(portfolio_validation)
    return {
        "status": "provided",
        "summary_type": "unknown",
        "overall_status": None,
        "failed_count": None,
        "warning_count": None,
        "result_count": 0,
    }


def _policy_validation_summary(portfolio_validation: dict[str, Any]) -> dict[str, Any]:
    validation = portfolio_validation.get("validation", {})
    return {
        "status": "provided",
        "summary_type": "policy_validation",
        "overall_status": validation.get("overall_status"),
        "failed_count": validation.get("failed_count"),
        "warning_count": validation.get("warning_count"),
        "result_count": len(portfolio_validation.get("results", [])),
    }


def _candidate_portfolio_summary(portfolio_validation: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in portfolio_validation.get("backtest_summary", [])
        if isinstance(row, dict)
    ]
    failed_count = sum(1 for row in rows if _number(row.get("total_return")) <= 0)
    warning_count = sum(
        1
        for row in rows
        if _number(row.get("total_return")) is not None
        and _number(row.get("total_return")) <= 0
    )
    overall_status = (
        "not_run"
        if not rows
        else "fail"
        if failed_count
        else "pass"
    )
    primary = rows[0] if rows else {}
    return {
        "status": "provided",
        "summary_type": "candidate_factor_portfolio",
        "overall_status": overall_status,
        "failed_count": failed_count,
        "warning_count": warning_count,
        "result_count": len(rows),
        "candidate_features": portfolio_validation.get("candidate_features", []),
        "method_count": len(portfolio_validation.get("methods", {})),
        "primary_result": {
            "method": primary.get("method"),
            "policy": primary.get("policy"),
            "total_return": primary.get("total_return"),
            "max_drawdown": primary.get("max_drawdown"),
            "gross_turnover": primary.get("gross_turnover"),
            "total_transaction_cost": primary.get("total_transaction_cost"),
            "trade_count": primary.get("trade_count"),
            "signal_count": primary.get("signal_count"),
        }
        if primary
        else {},
    }


def _check(name: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def _format_number(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.6g}"


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number
