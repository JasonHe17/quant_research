"""Monitoring reports for governed allocator evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_research.factors import FactorRegistry
from quant_research.portfolio.allocator_registry import (
    AllocatorRegistry,
    AllocatorRegistryEntry,
    validate_allocator_registry,
)


_CAPACITY_MONITOR_THRESHOLDS = {
    "min_total_return": ("total_return", "min"),
    "max_abs_drawdown": ("max_drawdown", "max_abs"),
    "max_unfilled_vs_traded_notional": ("capacity_unfilled_vs_traded", "max"),
    "max_unfilled_vs_desired_capacity_events": (
        "capacity_unfilled_vs_desired_capacity_events",
        "max",
    ),
}


@dataclass(frozen=True, slots=True)
class AllocatorMonitoringReport:
    """Machine-readable monitoring report for one governed allocator."""

    generated_at: str
    status: str
    allocator: dict[str, Any]
    sections: dict[str, Any]
    checks: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report payload."""

        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "allocator": self.allocator,
            "sections": self.sections,
            "checks": list(self.checks),
        }


def allocator_monitoring_history_row(
    report: AllocatorMonitoringReport,
    *,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten a monitoring report into one append-only history row."""

    payload = report.to_dict()
    sections = payload["sections"]
    validation = sections.get("validation", {})
    capacity = sections.get("capacity", {})
    event_gate = sections.get("event_state_gate", {})
    factor_health = sections.get("factor_health", {})
    primary = validation.get("primary_metrics") or {}
    capacity_scenarios = {
        str(row.get("scenario")): row
        for row in capacity.get("scenarios", [])
        if isinstance(row, dict)
    }
    cap_5 = capacity_scenarios.get("capacity_5pct", {})
    cap_2 = capacity_scenarios.get("capacity_2pct", {})
    row = {
        "generated_at": report.generated_at,
        "allocator_id": payload["allocator"].get("allocator_id"),
        "status": report.status,
        "registry_status": sections.get("registry", {}).get("status"),
        "validation_status": validation.get("status"),
        "capacity_status": capacity.get("status"),
        "event_state_gate_status": event_gate.get("status"),
        "factor_health_status": factor_health.get("status"),
        "total_return": primary.get("total_return"),
        "max_drawdown": primary.get("max_drawdown"),
        "gross_turnover": primary.get("gross_turnover"),
        "capacity_5pct_unfilled_vs_traded": cap_5.get(
            "capacity_unfilled_vs_traded"
        ),
        "capacity_2pct_unfilled_vs_traded": cap_2.get(
            "capacity_unfilled_vs_traded"
        ),
        "event_state_latest_scale": event_gate.get("latest_scale"),
        "event_state_latest_reason": event_gate.get("latest_reason"),
        "factor_health_latest_impaired_feature_count": factor_health.get(
            "latest_impaired_feature_count"
        ),
        "factor_health_latest_watch_feature_count": factor_health.get(
            "latest_watch_feature_count"
        ),
        "factor_health_latest_min_weight_scale": factor_health.get(
            "latest_min_weight_scale"
        ),
    }
    if extra_fields:
        row.update(extra_fields)
    return _json_safe(row)


def append_allocator_monitoring_history(
    report: AllocatorMonitoringReport,
    *,
    history_csv: str | Path,
    extra_fields: Mapping[str, Any] | None = None,
    replace_existing_on: str | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Append one report summary row to a CSV monitoring ledger."""

    path = Path(history_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = allocator_monitoring_history_row(report, extra_fields=extra_fields)
    frame = pd.DataFrame([row])
    replaced_count = 0
    if path.exists():
        existing = pd.read_csv(path)
        if replace_existing_on is not None and not existing.empty:
            key_columns = (
                (replace_existing_on,)
                if isinstance(replace_existing_on, str)
                else tuple(replace_existing_on)
            )
            mask = pd.Series([True] * len(existing), index=existing.index)
            for column in key_columns:
                if column not in existing.columns:
                    existing[column] = None
                observed = existing[column].fillna("").astype(str)
                expected = "" if row.get(column) is None else str(row.get(column))
                mask &= observed == expected
            replaced_count = int(mask.sum())
            existing = existing.loc[~mask].copy()
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "row_count": int(len(frame)),
        "latest_row": row,
        "replaced_count": replaced_count,
        "replace_existing_on": replace_existing_on,
    }


def allocator_monitoring_history_status(
    history_csv: str | Path,
    *,
    sustained_warning_window: int = 3,
) -> dict[str, Any]:
    """Summarize recent monitoring ledger status and sustained warning streaks."""

    path = Path(history_csv)
    if not path.exists():
        return {
            "status": "pending",
            "path": str(path),
            "row_count": 0,
            "sustained_warning_window": sustained_warning_window,
            "sustained_warning": False,
            "sustained_failure": False,
        }
    frame = pd.read_csv(path)
    if frame.empty:
        return {
            "status": "pending",
            "path": str(path),
            "row_count": 0,
            "sustained_warning_window": sustained_warning_window,
            "sustained_warning": False,
            "sustained_failure": False,
        }
    window = max(1, int(sustained_warning_window))
    recent = frame.tail(window)
    statuses = recent["status"].fillna("").astype(str).tolist()
    sustained_failure = len(recent) >= window and all(
        status == "fail" for status in statuses
    )
    sustained_warning = len(recent) >= window and all(
        status in {"warn", "fail"} for status in statuses
    )
    output_status = "fail" if sustained_failure else "warn" if sustained_warning else "pass"
    return {
        "status": output_status,
        "path": str(path),
        "row_count": int(len(frame)),
        "sustained_warning_window": window,
        "recent_statuses": statuses,
        "sustained_warning": sustained_warning,
        "sustained_failure": sustained_failure,
        "latest_status": str(frame.iloc[-1].get("status")),
        "latest_generated_at": str(frame.iloc[-1].get("generated_at")),
    }


def generate_allocator_monitoring_report(
    registry: AllocatorRegistry,
    *,
    allocator_id: str,
    factor_registry: FactorRegistry | None = None,
    project_root: str | Path = ".",
) -> AllocatorMonitoringReport:
    """Generate a read-only monitoring report from allocator evidence artifacts."""

    root = Path(project_root)
    registry_report = validate_allocator_registry(
        registry,
        factor_registry=factor_registry,
        project_root=root,
    )
    allocator = registry.get(allocator_id)
    checks: list[dict[str, Any]] = []
    sections: dict[str, Any] = {
        "registry": {
            "status": registry_report.status,
            "error_count": registry_report.summary.get("error_count", 0),
            "warning_count": registry_report.summary.get("warning_count", 0),
        }
    }
    checks.append(
        _check(
            "registry",
            "registry_validation",
            registry_report.status,
            {
                "error_count": registry_report.summary.get("error_count", 0),
                "warning_count": registry_report.summary.get("warning_count", 0),
            },
        )
    )

    for name, section_builder in (
        ("validation", _validation_section),
        ("capacity", _capacity_section),
        ("event_state_gate", _event_state_gate_section),
        ("factor_health", _factor_health_section),
    ):
        section, section_checks = section_builder(allocator, root)
        sections[name] = section
        checks.extend(section_checks)

    status = _combine_status(check["status"] for check in checks)
    return AllocatorMonitoringReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        allocator={
            "allocator_id": allocator.allocator_id,
            "display_name": allocator.display_name,
            "status": allocator.status,
            "validation_status": allocator.validation.get("status"),
            "registry_name": registry.registry_name,
            "registry_version": registry.version,
        },
        sections=_json_safe(sections),
        checks=tuple(_json_safe(check) for check in checks),
    )


def write_allocator_monitoring_report(
    report: AllocatorMonitoringReport,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON and Markdown monitoring reports."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "allocator_monitoring_report.json"
    markdown_path = output_path / "allocator_monitoring_report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_allocator_monitoring_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_allocator_monitoring_markdown(report: AllocatorMonitoringReport) -> str:
    """Render a compact Markdown monitoring report."""

    payload = report.to_dict()
    allocator = payload["allocator"]
    sections = payload["sections"]
    lines = [
        "# Allocator Monitoring Report",
        "",
        f"- Status: `{payload['status']}`",
        f"- Allocator: `{allocator['allocator_id']}`",
        f"- Validation status: `{allocator.get('validation_status')}`",
        f"- Generated at: `{payload['generated_at']}`",
        "",
        "## Sections",
        "",
        "| section | status | read |",
        "| --- | --- | --- |",
    ]
    for name in ("registry", "validation", "capacity", "event_state_gate", "factor_health"):
        section = sections.get(name, {})
        lines.append(
            f"| {name} | `{section.get('status')}` | {section.get('read', '')} |"
        )
    validation = sections.get("validation", {})
    primary = validation.get("primary_metrics") or {}
    if primary:
        lines.extend(
            [
                "",
                "## Primary Metrics",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        for key in (
            "total_return",
            "max_drawdown",
            "gross_turnover",
            "total_transaction_cost",
        ):
            lines.append(f"| {key} | {primary.get(key)} |")
    capacity = sections.get("capacity", {})
    scenarios = capacity.get("scenarios") or []
    if scenarios:
        lines.extend(
            [
                "",
                "## Capacity",
                "",
                "| scenario | status | return | max drawdown | unfilled/traded |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in scenarios:
            lines.append(
                "| {scenario} | `{status}` | {total_return} | {max_drawdown} | {capacity_unfilled_vs_traded} |".format(
                    **row
                )
            )
    factor_health = sections.get("factor_health", {})
    latest_features = factor_health.get("latest_features") or []
    if latest_features:
        lines.extend(
            [
                "",
                "## Latest Factor Health",
                "",
                "| feature | state | weight scale |",
                "| --- | --- | ---: |",
            ]
        )
        for row in latest_features:
            lines.append(
                f"| `{row.get('feature')}` | `{row.get('health_state')}` | {row.get('weight_scale')} |"
            )
    return "\n".join(lines) + "\n"


def _validation_section(
    allocator: AllocatorRegistryEntry,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = allocator.validation.get("robust_validation") or allocator.validation.get(
        "standard_validation"
    )
    path = _project_path(source, project_root)
    if path is None or not path.exists():
        section = {
            "status": "fail",
            "source": source,
            "read": "validation summary is missing",
        }
        return section, [_check("validation", "validation_summary_present", "fail", section)]
    payload = _read_json(path)
    validation = payload.get("validation") if isinstance(payload, dict) else {}
    if not isinstance(validation, dict):
        validation = {}
    status = _normalized_status(str(validation.get("overall_status", "pending")))
    params = payload.get("params") if isinstance(payload, dict) else {}
    if not isinstance(params, dict):
        params = {}
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list):
        results = []
    method = str(params.get("primary_method") or allocator.score.get("combination_method"))
    policy = str(params.get("policy") or allocator.execution_policy.get("policy"))
    full_base = _find_result(results, "full_base", method, policy)
    high_cost = _find_result(results, "full_high_cost", method, policy)
    zero_cost = _find_result(results, "full_zero_cost", method, policy)
    yearly = [
        row
        for row in results
        if isinstance(row, dict)
        and str(row.get("scenario", "")).startswith("year_")
        and row.get("method") == method
        and row.get("policy") == policy
    ]
    section = {
        "status": status,
        "source": str(path),
        "overall_status": validation.get("overall_status"),
        "failed_count": validation.get("failed_count", 0),
        "warning_count": validation.get("warning_count", 0),
        "primary_method": method,
        "policy": policy,
        "primary_metrics": _result_metrics(full_base),
        "high_cost_metrics": _result_metrics(high_cost),
        "zero_cost_metrics": _result_metrics(zero_cost),
        "yearly_returns": {
            str(row.get("scenario")): _finite_or_none(row.get("total_return"))
            for row in sorted(yearly, key=lambda item: str(item.get("scenario")))
        },
        "read": f"validation overall_status={validation.get('overall_status')}",
    }
    checks = [
        _check(
            "validation",
            "validation_overall_status",
            status,
            {
                "overall_status": validation.get("overall_status"),
                "failed_count": validation.get("failed_count", 0),
                "warning_count": validation.get("warning_count", 0),
            },
        )
    ]
    return section, checks


def _capacity_section(
    allocator: AllocatorRegistryEntry,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    monitoring = allocator.governance.get("capacity_monitoring")
    if not isinstance(monitoring, dict):
        section = {
            "status": "warn",
            "read": "capacity monitoring is not declared",
        }
        return section, [_check("capacity", "capacity_monitoring_declared", "warn", section)]
    path = _project_path(monitoring.get("diagnostic_summary"), project_root)
    if path is None or not path.exists():
        section = {
            "status": "fail",
            "source": monitoring.get("diagnostic_summary"),
            "read": "capacity diagnostic summary is missing",
        }
        return section, [_check("capacity", "capacity_summary_present", "fail", section)]
    payload = _read_json(path)
    rows = payload if isinstance(payload, list) else []
    rows_by_scenario = {
        str(row.get("scenario")): row for row in rows if isinstance(row, dict)
    }
    thresholds = monitoring.get("warning_thresholds")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    scenario_names = monitoring.get("stress_scenarios")
    scenario_names = scenario_names if isinstance(scenario_names, list) else []
    checks: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for scenario_name in scenario_names:
        scenario = str(scenario_name)
        row = rows_by_scenario.get(scenario)
        if row is None:
            checks.append(
                _check(
                    "capacity",
                    f"{scenario}_result_present",
                    "fail",
                    {"scenario": scenario},
                )
            )
            continue
        breaches = _capacity_threshold_breaches(row, thresholds)
        status = "warn" if breaches else "pass"
        checks.append(
            _check(
                "capacity",
                f"{scenario}_thresholds",
                status,
                {"scenario": scenario, "breaches": breaches},
            )
        )
        scenario_rows.append(
            {
                "scenario": scenario,
                "status": status,
                "total_return": _finite_or_none(row.get("total_return")),
                "max_drawdown": _finite_or_none(row.get("max_drawdown")),
                "gross_turnover": _finite_or_none(row.get("gross_turnover")),
                "capacity_limited_event_count": _finite_or_none(
                    row.get("capacity_limited_event_count")
                ),
                "capacity_unfilled_notional": _finite_or_none(
                    row.get("capacity_unfilled_notional")
                ),
                "capacity_unfilled_vs_traded": _finite_or_none(
                    row.get("capacity_unfilled_vs_traded")
                ),
                "capacity_unfilled_vs_desired_capacity_events": _finite_or_none(
                    row.get("capacity_unfilled_vs_desired_capacity_events")
                ),
                "breaches": breaches,
            }
        )
    status = _combine_status(check["status"] for check in checks) if checks else "warn"
    section = {
        "status": status,
        "source": str(path),
        "mode": monitoring.get("mode"),
        "thresholds": thresholds,
        "scenarios": scenario_rows,
        "read": f"{len(scenario_rows)} capacity stress scenarios checked",
    }
    return section, checks


def _event_state_gate_section(
    allocator: AllocatorRegistryEntry,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_gate = allocator.risk_controls.get("event_state_gate")
    if not isinstance(event_gate, dict):
        section = {"status": "fail", "read": "event_state_gate is missing"}
        return section, [_check("event_state_gate", "event_gate_present", "fail", section)]
    summary_path = _project_path(event_gate.get("summary_path"), project_root)
    if summary_path is None or not summary_path.exists():
        section = {
            "status": "fail",
            "source": event_gate.get("summary_path"),
            "read": "event-state gate summary is missing",
        }
        return section, [
            _check("event_state_gate", "event_gate_summary_present", "fail", section)
        ]
    summary = _read_json(summary_path)
    schedule_path = _project_path(event_gate.get("schedule_path"), project_root)
    latest = _latest_schedule_row(schedule_path)
    schedule_count = _number(summary.get("schedule_count")) if isinstance(summary, dict) else None
    scale_counts = summary.get("scale_counts", {}) if isinstance(summary, dict) else {}
    if not isinstance(scale_counts, dict):
        scale_counts = {}
    blocked_count = _number(scale_counts.get("0.0")) or 0.0
    blocked_share = blocked_count / schedule_count if schedule_count else None
    latest_scale = _number(latest.get("gross_exposure_scale")) if latest else None
    status = "warn" if latest_scale is not None and latest_scale < 1.0 else "pass"
    section = {
        "status": status,
        "source": str(summary_path),
        "schedule_path": str(schedule_path) if schedule_path is not None else None,
        "schedule_count": _finite_or_none(schedule_count),
        "blocked_count": _finite_or_none(blocked_count),
        "blocked_share": _finite_or_none(blocked_share),
        "latest_timestamp": latest.get("timestamp") if latest else None,
        "latest_scale": _finite_or_none(latest_scale),
        "latest_reason": latest.get("event_state_gate_reason") if latest else None,
        "read": "latest event-state gate is active" if status == "warn" else "event-state gate read cleanly",
    }
    return section, [
        _check(
            "event_state_gate",
            "latest_event_state_gate_scale",
            status,
            {
                "latest_timestamp": section["latest_timestamp"],
                "latest_scale": section["latest_scale"],
                "latest_reason": section["latest_reason"],
            },
        )
    ]


def _factor_health_section(
    allocator: AllocatorRegistryEntry,
    project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    health = allocator.risk_controls.get("factor_health")
    if not isinstance(health, dict):
        section = {"status": "fail", "read": "factor_health is missing"}
        return section, [_check("factor_health", "factor_health_present", "fail", section)]
    schedule_path = _project_path(health.get("schedule_path"), project_root)
    if schedule_path is None or not schedule_path.exists():
        section = {
            "status": "fail",
            "source": health.get("schedule_path"),
            "read": "factor-health schedule is missing",
        }
        return section, [
            _check("factor_health", "factor_health_schedule_present", "fail", section)
        ]
    frame = pd.read_csv(schedule_path)
    if frame.empty or "timestamp" not in frame.columns:
        section = {
            "status": "fail",
            "source": str(schedule_path),
            "read": "factor-health schedule is empty or malformed",
        }
        return section, [_check("factor_health", "factor_health_schedule_valid", "fail", section)]
    state_counts = (
        frame["health_state"].fillna("nan").astype(str).value_counts().to_dict()
        if "health_state" in frame.columns
        else {}
    )
    latest_timestamp = str(frame["timestamp"].max())
    latest = frame.loc[frame["timestamp"].astype(str) == latest_timestamp].copy()
    latest_features = []
    for row in latest.sort_values("feature").to_dict(orient="records"):
        latest_features.append(
            {
                "feature": row.get("feature"),
                "health_state": row.get("health_state"),
                "weight_scale": _finite_or_none(row.get("weight_scale")),
                "shrink_reason": row.get("shrink_reason"),
            }
        )
    latest_impaired = sum(row.get("health_state") == "impaired" for row in latest_features)
    latest_watch = sum(row.get("health_state") == "watch" for row in latest_features)
    latest_min_scale = min(
        (
            value
            for value in (_number(row.get("weight_scale")) for row in latest_features)
            if value is not None
        ),
        default=None,
    )
    status = "warn" if latest_impaired or (latest_min_scale is not None and latest_min_scale < 1.0) else "pass"
    section = {
        "status": status,
        "source": str(schedule_path),
        "observation_count": int(len(frame)),
        "feature_count": int(frame["feature"].nunique()) if "feature" in frame.columns else 0,
        "state_counts": state_counts,
        "latest_timestamp": latest_timestamp,
        "latest_impaired_feature_count": int(latest_impaired),
        "latest_watch_feature_count": int(latest_watch),
        "latest_min_weight_scale": _finite_or_none(latest_min_scale),
        "latest_features": latest_features,
        "read": "latest factor-health shrink is active" if status == "warn" else "factor-health schedule read cleanly",
    }
    return section, [
        _check(
            "factor_health",
            "latest_factor_health_state",
            status,
            {
                "latest_timestamp": latest_timestamp,
                "latest_impaired_feature_count": latest_impaired,
                "latest_watch_feature_count": latest_watch,
                "latest_min_weight_scale": latest_min_scale,
            },
        )
    ]


def _capacity_threshold_breaches(
    row: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    breaches: list[dict[str, Any]] = []
    for threshold_name, threshold_value in thresholds.items():
        threshold_name = str(threshold_name)
        if threshold_name not in _CAPACITY_MONITOR_THRESHOLDS:
            continue
        threshold = _number(threshold_value)
        if threshold is None:
            continue
        metric, comparator = _CAPACITY_MONITOR_THRESHOLDS[threshold_name]
        observed = _number(row.get(metric))
        if observed is None:
            breaches.append(
                {
                    "threshold": threshold_name,
                    "metric": metric,
                    "observed": None,
                    "limit": threshold,
                    "reason": "missing_metric",
                }
            )
            continue
        if _threshold_breached(observed, threshold, comparator):
            breaches.append(
                {
                    "threshold": threshold_name,
                    "metric": metric,
                    "observed": observed,
                    "limit": threshold,
                }
            )
    return breaches


def _threshold_breached(observed: float, threshold: float, comparator: str) -> bool:
    if comparator == "min":
        return observed < threshold
    if comparator == "max_abs":
        return abs(observed) > threshold
    return observed > threshold


def _find_result(
    results: list[Any],
    scenario: str,
    method: str,
    policy: str,
) -> dict[str, Any] | None:
    for row in results:
        if not isinstance(row, dict):
            continue
        if (
            row.get("scenario") == scenario
            and row.get("method") == method
            and row.get("policy") == policy
        ):
            return row
    return None


def _result_metrics(row: dict[str, Any] | None) -> dict[str, float | None]:
    if row is None:
        return {}
    return {
        key: _finite_or_none(row.get(key))
        for key in (
            "total_return",
            "max_drawdown",
            "gross_turnover",
            "trade_count",
            "total_transaction_cost",
            "final_equity",
        )
    }


def _latest_schedule_row(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "timestamp" not in frame.columns:
        return None
    latest_timestamp = str(frame["timestamp"].max())
    latest = frame.loc[frame["timestamp"].astype(str) == latest_timestamp]
    if latest.empty:
        return None
    return latest.iloc[-1].to_dict()


def _check(
    section: str,
    name: str,
    status: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "section": section,
        "name": name,
        "status": status,
        "details": _json_safe(details),
    }


def _combine_status(statuses: Any) -> str:
    values = list(statuses)
    if any(value == "fail" for value in values):
        return "fail"
    if any(value == "warn" for value in values):
        return "warn"
    return "pass"


def _normalized_status(status: str) -> str:
    if status in {"pass", "warn", "fail"}:
        return status
    return "warn"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _project_path(value: object, project_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.startswith(("http://", "https://")):
        return None
    path = Path(text)
    if not path.is_absolute():
        path = project_root / path
    return path


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(output):
        return None
    return output


def _finite_or_none(value: Any) -> float | None:
    return _number(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if value is not None and not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value
