"""Candidate allocator registry models and governance validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any

from quant_research.factors import FactorRegistry


ALLOCATOR_STATUSES = frozenset(
    {"planned", "candidate", "watchlist", "reject", "promoted", "deprecated"}
)
VALIDATION_STATUSES = frozenset({"pass", "warn", "fail", "pending"})
FEATURE_DIRECTIONS = frozenset({"long", "invert"})
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
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
class AllocatorRegistryEntry:
    """One governed candidate allocator definition."""

    allocator_id: str
    display_name: str
    status: str
    description: str
    hypothesis: str
    score: dict[str, Any]
    risk_controls: dict[str, Any]
    execution_policy: dict[str, Any]
    cost_model: dict[str, Any]
    validation: dict[str, Any]
    governance: dict[str, Any]
    data: dict[str, Any]
    references: tuple[str, ...]
    tags: tuple[str, ...] = ()
    owner: str = "quant_research"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AllocatorRegistryEntry":
        """Build an entry from a JSON-compatible dictionary."""

        return cls(
            allocator_id=str(payload.get("allocator_id", "")),
            display_name=str(payload.get("display_name", "")),
            status=str(payload.get("status", "")),
            description=str(payload.get("description", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            score=dict(payload.get("score") or {}),
            risk_controls=dict(payload.get("risk_controls") or {}),
            execution_policy=dict(payload.get("execution_policy") or {}),
            cost_model=dict(payload.get("cost_model") or {}),
            validation=dict(payload.get("validation") or {}),
            governance=dict(payload.get("governance") or {}),
            data=dict(payload.get("data") or {}),
            references=tuple(str(value) for value in payload.get("references", ())),
            tags=tuple(str(value) for value in payload.get("tags", ())),
            owner=str(payload.get("owner", "quant_research")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible entry payload."""

        return {
            "allocator_id": self.allocator_id,
            "display_name": self.display_name,
            "status": self.status,
            "owner": self.owner,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "score": self.score,
            "risk_controls": self.risk_controls,
            "execution_policy": self.execution_policy,
            "cost_model": self.cost_model,
            "data": self.data,
            "validation": self.validation,
            "governance": self.governance,
            "references": list(self.references),
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class AllocatorRegistry:
    """Structured allocator registry loaded from a versioned JSON file."""

    registry_name: str
    version: int
    allocators: tuple[AllocatorRegistryEntry, ...]
    owner: str = "quant_research"
    scope: str = ""
    updated_at: str = ""
    references: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AllocatorRegistry":
        """Build a registry from a JSON-compatible dictionary."""

        return cls(
            registry_name=str(payload.get("registry_name", "")),
            version=int(payload.get("version") or 0),
            allocators=tuple(
                AllocatorRegistryEntry.from_dict(entry)
                for entry in payload.get("allocators", ())
            ),
            owner=str(payload.get("owner", "quant_research")),
            scope=str(payload.get("scope", "")),
            updated_at=str(payload.get("updated_at", "")),
            references=tuple(str(value) for value in payload.get("references", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible registry payload."""

        return {
            "registry_name": self.registry_name,
            "version": self.version,
            "owner": self.owner,
            "scope": self.scope,
            "updated_at": self.updated_at,
            "references": list(self.references),
            "allocators": [entry.to_dict() for entry in self.allocators],
        }

    def get(self, allocator_id: str) -> AllocatorRegistryEntry:
        """Return an allocator entry by identifier."""

        for entry in self.allocators:
            if entry.allocator_id == allocator_id:
                return entry
        raise KeyError(allocator_id)


@dataclass(frozen=True, slots=True)
class AllocatorRegistryIssue:
    """One allocator registry validation issue."""

    severity: str
    code: str
    message: str
    allocator_id: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible issue payload."""

        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "allocator_id": self.allocator_id,
            "field": self.field,
        }


@dataclass(frozen=True, slots=True)
class AllocatorRegistryValidationReport:
    """Machine-readable validation report for an allocator registry."""

    generated_at: str
    status: str
    summary: dict[str, Any]
    issues: tuple[AllocatorRegistryIssue, ...]
    allocators: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report payload."""

        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
            "allocators": list(self.allocators),
        }


def load_allocator_registry(path: str | Path) -> AllocatorRegistry:
    """Read an allocator registry JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("allocator registry payload must be a JSON object")
    return AllocatorRegistry.from_dict(payload)


def validate_allocator_registry(
    registry: AllocatorRegistry,
    *,
    factor_registry: FactorRegistry | None = None,
    project_root: str | Path = ".",
) -> AllocatorRegistryValidationReport:
    """Validate allocator registry governance requirements."""

    root = Path(project_root)
    issues: list[AllocatorRegistryIssue] = []
    if not registry.registry_name:
        _add_issue(issues, "error", "missing_registry_name", "registry_name is required")
    if registry.version <= 0:
        _add_issue(issues, "error", "invalid_version", "version must be positive")
    if not registry.allocators:
        _add_issue(issues, "error", "empty_registry", "at least one allocator is required")

    seen_ids: set[str] = set()
    allocator_rows: list[dict[str, Any]] = []
    feature_to_factor = _feature_to_factor(factor_registry)
    for entry in registry.allocators:
        _validate_entry(
            entry,
            issues=issues,
            seen_ids=seen_ids,
            feature_to_factor=feature_to_factor,
            project_root=root,
        )
        allocator_rows.append(_allocator_row(entry))

    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    status = "fail" if error_count else "warn" if warning_count else "pass"
    return AllocatorRegistryValidationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        summary={
            "registry_name": registry.registry_name,
            "version": registry.version,
            "allocator_count": len(registry.allocators),
            "status_counts": _count_statuses(registry.allocators),
            "error_count": error_count,
            "warning_count": warning_count,
        },
        issues=tuple(issues),
        allocators=tuple(allocator_rows),
    )


def write_allocator_registry_report(
    report: AllocatorRegistryValidationReport,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON and Markdown allocator registry reports."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "allocator_registry_validation.json"
    markdown_path = output_path / "allocator_registry_validation.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_allocator_registry_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_allocator_registry_markdown(
    report: AllocatorRegistryValidationReport,
) -> str:
    """Render an allocator registry validation report as Markdown."""

    summary = report.summary
    lines = [
        "# Allocator Registry Validation",
        "",
        f"- Status: `{report.status}`",
        f"- Registry: `{summary.get('registry_name')}`",
        f"- Version: `{summary.get('version')}`",
        f"- Allocators: `{summary.get('allocator_count')}`",
        f"- Errors: `{summary.get('error_count')}`",
        f"- Warnings: `{summary.get('warning_count')}`",
        "",
        "## Allocators",
        "",
        "| allocator | status | validation | feature count | references |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in report.allocators:
        lines.append(
            "| {allocator_id} | {status} | {validation_status} | {feature_count} | {reference_count} |".format(
                **row
            )
        )
    if report.issues:
        lines.extend(["", "## Issues", ""])
        for issue in report.issues:
            lines.append(
                f"- `{issue.severity}` `{issue.code}`"
                f" `{issue.allocator_id or '-'}`: {issue.message}"
            )
    return "\n".join(lines) + "\n"


def _validate_entry(
    entry: AllocatorRegistryEntry,
    *,
    issues: list[AllocatorRegistryIssue],
    seen_ids: set[str],
    feature_to_factor: dict[str, Any],
    project_root: Path,
) -> None:
    allocator_id = entry.allocator_id
    if not allocator_id:
        _add_issue(issues, "error", "missing_allocator_id", "allocator_id is required")
    elif not _ID_RE.match(allocator_id):
        _add_issue(
            issues,
            "error",
            "invalid_allocator_id",
            "allocator_id must use lowercase letters, digits, and underscores",
            allocator_id=allocator_id,
            field="allocator_id",
        )
    elif allocator_id in seen_ids:
        _add_issue(
            issues,
            "error",
            "duplicate_allocator_id",
            f"duplicate allocator_id: {allocator_id}",
            allocator_id=allocator_id,
            field="allocator_id",
        )
    seen_ids.add(allocator_id)

    if entry.status not in ALLOCATOR_STATUSES:
        _add_issue(
            issues,
            "error",
            "invalid_status",
            f"status must be one of {sorted(ALLOCATOR_STATUSES)}",
            allocator_id=allocator_id,
            field="status",
        )
    for field_name in ("display_name", "description", "hypothesis"):
        if not getattr(entry, field_name):
            _add_issue(
                issues,
                "error",
                f"missing_{field_name}",
                f"{field_name} is required",
                allocator_id=allocator_id,
                field=field_name,
            )

    _validate_score(entry, issues=issues, feature_to_factor=feature_to_factor)
    _validate_execution_policy(entry, issues=issues)
    _validate_risk_controls(entry, issues=issues, project_root=project_root)
    _validate_cost_model(entry, issues=issues)
    _validate_validation(entry, issues=issues, project_root=project_root)
    _validate_capacity_monitoring(entry, issues=issues, project_root=project_root)
    _validate_references(entry, issues=issues, project_root=project_root)

    if entry.status in {"candidate", "promoted"}:
        validation_status = str(entry.validation.get("status", ""))
        if validation_status != "pass":
            _add_issue(
                issues,
                "error",
                "active_allocator_not_validated",
                "candidate/promoted allocators must have validation.status == pass",
                allocator_id=allocator_id,
                field="validation.status",
            )


def _validate_score(
    entry: AllocatorRegistryEntry,
    *,
    issues: list[AllocatorRegistryIssue],
    feature_to_factor: dict[str, Any],
) -> None:
    features = entry.score.get("features")
    if not isinstance(features, list) or not features:
        _add_issue(
            issues,
            "error",
            "missing_score_features",
            "score.features must be a non-empty list",
            allocator_id=entry.allocator_id,
            field="score.features",
        )
        return
    seen_features: set[str] = set()
    weights: list[float] = []
    for index, feature_payload in enumerate(features):
        if not isinstance(feature_payload, dict):
            _add_issue(
                issues,
                "error",
                "invalid_score_feature",
                "each score feature must be an object",
                allocator_id=entry.allocator_id,
                field=f"score.features[{index}]",
            )
            continue
        feature = str(feature_payload.get("feature", ""))
        direction = str(feature_payload.get("direction", ""))
        if not feature:
            _add_issue(
                issues,
                "error",
                "missing_feature",
                "feature is required",
                allocator_id=entry.allocator_id,
                field=f"score.features[{index}].feature",
            )
        elif feature in seen_features:
            _add_issue(
                issues,
                "error",
                "duplicate_feature",
                f"duplicate feature: {feature}",
                allocator_id=entry.allocator_id,
                field="score.features",
            )
        seen_features.add(feature)
        if direction not in FEATURE_DIRECTIONS:
            _add_issue(
                issues,
                "error",
                "invalid_feature_direction",
                "feature direction must be long or invert",
                allocator_id=entry.allocator_id,
                field=f"score.features[{index}].direction",
            )
        weight = _optional_float(feature_payload.get("weight"))
        if weight is None or weight < 0:
            _add_issue(
                issues,
                "error",
                "invalid_feature_weight",
                "feature weight must be non-negative",
                allocator_id=entry.allocator_id,
                field=f"score.features[{index}].weight",
            )
        else:
            weights.append(weight)
        factor_entry = feature_to_factor.get(feature)
        if feature_to_factor and factor_entry is None:
            _add_issue(
                issues,
                "error",
                "feature_missing_from_factor_registry",
                f"feature is not present in factor registry: {feature}",
                allocator_id=entry.allocator_id,
                field=f"score.features[{index}].feature",
            )
        elif factor_entry is not None and factor_entry.status in {"reject", "deprecated"}:
            _add_issue(
                issues,
                "error",
                "inactive_factor_feature",
                f"feature belongs to inactive factor {factor_entry.factor_id}",
                allocator_id=entry.allocator_id,
                field=f"score.features[{index}].feature",
            )
    if weights and abs(sum(weights) - 1.0) > 1e-6:
        _add_issue(
            issues,
            "error",
            "feature_weights_not_normalized",
            f"score feature weights must sum to 1.0, got {sum(weights):.12f}",
            allocator_id=entry.allocator_id,
            field="score.features",
        )


def _validate_execution_policy(
    entry: AllocatorRegistryEntry,
    *,
    issues: list[AllocatorRegistryIssue],
) -> None:
    policy = entry.execution_policy
    positive_ints = (
        "top_n",
        "rebalance_every_n_bars",
        "entry_rank",
        "exit_rank",
        "max_entries_per_rebalance",
        "max_exits_per_rebalance",
    )
    for key in positive_ints:
        value = _optional_int(policy.get(key))
        if value is None or value <= 0:
            _add_issue(
                issues,
                "error",
                "invalid_execution_policy_integer",
                f"execution_policy.{key} must be positive",
                allocator_id=entry.allocator_id,
                field=f"execution_policy.{key}",
            )
    entry_rank = _optional_int(policy.get("entry_rank"))
    exit_rank = _optional_int(policy.get("exit_rank"))
    if entry_rank is not None and exit_rank is not None and exit_rank < entry_rank:
        _add_issue(
            issues,
            "error",
            "invalid_rank_buffer",
            "exit_rank must be greater than or equal to entry_rank",
            allocator_id=entry.allocator_id,
            field="execution_policy.exit_rank",
        )
    partial = _optional_float(policy.get("partial_rebalance_rate"))
    if partial is None or not 0 < partial <= 1:
        _add_issue(
            issues,
            "error",
            "invalid_partial_rebalance_rate",
            "partial_rebalance_rate must be in (0, 1]",
            allocator_id=entry.allocator_id,
            field="execution_policy.partial_rebalance_rate",
        )
    band = _optional_float(policy.get("no_trade_weight_band"))
    if band is None or band < 0:
        _add_issue(
            issues,
            "error",
            "invalid_no_trade_weight_band",
            "no_trade_weight_band must be non-negative",
            allocator_id=entry.allocator_id,
            field="execution_policy.no_trade_weight_band",
        )


def _validate_risk_controls(
    entry: AllocatorRegistryEntry,
    *,
    issues: list[AllocatorRegistryIssue],
    project_root: Path,
) -> None:
    event_gate = entry.risk_controls.get("event_state_gate")
    if not isinstance(event_gate, dict):
        _add_issue(
            issues,
            "error",
            "missing_event_state_gate",
            "risk_controls.event_state_gate is required",
            allocator_id=entry.allocator_id,
            field="risk_controls.event_state_gate",
        )
    else:
        blocked = event_gate.get("blocked_states")
        if not isinstance(blocked, list) or not blocked:
            _add_issue(
                issues,
                "error",
                "missing_blocked_states",
                "event_state_gate.blocked_states must be non-empty",
                allocator_id=entry.allocator_id,
                field="risk_controls.event_state_gate.blocked_states",
            )
        _check_existing_path(
            event_gate.get("schedule_path"),
            issues=issues,
            allocator_id=entry.allocator_id,
            field="risk_controls.event_state_gate.schedule_path",
            project_root=project_root,
        )
    factor_health = entry.risk_controls.get("factor_health")
    if not isinstance(factor_health, dict):
        _add_issue(
            issues,
            "error",
            "missing_factor_health",
            "risk_controls.factor_health is required",
            allocator_id=entry.allocator_id,
            field="risk_controls.factor_health",
        )
    elif factor_health.get("mode") != "lagged_shrink":
        _add_issue(
            issues,
            "error",
            "invalid_factor_health_mode",
            "factor_health.mode must be lagged_shrink",
            allocator_id=entry.allocator_id,
            field="risk_controls.factor_health.mode",
        )


def _validate_cost_model(
    entry: AllocatorRegistryEntry,
    *,
    issues: list[AllocatorRegistryIssue],
) -> None:
    for key in ("commission_bps", "slippage_bps", "sell_stamp_tax_bps", "min_commission"):
        value = _optional_float(entry.cost_model.get(key))
        if value is None or value < 0:
            _add_issue(
                issues,
                "error",
                "invalid_cost_model_value",
                f"cost_model.{key} must be non-negative",
                allocator_id=entry.allocator_id,
                field=f"cost_model.{key}",
            )


def _validate_validation(
    entry: AllocatorRegistryEntry,
    *,
    issues: list[AllocatorRegistryIssue],
    project_root: Path,
) -> None:
    status = str(entry.validation.get("status", ""))
    if status not in VALIDATION_STATUSES:
        _add_issue(
            issues,
            "error",
            "invalid_validation_status",
            f"validation.status must be one of {sorted(VALIDATION_STATUSES)}",
            allocator_id=entry.allocator_id,
            field="validation.status",
        )
    for key in ("standard_validation", "robust_validation"):
        _check_existing_path(
            entry.validation.get(key),
            issues=issues,
            allocator_id=entry.allocator_id,
            field=f"validation.{key}",
            project_root=project_root,
        )


def _validate_capacity_monitoring(
    entry: AllocatorRegistryEntry,
    *,
    issues: list[AllocatorRegistryIssue],
    project_root: Path,
) -> None:
    monitoring = entry.governance.get("capacity_monitoring")
    capacity_checked = "capacity_checked" in entry.tags or any(
        str(key).startswith("capacity_") for key in entry.validation
    )
    if monitoring is None:
        if capacity_checked and entry.status in {"candidate", "promoted"}:
            _add_issue(
                issues,
                "warning",
                "missing_capacity_monitoring",
                "capacity-checked active allocators should declare governance.capacity_monitoring",
                allocator_id=entry.allocator_id,
                field="governance.capacity_monitoring",
            )
        return
    if not isinstance(monitoring, dict):
        _add_issue(
            issues,
            "error",
            "invalid_capacity_monitoring",
            "governance.capacity_monitoring must be an object",
            allocator_id=entry.allocator_id,
            field="governance.capacity_monitoring",
        )
        return
    if monitoring.get("mode") != "monitor_only":
        _add_issue(
            issues,
            "error",
            "invalid_capacity_monitoring_mode",
            "capacity_monitoring.mode must be monitor_only",
            allocator_id=entry.allocator_id,
            field="governance.capacity_monitoring.mode",
        )
    summary_value = monitoring.get("diagnostic_summary")
    _check_existing_path(
        summary_value,
        issues=issues,
        allocator_id=entry.allocator_id,
        field="governance.capacity_monitoring.diagnostic_summary",
        project_root=project_root,
    )
    stress_scenarios = monitoring.get("stress_scenarios")
    if not isinstance(stress_scenarios, list) or not stress_scenarios:
        _add_issue(
            issues,
            "error",
            "missing_capacity_stress_scenarios",
            "capacity_monitoring.stress_scenarios must be a non-empty list",
            allocator_id=entry.allocator_id,
            field="governance.capacity_monitoring.stress_scenarios",
        )
        return
    scenario_names = [str(value) for value in stress_scenarios]
    thresholds = _capacity_monitor_thresholds(entry, monitoring, issues)
    summary_path = _project_path(summary_value, project_root)
    if summary_path is None or not summary_path.exists():
        return
    rows = _load_capacity_monitor_rows(
        summary_path,
        entry=entry,
        issues=issues,
    )
    if rows is None:
        return
    rows_by_scenario = {
        str(row.get("scenario")): row for row in rows if isinstance(row, dict)
    }
    for scenario in scenario_names:
        row = rows_by_scenario.get(scenario)
        if row is None:
            _add_issue(
                issues,
                "error",
                "missing_capacity_stress_result",
                f"capacity diagnostic summary is missing scenario: {scenario}",
                allocator_id=entry.allocator_id,
                field="governance.capacity_monitoring.diagnostic_summary",
            )
            continue
        _check_capacity_monitor_thresholds(
            row,
            thresholds=thresholds,
            scenario=scenario,
            entry=entry,
            issues=issues,
        )


def _capacity_monitor_thresholds(
    entry: AllocatorRegistryEntry,
    monitoring: dict[str, Any],
    issues: list[AllocatorRegistryIssue],
) -> dict[str, float]:
    thresholds = monitoring.get("warning_thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        _add_issue(
            issues,
            "error",
            "missing_capacity_warning_thresholds",
            "capacity_monitoring.warning_thresholds must be a non-empty object",
            allocator_id=entry.allocator_id,
            field="governance.capacity_monitoring.warning_thresholds",
        )
        return {}
    output: dict[str, float] = {}
    for key, value in thresholds.items():
        key_text = str(key)
        threshold = _optional_float(value)
        if threshold is None:
            _add_issue(
                issues,
                "error",
                "invalid_capacity_warning_threshold",
                f"capacity warning threshold is not numeric: {key_text}",
                allocator_id=entry.allocator_id,
                field=f"governance.capacity_monitoring.warning_thresholds.{key_text}",
            )
            continue
        if key_text not in _CAPACITY_MONITOR_THRESHOLDS:
            _add_issue(
                issues,
                "warning",
                "unknown_capacity_warning_threshold",
                f"capacity warning threshold is not enforced by validator: {key_text}",
                allocator_id=entry.allocator_id,
                field=f"governance.capacity_monitoring.warning_thresholds.{key_text}",
            )
            continue
        if key_text.startswith("max_") and threshold < 0:
            _add_issue(
                issues,
                "error",
                "invalid_capacity_warning_threshold",
                f"capacity warning threshold must be non-negative: {key_text}",
                allocator_id=entry.allocator_id,
                field=f"governance.capacity_monitoring.warning_thresholds.{key_text}",
            )
            continue
        output[key_text] = threshold
    return output


def _load_capacity_monitor_rows(
    path: Path,
    *,
    entry: AllocatorRegistryEntry,
    issues: list[AllocatorRegistryIssue],
) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _add_issue(
            issues,
            "error",
            "invalid_capacity_monitoring_summary",
            f"capacity diagnostic summary cannot be read: {exc}",
            allocator_id=entry.allocator_id,
            field="governance.capacity_monitoring.diagnostic_summary",
        )
        return None
    if not isinstance(payload, list):
        _add_issue(
            issues,
            "error",
            "invalid_capacity_monitoring_summary",
            "capacity diagnostic summary must be a list of scenario rows",
            allocator_id=entry.allocator_id,
            field="governance.capacity_monitoring.diagnostic_summary",
        )
        return None
    return [row for row in payload if isinstance(row, dict)]


def _check_capacity_monitor_thresholds(
    row: dict[str, Any],
    *,
    thresholds: dict[str, float],
    scenario: str,
    entry: AllocatorRegistryEntry,
    issues: list[AllocatorRegistryIssue],
) -> None:
    for threshold_name, threshold in thresholds.items():
        field, comparator = _CAPACITY_MONITOR_THRESHOLDS[threshold_name]
        observed = _optional_float(row.get(field))
        if observed is None:
            _add_issue(
                issues,
                "error",
                "missing_capacity_monitor_metric",
                f"capacity diagnostic row {scenario} is missing metric: {field}",
                allocator_id=entry.allocator_id,
                field="governance.capacity_monitoring.diagnostic_summary",
            )
            continue
        if _capacity_threshold_breached(observed, threshold, comparator):
            _add_issue(
                issues,
                "warning",
                "capacity_monitor_threshold_breach",
                (
                    f"{scenario}.{field}={observed:.6f} breaches "
                    f"{threshold_name}={threshold:.6f}"
                ),
                allocator_id=entry.allocator_id,
                field=f"governance.capacity_monitoring.warning_thresholds.{threshold_name}",
            )


def _capacity_threshold_breached(
    observed: float,
    threshold: float,
    comparator: str,
) -> bool:
    if comparator == "min":
        return observed < threshold
    if comparator == "max_abs":
        return abs(observed) > threshold
    return observed > threshold


def _validate_references(
    entry: AllocatorRegistryEntry,
    *,
    issues: list[AllocatorRegistryIssue],
    project_root: Path,
) -> None:
    if not entry.references:
        _add_issue(
            issues,
            "error",
            "missing_references",
            "allocator references must be non-empty",
            allocator_id=entry.allocator_id,
            field="references",
        )
        return
    for index, reference in enumerate(entry.references):
        if reference.startswith(("http://", "https://")):
            continue
        _check_existing_path(
            reference,
            issues=issues,
            allocator_id=entry.allocator_id,
            field=f"references[{index}]",
            project_root=project_root,
        )


def _allocator_row(entry: AllocatorRegistryEntry) -> dict[str, Any]:
    features = entry.score.get("features")
    return {
        "allocator_id": entry.allocator_id,
        "display_name": entry.display_name,
        "status": entry.status,
        "validation_status": entry.validation.get("status"),
        "feature_count": len(features) if isinstance(features, list) else 0,
        "reference_count": len(entry.references),
    }


def _feature_to_factor(factor_registry: FactorRegistry | None) -> dict[str, Any]:
    if factor_registry is None:
        return {}
    mapping: dict[str, Any] = {}
    for entry in factor_registry.entries:
        for feature in entry.feature_columns:
            mapping[feature] = entry
    return mapping


def _check_existing_path(
    value: object,
    *,
    issues: list[AllocatorRegistryIssue],
    allocator_id: str,
    field: str,
    project_root: Path,
) -> None:
    if value in (None, ""):
        _add_issue(
            issues,
            "error",
            "missing_artifact_path",
            f"{field} is required",
            allocator_id=allocator_id,
            field=field,
        )
        return
    path_text = str(value)
    if path_text.startswith(("http://", "https://")):
        return
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        _add_issue(
            issues,
            "error",
            "missing_artifact",
            f"artifact does not exist: {value}",
            allocator_id=allocator_id,
            field=field,
        )


def _project_path(value: object, project_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path_text = str(value)
    if path_text.startswith(("http://", "https://")):
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root / path
    return path


def _add_issue(
    issues: list[AllocatorRegistryIssue],
    severity: str,
    code: str,
    message: str,
    *,
    allocator_id: str | None = None,
    field: str | None = None,
) -> None:
    issues.append(
        AllocatorRegistryIssue(
            severity=severity,
            code=code,
            message=message,
            allocator_id=allocator_id,
            field=field,
        )
    )


def _count_statuses(entries: tuple[AllocatorRegistryEntry, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return dict(sorted(counts.items()))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
