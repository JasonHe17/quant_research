"""Factor registry models and governance validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


FACTOR_FAMILIES = frozenset(
    {
        "event",
        "fundamental",
        "liquidity",
        "market_regime",
        "momentum",
        "quality",
        "reversal",
        "risk",
        "turnover",
        "value",
        "volatility",
        "volume",
        "other",
    }
)
FACTOR_STATUSES = frozenset(
    {"planned", "candidate", "watchlist", "reject", "promoted", "deprecated"}
)
EXPECTED_DIRECTIONS = frozenset({"long", "invert", "neutral", "mixed"})
DECISION_REASONS = frozenset(
    {
        "weak_ic",
        "unstable_years",
        "weak_hit_rate",
        "cost_fragile",
        "portfolio_negative",
        "duplicate_like",
        "implementation_issue",
        "data_quality",
        "risk_concentration",
        "other",
    }
)
REQUIRED_A_SHARE_FLAGS = (
    "long_only",
    "price_limit_aware",
    "st_aware",
    "t_plus_one_safe",
)
ACTIVE_STATUSES = frozenset({"candidate", "watchlist", "promoted"})
MEMORY_REQUIRED_STATUSES = frozenset({"watchlist", "reject", "deprecated"})


@dataclass(frozen=True, slots=True)
class FactorRegistryEntry:
    """One registered factor or feature-column family member."""

    factor_id: str
    display_name: str
    family: str
    status: str
    expected_direction: str
    feature_columns: tuple[str, ...]
    required_inputs: tuple[str, ...]
    frequency: str
    description: str
    hypothesis: str
    implementation: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    research_memory: dict[str, Any] = field(default_factory=dict)
    a_share_constraints: dict[str, bool] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    lookback_bars: int | None = None
    label_lag_bars: int | None = None
    point_in_time_safe: bool = False
    live_available: bool = False
    owner: str = "research"
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorRegistryEntry":
        """Build an entry from a JSON-compatible dictionary."""

        return cls(
            factor_id=str(payload.get("factor_id", "")),
            display_name=str(payload.get("display_name", "")),
            family=str(payload.get("family", "")),
            status=str(payload.get("status", "")),
            expected_direction=str(payload.get("expected_direction", "")),
            feature_columns=tuple(str(value) for value in payload.get("feature_columns", ())),
            required_inputs=tuple(str(value) for value in payload.get("required_inputs", ())),
            frequency=str(payload.get("frequency", "")),
            description=str(payload.get("description", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            implementation=dict(payload.get("implementation") or {}),
            evaluation=dict(payload.get("evaluation") or {}),
            research_memory=dict(payload.get("research_memory") or {}),
            a_share_constraints={
                str(key): bool(value)
                for key, value in dict(payload.get("a_share_constraints") or {}).items()
            },
            tags=tuple(str(value) for value in payload.get("tags", ())),
            references=tuple(str(value) for value in payload.get("references", ())),
            lookback_bars=_optional_int(payload.get("lookback_bars")),
            label_lag_bars=_optional_int(payload.get("label_lag_bars")),
            point_in_time_safe=bool(payload.get("point_in_time_safe", False)),
            live_available=bool(payload.get("live_available", False)),
            owner=str(payload.get("owner", "research")),
            notes=str(payload.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible entry payload."""

        return {
            "factor_id": self.factor_id,
            "display_name": self.display_name,
            "family": self.family,
            "status": self.status,
            "expected_direction": self.expected_direction,
            "feature_columns": list(self.feature_columns),
            "required_inputs": list(self.required_inputs),
            "frequency": self.frequency,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "implementation": self.implementation,
            "evaluation": self.evaluation,
            "research_memory": self.research_memory,
            "a_share_constraints": self.a_share_constraints,
            "tags": list(self.tags),
            "references": list(self.references),
            "lookback_bars": self.lookback_bars,
            "label_lag_bars": self.label_lag_bars,
            "point_in_time_safe": self.point_in_time_safe,
            "live_available": self.live_available,
            "owner": self.owner,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class FactorRegistry:
    """Structured factor registry loaded from a versioned JSON file."""

    registry_name: str
    version: int
    entries: tuple[FactorRegistryEntry, ...]
    owner: str = "quant_research"
    scope: str = ""
    updated_at: str = ""
    references: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorRegistry":
        """Build a registry from a JSON-compatible dictionary."""

        return cls(
            registry_name=str(payload.get("registry_name", "")),
            version=int(payload.get("version") or 0),
            entries=tuple(
                FactorRegistryEntry.from_dict(entry)
                for entry in payload.get("entries", ())
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
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def get(self, factor_id: str) -> FactorRegistryEntry:
        """Return a factor entry by identifier."""

        for entry in self.entries:
            if entry.factor_id == factor_id:
                return entry
        raise KeyError(factor_id)


@dataclass(frozen=True, slots=True)
class FactorResearchMemoryMatch:
    """One historical factor similar to a proposed research idea."""

    factor_id: str
    status: str
    family: str
    feature_columns: tuple[str, ...]
    decision_reason: str | None
    similarity_score: float
    matched_fields: tuple[str, ...]
    negative_findings: str
    retry_conditions: str
    evidence_artifacts: tuple[str, ...]
    blocking: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible match payload."""

        return {
            "factor_id": self.factor_id,
            "status": self.status,
            "family": self.family,
            "feature_columns": list(self.feature_columns),
            "decision_reason": self.decision_reason,
            "similarity_score": self.similarity_score,
            "matched_fields": list(self.matched_fields),
            "negative_findings": self.negative_findings,
            "retry_conditions": self.retry_conditions,
            "evidence_artifacts": list(self.evidence_artifacts),
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class FactorRegistryIssue:
    """One registry validation issue."""

    severity: str
    code: str
    message: str
    factor_id: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible issue payload."""

        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "factor_id": self.factor_id,
            "field": self.field,
        }


@dataclass(frozen=True, slots=True)
class FactorRegistryValidationReport:
    """Machine-readable validation report for a factor registry."""

    generated_at: str
    status: str
    summary: dict[str, Any]
    issues: tuple[FactorRegistryIssue, ...]
    entries: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report payload."""

        return {
            "generated_at": self.generated_at,
            "status": self.status,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
            "entries": list(self.entries),
        }


def load_factor_registry(path: str | Path) -> FactorRegistry:
    """Read a factor registry JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("factor registry payload must be a JSON object")
    return FactorRegistry.from_dict(payload)


def find_factor_research_memory_matches(
    registry: FactorRegistry,
    *,
    factor_id: str,
    family: str,
    required_inputs: tuple[str, ...] = (),
    lookback_bars: int | None = None,
    keywords: tuple[str, ...] = (),
    min_score: float = 0.35,
    statuses: tuple[str, ...] = ("watchlist", "reject", "deprecated"),
) -> tuple[FactorResearchMemoryMatch, ...]:
    """Find historical watchlist/rejected factors similar to a proposed idea."""

    if not factor_id:
        raise ValueError("factor_id must be non-empty")
    if min_score < 0:
        raise ValueError("min_score must be non-negative")
    requested_inputs = frozenset(required_inputs)
    requested_keywords = _normalize_keywords((factor_id, family, *keywords))
    matches: list[FactorResearchMemoryMatch] = []
    for entry in registry.entries:
        if entry.factor_id == factor_id:
            continue
        if entry.status not in statuses:
            continue
        matched_fields: list[str] = []
        score = 0.0
        if family and entry.family == family:
            score += 0.30
            matched_fields.append("family")
        input_overlap = _jaccard(requested_inputs, frozenset(entry.required_inputs))
        if input_overlap > 0:
            score += 0.25 * input_overlap
            matched_fields.append("required_inputs")
        if lookback_bars is not None and entry.lookback_bars is not None:
            lookback_similarity = 1.0 - min(
                abs(float(lookback_bars) - float(entry.lookback_bars))
                / max(float(lookback_bars), float(entry.lookback_bars), 1.0),
                1.0,
            )
            if lookback_similarity >= 0.50:
                score += 0.15 * lookback_similarity
                matched_fields.append("lookback_bars")
        entry_keywords = _normalize_keywords(
            (
                entry.factor_id,
                entry.display_name,
                *entry.feature_columns,
                entry.description,
                entry.hypothesis,
                *entry.tags,
                *entry.research_memory.get("similar_to", ()),
                entry.notes,
            )
        )
        keyword_overlap = _jaccard(requested_keywords, entry_keywords)
        if keyword_overlap > 0:
            score += 0.20 * keyword_overlap
            matched_fields.append("keywords")
        similar_to = set(str(value) for value in entry.research_memory.get("similar_to", ()))
        if factor_id in similar_to:
            score += 0.30
            matched_fields.append("similar_to")
        if score < min_score:
            continue
        matches.append(
            FactorResearchMemoryMatch(
                factor_id=entry.factor_id,
                status=entry.status,
                family=entry.family,
                feature_columns=entry.feature_columns,
                decision_reason=entry.research_memory.get("decision_reason"),
                similarity_score=round(score, 6),
                matched_fields=tuple(matched_fields),
                negative_findings=str(entry.research_memory.get("negative_findings", "")),
                retry_conditions=str(entry.research_memory.get("retry_conditions", "")),
                evidence_artifacts=tuple(
                    str(value)
                    for value in entry.research_memory.get("evidence_artifacts", ())
                ),
                blocking=entry.status in {"reject", "deprecated"},
            )
        )
    return tuple(
        sorted(matches, key=lambda match: (-match.similarity_score, match.factor_id))
    )


def validate_factor_registry(registry: FactorRegistry) -> FactorRegistryValidationReport:
    """Validate a factor registry for governance completeness."""

    issues: list[FactorRegistryIssue] = []
    if not registry.registry_name:
        _issue(issues, "error", "missing_registry_name", "registry_name is required")
    if registry.version <= 0:
        _issue(issues, "error", "invalid_version", "version must be positive")
    if not registry.entries:
        _issue(issues, "error", "empty_registry", "at least one factor entry is required")

    _validate_unique_ids(registry.entries, issues)
    _validate_unique_feature_columns(registry.entries, issues)
    for entry in registry.entries:
        _validate_entry(entry, issues)

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    status = "fail" if error_count else "warn" if warning_count else "pass"
    return FactorRegistryValidationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        summary={
            "registry_name": registry.registry_name,
            "version": registry.version,
            "entry_count": len(registry.entries),
            "status_counts": _count_by(registry.entries, "status"),
            "family_counts": _count_by(registry.entries, "family"),
            "error_count": error_count,
            "warning_count": warning_count,
        },
        issues=tuple(issues),
        entries=tuple(_entry_summary(entry, issues) for entry in registry.entries),
    )


def write_factor_registry_report(
    report: FactorRegistryValidationReport,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON and Markdown registry validation reports."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "factor_registry_validation.json"
    markdown_path = output / "factor_registry_validation.md"
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_factor_registry_markdown(report),
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path)}


def render_factor_registry_markdown(report: FactorRegistryValidationReport) -> str:
    """Render a human-readable registry validation report."""

    summary = report.summary
    lines = [
        "# Factor Registry Validation",
        "",
        f"- Status: `{report.status}`",
        f"- Registry: `{summary.get('registry_name')}`",
        f"- Version: `{summary.get('version')}`",
        f"- Entries: `{summary.get('entry_count')}`",
        f"- Errors: `{summary.get('error_count')}`",
        f"- Warnings: `{summary.get('warning_count')}`",
        "",
        "## Entries",
        "",
        "| factor_id | status | family | direction | features | issues |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report.entries:
        lines.append(
            "| {factor_id} | {status} | {family} | {direction} | {features} | {issues} |".format(
                factor_id=entry["factor_id"],
                status=entry["status"],
                family=entry["family"],
                direction=entry["expected_direction"],
                features=", ".join(entry["feature_columns"]),
                issues=", ".join(entry["issues"]) or "-",
            )
        )
    lines.extend(["", "## Issues", ""])
    if not report.issues:
        lines.append("No registry issues.")
    else:
        lines.extend(
            [
                "| severity | factor_id | field | code | message |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for issue in report.issues:
            lines.append(
                "| {severity} | {factor_id} | {field} | {code} | {message} |".format(
                    severity=issue.severity,
                    factor_id=issue.factor_id or "-",
                    field=issue.field or "-",
                    code=issue.code,
                    message=issue.message.replace("|", "\\|"),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _validate_unique_ids(
    entries: tuple[FactorRegistryEntry, ...],
    issues: list[FactorRegistryIssue],
) -> None:
    seen: dict[str, str] = {}
    for entry in entries:
        if not entry.factor_id:
            continue
        if entry.factor_id in seen:
            _issue(
                issues,
                "error",
                "duplicate_factor_id",
                f"factor_id duplicates {seen[entry.factor_id]}",
                factor_id=entry.factor_id,
                field="factor_id",
            )
        else:
            seen[entry.factor_id] = entry.factor_id


def _validate_unique_feature_columns(
    entries: tuple[FactorRegistryEntry, ...],
    issues: list[FactorRegistryIssue],
) -> None:
    seen: dict[str, str] = {}
    for entry in entries:
        for feature in entry.feature_columns:
            if feature in seen:
                _issue(
                    issues,
                    "error",
                    "duplicate_feature_column",
                    f"feature column is already registered by {seen[feature]}",
                    factor_id=entry.factor_id,
                    field="feature_columns",
                )
            else:
                seen[feature] = entry.factor_id


def _validate_entry(
    entry: FactorRegistryEntry,
    issues: list[FactorRegistryIssue],
) -> None:
    _validate_required_strings(entry, issues)
    if entry.family not in FACTOR_FAMILIES:
        _issue(
            issues,
            "error",
            "unknown_family",
            f"family must be one of {sorted(FACTOR_FAMILIES)}",
            factor_id=entry.factor_id,
            field="family",
        )
    if entry.status not in FACTOR_STATUSES:
        _issue(
            issues,
            "error",
            "unknown_status",
            f"status must be one of {sorted(FACTOR_STATUSES)}",
            factor_id=entry.factor_id,
            field="status",
        )
    if entry.expected_direction not in EXPECTED_DIRECTIONS:
        _issue(
            issues,
            "error",
            "unknown_expected_direction",
            f"expected_direction must be one of {sorted(EXPECTED_DIRECTIONS)}",
            factor_id=entry.factor_id,
            field="expected_direction",
        )
    if not entry.feature_columns:
        _issue(
            issues,
            "error",
            "missing_feature_columns",
            "at least one feature column is required",
            factor_id=entry.factor_id,
            field="feature_columns",
        )
    if not entry.required_inputs:
        _issue(
            issues,
            "error",
            "missing_required_inputs",
            "at least one raw input field is required",
            factor_id=entry.factor_id,
            field="required_inputs",
        )
    if entry.lookback_bars is not None and entry.lookback_bars <= 0:
        _issue(
            issues,
            "error",
            "invalid_lookback_bars",
            "lookback_bars must be positive when provided",
            factor_id=entry.factor_id,
            field="lookback_bars",
        )
    if entry.label_lag_bars is not None and entry.label_lag_bars < 0:
        _issue(
            issues,
            "error",
            "invalid_label_lag_bars",
            "label_lag_bars must be non-negative when provided",
            factor_id=entry.factor_id,
            field="label_lag_bars",
        )
    if entry.status in ACTIVE_STATUSES:
        _validate_active_entry(entry, issues)
    if entry.status in MEMORY_REQUIRED_STATUSES:
        _validate_research_memory(entry, issues)


def _validate_required_strings(
    entry: FactorRegistryEntry,
    issues: list[FactorRegistryIssue],
) -> None:
    for field_name in (
        "factor_id",
        "display_name",
        "family",
        "status",
        "expected_direction",
        "frequency",
        "description",
        "hypothesis",
    ):
        if not str(getattr(entry, field_name)).strip():
            _issue(
                issues,
                "error",
                "missing_required_field",
                f"{field_name} is required",
                factor_id=entry.factor_id or None,
                field=field_name,
            )


def _validate_active_entry(
    entry: FactorRegistryEntry,
    issues: list[FactorRegistryIssue],
) -> None:
    if not entry.point_in_time_safe:
        _issue(
            issues,
            "error",
            "point_in_time_not_confirmed",
            "active factors must explicitly confirm point-in-time safety",
            factor_id=entry.factor_id,
            field="point_in_time_safe",
        )
    if not entry.live_available:
        _issue(
            issues,
            "error",
            "live_availability_not_confirmed",
            "active factors must be computable from live-available data",
            factor_id=entry.factor_id,
            field="live_available",
        )
    for flag in REQUIRED_A_SHARE_FLAGS:
        if entry.a_share_constraints.get(flag) is not True:
            _issue(
                issues,
                "error",
                "missing_a_share_constraint",
                f"a_share_constraints.{flag} must be true for active factors",
                factor_id=entry.factor_id,
                field=f"a_share_constraints.{flag}",
            )
    if not entry.implementation.get("module"):
        _issue(
            issues,
            "error",
            "missing_implementation_module",
            "active factors must declare the implementation module",
            factor_id=entry.factor_id,
            field="implementation.module",
        )
    if not (
        entry.implementation.get("builder")
        or entry.implementation.get("class")
        or entry.implementation.get("function")
    ):
        _issue(
            issues,
            "error",
            "missing_implementation_callable",
            "active factors must declare a builder, function, or class",
            factor_id=entry.factor_id,
            field="implementation",
        )
    if not entry.evaluation.get("admission_status"):
        _issue(
            issues,
            "warning",
            "missing_admission_status",
            "active factors should record latest admission status",
            factor_id=entry.factor_id,
            field="evaluation.admission_status",
        )
    if not entry.references:
        _issue(
            issues,
            "warning",
            "missing_references",
            "active factors should reference research notes or validation artifacts",
            factor_id=entry.factor_id,
            field="references",
        )


def _validate_research_memory(
    entry: FactorRegistryEntry,
    issues: list[FactorRegistryIssue],
) -> None:
    memory = entry.research_memory
    if not memory:
        _issue(
            issues,
            "error",
            "missing_research_memory",
            "watchlist, reject, and deprecated factors must record research_memory",
            factor_id=entry.factor_id,
            field="research_memory",
        )
        return
    decision_reason = str(memory.get("decision_reason", "")).strip()
    if decision_reason not in DECISION_REASONS:
        _issue(
            issues,
            "error",
            "invalid_decision_reason",
            f"research_memory.decision_reason must be one of {sorted(DECISION_REASONS)}",
            factor_id=entry.factor_id,
            field="research_memory.decision_reason",
        )
    negative_findings = str(memory.get("negative_findings", "")).strip()
    if not negative_findings:
        _issue(
            issues,
            "error",
            "missing_negative_findings",
            "research_memory.negative_findings must summarize the failed or limited result",
            factor_id=entry.factor_id,
            field="research_memory.negative_findings",
        )
    retry_conditions = str(memory.get("retry_conditions", "")).strip()
    if not retry_conditions:
        _issue(
            issues,
            "error",
            "missing_retry_conditions",
            "research_memory.retry_conditions must state when the idea may be retried",
            factor_id=entry.factor_id,
            field="research_memory.retry_conditions",
        )
    evidence = memory.get("evidence_artifacts")
    if not isinstance(evidence, list) or not evidence or not all(
        str(item).strip() for item in evidence
    ):
        _issue(
            issues,
            "error",
            "missing_evidence_artifacts",
            "research_memory.evidence_artifacts must list validation artifacts",
            factor_id=entry.factor_id,
            field="research_memory.evidence_artifacts",
        )
    similar_to = memory.get("similar_to", [])
    if similar_to is None:
        similar_to = []
    if not isinstance(similar_to, list) or not all(
        isinstance(item, str) and item.strip() for item in similar_to
    ):
        _issue(
            issues,
            "error",
            "invalid_similar_to",
            "research_memory.similar_to must be a list of factor ids",
            factor_id=entry.factor_id,
            field="research_memory.similar_to",
        )


def _entry_summary(
    entry: FactorRegistryEntry,
    issues: list[FactorRegistryIssue],
) -> dict[str, Any]:
    return {
        "factor_id": entry.factor_id,
        "display_name": entry.display_name,
        "status": entry.status,
        "family": entry.family,
        "expected_direction": entry.expected_direction,
        "feature_columns": list(entry.feature_columns),
        "required_inputs": list(entry.required_inputs),
        "point_in_time_safe": entry.point_in_time_safe,
        "live_available": entry.live_available,
        "decision_reason": entry.research_memory.get("decision_reason"),
        "issues": [
            issue.code
            for issue in issues
            if issue.factor_id == entry.factor_id
        ],
    }


def _count_by(entries: tuple[FactorRegistryEntry, ...], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = str(getattr(entry, field_name))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _normalize_keywords(values: tuple[str, ...]) -> frozenset[str]:
    tokens: set[str] = set()
    for value in values:
        for token in re.split(r"[^A-Za-z0-9]+", value.lower()):
            if len(token) >= 3:
                tokens.add(token)
    return frozenset(tokens)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _issue(
    issues: list[FactorRegistryIssue],
    severity: str,
    code: str,
    message: str,
    *,
    factor_id: str | None = None,
    field: str | None = None,
) -> None:
    issues.append(
        FactorRegistryIssue(
            severity=severity,
            code=code,
            message=message,
            factor_id=factor_id,
            field=field,
        )
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
