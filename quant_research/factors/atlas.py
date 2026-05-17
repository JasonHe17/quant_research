"""Factor failure atlas builders.

The atlas is a governance diagnostic: it joins registry metadata with available
admission and portfolio artifacts, then summarizes where factor research is
getting stuck.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

from quant_research.factors.registry import FactorRegistry


def build_factor_failure_atlas(
    registry: FactorRegistry,
    *,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Build a factor-level failure atlas from registry and local artifacts."""

    base_path = Path(base_dir)
    records = [
        _factor_record(entry, base_dir=base_path)
        for entry in registry.entries
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "registry_name": registry.registry_name,
            "registry_version": registry.version,
            "base_dir": str(base_path),
        },
        "summary": _summary(records),
        "family_status_counts": _nested_counts(records, "family", "status"),
        "decision_reason_counts": _counts_present(records, "decision_reason"),
        "failed_check_counts": _failed_check_counts(records),
        "portfolio_status_counts": _counts(records, "portfolio_validation_status"),
        "records": records,
        "recommendations": _recommendations(records),
    }


def write_factor_failure_atlas_outputs(
    atlas: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, CSV, and Markdown atlas artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "factor_failure_atlas.json"
    csv_path = output / "factor_failure_atlas.csv"
    modes_path = output / "factor_failure_modes.csv"
    markdown_path = output / "factor_failure_atlas.md"

    json_path.write_text(
        json.dumps(atlas, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_records_csv(atlas["records"], csv_path)
    _write_modes_csv(atlas, modes_path)
    markdown_path.write_text(render_factor_failure_atlas_markdown(atlas), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "modes_csv": str(modes_path),
        "markdown": str(markdown_path),
    }


def render_factor_failure_atlas_markdown(atlas: dict[str, Any]) -> str:
    """Render the atlas as a concise Markdown report."""

    summary = atlas["summary"]
    lines = [
        "# Factor Failure Atlas",
        "",
        f"- Registry: `{atlas['source']['registry_name']}` v`{atlas['source']['registry_version']}`",
        f"- Factors: `{summary['factor_count']}`",
        f"- Candidates: `{summary['status_counts'].get('candidate', 0)}`",
        f"- Watchlist: `{summary['status_counts'].get('watchlist', 0)}`",
        f"- Rejects: `{summary['status_counts'].get('reject', 0)}`",
        f"- Admission evidence coverage: `{summary['admission_evidence_coverage']:.1%}`",
        f"- Portfolio evidence coverage: `{summary['portfolio_evidence_coverage']:.1%}`",
        "",
        "## Failure Modes",
        "",
    ]
    failed_counts = atlas["failed_check_counts"]
    decision_counts = atlas["decision_reason_counts"]
    if not failed_counts and not decision_counts:
        lines.append("No failure evidence found.")
    else:
        lines.extend(
            [
                "| mode | count |",
                "| --- | ---: |",
            ]
        )
        combined = Counter(decision_counts)
        combined.update({f"failed_check:{key}": value for key, value in failed_counts.items()})
        for key, value in combined.most_common():
            lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Family Status", ""])
    lines.extend(["| family | candidate | watchlist | reject | other |", "| --- | ---: | ---: | ---: | ---: |"])
    for family, counts in sorted(atlas["family_status_counts"].items()):
        known = sum(counts.get(status, 0) for status in ("candidate", "watchlist", "reject"))
        other = sum(counts.values()) - known
        lines.append(
            "| {family} | {candidate} | {watchlist} | {reject} | {other} |".format(
                family=family,
                candidate=counts.get("candidate", 0),
                watchlist=counts.get("watchlist", 0),
                reject=counts.get("reject", 0),
                other=other,
            )
        )

    lines.extend(["", "## Highest-Risk Records", ""])
    risky = sorted(
        atlas["records"],
        key=lambda row: (
            row["status"] not in {"reject", "watchlist"},
            -(len(row["admission_failed_checks"]) + len(row["failure_modes"])),
            row["factor_id"],
        ),
    )[:12]
    lines.extend(
        [
            "| factor_id | status | family | decision | failed_checks | retry_condition |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in risky:
        lines.append(
            "| {factor_id} | {status} | {family} | {decision} | {failed} | {retry} |".format(
                factor_id=row["factor_id"],
                status=row["status"],
                family=row["family"],
                decision=row.get("decision_reason") or "-",
                failed=", ".join(row["admission_failed_checks"]) or "-",
                retry=_markdown_cell(row.get("retry_conditions") or "-"),
            )
        )

    lines.extend(["", "## Recommendations", ""])
    for recommendation in atlas["recommendations"]:
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def _factor_record(entry: Any, *, base_dir: Path) -> dict[str, Any]:
    evaluation = dict(entry.evaluation or {})
    memory = dict(entry.research_memory or {})
    admission_report_path = evaluation.get("admission_report")
    admission_row = _load_admission_row(
        admission_report_path,
        feature_columns=entry.feature_columns,
        base_dir=base_dir,
    )
    portfolio_artifacts = _portfolio_artifacts(evaluation, base_dir=base_dir)
    failure_modes = _failure_modes(
        status=entry.status,
        decision_reason=memory.get("decision_reason"),
        admission_failed_checks=tuple(admission_row.get("failed_checks", ()) if admission_row else ()),
        portfolio_status=str(evaluation.get("portfolio_validation_status") or ""),
    )
    return {
        "factor_id": entry.factor_id,
        "display_name": entry.display_name,
        "family": entry.family,
        "status": entry.status,
        "expected_direction": entry.expected_direction,
        "admission_status": evaluation.get("admission_status"),
        "admission_direction": evaluation.get("admission_direction"),
        "feature_columns": list(entry.feature_columns),
        "lookback_bars": entry.lookback_bars,
        "decision_reason": memory.get("decision_reason"),
        "failure_modes": failure_modes,
        "retry_conditions": memory.get("retry_conditions", ""),
        "negative_findings": memory.get("negative_findings", ""),
        "similar_to": list(memory.get("similar_to", ())),
        "admission_report": str(admission_report_path or ""),
        "admission_evidence_found": admission_row is not None,
        "admission_failed_checks": list(admission_row.get("failed_checks", ()) if admission_row else ()),
        "rank_ic": admission_row.get("spearman_rank_ic_mean") if admission_row else None,
        "rank_ic_t_stat": admission_row.get("spearman_rank_ic_t_stat") if admission_row else None,
        "hit_rate": admission_row.get("directional_ic_hit_rate") if admission_row else None,
        "stable_year_count": admission_row.get("stable_year_count") if admission_row else None,
        "cost_adjusted_spread": (
            admission_row.get("cost_adjusted_top_minus_bottom_label")
            if admission_row
            else None
        ),
        "top_n_turnover": admission_row.get("top_n_turnover") if admission_row else None,
        "yearly_rank_ic": admission_row.get("yearly_spearman_rank_ic_mean", {}) if admission_row else {},
        "portfolio_validation_status": evaluation.get("portfolio_validation_status"),
        "portfolio_evidence_found": bool(portfolio_artifacts),
        "portfolio_artifacts": portfolio_artifacts,
    }


def _load_admission_row(
    path: object,
    *,
    feature_columns: tuple[str, ...],
    base_dir: Path,
) -> dict[str, Any] | None:
    if not path:
        return None
    payload = _read_json(base_dir / str(path))
    if not isinstance(payload, dict):
        return None
    wanted = set(feature_columns)
    for row in payload.get("factors", ()):
        if isinstance(row, dict) and row.get("feature") in wanted:
            return row
    return None


def _portfolio_artifacts(evaluation: dict[str, Any], *, base_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for key, value in sorted(evaluation.items()):
        if "portfolio_validation" not in key and "risk_gate_validation" not in key:
            continue
        if not isinstance(value, str) or not value.endswith(".json"):
            continue
        payload = _read_json(base_dir / value)
        if not isinstance(payload, dict):
            continue
        artifacts.append(
            {
                "key": key,
                "path": value,
                "status": payload.get("status"),
                "overall_status": _portfolio_overall_status(payload),
                "primary_result": _portfolio_primary_result(payload),
            }
        )
    return artifacts


def _portfolio_overall_status(payload: dict[str, Any]) -> str | None:
    validation = payload.get("validation")
    if isinstance(validation, dict) and validation.get("overall_status") is not None:
        return str(validation.get("overall_status"))
    return str(payload.get("status")) if payload.get("status") is not None else None


def _portfolio_primary_result(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("results")
    if not isinstance(rows, list):
        rows = payload.get("backtest_summary")
    if not isinstance(rows, list):
        return {}
    preferred = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("scenario") in {None, "full_base"}
        and row.get("method") in {None, "decorrelated"}
    ]
    row = preferred[0] if preferred else rows[0] if rows else {}
    if not isinstance(row, dict):
        return {}
    return {
        key: row.get(key)
        for key in (
            "scenario",
            "method",
            "policy",
            "total_return",
            "max_drawdown",
            "gross_turnover",
            "trade_count",
            "total_transaction_cost",
        )
        if key in row
    }


def _failure_modes(
    *,
    status: str,
    decision_reason: object,
    admission_failed_checks: tuple[str, ...],
    portfolio_status: str,
) -> list[str]:
    modes: list[str] = []
    if decision_reason:
        modes.append(str(decision_reason))
    for check in admission_failed_checks:
        modes.append(f"admission:{check}")
    if "negative" in portfolio_status or "failed" in portfolio_status:
        modes.append("portfolio_negative")
    if "unstable" in portfolio_status:
        modes.append("portfolio_unstable")
    if "risk_gate" in portfolio_status:
        modes.append("risk_gate_evidence")
    if status == "candidate" and not modes:
        modes.append("candidate_or_promotable")
    return sorted(set(modes))


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    factor_count = len(records)
    admission_count = sum(1 for row in records if row["admission_evidence_found"])
    portfolio_count = sum(1 for row in records if row["portfolio_evidence_found"])
    return {
        "factor_count": factor_count,
        "status_counts": _counts(records, "status"),
        "family_counts": _counts(records, "family"),
        "admission_evidence_count": admission_count,
        "admission_evidence_coverage": admission_count / factor_count if factor_count else 0.0,
        "portfolio_evidence_count": portfolio_count,
        "portfolio_evidence_coverage": portfolio_count / factor_count if factor_count else 0.0,
    }


def _recommendations(records: list[dict[str, Any]]) -> list[str]:
    decision_counts = Counter(
        row["decision_reason"]
        for row in records
        if row.get("decision_reason")
    )
    failed_counts = Counter(
        check
        for row in records
        for check in row.get("admission_failed_checks", ())
    )
    recommendations = [
        "Stop proposing raw variants whose nearest rejected/watchlist parent has the same inputs, horizon, and transform family unless the registry retry condition is explicitly satisfied.",
        "Route risk-like factors with strong IC but failed cost-adjusted spread to exposure-gate validation, not standalone alpha validation.",
        "Require annual top-bucket health before portfolio validation when the factor is intended for long-only stock selection.",
    ]
    if decision_counts.get("weak_hit_rate", 0) + failed_counts.get("directional_ic_hit_rate", 0) >= 3:
        recommendations.append(
            "Prioritize regime-conditioned or sector-relative designs because weak hit-rate failures suggest the all-market direction is unstable."
        )
    if decision_counts.get("cost_fragile", 0) + failed_counts.get("cost_adjusted_spread", 0) >= 3:
        recommendations.append(
            "Add turnover and execution diagnostics before implementation for any new high-frequency price-pressure factor."
        )
    if decision_counts.get("portfolio_negative", 0) >= 1:
        recommendations.append(
            "Separate factors that only avoid weak bottom buckets from factors that produce positive long-only top buckets."
        )
    return recommendations


def _counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key) or "missing") for row in records)
    return dict(sorted(counter.items()))


def _counts_present(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row[key]) for row in records if row.get(key))
    return dict(sorted(counter.items()))


def _nested_counts(
    records: list[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> dict[str, dict[str, int]]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        buckets[str(row.get(left_key) or "missing")][str(row.get(right_key) or "missing")] += 1
    return {key: dict(sorted(counter.items())) for key, counter in sorted(buckets.items())}


def _failed_check_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in records:
        counter.update(str(check) for check in row.get("admission_failed_checks", ()))
    return dict(counter.most_common())


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "factor_id",
        "status",
        "family",
        "decision_reason",
        "admission_status",
        "admission_direction",
        "rank_ic",
        "rank_ic_t_stat",
        "hit_rate",
        "stable_year_count",
        "cost_adjusted_spread",
        "top_n_turnover",
        "admission_failed_checks",
        "portfolio_validation_status",
        "failure_modes",
        "retry_conditions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    column: _csv_value(row.get(column))
                    for column in columns
                }
            )


def _write_modes_csv(atlas: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for key, value in atlas["decision_reason_counts"].items():
        rows.append({"source": "decision_reason", "mode": key, "count": value})
    for key, value in atlas["failed_check_counts"].items():
        rows.append({"source": "failed_check", "mode": key, "count": value})
    rows.sort(key=lambda row: (-int(row["count"]), str(row["source"]), str(row["mode"])))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "mode", "count"))
        writer.writeheader()
        writer.writerows(rows)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
