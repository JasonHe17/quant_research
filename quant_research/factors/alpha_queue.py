"""Candidate alpha queue review.

This module consumes the factor opportunity map and registry metadata to decide
which standalone alpha candidates are ready for portfolio validation and which
ones still need missing artifacts or narrower admission inputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

from quant_research.factors.opportunity import build_factor_opportunity_map
from quant_research.factors.registry import FactorRegistry


def build_candidate_alpha_queue_review(
    registry: FactorRegistry,
    *,
    base_dir: str | Path = ".",
    opportunity_map: dict[str, Any] | None = None,
    opportunity_class: str = "long_alpha_candidate",
    output_root: str = "runs/candidate_factor_portfolios",
) -> dict[str, Any]:
    """Review standalone-alpha candidates and identify validation gaps."""

    base_path = Path(base_dir)
    opportunity_map = opportunity_map or build_factor_opportunity_map(
        registry,
        base_dir=base_path,
    )
    registry_lookup = {entry.factor_id: entry for entry in registry.entries}
    registry_lookup.update(
        {
            feature: entry
            for entry in registry.entries
            for feature in entry.feature_columns
        }
    )
    rows = [
        _queue_row(
            record,
            registry_lookup=registry_lookup,
            base_dir=base_path,
            output_root=output_root,
        )
        for record in opportunity_map["records"]
        if record.get("opportunity_class") == opportunity_class
    ]
    rows.sort(key=lambda row: (_queue_rank(row["queue_status"]), row["factor_id"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": opportunity_map["source"],
        "params": {
            "opportunity_class": opportunity_class,
            "output_root": output_root,
        },
        "summary": _summary(rows),
        "queue": rows,
        "recommendations": _recommendations(rows),
    }


def write_candidate_alpha_queue_review_outputs(
    review: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, CSV, and Markdown queue-review artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "candidate_alpha_queue_review.json"
    csv_path = output / "candidate_alpha_queue_review.csv"
    markdown_path = output / "candidate_alpha_queue_review.md"
    json_path.write_text(
        json.dumps(review, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(review["queue"], csv_path)
    markdown_path.write_text(render_candidate_alpha_queue_review_markdown(review), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}


def render_candidate_alpha_queue_review_markdown(review: dict[str, Any]) -> str:
    """Render a concise alpha queue review."""

    summary = review["summary"]
    lines = [
        "# Candidate Alpha Queue Review",
        "",
        f"- Registry: `{review['source']['registry_name']}` v`{review['source']['registry_version']}`",
        f"- Queue size: `{summary['queue_count']}`",
        f"- Needs validation: `{summary['status_counts'].get('needs_portfolio_validation', 0)}`",
        f"- Validated/watch: `{summary['status_counts'].get('portfolio_validated_watch', 0)}`",
        f"- Promotion review: `{summary['status_counts'].get('candidate_for_promotion_review', 0)}`",
        f"- Shared admission commands: `{summary['status_counts'].get('needs_shared_admission_filtered_validation', 0)}`",
        "",
        "## Queue",
        "",
        "| factor_id | status | selected_mean | positive_years | portfolio_status | queue_status | next_step |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in review["queue"]:
        lines.append(
            "| {factor_id} | {registry_status} | {selected} | {years} | {portfolio_status} | {queue_status} | {next_step} |".format(
                factor_id=row["factor_id"],
                registry_status=row["registry_status"],
                selected=_format_number(row.get("selected_mean_label")),
                years=row.get("selected_positive_years"),
                portfolio_status=row.get("portfolio_validation_status") or "-",
                queue_status=row["queue_status"],
                next_step=_markdown_cell(row["next_step"]),
            )
        )
    lines.extend(["", "## Recommended Commands", ""])
    for row in review["queue"]:
        command = row.get("recommended_command")
        if not command:
            continue
        lines.extend([f"### {row['factor_id']}", "", "```bash", command, "```", ""])
    lines.extend(["## Recommendations", ""])
    for recommendation in review["recommendations"]:
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def _queue_row(
    record: dict[str, Any],
    *,
    registry_lookup: dict[str, Any],
    base_dir: Path,
    output_root: str,
) -> dict[str, Any]:
    feature = record["factor_id"]
    entry = registry_lookup.get(feature)
    if entry is None:
        raise KeyError(f"no registry entry found for opportunity record: {feature}")
    evaluation = dict(entry.evaluation or {})
    portfolio_status = str(evaluation.get("portfolio_validation_status") or "")
    portfolio_artifacts = _portfolio_artifacts(evaluation, base_dir=base_dir)
    dataset_dir = _dataset_dir_from_admission(evaluation.get("admission_report"))
    admission_is_single_feature = _admission_is_single_feature(
        evaluation.get("admission_report"),
        feature_columns=entry.feature_columns,
        base_dir=base_dir,
    )
    queue_status = _queue_status(
        portfolio_status=portfolio_status,
        portfolio_artifacts=portfolio_artifacts,
        admission_is_single_feature=admission_is_single_feature,
    )
    include_features = () if admission_is_single_feature else entry.feature_columns
    return {
        "factor_id": entry.factor_id,
        "feature_columns": list(entry.feature_columns),
        "registry_status": entry.status,
        "family": entry.family,
        "selected_mean_label": record.get("selected_mean_label"),
        "selected_positive_years": record.get("selected_positive_years"),
        "cost_adjusted_spread": record.get("cost_adjusted_spread"),
        "portfolio_validation_status": portfolio_status,
        "portfolio_artifacts": portfolio_artifacts,
        "admission_report": evaluation.get("admission_report"),
        "dataset_dir": dataset_dir,
        "admission_is_single_feature": admission_is_single_feature,
        "requires_include_feature_filter": not admission_is_single_feature,
        "queue_status": queue_status,
        "next_step": _next_step(queue_status),
        "recommended_command": _recommended_command(
            queue_status,
            admission_report=evaluation.get("admission_report"),
            dataset_dir=dataset_dir,
            output_dir=f"{output_root}/{entry.factor_id}_validation",
            include_features=include_features,
        ),
    }


def _queue_status(
    *,
    portfolio_status: str,
    portfolio_artifacts: list[dict[str, Any]],
    admission_is_single_feature: bool,
) -> str:
    if portfolio_status in {"standalone_validated_drawdown_watch", "watchlist_portfolio_unstable"}:
        return "portfolio_validated_watch"
    if portfolio_status in {"risk_gate_validated", "standalone_validated"}:
        return "candidate_for_promotion_review"
    if portfolio_status and ("negative" in portfolio_status or "failed" in portfolio_status):
        return "portfolio_failed"
    if portfolio_artifacts:
        return "portfolio_evidence_needs_review"
    if not admission_is_single_feature:
        return "needs_shared_admission_filtered_validation"
    return "needs_portfolio_validation"


def _next_step(queue_status: str) -> str:
    return {
        "needs_portfolio_validation": "Run standard decorrelated portfolio validation for this single-factor admission dataset.",
        "needs_shared_admission_filtered_validation": "Run standard decorrelated portfolio validation with an explicit include-feature allowlist.",
        "portfolio_validated_watch": "Review drawdown and annual-slice issues before promotion; do not rerun blindly.",
        "candidate_for_promotion_review": "Open promotion review against current promoted policy and risk gates.",
        "portfolio_failed": "Keep out of promotion queue; use diagnostics only.",
        "portfolio_evidence_needs_review": "Inspect linked portfolio artifacts and update registry status.",
    }[queue_status]


def _recommended_command(
    queue_status: str,
    *,
    admission_report: object,
    dataset_dir: str | None,
    output_dir: str,
    include_features: tuple[str, ...] = (),
) -> str:
    if queue_status not in {
        "needs_portfolio_validation",
        "needs_shared_admission_filtered_validation",
    } or not admission_report or not dataset_dir:
        return ""
    command = [
        "conda run -n quant python examples/run_candidate_factor_portfolios.py",
        f"--dataset-dir {dataset_dir}",
        f"--admission-report {admission_report}",
        f"--output-dir {output_dir}",
        "--methods decorrelated",
        "--statuses candidate",
        "--registry-statuses candidate",
    ]
    if include_features:
        command.append("--include-features " + " ".join(include_features))
    command.extend(
        [
            "--run-backtests",
            "--backtest-policy-set single",
            "--trade-policy cost_aware_optimizer",
            "--rebalance-every-n-bars 48",
            "--policy-partial-rebalance-rate 1.0",
            "--policy-total-gross-turnover-budget 52",
            "--policy-turnover-budget-period path",
            "--optimizer-candidate-rank 150",
            "--optimizer-score-to-edge-bps 0",
            "--optimizer-min-net-edge-bps 1",
            "--optimizer-risk-penalty-multiplier 0",
            "--optimizer-weighting equal",
            "--backtest-workers 1",
            "--resume-existing",
        ]
    )
    return " ".join(command)


def _portfolio_artifacts(evaluation: dict[str, Any], *, base_dir: Path) -> list[str]:
    artifacts: list[str] = []
    for key, value in sorted(evaluation.items()):
        if "portfolio_validation" not in key and "risk_gate_validation" not in key:
            continue
        if isinstance(value, str) and value.endswith(".json") and (base_dir / value).exists():
            artifacts.append(value)
    return artifacts


def _dataset_dir_from_admission(admission_report: object) -> str | None:
    if not admission_report:
        return None
    path = Path(str(admission_report))
    if path.name != "factor_admission_report.json":
        return None
    return str(path.parent.parent / "alpha_dataset")


def _admission_is_single_feature(
    admission_report: object,
    *,
    feature_columns: tuple[str, ...],
    base_dir: Path,
) -> bool:
    if not admission_report:
        return False
    path = base_dir / str(admission_report)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    rows = payload.get("factors", ())
    if not isinstance(rows, list):
        return False
    features = {row.get("feature") for row in rows if isinstance(row, dict)}
    return features == set(feature_columns)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["queue_status"]] = counts.get(row["queue_status"], 0) + 1
    return {
        "queue_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
    }


def _recommendations(rows: list[dict[str, Any]]) -> list[str]:
    recommendations = [
        "Validate shared admission candidates only with explicit include-feature allowlists; broad reports can accidentally test a basket.",
        "Treat drawdown-watch candidates as research follow-ups, not immediate promotion candidates.",
    ]
    if any(row["queue_status"] == "needs_shared_admission_filtered_validation" for row in rows):
        recommendations.append(
            "Use the generated include-feature validation commands for shared standard datasets."
        )
    if any(row["queue_status"] == "needs_portfolio_validation" for row in rows):
        recommendations.append(
            "Run the generated validation commands before developing more standalone alpha factors."
        )
    return recommendations


def _queue_rank(status: str) -> int:
    ranks = {
        "needs_portfolio_validation": 0,
        "needs_shared_admission_filtered_validation": 1,
        "portfolio_validated_watch": 2,
        "portfolio_evidence_needs_review": 3,
        "candidate_for_promotion_review": 4,
        "portfolio_failed": 5,
    }
    return ranks.get(status, 99)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "factor_id",
        "registry_status",
        "family",
        "selected_mean_label",
        "selected_positive_years",
        "cost_adjusted_spread",
        "portfolio_validation_status",
        "admission_is_single_feature",
        "queue_status",
        "next_step",
        "recommended_command",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6g}"


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
