"""Factor opportunity-map builders.

The opportunity map extends the failure atlas with long-only top-bucket health.
It answers a narrower question than admission: did the selected side of the
factor actually make money across years, or did the signal only avoid weak
bottom buckets?
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research.factors.atlas import build_factor_failure_atlas
from quant_research.factors.registry import FactorRegistry


def build_factor_opportunity_map(
    registry: FactorRegistry,
    *,
    base_dir: str | Path = ".",
    min_positive_years: int = 2,
    min_selected_mean_label: float = 0.0,
    min_cost_adjusted_spread: float = 0.0,
) -> dict[str, Any]:
    """Build a top-bucket opportunity map for registered factors."""

    if min_positive_years <= 0:
        raise ValueError("min_positive_years must be positive")
    base_path = Path(base_dir)
    atlas = build_factor_failure_atlas(registry, base_dir=base_path)
    by_timestamp_cache: dict[Path, pd.DataFrame | None] = {}
    records = [
        _opportunity_record(
            atlas_record,
            base_dir=base_path,
            by_timestamp_cache=by_timestamp_cache,
            min_positive_years=min_positive_years,
            min_selected_mean_label=min_selected_mean_label,
            min_cost_adjusted_spread=min_cost_adjusted_spread,
        )
        for atlas_record in atlas["records"]
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": atlas["source"],
        "params": {
            "min_positive_years": min_positive_years,
            "min_selected_mean_label": min_selected_mean_label,
            "min_cost_adjusted_spread": min_cost_adjusted_spread,
        },
        "summary": _summary(records),
        "class_counts": _counts(records, "opportunity_class"),
        "family_class_counts": _family_class_counts(records),
        "records": records,
        "recommendations": _recommendations(records),
    }


def write_factor_opportunity_map_outputs(
    opportunity_map: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON, CSV, and Markdown opportunity-map artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "factor_opportunity_map.json"
    csv_path = output / "factor_opportunity_map.csv"
    markdown_path = output / "factor_opportunity_map.md"
    json_path.write_text(
        json.dumps(opportunity_map, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(opportunity_map["records"], csv_path)
    markdown_path.write_text(
        render_factor_opportunity_map_markdown(opportunity_map),
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(markdown_path)}


def render_factor_opportunity_map_markdown(opportunity_map: dict[str, Any]) -> str:
    """Render a human-readable opportunity map."""

    summary = opportunity_map["summary"]
    lines = [
        "# Factor Opportunity Map",
        "",
        f"- Registry: `{opportunity_map['source']['registry_name']}` v`{opportunity_map['source']['registry_version']}`",
        f"- Factors: `{summary['factor_count']}`",
        f"- Top-bucket evidence coverage: `{summary['top_bucket_evidence_coverage']:.1%}`",
        "",
        "## Opportunity Classes",
        "",
        "| class | count |",
        "| --- | ---: |",
    ]
    for name, count in sorted(opportunity_map["class_counts"].items()):
        lines.append(f"| `{name}` | {count} |")

    lines.extend(["", "## Research Queue", ""])
    lines.extend(
        [
            "| factor_id | class | family | selected_mean | positive_years | cost_adj_spread | action |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _sorted_records(opportunity_map["records"]):
        lines.append(
            "| {factor_id} | {klass} | {family} | {selected} | {years} | {spread} | {action} |".format(
                factor_id=row["factor_id"],
                klass=row["opportunity_class"],
                family=row["family"],
                selected=_format_number(row.get("selected_mean_label")),
                years=row.get("selected_positive_years"),
                spread=_format_number(row.get("cost_adjusted_spread")),
                action=_markdown_cell(row["recommended_action"]),
            )
        )

    lines.extend(["", "## Recommendations", ""])
    for recommendation in opportunity_map["recommendations"]:
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def _opportunity_record(
    atlas_record: dict[str, Any],
    *,
    base_dir: Path,
    by_timestamp_cache: dict[Path, pd.DataFrame | None],
    min_positive_years: int,
    min_selected_mean_label: float,
    min_cost_adjusted_spread: float,
) -> dict[str, Any]:
    direction = _effective_direction(atlas_record)
    yearly_health = _yearly_top_bucket_health(
        atlas_record,
        direction=direction,
        base_dir=base_dir,
        by_timestamp_cache=by_timestamp_cache,
    )
    selected_values = [row["selected_mean_label"] for row in yearly_health.values()]
    selected_positive_years = sum(value > 0.0 for value in selected_values)
    selected_negative_years = sum(value <= 0.0 for value in selected_values)
    selected_mean = _mean(selected_values)
    avoided_mean = _mean(row["avoided_mean_label"] for row in yearly_health.values())
    selected_minus_avoided = _mean(
        row["selected_minus_avoided_label"] for row in yearly_health.values()
    )
    klass = _classify(
        atlas_record,
        selected_mean=selected_mean,
        selected_positive_years=selected_positive_years,
        selected_minus_avoided=selected_minus_avoided,
        min_positive_years=min_positive_years,
        min_selected_mean_label=min_selected_mean_label,
        min_cost_adjusted_spread=min_cost_adjusted_spread,
    )
    return {
        "factor_id": atlas_record["factor_id"],
        "status": atlas_record["status"],
        "family": atlas_record["family"],
        "effective_direction": direction,
        "opportunity_class": klass,
        "selected_mean_label": selected_mean,
        "avoided_mean_label": avoided_mean,
        "selected_minus_avoided_label": selected_minus_avoided,
        "selected_positive_years": selected_positive_years,
        "selected_negative_years": selected_negative_years,
        "year_count": len(yearly_health),
        "yearly_top_bucket_health": yearly_health,
        "rank_ic": atlas_record.get("rank_ic"),
        "hit_rate": atlas_record.get("hit_rate"),
        "stable_year_count": atlas_record.get("stable_year_count"),
        "cost_adjusted_spread": atlas_record.get("cost_adjusted_spread"),
        "portfolio_validation_status": atlas_record.get("portfolio_validation_status"),
        "failure_modes": atlas_record.get("failure_modes", []),
        "recommended_action": _recommended_action(klass),
    }


def _yearly_top_bucket_health(
    atlas_record: dict[str, Any],
    *,
    direction: str,
    base_dir: Path,
    by_timestamp_cache: dict[Path, pd.DataFrame | None],
) -> dict[str, dict[str, float]]:
    path = _by_timestamp_path(atlas_record, base_dir=base_dir)
    if path is None:
        return {}
    if path not in by_timestamp_cache:
        by_timestamp_cache[path] = _read_by_timestamp(path)
    frame = by_timestamp_cache[path]
    if frame is None or frame.empty:
        return {}
    features = set(atlas_record.get("feature_columns") or [atlas_record["factor_id"]])
    factor_frame = frame.loc[frame["feature"].isin(features)].copy()
    if factor_frame.empty:
        return {}
    factor_frame["year"] = pd.to_datetime(
        factor_frame["timestamp"],
        errors="coerce",
        utc=True,
    ).dt.year
    factor_frame = factor_frame.loc[factor_frame["year"].notna()]
    if direction == "invert":
        selected = factor_frame["bottom_n_mean_label"]
        avoided = factor_frame["top_n_mean_label"]
        spread = -factor_frame["top_minus_bottom_label"]
    else:
        selected = factor_frame["top_n_mean_label"]
        avoided = factor_frame["bottom_n_mean_label"]
        spread = factor_frame["top_minus_bottom_label"]
    factor_frame = factor_frame.assign(
        selected_mean_label=selected,
        avoided_mean_label=avoided,
        selected_minus_avoided_label=spread,
    )
    yearly: dict[str, dict[str, float]] = {}
    for year, group in factor_frame.groupby("year", sort=True):
        yearly[str(int(year))] = {
            "selected_mean_label": _nullable_float(group["selected_mean_label"].mean()),
            "avoided_mean_label": _nullable_float(group["avoided_mean_label"].mean()),
            "selected_minus_avoided_label": _nullable_float(
                group["selected_minus_avoided_label"].mean()
            ),
            "timestamp_count": int(group["timestamp"].nunique()),
        }
    return yearly


def _classify(
    atlas_record: dict[str, Any],
    *,
    selected_mean: float | None,
    selected_positive_years: int,
    selected_minus_avoided: float | None,
    min_positive_years: int,
    min_selected_mean_label: float,
    min_cost_adjusted_spread: float,
) -> str:
    modes = set(atlas_record.get("failure_modes", ()))
    portfolio_status = str(atlas_record.get("portfolio_validation_status") or "")
    cost_adjusted_spread = atlas_record.get("cost_adjusted_spread")
    if "risk_gate_evidence" in modes or "risk_gate" in portfolio_status:
        return "risk_gate_only"
    admission_clean = (
        atlas_record.get("status") != "reject"
        and atlas_record.get("admission_status") == "candidate"
        and not atlas_record.get("admission_failed_checks")
    )
    if (
        admission_clean
        and
        selected_mean is not None
        and selected_mean > min_selected_mean_label
        and selected_positive_years >= min_positive_years
        and cost_adjusted_spread is not None
        and cost_adjusted_spread > min_cost_adjusted_spread
    ):
        return "long_alpha_candidate"
    if (
        selected_minus_avoided is not None
        and selected_minus_avoided > 0.0
        and (
            selected_mean is None
            or selected_mean <= min_selected_mean_label
            or selected_positive_years < min_positive_years
        )
    ):
        return "bottom_avoidance_only"
    if (
        atlas_record.get("family") in {"risk", "volatility"}
        and atlas_record.get("stable_year_count", 0) >= min_positive_years
        and abs(float(atlas_record.get("rank_ic") or 0.0)) >= 0.01
    ):
        return "risk_gate_only"
    return "dead_zone"


def _recommended_action(klass: str) -> str:
    return {
        "long_alpha_candidate": "Run or refresh portfolio validation; require annual top-bucket health before promotion.",
        "bottom_avoidance_only": "Do not promote as standalone long alpha; test only as exclusion, score cap, or risk overlay.",
        "risk_gate_only": "Route to exposure-gate validation against the promoted policy rather than standalone stock selection.",
        "dead_zone": "Block near-duplicate follow-ups unless a materially new data source or regime split is added.",
    }[klass]


def _effective_direction(atlas_record: dict[str, Any]) -> str:
    direction = atlas_record.get("admission_direction")
    if direction in {"long", "invert"}:
        return str(direction)
    rank_ic = atlas_record.get("rank_ic")
    if rank_ic is not None and float(rank_ic) < 0.0:
        return "invert"
    return "long"


def _by_timestamp_path(atlas_record: dict[str, Any], *, base_dir: Path) -> Path | None:
    admission_report = atlas_record.get("admission_report")
    if not admission_report:
        return None
    report_path = base_dir / str(admission_report)
    batch_dir = report_path.parent.parent
    summary_path = batch_dir / "factor_evaluation" / "summary.json"
    payload = _read_json(summary_path)
    if not isinstance(payload, dict):
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts.get("by_timestamp"):
        return None
    return base_dir / str(artifacts["by_timestamp"])


def _read_by_timestamp(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    columns = [
        "feature",
        "timestamp",
        "top_n_mean_label",
        "bottom_n_mean_label",
        "top_minus_bottom_label",
    ]
    try:
        return pd.read_csv(path, usecols=columns)
    except (ValueError, OSError):
        return None


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    factor_count = len(records)
    evidence_count = sum(1 for row in records if row["year_count"] > 0)
    return {
        "factor_count": factor_count,
        "top_bucket_evidence_count": evidence_count,
        "top_bucket_evidence_coverage": evidence_count / factor_count if factor_count else 0.0,
        "class_counts": _counts(records, "opportunity_class"),
    }


def _recommendations(records: list[dict[str, Any]]) -> list[str]:
    counts = Counter(row["opportunity_class"] for row in records)
    recommendations = [
        "Use top-bucket health as a pre-portfolio gate for every standalone long-alpha proposal.",
        "Block bottom-avoidance-only signals from standalone promotion even when top-minus-bottom spread is positive.",
    ]
    if counts.get("risk_gate_only", 0):
        recommendations.append(
            "Maintain a separate risk-gate queue for volatility and downside-pressure signals with strong IC but weak standalone economics."
        )
    if counts.get("dead_zone", 0):
        recommendations.append(
            "Require materially new information for dead-zone families; changing only lookback windows is not enough."
        )
    return recommendations


def _family_class_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in records:
        family = row["family"]
        counts.setdefault(family, Counter())[row["opportunity_class"]] += 1
    return {family: dict(counter) for family, counter in sorted(counts.items())}


def _counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key) or "missing") for row in records))


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {
        "long_alpha_candidate": 0,
        "risk_gate_only": 1,
        "bottom_avoidance_only": 2,
        "dead_zone": 3,
    }
    return sorted(
        records,
        key=lambda row: (
            rank.get(row["opportunity_class"], 99),
            -(row.get("selected_positive_years") or 0),
            -(row.get("selected_mean_label") or -999.0),
            row["factor_id"],
        ),
    )


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "factor_id",
        "status",
        "family",
        "effective_direction",
        "opportunity_class",
        "selected_mean_label",
        "avoided_mean_label",
        "selected_minus_avoided_label",
        "selected_positive_years",
        "selected_negative_years",
        "year_count",
        "rank_ic",
        "hit_rate",
        "stable_year_count",
        "cost_adjusted_spread",
        "portfolio_validation_status",
        "recommended_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({column: row.get(column) for column in columns})


def _mean(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None and pd.notna(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6g}"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
