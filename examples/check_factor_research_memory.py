"""Check whether a proposed factor repeats known weak or failed ideas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import (
    find_factor_research_memory_matches,
    load_factor_registry,
)


def main() -> None:
    args = _parse_args()
    registry = load_factor_registry(args.registry)
    matches = find_factor_research_memory_matches(
        registry,
        factor_id=args.factor_id,
        family=args.family,
        required_inputs=tuple(args.required_inputs),
        lookback_bars=args.lookback_bars,
        keywords=tuple(args.keywords),
        min_score=args.min_score,
        statuses=tuple(args.statuses),
    )
    blocking_count = sum(1 for match in matches if match.blocking)
    warning_count = len(matches) - blocking_count
    status = "blocked" if blocking_count else "warn" if warning_count else "clear"
    payload: dict[str, Any] = {
        "params": {
            "registry": str(args.registry),
            "factor_id": args.factor_id,
            "family": args.family,
            "required_inputs": list(args.required_inputs),
            "lookback_bars": args.lookback_bars,
            "keywords": list(args.keywords),
            "min_score": args.min_score,
            "statuses": list(args.statuses),
        },
        "summary": {
            "status": status,
            "match_count": len(matches),
            "blocking_count": blocking_count,
            "warning_count": warning_count,
        },
        "matches": [match.to_dict() for match in matches],
    }
    artifacts = _write_report(payload, output_dir=args.output_dir)
    payload["artifacts"] = artifacts
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    if args.enforce_no_blocking and blocking_count:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/factors/factor_registry.json",
        help="path to factor registry JSON",
    )
    parser.add_argument("--factor-id", required=True, help="proposed factor_id")
    parser.add_argument("--family", required=True, help="proposed factor family")
    parser.add_argument(
        "--required-inputs",
        nargs="*",
        default=(),
        help="raw fields required by the proposed factor",
    )
    parser.add_argument(
        "--lookback-bars",
        type=int,
        default=None,
        help="proposed lookback horizon in bars",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=(),
        help="hypothesis or transform keywords used for similarity search",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="minimum similarity score to report",
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=("watchlist", "reject", "deprecated"),
        help="historical statuses to search",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/factor_research_memory/current",
        help="directory for JSON and Markdown reports",
    )
    parser.add_argument(
        "--enforce-no-blocking",
        action="store_true",
        help="exit non-zero if any reject/deprecated match is found",
    )
    return parser.parse_args()


def _write_report(payload: dict[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "factor_research_memory_check.json"
    markdown_path = output / "factor_research_memory_check.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _render_markdown(payload: dict[str, Any]) -> str:
    params = payload["params"]
    summary = payload["summary"]
    lines = [
        "# Factor Research Memory Check",
        "",
        f"- Status: `{summary['status']}`",
        f"- Proposed factor: `{params['factor_id']}`",
        f"- Family: `{params['family']}`",
        f"- Matches: `{summary['match_count']}`",
        f"- Blocking matches: `{summary['blocking_count']}`",
        "",
        "## Matches",
        "",
    ]
    if not payload["matches"]:
        lines.append("No similar watchlist/rejected/deprecated factors found.")
        lines.append("")
        return "\n".join(lines)
    lines.extend(
        [
            "| factor_id | status | score | decision_reason | matched_fields | blocking | retry_conditions |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for match in payload["matches"]:
        lines.append(
            "| {factor_id} | {status} | {score:.3f} | {decision} | {fields} | {blocking} | {retry} |".format(
                factor_id=match["factor_id"],
                status=match["status"],
                score=float(match["similarity_score"]),
                decision=match.get("decision_reason") or "-",
                fields=", ".join(match["matched_fields"]) or "-",
                blocking="yes" if match["blocking"] else "no",
                retry=str(match.get("retry_conditions") or "-").replace("|", "\\|"),
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
