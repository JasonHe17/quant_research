"""Render a unified candidate-factor review report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import (
    build_factor_candidate_review,
    load_factor_registry,
    load_optional_json,
    write_factor_candidate_review,
)


def main() -> None:
    args = _parse_args()
    registry = load_factor_registry(args.registry)
    entry = registry.get(args.factor_id)
    admission_report_path = args.admission_report or entry.evaluation.get(
        "admission_report"
    )
    review = build_factor_candidate_review(
        registry,
        factor_id=args.factor_id,
        admission_report=load_optional_json(admission_report_path),
        portfolio_validation=load_optional_json(args.portfolio_validation),
    )
    artifacts = write_factor_candidate_review(review, output_dir=args.output_dir)
    payload = {**review, "artifacts": artifacts}
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    if args.enforce_ready and review["status"] != "ready_for_portfolio_review":
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/factors/factor_registry.json",
        help="path to factor registry JSON",
    )
    parser.add_argument("--factor-id", required=True, help="registered factor_id")
    parser.add_argument(
        "--admission-report",
        default=None,
        help="optional factor_admission_report.json; defaults to registry metadata",
    )
    parser.add_argument(
        "--portfolio-validation",
        default=None,
        help="optional candidate-policy validation_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for review outputs; defaults to runs/factor_candidate_reviews/<factor-id>",
    )
    parser.add_argument(
        "--enforce-ready",
        action="store_true",
        help="exit non-zero unless the factor is ready for portfolio review",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f"runs/factor_candidate_reviews/{args.factor_id}"
    return args


if __name__ == "__main__":
    main()
