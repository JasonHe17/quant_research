"""Validate the factor registry governance file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import (
    load_factor_registry,
    validate_factor_registry,
    write_factor_registry_report,
)


def main() -> None:
    args = _parse_args()
    registry = load_factor_registry(args.registry)
    report = validate_factor_registry(registry)
    artifacts = write_factor_registry_report(report, output_dir=args.output_dir)
    payload = report.to_dict()
    payload["artifacts"] = artifacts
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    if args.enforce_clean and report.status != "pass":
        raise SystemExit(1)
    if args.enforce_no_errors and report.status == "fail":
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/factors/factor_registry.json",
        help="path to factor registry JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/factor_registry_validation/current",
        help="directory for JSON and Markdown validation reports",
    )
    parser.add_argument(
        "--enforce-no-errors",
        action="store_true",
        help="exit non-zero when registry validation has hard errors",
    )
    parser.add_argument(
        "--enforce-clean",
        action="store_true",
        help="exit non-zero when registry validation has errors or warnings",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
