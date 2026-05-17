"""Build a factor failure atlas from registry and research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import (
    build_factor_failure_atlas,
    load_factor_registry,
    write_factor_failure_atlas_outputs,
)


def main() -> None:
    args = _parse_args()
    registry = load_factor_registry(args.registry)
    atlas = build_factor_failure_atlas(registry, base_dir=args.base_dir)
    artifacts = write_factor_failure_atlas_outputs(
        atlas,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "summary": atlas["summary"],
                "decision_reason_counts": atlas["decision_reason_counts"],
                "failed_check_counts": atlas["failed_check_counts"],
                "artifacts": artifacts,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/factors/factor_registry.json",
        help="path to factor registry JSON",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="base directory used to resolve relative artifact paths",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/factor_failure_atlas/current",
        help="directory for atlas JSON, CSV, and Markdown outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
