"""Review standalone-alpha candidates selected by the opportunity map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import (
    build_candidate_alpha_queue_review,
    build_factor_opportunity_map,
    load_factor_registry,
    write_candidate_alpha_queue_review_outputs,
)


def main() -> None:
    args = _parse_args()
    registry = load_factor_registry(args.registry)
    opportunity_map = None
    if args.opportunity_map:
        opportunity_map = json.loads(Path(args.opportunity_map).read_text(encoding="utf-8"))
    else:
        opportunity_map = build_factor_opportunity_map(
            registry,
            base_dir=args.base_dir,
            min_positive_years=args.min_positive_years,
        )
    review = build_candidate_alpha_queue_review(
        registry,
        base_dir=args.base_dir,
        opportunity_map=opportunity_map,
        opportunity_class=args.opportunity_class,
        output_root=args.validation_output_root,
    )
    artifacts = write_candidate_alpha_queue_review_outputs(
        review,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "summary": review["summary"],
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
        "--opportunity-map",
        help="optional factor_opportunity_map.json to reuse instead of rebuilding",
    )
    parser.add_argument("--base-dir", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/candidate_alpha_queue/current",
        help="directory for queue review outputs",
    )
    parser.add_argument(
        "--validation-output-root",
        default="runs/candidate_factor_portfolios",
        help="root directory used in generated portfolio-validation commands",
    )
    parser.add_argument("--opportunity-class", default="long_alpha_candidate")
    parser.add_argument("--min-positive-years", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
