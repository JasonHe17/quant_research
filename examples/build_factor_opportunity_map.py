"""Build a factor opportunity map with annual top-bucket health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import (
    build_factor_opportunity_map,
    load_factor_registry,
    write_factor_opportunity_map_outputs,
)


def main() -> None:
    args = _parse_args()
    registry = load_factor_registry(args.registry)
    opportunity_map = build_factor_opportunity_map(
        registry,
        base_dir=args.base_dir,
        min_positive_years=args.min_positive_years,
        min_selected_mean_label=args.min_selected_mean_label,
        min_cost_adjusted_spread=args.min_cost_adjusted_spread,
    )
    artifacts = write_factor_opportunity_map_outputs(
        opportunity_map,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "summary": opportunity_map["summary"],
                "class_counts": opportunity_map["class_counts"],
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
        default="runs/factor_opportunity_map/current",
        help="directory for opportunity-map JSON, CSV, and Markdown outputs",
    )
    parser.add_argument("--min-positive-years", type=int, default=2)
    parser.add_argument("--min-selected-mean-label", type=float, default=0.0)
    parser.add_argument("--min-cost-adjusted-spread", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
