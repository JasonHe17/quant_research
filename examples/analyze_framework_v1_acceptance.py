"""Analyze Framework v1 acceptance results and classify factor candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.validation import (
    FactorAdmissionThresholds,
    build_factor_admission_report,
    write_factor_admission_outputs,
)


def main() -> None:
    args = _parse_args()
    benchmark_summary = _read_json(Path(args.benchmark_summary))
    factor_summary_path = _factor_summary_path(args, benchmark_summary)
    factor_summary_payload = _read_json(factor_summary_path)
    factor_summary = pd.DataFrame(factor_summary_payload.get("summary", []))
    by_timestamp_path = _by_timestamp_path(args, factor_summary_payload)
    by_timestamp = pd.read_csv(by_timestamp_path)
    thresholds = FactorAdmissionThresholds(
        min_coverage=args.min_coverage,
        min_timestamp_count=args.min_timestamp_count,
        min_abs_rank_ic_mean=args.min_abs_rank_ic_mean,
        min_abs_rank_ic_t_stat=args.min_abs_rank_ic_t_stat,
        min_directional_ic_hit_rate=args.min_directional_ic_hit_rate,
        min_stable_years=args.min_stable_years,
        min_years_observed=args.min_years_observed,
        min_cost_adjusted_spread=args.min_cost_adjusted_spread,
        max_top_n_turnover=args.max_top_n_turnover,
        cost_bps=args.cost_bps,
    )
    report = build_factor_admission_report(
        benchmark_summary=benchmark_summary,
        factor_summary=factor_summary,
        by_timestamp=by_timestamp,
        thresholds=thresholds,
    )
    artifacts = write_factor_admission_outputs(
        report,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({"summary": report["summary"], "artifacts": artifacts}, indent=2))
    if args.enforce_candidates and report["summary"]["candidate_count"] == 0:
        raise RuntimeError("factor admission produced no candidates")


def _factor_summary_path(args: argparse.Namespace, benchmark_summary: dict[str, object]) -> Path:
    if args.factor_summary:
        return Path(args.factor_summary)
    artifacts = benchmark_summary.get("artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts.get("factor_evaluation_summary"):
        raise ValueError("benchmark summary does not reference factor_evaluation_summary")
    return Path(str(artifacts["factor_evaluation_summary"]))


def _by_timestamp_path(
    args: argparse.Namespace,
    factor_summary_payload: dict[str, object],
) -> Path:
    if args.by_timestamp:
        return Path(args.by_timestamp)
    artifacts = factor_summary_payload.get("artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts.get("by_timestamp"):
        raise ValueError("factor evaluation summary does not reference by_timestamp")
    return Path(str(artifacts["by_timestamp"]))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-summary",
        default="runs/framework_v1_acceptance/standard/benchmark_summary.json",
    )
    parser.add_argument("--factor-summary")
    parser.add_argument("--by-timestamp")
    parser.add_argument(
        "--output-dir",
        default="runs/framework_v1_acceptance/standard/factor_admission",
    )
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-timestamp-count", type=int, default=1_000)
    parser.add_argument("--min-abs-rank-ic-mean", type=float, default=0.001)
    parser.add_argument("--min-abs-rank-ic-t-stat", type=float, default=2.0)
    parser.add_argument("--min-directional-ic-hit-rate", type=float, default=0.52)
    parser.add_argument("--min-stable-years", type=int, default=2)
    parser.add_argument("--min-years-observed", type=int, default=3)
    parser.add_argument("--min-cost-adjusted-spread", type=float, default=0.0)
    parser.add_argument("--max-top-n-turnover", type=float, default=0.95)
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=13.0,
        help="Round-trip cost proxy used to adjust top-minus-bottom spread.",
    )
    parser.add_argument(
        "--enforce-candidates",
        action="store_true",
        help="exit non-zero if no factor reaches candidate status",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
