"""Generate a read-only monitoring report for a governed allocator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import load_factor_registry  # noqa: E402
from quant_research.portfolio import (  # noqa: E402
    allocator_monitoring_history_status,
    append_allocator_monitoring_history,
    generate_allocator_monitoring_report,
    load_allocator_registry,
    write_allocator_monitoring_report,
)


def main() -> None:
    args = _parse_args()
    registry = load_allocator_registry(args.registry)
    factor_registry = (
        load_factor_registry(args.factor_registry)
        if args.factor_registry
        else None
    )
    report = generate_allocator_monitoring_report(
        registry,
        allocator_id=args.allocator_id,
        factor_registry=factor_registry,
        project_root=PROJECT_ROOT,
    )
    artifacts = write_allocator_monitoring_report(report, output_dir=args.output_dir)
    payload = report.to_dict()
    payload["artifacts"] = artifacts
    if args.append_history:
        history = append_allocator_monitoring_history(
            report,
            history_csv=args.history_csv,
        )
        history["summary"] = allocator_monitoring_history_status(
            args.history_csv,
            sustained_warning_window=args.sustained_warning_window,
        )
        payload["history"] = history
    run_summary_path = Path(args.output_dir) / "allocator_monitoring_run_summary.json"
    payload["artifacts"]["run_summary"] = str(run_summary_path)
    run_summary_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    if args.enforce_no_failures and report.status == "fail":
        raise SystemExit(1)
    if args.enforce_clean and report.status != "pass":
        raise SystemExit(1)
    if (
        args.enforce_no_sustained_warnings
        and payload.get("history", {}).get("summary", {}).get("sustained_warning")
    ):
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/allocators/candidate_allocator_registry.json",
        help="path to allocator registry JSON",
    )
    parser.add_argument(
        "--factor-registry",
        default="configs/factors/factor_registry.json",
        help="optional factor registry used to verify allocator feature references",
    )
    parser.add_argument(
        "--allocator-id",
        default="event_limit_diffusion_complementary_health_shrink_48b",
        help="allocator identifier to monitor",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/allocator_monitoring/current",
        help="directory for JSON and Markdown monitoring reports",
    )
    parser.add_argument(
        "--history-csv",
        default="runs/allocator_monitoring/history.csv",
        help="append-only CSV ledger for monitoring snapshots",
    )
    parser.add_argument(
        "--append-history",
        action="store_true",
        help="append this monitoring run to --history-csv",
    )
    parser.add_argument(
        "--sustained-warning-window",
        type=int,
        default=3,
        help="number of recent history rows used to flag sustained warnings",
    )
    parser.add_argument(
        "--enforce-no-failures",
        action="store_true",
        help="exit non-zero when monitoring status is fail",
    )
    parser.add_argument(
        "--enforce-no-sustained-warnings",
        action="store_true",
        help="exit non-zero when appended history has sustained warn/fail rows",
    )
    parser.add_argument(
        "--enforce-clean",
        action="store_true",
        help="exit non-zero when monitoring status is warn or fail",
    )
    args = parser.parse_args()
    if args.sustained_warning_window <= 0:
        raise ValueError("--sustained-warning-window must be positive")
    if args.enforce_no_sustained_warnings and not args.append_history:
        raise ValueError("--enforce-no-sustained-warnings requires --append-history")
    return args


if __name__ == "__main__":
    main()
