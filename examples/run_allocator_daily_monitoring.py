"""Run a daily read-only monitoring job for a governed allocator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import load_factor_registry  # noqa: E402
from quant_research.portfolio import (  # noqa: E402
    allocator_monitoring_history_status,
    append_allocator_monitoring_history,
    generate_allocator_monitoring_report,
    load_allocator_registry,
    validate_allocator_registry,
    write_allocator_monitoring_report,
    write_allocator_registry_report,
)


def main() -> None:
    args = _parse_args()
    run_id = args.run_id or _default_run_id(args.timezone)
    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir) if args.output_dir else output_root / run_id
    registry_output_dir = output_dir / "allocator_registry_validation"
    monitoring_output_dir = output_dir / "allocator_monitoring"
    history_csv = Path(args.history_csv) if args.history_csv else output_root / "history.csv"

    registry = load_allocator_registry(args.registry)
    factor_registry = (
        load_factor_registry(args.factor_registry)
        if args.factor_registry
        else None
    )

    registry_report = validate_allocator_registry(
        registry,
        factor_registry=factor_registry,
        project_root=PROJECT_ROOT,
    )
    registry_artifacts = write_allocator_registry_report(
        registry_report,
        output_dir=registry_output_dir,
    )

    monitoring_report = generate_allocator_monitoring_report(
        registry,
        allocator_id=args.allocator_id,
        factor_registry=factor_registry,
        project_root=PROJECT_ROOT,
    )
    monitoring_artifacts = write_allocator_monitoring_report(
        monitoring_report,
        output_dir=monitoring_output_dir,
    )

    if args.append_history:
        history = append_allocator_monitoring_history(
            monitoring_report,
            history_csv=history_csv,
            extra_fields={"run_id": run_id, "mode": args.mode},
            replace_existing_on=("allocator_id", "run_id"),
        )
    else:
        history = {"path": str(history_csv), "row_count": None, "latest_row": None}
    history["summary"] = allocator_monitoring_history_status(
        history_csv,
        sustained_warning_window=args.sustained_warning_window,
    )

    statuses = [registry_report.status, monitoring_report.status]
    if args.append_history:
        statuses.append(history["summary"].get("status", "pending"))
    status = _combine_status(statuses)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "run_id": run_id,
        "status": status,
        "allocator_id": args.allocator_id,
        "registry": {
            "status": registry_report.status,
            "error_count": registry_report.summary.get("error_count", 0),
            "warning_count": registry_report.summary.get("warning_count", 0),
            "artifacts": registry_artifacts,
        },
        "monitoring": {
            "status": monitoring_report.status,
            "artifacts": monitoring_artifacts,
            "section_statuses": {
                name: section.get("status")
                for name, section in monitoring_report.sections.items()
                if isinstance(section, dict)
            },
        },
        "history": history,
        "exit_policy": {
            "enforce_no_failures": args.enforce_no_failures,
            "enforce_no_sustained_warnings": args.enforce_no_sustained_warnings,
            "enforce_clean": args.enforce_clean,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "daily_monitoring_summary.json"
    payload["artifacts"] = {"summary": str(summary_path)}
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))

    if args.enforce_no_failures and _has_failure(payload):
        raise SystemExit(1)
    if (
        args.enforce_no_sustained_warnings
        and history["summary"].get("sustained_warning")
    ):
        raise SystemExit(1)
    if args.enforce_clean and status != "pass":
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
        "--mode",
        default="paper",
        choices=("paper", "live_simulation"),
        help="monitoring mode label written to the run summary",
    )
    parser.add_argument(
        "--output-root",
        default="runs/allocator_monitoring/daily",
        help="root directory for dated daily monitoring outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="exact output directory; overrides --output-root/--run-id",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run identifier; defaults to the current date in --timezone",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Shanghai",
        help="timezone used to derive the default run-id",
    )
    parser.add_argument(
        "--history-csv",
        default=None,
        help="append-only CSV ledger; defaults to --output-root/history.csv",
    )
    parser.add_argument(
        "--no-append-history",
        action="store_false",
        dest="append_history",
        help="render reports without appending a monitoring history row",
    )
    parser.set_defaults(append_history=True)
    parser.add_argument(
        "--sustained-warning-window",
        type=int,
        default=3,
        help="number of recent history rows used to flag sustained warnings",
    )
    parser.add_argument(
        "--enforce-no-failures",
        action="store_true",
        help="exit non-zero when registry or monitoring status is fail",
    )
    parser.add_argument(
        "--enforce-no-sustained-warnings",
        action="store_true",
        help="exit non-zero when appended history has sustained warn/fail rows",
    )
    parser.add_argument(
        "--enforce-clean",
        action="store_true",
        help="exit non-zero when the daily monitoring summary is warn or fail",
    )
    args = parser.parse_args()
    if args.sustained_warning_window <= 0:
        raise ValueError("--sustained-warning-window must be positive")
    if args.enforce_no_sustained_warnings and not args.append_history:
        raise ValueError("--enforce-no-sustained-warnings requires history appends")
    return args


def _default_run_id(timezone_name: str) -> str:
    try:
        tzinfo = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tzinfo = timezone.utc
    return datetime.now(tzinfo).date().isoformat()


def _combine_status(statuses: list[str]) -> str:
    normalized = [str(status) for status in statuses]
    if any(status == "fail" for status in normalized):
        return "fail"
    if any(status == "warn" for status in normalized):
        return "warn"
    if any(status == "pending" for status in normalized):
        return "pending"
    return "pass"


def _has_failure(payload: dict[str, object]) -> bool:
    registry = payload.get("registry")
    monitoring = payload.get("monitoring")
    history = payload.get("history")
    registry_failed = isinstance(registry, dict) and registry.get("status") == "fail"
    monitoring_failed = (
        isinstance(monitoring, dict) and monitoring.get("status") == "fail"
    )
    history_failed = (
        isinstance(history, dict)
        and isinstance(history.get("summary"), dict)
        and history["summary"].get("sustained_failure")
    )
    return bool(registry_failed or monitoring_failed or history_failed)


if __name__ == "__main__":
    main()
