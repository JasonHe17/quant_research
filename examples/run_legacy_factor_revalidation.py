"""Revalidate legacy registry factors under the current framework."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import FactorRegistryEntry, load_factor_registry


EXAMPLES_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class RevalidationJob:
    """One per-factor policy validation job."""

    factor_id: str
    feature_columns: tuple[str, ...]
    command: list[str]
    output_dir: Path
    summary_path: Path
    log_path: Path
    memory_estimate_gb: float


def main() -> None:
    args = _parse_args()
    summary = run_legacy_factor_revalidation(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def run_legacy_factor_revalidation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    factors = _selected_factors(args)
    shared_command = _shared_benchmark_command(args)
    jobs = _factor_jobs(args, factors)
    commands = {
        "shared_benchmark": shared_command,
        "factor_revalidations": {
            job.factor_id: job.command
            for job in jobs
        },
    }
    (output_dir / "commands.json").write_text(
        json.dumps(commands, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        summary = _summary_payload(
            args,
            factors=factors,
            jobs=jobs,
            status="dry_run",
            commands=commands,
            results=[],
        )
        _write_summary(output_dir, summary)
        return summary

    if not args.skip_shared_benchmark:
        if not (args.resume_existing and _shared_benchmark_complete(args)):
            _run_command(shared_command, log_path=logs_dir / "shared_benchmark.log")
    runnable_jobs = _admission_eligible_jobs(args, factors, jobs)
    _run_factor_jobs(args, runnable_jobs)
    results = _collect_factor_results(args, factors, jobs)
    summary = _summary_payload(
        args,
        factors=factors,
        jobs=jobs,
        status="completed",
        commands=commands,
        results=results,
    )
    _write_summary(output_dir, summary)
    return summary


def _selected_factors(args: argparse.Namespace) -> list[FactorRegistryEntry]:
    registry = load_factor_registry(Path(args.registry))
    statuses = set(args.statuses)
    factor_ids = set(args.factor_ids or [])
    factors = [
        entry
        for entry in registry.entries
        if entry.status in statuses and (not factor_ids or entry.factor_id in factor_ids)
    ]
    if args.max_factors is not None:
        factors = factors[: args.max_factors]
    if not factors:
        raise ValueError("no registry factors selected for revalidation")
    return factors


def _shared_benchmark_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(EXAMPLES_DIR / "run_framework_v1_benchmark.py"),
        "--output-dir",
        str(_shared_benchmark_dir(args)),
        "--start",
        args.start,
        "--end",
        args.end,
        "--profile",
        args.profile,
        "--label-horizon-bars",
        *[str(value) for value in args.label_horizon_bars],
        "--auto-factor-admission",
        "--candidate-policy-validation-methods",
        *args.methods,
        "--candidate-policy-validation-policy",
        args.primary_policy,
        "--top-n",
        str(args.top_n),
        "--commission-bps",
        str(args.commission_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--sell-stamp-tax-bps",
        str(args.sell_stamp_tax_bps),
        "--min-commission",
        str(args.min_commission),
        "--cost-stress-multiplier",
        str(args.cost_stress_multiplier),
        "--backtest-workers",
        str(args.shared_backtest_workers),
        "--full-backtest-memory-gb",
        str(args.full_backtest_memory_gb),
        "--yearly-backtest-memory-gb",
        str(args.yearly_backtest_memory_gb),
        "--dataset-workers",
        str(args.dataset_workers),
        "--dataset-worker-memory-estimate-gb",
        str(args.dataset_worker_memory_estimate_gb),
        "--evaluation-workers",
        str(args.evaluation_workers),
        "--data-access-mode",
        args.data_access_mode,
        "--streaming-chunk",
        args.streaming_chunk,
        "--streaming-chunk-padding-days",
        str(args.streaming_chunk_padding_days),
    ]
    if args.catalog_path:
        command.extend(["--catalog-path", args.catalog_path])
    if args.data_snapshot:
        command.extend(["--data-snapshot", args.data_snapshot])
    if args.max_symbols is not None:
        command.extend(["--max-symbols", str(args.max_symbols)])
    if args.resume_existing:
        command.append("--resume-existing")
    return command


def _factor_jobs(
    args: argparse.Namespace,
    factors: list[FactorRegistryEntry],
) -> list[RevalidationJob]:
    jobs = []
    for entry in factors:
        factor_dir = Path(args.output_dir) / "factors" / entry.factor_id
        summary_path = factor_dir / "validation_summary.json"
        command = _factor_revalidation_command(args, entry, factor_dir)
        jobs.append(
            RevalidationJob(
                factor_id=entry.factor_id,
                feature_columns=entry.feature_columns,
                command=command,
                output_dir=factor_dir,
                summary_path=summary_path,
                log_path=Path(args.output_dir) / "logs" / f"{entry.factor_id}.log",
                memory_estimate_gb=args.factor_job_memory_gb,
            )
        )
    return jobs


def _factor_revalidation_command(
    args: argparse.Namespace,
    entry: FactorRegistryEntry,
    output_dir: Path,
) -> list[str]:
    shared_dir = _shared_benchmark_dir(args)
    command = [
        sys.executable,
        str(EXAMPLES_DIR / "run_candidate_policy_validation.py"),
        "--dataset-dir",
        str(shared_dir / "alpha_dataset"),
        "--label-column",
        _primary_label_column(args),
        "--admission-report",
        str(shared_dir / "factor_admission" / "factor_admission_report.json"),
        "--factor-correlation",
        str(shared_dir / "factor_evaluation" / "feature_correlation.csv"),
        "--registry",
        args.registry,
        "--registry-statuses",
        *args.statuses,
        "--admission-statuses",
        *args.admission_statuses,
        "--output-dir",
        str(output_dir),
        "--profile",
        args.profile,
        "--methods",
        *args.methods,
        "--primary-method",
        args.methods[0],
        "--policy",
        args.primary_policy,
        "--factor-health-mode",
        args.factor_health_mode,
        "--include-features",
        *entry.feature_columns,
        "--top-n",
        str(args.top_n),
        "--commission-bps",
        str(args.commission_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--sell-stamp-tax-bps",
        str(args.sell_stamp_tax_bps),
        "--min-commission",
        str(args.min_commission),
        "--cost-stress-multiplier",
        str(args.cost_stress_multiplier),
        "--backtest-workers",
        str(args.factor_backtest_workers),
        "--full-backtest-memory-gb",
        str(args.factor_job_memory_gb),
        "--yearly-backtest-memory-gb",
        str(args.factor_job_memory_gb),
        "--data-access-mode",
        args.data_access_mode,
        "--streaming-chunk",
        args.streaming_chunk,
        "--streaming-chunk-padding-days",
        str(args.streaming_chunk_padding_days),
    ]
    if args.backtest_policies:
        command.extend(["--backtest-policies", *args.backtest_policies])
    if args.resume_existing:
        command.append("--resume-existing")
    return command


def _run_factor_jobs(args: argparse.Namespace, jobs: list[RevalidationJob]) -> None:
    pending = [
        job
        for job in jobs
        if not (args.resume_existing and job.summary_path.exists())
    ]
    if not pending:
        return
    if args.factor_workers == 1 or len(pending) == 1:
        for job in pending:
            _run_command(job.command, log_path=job.log_path)
        return
    _run_jobs_with_budget(
        pending,
        max_workers=args.factor_workers,
        memory_budget_gb=_effective_memory_budget(args),
    )


def _run_jobs_with_budget(
    jobs: list[RevalidationJob],
    *,
    max_workers: int,
    memory_budget_gb: float,
) -> None:
    pending = list(jobs)
    running: dict[Future[None], RevalidationJob] = {}
    running_memory_gb = 0.0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while pending or running:
            while pending and len(running) < max_workers:
                job = pending[0]
                if running_memory_gb + job.memory_estimate_gb > memory_budget_gb:
                    break
                pending.pop(0)
                future = executor.submit(_run_command, job.command, log_path=job.log_path)
                running[future] = job
                running_memory_gb += job.memory_estimate_gb
            if not running:
                job = pending[0]
                raise RuntimeError(
                    f"factor job {job.factor_id} requires {job.memory_estimate_gb:.2f} GB, "
                    f"above memory budget {memory_budget_gb:.2f} GB"
                )
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                running_memory_gb -= job.memory_estimate_gb
                future.result()


def _admission_eligible_jobs(
    args: argparse.Namespace,
    factors: list[FactorRegistryEntry],
    jobs: list[RevalidationJob],
) -> list[RevalidationJob]:
    admission_by_feature = _admission_by_feature(args)
    allowed_statuses = set(args.admission_statuses)
    factor_by_id = {entry.factor_id: entry for entry in factors}
    eligible_jobs = []
    for job in jobs:
        entry = factor_by_id[job.factor_id]
        statuses = [
            str(admission_by_feature[column].get("admission_status"))
            for column in entry.feature_columns
            if column in admission_by_feature
        ]
        if _combined_admission_status(statuses) in allowed_statuses:
            eligible_jobs.append(job)
    return eligible_jobs


def _run_command(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with code {result.returncode}: {command[1]} "
            f"(see {log_path})"
        )


def _collect_factor_results(
    args: argparse.Namespace,
    factors: list[FactorRegistryEntry],
    jobs: list[RevalidationJob],
) -> list[dict[str, Any]]:
    admission_by_feature = _admission_by_feature(args)
    job_by_factor = {job.factor_id: job for job in jobs}
    rows = []
    for entry in factors:
        job = job_by_factor[entry.factor_id]
        payload = _read_json(job.summary_path) if job.summary_path.exists() else {}
        leaderboard = payload.get("policy_leaderboard", [])
        best_policy = leaderboard[0] if isinstance(leaderboard, list) and leaderboard else {}
        new_rows = [
            admission_by_feature[column]
            for column in entry.feature_columns
            if column in admission_by_feature
        ]
        new_statuses = [str(row.get("admission_status")) for row in new_rows]
        new_status = _combined_admission_status(new_statuses)
        if job.summary_path.exists():
            validation_status = payload.get("status", "completed")
        elif new_status not in set(args.admission_statuses):
            validation_status = "admission_filtered"
        else:
            validation_status = "missing"
        rows.append(
            {
                "factor_id": entry.factor_id,
                "feature_columns": list(entry.feature_columns),
                "legacy_status": entry.status,
                "legacy_admission_status": entry.evaluation.get("admission_status"),
                "new_admission_status": new_status,
                "new_admission_rows": new_rows,
                "validation_status": validation_status,
                "best_method": best_policy.get("method"),
                "best_policy": best_policy.get("policy"),
                "best_full_base_return": best_policy.get("full_base_return"),
                "best_full_high_cost_return": best_policy.get("full_high_cost_return"),
                "best_mean_gross_turnover": best_policy.get("mean_gross_turnover"),
                "recommended_action": _recommended_action(
                    legacy_status=entry.status,
                    new_status=new_status,
                    best_policy=best_policy,
                ),
                "validation_summary": str(job.summary_path),
            }
        )
    return rows


def _admission_by_feature(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    admission = _read_json(
        _shared_benchmark_dir(args) / "factor_admission" / "factor_admission_report.json"
    )
    return {
        str(row.get("feature")): row
        for row in admission.get("factors", [])
        if isinstance(row, dict)
    }


def _combined_admission_status(statuses: list[str]) -> str:
    if not statuses:
        return "missing"
    if "candidate" in statuses:
        return "candidate"
    if "watchlist" in statuses:
        return "watchlist"
    if "reject" in statuses:
        return "reject"
    return statuses[0]


def _recommended_action(
    *,
    legacy_status: str,
    new_status: str,
    best_policy: dict[str, Any],
) -> str:
    full_return = _number(best_policy.get("full_base_return"))
    high_cost_return = _number(best_policy.get("full_high_cost_return"))
    if new_status == "candidate" and full_return is not None and full_return > 0:
        if legacy_status in {"candidate", "promoted"}:
            return "confirmed"
        return "upgrade_to_candidate"
    if new_status == "candidate":
        return "policy_dependent_review"
    if new_status == "watchlist":
        return "horizon_or_policy_review"
    if high_cost_return is not None and high_cost_return <= 0:
        return "cost_fragile_review"
    return "deprecated_review"


def _summary_payload(
    args: argparse.Namespace,
    *,
    factors: list[FactorRegistryEntry],
    jobs: list[RevalidationJob],
    status: str,
    commands: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "params": {
            "registry": args.registry,
            "statuses": args.statuses,
            "admission_statuses": args.admission_statuses,
            "factor_ids": args.factor_ids,
            "profile": args.profile,
            "label_horizon_bars": args.label_horizon_bars,
            "methods": args.methods,
            "backtest_policies": args.backtest_policies,
            "factor_health_mode": args.factor_health_mode,
            "factor_workers": args.factor_workers,
            "factor_memory_budget_gb": args.factor_memory_budget_gb,
            "factor_job_memory_gb": args.factor_job_memory_gb,
        },
        "shared_benchmark_dir": str(_shared_benchmark_dir(args)),
        "factor_count": len(factors),
        "commands": commands,
        "jobs": [
            {
                "factor_id": job.factor_id,
                "feature_columns": list(job.feature_columns),
                "output_dir": str(job.output_dir),
                "summary": str(job.summary_path),
                "log": str(job.log_path),
                "memory_estimate_gb": job.memory_estimate_gb,
            }
            for job in jobs
        ],
        "results": results,
    }


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "legacy_factor_revalidation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results = summary.get("results", [])
    if results:
        try:
            import pandas as pd
        except ImportError:
            return
        table_rows = [
            {
                key: value
                for key, value in row.items()
                if key not in {"new_admission_rows"}
            }
            for row in results
        ]
        pd.DataFrame(table_rows).to_csv(
            output_dir / "legacy_factor_revalidation_summary.csv",
            index=False,
        )


def _shared_benchmark_complete(args: argparse.Namespace) -> bool:
    shared_dir = _shared_benchmark_dir(args)
    return (
        (shared_dir / "benchmark_summary.json").exists()
        and (shared_dir / "factor_admission" / "factor_admission_report.json").exists()
    )


def _shared_benchmark_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / "shared_benchmark"


def _primary_label_column(args: argparse.Namespace) -> str:
    horizons = list(dict.fromkeys(args.label_horizon_bars))
    if len(horizons) == 1:
        return "forward_return"
    return f"forward_return_{horizons[0]}b"


def _effective_memory_budget(args: argparse.Namespace) -> float:
    if args.factor_memory_budget_gb > 0:
        return args.factor_memory_budget_gb
    available = _available_memory_gb()
    if available is None:
        return max(args.factor_workers * args.factor_job_memory_gb, args.factor_job_memory_gb)
    return max(min(available * 0.55, available - 4.0), args.factor_job_memory_gb)


def _available_memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[1]) / 1024.0 / 1024.0
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/factors/factor_registry.json")
    parser.add_argument("--output-dir", default="runs/legacy_factor_revalidation/current")
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["candidate", "promoted", "watchlist"],
    )
    parser.add_argument(
        "--admission-statuses",
        nargs="+",
        default=["candidate", "watchlist"],
    )
    parser.add_argument("--factor-ids", nargs="+")
    parser.add_argument("--max-factors", type=int)
    parser.add_argument(
        "--profile",
        choices=("quick", "standard", "robust"),
        default="standard",
    )
    parser.add_argument("--start", default="2023-01-03T09:35:00+08:00")
    parser.add_argument("--end", default="2025-12-31T15:00:00+08:00")
    parser.add_argument("--catalog-path")
    parser.add_argument("--data-snapshot", default="2026-05-09")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--label-horizon-bars", type=int, nargs="+", default=[48, 240, 960])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("equal", "ic_weighted", "decorrelated"),
        default=["decorrelated", "equal", "ic_weighted"],
    )
    parser.add_argument(
        "--backtest-policies",
        nargs="+",
        default=["partial_rebalance_daily", "cost_aware_optimizer_daily"],
    )
    parser.add_argument("--primary-policy", default="partial_rebalance_daily")
    parser.add_argument(
        "--factor-health-mode",
        choices=("off", "monitor", "shrink"),
        default="monitor",
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--cost-stress-multiplier", type=float, default=2.0)
    parser.add_argument("--dataset-workers", type=int, default=1)
    parser.add_argument("--dataset-worker-memory-estimate-gb", type=float, default=10.0)
    parser.add_argument("--evaluation-workers", type=int, default=6)
    parser.add_argument("--shared-backtest-workers", type=int, default=6)
    parser.add_argument("--full-backtest-memory-gb", type=float, default=8.0)
    parser.add_argument("--yearly-backtest-memory-gb", type=float, default=6.0)
    parser.add_argument("--factor-workers", type=int, default=2)
    parser.add_argument("--factor-backtest-workers", type=int, default=1)
    parser.add_argument("--factor-job-memory-gb", type=float, default=5.0)
    parser.add_argument("--factor-memory-budget-gb", type=float, default=0.0)
    parser.add_argument(
        "--data-access-mode",
        choices=("data_portal", "fast_parquet"),
        default="fast_parquet",
    )
    parser.add_argument(
        "--streaming-chunk",
        choices=("year", "month", "week", "day"),
        default="month",
    )
    parser.add_argument("--streaming-chunk-padding-days", type=int, default=10)
    parser.add_argument("--skip-shared-benchmark", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_factors is not None and args.max_factors <= 0:
        raise ValueError("--max-factors must be positive")
    for name in (
        "top_n",
        "dataset_workers",
        "evaluation_workers",
        "shared_backtest_workers",
        "factor_workers",
        "factor_backtest_workers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "commission_bps",
        "slippage_bps",
        "sell_stamp_tax_bps",
        "min_commission",
        "dataset_worker_memory_estimate_gb",
        "full_backtest_memory_gb",
        "yearly_backtest_memory_gb",
        "factor_job_memory_gb",
        "factor_memory_budget_gb",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.cost_stress_multiplier < 1:
        raise ValueError("--cost-stress-multiplier must be at least 1")
    if any(value <= 0 for value in args.label_horizon_bars):
        raise ValueError("--label-horizon-bars values must be positive")
    if args.streaming_chunk_padding_days < 0:
        raise ValueError("--streaming-chunk-padding-days must be non-negative")


if __name__ == "__main__":
    main()
