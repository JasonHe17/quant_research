"""Run the fixed Framework v1 benchmark workflow."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class FrameworkV1BenchmarkConfig:
    """Configuration for the reproducible Framework v1 acceptance benchmark."""

    profile: str
    catalog_path: Path
    output_dir: Path
    start: str
    end: str
    data_snapshot: str | None
    max_symbols: int | None
    initial_cash: float
    top_n: int
    benchmark_lookback_bars: int
    label_horizon_bars: tuple[int, ...]
    label_entry_lag_bars: int
    commission_bps: float
    slippage_bps: float
    sell_stamp_tax_bps: float
    min_commission: float
    min_trade_weight: float
    lot_size: int
    exclude_st: bool
    limit_up_bps: float
    limit_down_bps: float
    data_access_mode: str
    streaming_chunk: str
    streaming_chunk_padding_days: int
    dataset_workers: int
    dataset_worker_memory_estimate_gb: float
    dataset_memory_budget_gb: float | None
    evaluation_workers: int
    evaluation_backend: str
    skip_feature_correlation: bool
    partition: str
    padding_days: int
    cost_stress_multiplier: float
    auto_factor_admission: bool
    candidate_admission_report: Path | None
    candidate_policy_validation_methods: tuple[str, ...]
    candidate_policy_validation_policy: str
    candidate_policy_validation_memory_gb: float
    backtest_workers: int
    backtest_memory_budget_gb: float | None
    full_backtest_memory_gb: float
    yearly_backtest_memory_gb: float
    enforce_gates: bool
    resume_existing: bool


@dataclass(frozen=True, slots=True)
class BacktestScenario:
    """One acceptance backtest scenario."""

    name: str
    start: str
    end: str
    commission_bps: float
    slippage_bps: float
    sell_stamp_tax_bps: float
    min_commission: float
    min_trade_weight: float
    description: str


@dataclass(frozen=True, slots=True)
class BacktestJob:
    """One scheduled backtest command with a memory estimate."""

    stage_name: str
    scenario_name: str
    command: list[str]
    log_path: Path
    memory_estimate_gb: float


def main() -> None:
    args = _parse_args()
    config = _config_from_args(args)
    result = run_framework_v1_benchmark(
        config,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


def run_framework_v1_benchmark(
    config: FrameworkV1BenchmarkConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or plan the fixed benchmark workflow."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    commands = _benchmark_commands(config)
    (output_dir / "commands.json").write_text(
        json.dumps(commands, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if dry_run:
        summary = _benchmark_summary(
            config,
            commands=commands,
            status="dry_run",
            artifacts=_artifact_paths(config),
            acceptance_plan=_acceptance_plan(config),
            backtests={
                scenario.name: {
                    "description": scenario.description,
                    "start": scenario.start,
                    "end": scenario.end,
                    "commission_bps": scenario.commission_bps,
                    "slippage_bps": scenario.slippage_bps,
                    "sell_stamp_tax_bps": scenario.sell_stamp_tax_bps,
                    "min_commission": scenario.min_commission,
                    "min_trade_weight": scenario.min_trade_weight,
                }
                for scenario in _backtest_scenarios(config)
            },
        )
        _write_summary(output_dir, summary)
        return summary
    _run_required_stage(config, commands, "dataset", logs_dir)
    _run_required_stage(config, commands, "factor_evaluation", logs_dir)
    _run_backtest_commands(config, commands=commands, logs_dir=logs_dir)
    if _should_run_factor_admission(config):
        summary = _collect_completed_summary(config, commands=commands)
        _write_summary(output_dir, summary)
        _run_required_stage(config, commands, "factor_admission", logs_dir)
    if _should_run_candidate_policy_validation(config):
        _run_required_stage(config, commands, "candidate_policy_validation", logs_dir)
    summary = _collect_completed_summary(config, commands=commands)
    _write_summary(output_dir, summary)
    if config.enforce_gates and summary["acceptance"]["overall_status"] == "fail":
        raise RuntimeError(
            "Framework v1 acceptance gates failed; see "
            f"{output_dir / 'benchmark_summary.json'}"
        )
    return summary


def _benchmark_commands(
    config: FrameworkV1BenchmarkConfig,
) -> dict[str, list[str]]:
    dataset_dir = config.output_dir / "alpha_dataset"
    factor_eval_dir = config.output_dir / "factor_evaluation"
    dataset_command = [
        sys.executable,
        str(EXAMPLES_DIR / "build_baseline_a_alpha_dataset.py"),
        "--catalog-path",
        str(config.catalog_path),
        "--start",
        config.start,
        "--end",
        config.end,
        "--output-dir",
        str(dataset_dir),
        "--factor-groups",
        "all",
        "--label-name",
        "forward_return",
        "--horizon-bars",
        *[str(value) for value in config.label_horizon_bars],
        "--entry-lag-bars",
        str(config.label_entry_lag_bars),
        "--limit-up-bps",
        str(config.limit_up_bps),
        "--limit-down-bps",
        str(config.limit_down_bps),
        "--partition",
        config.partition,
        "--padding-days",
        str(config.padding_days),
        "--workers",
        str(config.dataset_workers),
    ]
    if config.data_snapshot:
        dataset_command.extend(["--data-snapshot", config.data_snapshot])
    if config.max_symbols is not None:
        dataset_command.extend(["--max-symbols", str(config.max_symbols)])
    if config.exclude_st:
        dataset_command.append("--exclude-st")
    else:
        dataset_command.append("--no-exclude-st")
    evaluation_command = [
        sys.executable,
        str(EXAMPLES_DIR / "evaluate_alpha_dataset.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(factor_eval_dir),
        "--label-column",
        _primary_label_column(config),
        "--top-n",
        str(config.top_n),
        "--quantiles",
        "5",
        "--workers",
        str(config.evaluation_workers),
        "--backend",
        config.evaluation_backend,
    ]
    horizon_labels = _horizon_label_columns(config)
    if horizon_labels:
        evaluation_command.extend(["--horizon-label-columns", *horizon_labels])
    if config.skip_feature_correlation:
        evaluation_command.append("--skip-feature-correlation")
    if config.dataset_memory_budget_gb is not None:
        dataset_command.extend(["--memory-budget-gb", str(config.dataset_memory_budget_gb)])
    dataset_command.extend(
        [
            "--worker-memory-estimate-gb",
            str(config.dataset_worker_memory_estimate_gb),
        ]
    )
    commands = {
        "dataset": dataset_command,
        "factor_evaluation": evaluation_command,
    }
    for scenario in _backtest_scenarios(config):
        commands[f"backtest_{scenario.name}"] = _backtest_command(config, scenario)
    if _should_run_factor_admission(config):
        commands["factor_admission"] = _factor_admission_command(config)
    if _should_run_candidate_policy_validation(config):
        commands["candidate_policy_validation"] = _candidate_policy_validation_command(config)
    return commands


def _backtest_command(
    config: FrameworkV1BenchmarkConfig,
    scenario: BacktestScenario,
) -> list[str]:
    backtest_dir = config.output_dir / "backtests" / scenario.name
    command = [
        sys.executable,
        str(EXAMPLES_DIR / "run_baseline_a_real_backtest.py"),
        "--catalog-path",
        str(config.catalog_path),
        "--start",
        scenario.start,
        "--end",
        scenario.end,
        "--top-n",
        str(config.top_n),
        "--initial-cash",
        str(config.initial_cash),
        "--lookback-bars",
        str(config.benchmark_lookback_bars),
        "--commission-bps",
        str(scenario.commission_bps),
        "--slippage-bps",
        str(scenario.slippage_bps),
        "--sell-stamp-tax-bps",
        str(scenario.sell_stamp_tax_bps),
        "--min-commission",
        str(scenario.min_commission),
        "--min-trade-weight",
        str(scenario.min_trade_weight),
        "--lot-size",
        str(config.lot_size),
        "--limit-up-bps",
        str(config.limit_up_bps),
        "--limit-down-bps",
        str(config.limit_down_bps),
        "--data-access-mode",
        config.data_access_mode,
        "--streaming-chunk",
        config.streaming_chunk,
        "--streaming-chunk-padding-days",
        str(config.streaming_chunk_padding_days),
        "--output-dir",
        str(backtest_dir),
    ]
    if config.max_symbols is not None:
        command.extend(["--max-symbols", str(config.max_symbols)])
    if config.exclude_st:
        command.append("--exclude-st")
    return command


def _backtest_scenarios(config: FrameworkV1BenchmarkConfig) -> list[BacktestScenario]:
    scenarios = [
        BacktestScenario(
            name="full_base",
            start=config.start,
            end=config.end,
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
            sell_stamp_tax_bps=config.sell_stamp_tax_bps,
            min_commission=config.min_commission,
            min_trade_weight=config.min_trade_weight,
            description="Full-window baseline with production-like costs.",
        )
    ]
    if config.profile in {"standard", "robust"}:
        for year in _years_in_window(config.start, config.end):
            scenarios.append(
                BacktestScenario(
                    name=f"year_{year}_base",
                    start=max(config.start, f"{year}-01-01T00:00:00+08:00"),
                    end=min(config.end, f"{year}-12-31T23:59:59+08:00"),
                    commission_bps=config.commission_bps,
                    slippage_bps=config.slippage_bps,
                    sell_stamp_tax_bps=config.sell_stamp_tax_bps,
                    min_commission=config.min_commission,
                    min_trade_weight=config.min_trade_weight,
                    description=f"Calendar-year stability slice for {year}.",
                )
            )
        scenarios.append(
            BacktestScenario(
                name="full_high_cost",
                start=config.start,
                end=config.end,
                commission_bps=config.commission_bps * config.cost_stress_multiplier,
                slippage_bps=config.slippage_bps * config.cost_stress_multiplier,
                sell_stamp_tax_bps=(
                    config.sell_stamp_tax_bps * config.cost_stress_multiplier
                ),
                min_commission=config.min_commission,
                min_trade_weight=config.min_trade_weight,
                description="Full-window transaction-cost stress.",
            )
        )
    if config.profile == "robust":
        scenarios.append(
            BacktestScenario(
                name="full_zero_cost",
                start=config.start,
                end=config.end,
                commission_bps=0.0,
                slippage_bps=0.0,
                sell_stamp_tax_bps=0.0,
                min_commission=0.0,
                min_trade_weight=config.min_trade_weight,
                description="Full-window zero-cost diagnostic upper bound.",
            )
        )
        scenarios.append(
            BacktestScenario(
                name="full_trade_filter_stress",
                start=config.start,
                end=config.end,
                commission_bps=config.commission_bps,
                slippage_bps=config.slippage_bps,
                sell_stamp_tax_bps=config.sell_stamp_tax_bps,
                min_commission=config.min_commission,
                min_trade_weight=min(config.min_trade_weight * 4.0, 1.0),
                description="Full-window minimum-trade-weight stress.",
            )
        )
    return scenarios


def _candidate_policy_validation_command(config: FrameworkV1BenchmarkConfig) -> list[str]:
    output_dir = config.output_dir / "candidate_policy_validation"
    command = [
        sys.executable,
        str(EXAMPLES_DIR / "run_candidate_policy_validation.py"),
        "--dataset-dir",
        str(config.output_dir / "alpha_dataset"),
        "--label-column",
        _primary_label_column(config),
        "--admission-report",
        str(_effective_candidate_admission_report(config)),
        "--factor-correlation",
        str(config.output_dir / "factor_evaluation" / "feature_correlation.csv"),
        "--output-dir",
        str(output_dir),
        "--profile",
        config.profile,
        "--methods",
        *config.candidate_policy_validation_methods,
        "--primary-method",
        config.candidate_policy_validation_methods[0],
        "--policy",
        config.candidate_policy_validation_policy,
        "--top-n",
        str(config.top_n),
        "--initial-cash",
        str(config.initial_cash),
        "--commission-bps",
        str(config.commission_bps),
        "--slippage-bps",
        str(config.slippage_bps),
        "--sell-stamp-tax-bps",
        str(config.sell_stamp_tax_bps),
        "--min-commission",
        str(config.min_commission),
        "--cost-stress-multiplier",
        str(config.cost_stress_multiplier),
        "--lot-size",
        str(config.lot_size),
        "--min-trade-weight",
        str(config.min_trade_weight),
        "--limit-up-bps",
        str(config.limit_up_bps),
        "--limit-down-bps",
        str(config.limit_down_bps),
        "--data-access-mode",
        config.data_access_mode,
        "--streaming-chunk",
        config.streaming_chunk,
        "--streaming-chunk-padding-days",
        str(config.streaming_chunk_padding_days),
        "--backtest-workers",
        str(config.backtest_workers),
        "--full-backtest-memory-gb",
        str(config.candidate_policy_validation_memory_gb),
        "--yearly-backtest-memory-gb",
        str(config.candidate_policy_validation_memory_gb),
    ]
    if config.exclude_st:
        command.append("--exclude-st")
    else:
        command.append("--no-exclude-st")
    if config.resume_existing:
        command.append("--resume-existing")
    return command


def _factor_admission_command(config: FrameworkV1BenchmarkConfig) -> list[str]:
    command = [
        sys.executable,
        str(EXAMPLES_DIR / "analyze_framework_v1_acceptance.py"),
        "--benchmark-summary",
        str(config.output_dir / "benchmark_summary.json"),
        "--factor-summary",
        str(config.output_dir / "factor_evaluation" / "summary.json"),
        "--output-dir",
        str(config.output_dir / "factor_admission"),
        "--cost-bps",
        str(_estimated_round_trip_cost_bps(config)),
    ]
    return command


def _should_run_factor_admission(config: FrameworkV1BenchmarkConfig) -> bool:
    return config.auto_factor_admission


def _should_run_candidate_policy_validation(config: FrameworkV1BenchmarkConfig) -> bool:
    return _effective_candidate_admission_report(config) is not None


def _effective_candidate_admission_report(config: FrameworkV1BenchmarkConfig) -> Path | None:
    if config.candidate_admission_report is not None:
        return config.candidate_admission_report
    if config.auto_factor_admission:
        return config.output_dir / "factor_admission" / "factor_admission_report.json"
    return None


def _estimated_round_trip_cost_bps(config: FrameworkV1BenchmarkConfig) -> float:
    return float(
        2.0 * config.commission_bps
        + 2.0 * config.slippage_bps
        + config.sell_stamp_tax_bps
    )


def _years_in_window(start: str, end: str) -> list[int]:
    start_year = int(start[:4])
    end_year = int(end[:4])
    return list(range(start_year, end_year + 1))


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
            f"benchmark command failed with code {result.returncode}: "
            f"{command[1]} (see {log_path})"
        )


def _run_required_stage(
    config: FrameworkV1BenchmarkConfig,
    commands: dict[str, list[str]],
    stage_name: str,
    logs_dir: Path,
) -> None:
    if config.resume_existing and _stage_complete(config, stage_name):
        return
    _run_command(commands[stage_name], log_path=logs_dir / f"{stage_name}.log")


def _run_backtest_commands(
    config: FrameworkV1BenchmarkConfig,
    *,
    commands: dict[str, list[str]],
    logs_dir: Path,
) -> None:
    jobs = _pending_backtest_jobs(config, commands=commands, logs_dir=logs_dir)
    if not jobs:
        return
    if config.backtest_workers == 1 or len(jobs) == 1:
        for job in jobs:
            _run_command(job.command, log_path=job.log_path)
        return
    _run_backtest_jobs_with_budget(
        jobs,
        max_workers=config.backtest_workers,
        memory_budget_gb=_effective_backtest_memory_budget_gb(config),
    )


def _pending_backtest_jobs(
    config: FrameworkV1BenchmarkConfig,
    *,
    commands: dict[str, list[str]],
    logs_dir: Path,
) -> list[BacktestJob]:
    jobs: list[BacktestJob] = []
    for scenario in _backtest_scenarios(config):
        stage_name = f"backtest_{scenario.name}"
        if config.resume_existing and _stage_complete(config, stage_name):
            continue
        jobs.append(
            BacktestJob(
                stage_name=stage_name,
                scenario_name=scenario.name,
                command=commands[stage_name],
                log_path=logs_dir / f"{stage_name}.log",
                memory_estimate_gb=_backtest_memory_estimate_gb(config, scenario),
            )
        )
    return jobs


def _run_backtest_jobs_with_budget(
    jobs: list[BacktestJob],
    *,
    max_workers: int,
    memory_budget_gb: float,
) -> None:
    pending = list(jobs)
    running: dict[Future[None], BacktestJob] = {}
    running_memory_gb = 0.0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while pending or running:
            while pending and len(running) < max_workers:
                job = pending[0]
                if not _can_launch_backtest_job(
                    job,
                    running_memory_gb=running_memory_gb,
                    memory_budget_gb=memory_budget_gb,
                ):
                    break
                pending.pop(0)
                future = executor.submit(_run_command, job.command, log_path=job.log_path)
                running[future] = job
                running_memory_gb += job.memory_estimate_gb
            if not running:
                job = pending[0]
                raise RuntimeError(
                    f"backtest job {job.stage_name} requires an estimated "
                    f"{job.memory_estimate_gb:.2f} GB, above the configured "
                    f"budget of {memory_budget_gb:.2f} GB"
                )
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                running_memory_gb -= job.memory_estimate_gb
                future.result()


def _can_launch_backtest_job(
    job: BacktestJob,
    *,
    running_memory_gb: float,
    memory_budget_gb: float,
) -> bool:
    return running_memory_gb + job.memory_estimate_gb <= memory_budget_gb


def _backtest_memory_estimate_gb(
    config: FrameworkV1BenchmarkConfig,
    scenario: BacktestScenario,
) -> float:
    if scenario.name.startswith("year_"):
        return config.yearly_backtest_memory_gb
    return config.full_backtest_memory_gb


def _effective_backtest_memory_budget_gb(config: FrameworkV1BenchmarkConfig) -> float:
    if config.backtest_memory_budget_gb is not None:
        return config.backtest_memory_budget_gb
    available = _available_memory_gb()
    if available is None:
        return max(config.full_backtest_memory_gb, config.yearly_backtest_memory_gb)
    return max(
        min(available * 0.60, available - 2.0),
        min(config.full_backtest_memory_gb, config.yearly_backtest_memory_gb),
    )


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


def _stage_complete(config: FrameworkV1BenchmarkConfig, stage_name: str) -> bool:
    artifacts = _artifact_paths(config)
    if stage_name == "dataset":
        return Path(str(artifacts["dataset_summary"])).exists()
    if stage_name == "factor_evaluation":
        return Path(str(artifacts["factor_evaluation_summary"])).exists()
    if stage_name == "factor_admission":
        return Path(str(artifacts["factor_admission_summary"])).exists()
    if stage_name.startswith("backtest_"):
        scenario_name = stage_name.removeprefix("backtest_")
        summaries = artifacts["backtest_summaries"]
        if isinstance(summaries, dict):
            return Path(str(summaries.get(scenario_name, ""))).exists()
    if stage_name == "candidate_policy_validation":
        return Path(str(artifacts["candidate_policy_validation_summary"])).exists()
    return False


def _collect_completed_summary(
    config: FrameworkV1BenchmarkConfig,
    *,
    commands: dict[str, list[str]],
) -> dict[str, Any]:
    artifacts = _artifact_paths(config)
    dataset_summary = _read_json(Path(artifacts["dataset_summary"]))
    evaluation_summary = _read_json(Path(artifacts["factor_evaluation_summary"]))
    backtest_summaries = _read_backtest_summaries(config)
    factor_admission = _read_factor_admission_summary(config)
    candidate_policy_validation = _read_candidate_policy_validation_summary(config)
    dataset_partitions = dataset_summary.get("partitions", [])
    dataset_rows = sum(int(row.get("dataset_row_count", 0)) for row in dataset_partitions)
    label_rows = sum(int(row.get("label_row_count", 0)) for row in dataset_partitions)
    bar_rows = sum(int(row.get("bar_count", 0)) for row in dataset_partitions)
    factor_rows = evaluation_summary.get("summary", [])
    top_factors = factor_rows[:10] if isinstance(factor_rows, list) else []
    dataset = {
        "bar_count": bar_rows,
        "dataset_row_count": dataset_rows,
        "label_row_count": label_rows,
        "partition_count": len(dataset_partitions),
        "partitions": dataset_partitions,
    }
    factor_evaluation = {
        "top_factors": top_factors,
        "feature_count": len(factor_rows) if isinstance(factor_rows, list) else 0,
    }
    acceptance = _acceptance_checks(
        config,
        dataset=dataset,
        factor_evaluation=factor_evaluation,
        factor_admission=factor_admission,
        backtests=backtest_summaries,
    )
    status = "completed" if acceptance["overall_status"] != "fail" else "failed_gates"
    return _benchmark_summary(
        config,
        commands=commands,
        status=status,
        artifacts=artifacts,
        dataset=dataset,
        factor_evaluation=factor_evaluation,
        backtests=backtest_summaries,
        candidate_policy_validation=candidate_policy_validation,
        acceptance=acceptance,
    )


def _read_factor_admission_summary(config: FrameworkV1BenchmarkConfig) -> dict[str, Any]:
    path = config.output_dir / "factor_admission" / "factor_admission_report.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    return {
        "artifact_dir": str(path.parent),
        "summary": str(path),
        "generated_at": payload.get("generated_at"),
        "source": payload.get("source", {}),
        "thresholds": payload.get("thresholds", {}),
        "summary_metrics": payload.get("summary", {}),
    }


def _read_candidate_policy_validation_summary(config: FrameworkV1BenchmarkConfig) -> dict[str, Any]:
    if not _should_run_candidate_policy_validation(config):
        return {}
    path = config.output_dir / "candidate_policy_validation" / "validation_summary.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    return {
        "artifact_dir": str(path.parent),
        "summary": str(path),
        "status": payload.get("status"),
        "policy_leaderboard": payload.get("policy_leaderboard", []),
        "validation": payload.get("validation", {}),
    }


def _read_backtest_summaries(config: FrameworkV1BenchmarkConfig) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for scenario in _backtest_scenarios(config):
        summary_path = config.output_dir / "backtests" / scenario.name / "summary.json"
        summary = _read_json(summary_path)
        summaries[scenario.name] = {
            "description": scenario.description,
            "start": scenario.start,
            "end": scenario.end,
            "artifact_dir": str(summary_path.parent),
            "bar_count": summary.get("bar_count"),
            "instrument_count": summary.get("instrument_count"),
            "signal_count": summary.get("signal_count"),
            "metrics": summary.get("metrics", {}),
            "execution_constraint_counts": summary.get(
                "execution_constraint_counts",
                {},
            ),
        }
    return summaries


def _acceptance_plan(config: FrameworkV1BenchmarkConfig) -> dict[str, Any]:
    return {
        "profile": config.profile,
        "window": {
            "start": config.start,
            "end": config.end,
            "recommended_min_years": 3,
        },
        "scenarios": {
            scenario.name: {
                "description": scenario.description,
                "start": scenario.start,
                "end": scenario.end,
                "commission_bps": scenario.commission_bps,
                "slippage_bps": scenario.slippage_bps,
                "sell_stamp_tax_bps": scenario.sell_stamp_tax_bps,
                "min_commission": scenario.min_commission,
                "min_trade_weight": scenario.min_trade_weight,
            }
            for scenario in _backtest_scenarios(config)
        },
        "failure_gates": [
            "dataset rows and labels must be non-empty",
            "factor evaluation must produce at least one feature summary",
            "all backtest scenarios must emit finite metrics and positive equity",
            "full_base must have signals, execution rows, and trades",
        ],
        "warning_gates": [
            "validation window should span at least three calendar years",
            "production acceptance should use a broad universe, not max-symbol smoke",
            "standard profile should include full, yearly, and high-cost scenarios",
            "robust profile should include zero-cost and trade-filter stress scenarios",
        ],
    }


def _acceptance_checks(
    config: FrameworkV1BenchmarkConfig,
    *,
    dataset: dict[str, Any],
    factor_evaluation: dict[str, Any],
    backtests: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        name="dataset_non_empty",
        status=(
            "pass"
            if int(dataset.get("dataset_row_count", 0)) > 0
            and int(dataset.get("label_row_count", 0)) > 0
            else "fail"
        ),
        details={
            "dataset_row_count": dataset.get("dataset_row_count", 0),
            "label_row_count": dataset.get("label_row_count", 0),
        },
    )
    _add_check(
        checks,
        name="factor_summary_non_empty",
        status="pass" if int(factor_evaluation.get("feature_count", 0)) > 0 else "fail",
        details={"feature_count": factor_evaluation.get("feature_count", 0)},
    )
    _add_check(
        checks,
        name="backtest_scenarios_present",
        status=(
            "pass"
            if set(_expected_scenario_names(config)).issubset(backtests)
            else "fail"
        ),
        details={
            "expected": _expected_scenario_names(config),
            "actual": sorted(backtests),
        },
    )
    for scenario_name, summary in backtests.items():
        metrics = summary.get("metrics", {})
        _add_check(
            checks,
            name=f"{scenario_name}_metrics_finite",
            status="pass" if _metrics_are_finite(metrics) else "fail",
            details={"metrics": metrics},
        )
        _add_check(
            checks,
            name=f"{scenario_name}_positive_equity",
            status=(
                "pass"
                if _number(metrics.get("final_equity")) is not None
                and _number(metrics.get("final_equity")) > 0
                else "fail"
            ),
            details={"final_equity": metrics.get("final_equity")},
        )
        drawdown = _number(metrics.get("max_drawdown"))
        _add_check(
            checks,
            name=f"{scenario_name}_drawdown_range",
            status="pass" if drawdown is not None and -1.0 <= drawdown <= 0.0 else "fail",
            details={"max_drawdown": metrics.get("max_drawdown")},
        )
    full_base = backtests.get("full_base", {})
    full_base_metrics = full_base.get("metrics", {})
    full_base_counts = full_base.get("execution_constraint_counts", {})
    _add_check(
        checks,
        name="full_base_execution_activity",
        status=(
            "pass"
            if int(full_base.get("signal_count") or 0) > 0
            and int(full_base_counts.get("execution_row_count") or 0) > 0
            and _number(full_base_metrics.get("trade_count")) is not None
            and _number(full_base_metrics.get("trade_count")) > 0
            else "fail"
        ),
        details={
            "signal_count": full_base.get("signal_count"),
            "execution_row_count": full_base_counts.get("execution_row_count"),
            "trade_count": full_base_metrics.get("trade_count"),
        },
    )
    high_cost = backtests.get("full_high_cost", {}).get("metrics", {})
    if high_cost:
        base_cost = _number(full_base_metrics.get("total_transaction_cost"))
        stress_cost = _number(high_cost.get("total_transaction_cost"))
        _add_check(
            checks,
            name="high_cost_stress_costs_not_lower",
            status=(
                "pass"
                if base_cost is not None
                and stress_cost is not None
                and stress_cost >= base_cost
                else "warn"
            ),
            details={
                "full_base_total_transaction_cost": base_cost,
                "full_high_cost_total_transaction_cost": stress_cost,
            },
        )
    zero_cost = backtests.get("full_zero_cost", {}).get("metrics", {})
    if zero_cost:
        base_return = _number(full_base_metrics.get("total_return"))
        zero_return = _number(zero_cost.get("total_return"))
        _add_check(
            checks,
            name="zero_cost_return_not_lower_than_base",
            status=(
                "pass"
                if base_return is not None
                and zero_return is not None
                and zero_return >= base_return
                else "warn"
            ),
            details={
                "full_base_total_return": base_return,
                "full_zero_cost_total_return": zero_return,
            },
        )
    _add_check(
        checks,
        name="validation_window_length",
        status="pass" if len(_years_in_window(config.start, config.end)) >= 3 else "warn",
        details={"years": _years_in_window(config.start, config.end)},
    )
    full_base_instruments = int(full_base.get("instrument_count") or 0)
    _add_check(
        checks,
        name="universe_breadth",
        status="pass" if config.max_symbols is None and full_base_instruments >= 100 else "warn",
        details={
            "max_symbols": config.max_symbols,
            "full_base_instrument_count": full_base_instruments,
        },
    )
    failed = [check for check in checks if check["status"] == "fail"]
    warned = [check for check in checks if check["status"] == "warn"]
    return {
        "overall_status": "fail" if failed else "warn" if warned else "pass",
        "checks": checks,
        "failed_count": len(failed),
        "warning_count": len(warned),
    }


def _expected_scenario_names(config: FrameworkV1BenchmarkConfig) -> list[str]:
    return [scenario.name for scenario in _backtest_scenarios(config)]


def _add_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    status: str,
    details: dict[str, Any],
) -> None:
    checks.append({"name": name, "status": status, "details": details})


def _metrics_are_finite(metrics: dict[str, Any]) -> bool:
    if not metrics:
        return False
    values = [_number(value) for value in metrics.values()]
    numeric_values = [value for value in values if value is not None]
    return bool(numeric_values) and all(math.isfinite(value) for value in numeric_values)


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(output):
        return None
    return output


def _benchmark_summary(
    config: FrameworkV1BenchmarkConfig,
    *,
    commands: dict[str, list[str]],
    status: str,
    artifacts: dict[str, str],
    dataset: dict[str, Any] | None = None,
    factor_evaluation: dict[str, Any] | None = None,
    factor_admission: dict[str, Any] | None = None,
    backtests: dict[str, Any] | None = None,
    candidate_policy_validation: dict[str, Any] | None = None,
    acceptance: dict[str, Any] | None = None,
    acceptance_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "benchmark": "framework_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "config": _jsonable_config(config),
        "commands": commands,
        "artifacts": artifacts,
        "dataset": dataset or {},
        "factor_evaluation": factor_evaluation or {},
        "factor_admission": factor_admission or {},
        "backtests": backtests or {},
        "candidate_policy_validation": candidate_policy_validation or {},
        "acceptance": acceptance or {},
        "acceptance_plan": acceptance_plan or {},
    }


def _artifact_paths(config: FrameworkV1BenchmarkConfig) -> dict[str, str]:
    return {
        "commands": str(config.output_dir / "commands.json"),
        "summary": str(config.output_dir / "benchmark_summary.json"),
        "dataset_dir": str(config.output_dir / "alpha_dataset"),
        "dataset_summary": str(config.output_dir / "alpha_dataset" / "summary.json"),
        "factor_evaluation_dir": str(config.output_dir / "factor_evaluation"),
        "factor_evaluation_summary": str(
            config.output_dir / "factor_evaluation" / "summary.json"
        ),
        "factor_admission_dir": str(config.output_dir / "factor_admission"),
        "factor_admission_summary": str(
            config.output_dir / "factor_admission" / "factor_admission_report.json"
        ),
        "backtests_dir": str(config.output_dir / "backtests"),
        "backtest_summary": str(
            config.output_dir / "backtests" / "full_base" / "summary.json"
        ),
        "backtest_summaries": {
            scenario.name: str(
                config.output_dir / "backtests" / scenario.name / "summary.json"
            )
            for scenario in _backtest_scenarios(config)
        },
        "candidate_policy_validation_dir": str(
            config.output_dir / "candidate_policy_validation"
        ),
        "candidate_policy_validation_summary": str(
            config.output_dir / "candidate_policy_validation" / "validation_summary.json"
        ),
        "logs_dir": str(config.output_dir / "logs"),
    }


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable_config(config: FrameworkV1BenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["catalog_path"] = str(config.catalog_path)
    payload["output_dir"] = str(config.output_dir)
    payload["label_horizon_bars"] = list(config.label_horizon_bars)
    payload["auto_factor_admission"] = config.auto_factor_admission
    payload["candidate_admission_report"] = (
        str(config.candidate_admission_report)
        if config.candidate_admission_report is not None
        else None
    )
    effective_admission_report = _effective_candidate_admission_report(config)
    payload["effective_candidate_admission_report"] = (
        str(effective_admission_report)
        if effective_admission_report is not None
        else None
    )
    payload["candidate_policy_validation_methods"] = list(
        config.candidate_policy_validation_methods
    )
    return payload


def _label_columns(config: FrameworkV1BenchmarkConfig) -> tuple[str, ...]:
    if len(config.label_horizon_bars) == 1:
        return ("forward_return",)
    return tuple(f"forward_return_{horizon}b" for horizon in config.label_horizon_bars)


def _primary_label_column(config: FrameworkV1BenchmarkConfig) -> str:
    return _label_columns(config)[0]


def _horizon_label_columns(config: FrameworkV1BenchmarkConfig) -> list[str]:
    return list(_label_columns(config)[1:])


def _config_from_args(args: argparse.Namespace) -> FrameworkV1BenchmarkConfig:
    return FrameworkV1BenchmarkConfig(
        profile=args.profile,
        catalog_path=Path(args.catalog_path),
        output_dir=Path(args.output_dir),
        start=args.start,
        end=args.end,
        data_snapshot=args.data_snapshot,
        max_symbols=args.max_symbols,
        initial_cash=args.initial_cash,
        top_n=args.top_n,
        benchmark_lookback_bars=args.benchmark_lookback_bars,
        label_horizon_bars=tuple(dict.fromkeys(args.label_horizon_bars)),
        label_entry_lag_bars=args.label_entry_lag_bars,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        sell_stamp_tax_bps=args.sell_stamp_tax_bps,
        min_commission=args.min_commission,
        min_trade_weight=args.min_trade_weight,
        lot_size=args.lot_size,
        exclude_st=args.exclude_st,
        limit_up_bps=args.limit_up_bps,
        limit_down_bps=args.limit_down_bps,
        data_access_mode=args.data_access_mode,
        streaming_chunk=args.streaming_chunk,
        streaming_chunk_padding_days=args.streaming_chunk_padding_days,
        dataset_workers=args.dataset_workers,
        dataset_worker_memory_estimate_gb=args.dataset_worker_memory_estimate_gb,
        dataset_memory_budget_gb=(
            args.dataset_memory_budget_gb
            if args.dataset_memory_budget_gb > 0
            else None
        ),
        evaluation_workers=args.evaluation_workers,
        evaluation_backend=args.evaluation_backend,
        skip_feature_correlation=args.skip_feature_correlation,
        partition=args.partition,
        padding_days=args.padding_days,
        cost_stress_multiplier=args.cost_stress_multiplier,
        auto_factor_admission=args.auto_factor_admission,
        candidate_admission_report=(
            Path(args.candidate_admission_report)
            if args.candidate_admission_report
            else None
        ),
        candidate_policy_validation_methods=tuple(args.candidate_policy_validation_methods),
        candidate_policy_validation_policy=args.candidate_policy_validation_policy,
        candidate_policy_validation_memory_gb=args.candidate_policy_validation_memory_gb,
        backtest_workers=args.backtest_workers,
        backtest_memory_budget_gb=(
            args.backtest_memory_budget_gb
            if args.backtest_memory_budget_gb > 0
            else None
        ),
        full_backtest_memory_gb=args.full_backtest_memory_gb,
        yearly_backtest_memory_gb=args.yearly_backtest_memory_gb,
        enforce_gates=args.enforce_gates,
        resume_existing=args.resume_existing,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-path",
        default="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "standard", "robust"),
        default="standard",
        help=(
            "quick runs only full_base, standard adds calendar-year and high-cost "
            "scenarios, robust also adds zero-cost and trade-filter stress"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2023-01-03T09:35:00+08:00")
    parser.add_argument("--end", default="2025-12-31T15:00:00+08:00")
    parser.add_argument("--data-snapshot", default="2026-05-09")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--benchmark-lookback-bars", type=int, default=1)
    parser.add_argument("--label-horizon-bars", type=int, nargs="+", default=[48])
    parser.add_argument("--label-entry-lag-bars", type=int, default=1)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--min-trade-weight", type=float, default=0.0005)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument(
        "--exclude-st",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--limit-up-bps", type=float, default=980.0)
    parser.add_argument("--limit-down-bps", type=float, default=980.0)
    parser.add_argument(
        "--data-access-mode",
        choices=("data_portal", "fast_parquet"),
        default="fast_parquet",
    )
    parser.add_argument(
        "--streaming-chunk",
        choices=("year", "month", "week", "day"),
        default="month",
        help="Chunk size passed to fast_parquet streaming backtests.",
    )
    parser.add_argument(
        "--streaming-chunk-padding-days",
        type=int,
        default=10,
        help="Padding passed to fast_parquet streaming chunks.",
    )
    parser.add_argument("--dataset-workers", type=int, default=1)
    parser.add_argument(
        "--dataset-worker-memory-estimate-gb",
        type=float,
        default=10.0,
        help="estimated memory footprint for each dataset build worker",
    )
    parser.add_argument(
        "--dataset-memory-budget-gb",
        type=float,
        default=0.0,
        help=(
            "memory budget for concurrent dataset workers; 0 lets the dataset "
            "builder auto-detect available memory"
        ),
    )
    parser.add_argument("--evaluation-workers", type=int, default=6)
    parser.add_argument(
        "--evaluation-backend",
        choices=("process",),
        default="process",
    )
    parser.add_argument("--skip-feature-correlation", action="store_true")
    parser.add_argument("--partition", choices=("monthly", "yearly"), default="monthly")
    parser.add_argument("--padding-days", type=int, default=30)
    parser.add_argument("--cost-stress-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--auto-factor-admission",
        action="store_true",
        help=(
            "after benchmark backtests finish, generate factor admission outputs "
            "from the benchmark artifacts and use them for candidate policy validation"
        ),
    )
    parser.add_argument(
        "--candidate-admission-report",
        help=(
            "optional factor_admission_report.json; when provided, run candidate "
            "policy validation as part of the benchmark"
        ),
    )
    parser.add_argument(
        "--candidate-policy-validation-methods",
        nargs="+",
        choices=("equal", "ic_weighted", "decorrelated"),
        default=["decorrelated", "equal", "ic_weighted"],
    )
    parser.add_argument(
        "--candidate-policy-validation-policy",
        default="partial_rebalance_daily",
        help="primary policy name used for candidate policy validation gates",
    )
    parser.add_argument(
        "--candidate-policy-validation-memory-gb",
        type=float,
        default=5.0,
        help="estimated memory footprint for each candidate policy validation backtest",
    )
    parser.add_argument(
        "--backtest-workers",
        type=int,
        default=6,
        help="maximum number of backtest subprocesses to run concurrently",
    )
    parser.add_argument(
        "--backtest-memory-budget-gb",
        type=float,
        default=0.0,
        help=(
            "memory budget for concurrent backtests; 0 auto-detects available "
            "memory and uses a conservative fraction"
        ),
    )
    parser.add_argument(
        "--full-backtest-memory-gb",
        type=float,
        default=8.0,
        help="estimated memory footprint for each full-window backtest",
    )
    parser.add_argument(
        "--yearly-backtest-memory-gb",
        type=float,
        default=6.0,
        help="estimated memory footprint for each yearly backtest",
    )
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="exit non-zero when acceptance failure gates fail",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="skip benchmark stages whose expected summary artifacts already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write benchmark commands and summary without running data jobs",
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_symbols is not None and args.max_symbols <= 0:
        raise ValueError("--max-symbols must be positive")
    for name in (
        "initial_cash",
        "commission_bps",
        "slippage_bps",
        "sell_stamp_tax_bps",
        "min_commission",
        "limit_up_bps",
        "limit_down_bps",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in (
        "top_n",
        "benchmark_lookback_bars",
        "label_entry_lag_bars",
        "lot_size",
        "dataset_workers",
        "evaluation_workers",
        "backtest_workers",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if any(value <= 0 for value in args.label_horizon_bars):
        raise ValueError("--label-horizon-bars values must be positive")
    if not 0 <= args.min_trade_weight <= 1:
        raise ValueError("--min-trade-weight must be in [0, 1]")
    if args.padding_days < 0:
        raise ValueError("--padding-days must be non-negative")
    if args.streaming_chunk_padding_days < 0:
        raise ValueError("--streaming-chunk-padding-days must be non-negative")
    if args.cost_stress_multiplier < 1:
        raise ValueError("--cost-stress-multiplier must be at least 1")
    if (
        (args.candidate_admission_report or args.auto_factor_admission)
        and args.skip_feature_correlation
    ):
        raise ValueError(
            "candidate policy validation requires feature correlation; "
            "do not pass --skip-feature-correlation"
        )
    for name in (
        "dataset_memory_budget_gb",
        "dataset_worker_memory_estimate_gb",
        "backtest_memory_budget_gb",
        "full_backtest_memory_gb",
        "yearly_backtest_memory_gb",
        "candidate_policy_validation_memory_gb",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not args.candidate_policy_validation_methods:
        raise ValueError("--candidate-policy-validation-methods must be non-empty")


if __name__ == "__main__":
    main()
