"""Run multi-window validation for a promoted candidate portfolio policy."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.build_factor_risk_gate import build_factor_risk_gate


@dataclass(frozen=True, slots=True)
class ValidationScenario:
    """One candidate-policy validation scenario."""

    name: str
    partition_start: str
    partition_end: str
    start: str
    end: str
    commission_bps: float
    slippage_bps: float
    sell_stamp_tax_bps: float
    min_commission: float
    memory_estimate_gb: float
    description: str


def main() -> None:
    args = _parse_args()
    summary = run_candidate_policy_validation(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def run_candidate_policy_validation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    factor_risk_gate_summary = _prepare_factor_risk_gate(args, output_dir)
    years = args.years or _infer_full_years(Path(args.dataset_dir))
    scenarios = _validation_scenarios(args, years=years)
    commands = {
        scenario.name: _scenario_command(args, scenario)
        for scenario in scenarios
    }
    (output_dir / "commands.json").write_text(
        json.dumps(commands, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        summary = _validation_summary(
            args,
            years=years,
            scenarios=scenarios,
            commands=commands,
            status="dry_run",
            rows=[],
            monthly_rows=[],
            factor_health_rows=[],
            factor_contribution_rows=[],
            factor_risk_gate_summary=factor_risk_gate_summary,
        )
        _write_summary(output_dir, summary)
        return summary
    pending_scenarios = [
        scenario
        for scenario in scenarios
        if not _scenario_outputs_exist(args, scenario)
    ]
    _run_scenarios(args, pending_scenarios, commands, logs_dir)
    rows = _collect_summary_rows(args, scenarios)
    monthly_rows = _collect_monthly_summary_rows(args, scenarios)
    factor_health_rows = _collect_factor_health_summary_rows(args, scenarios)
    factor_contribution_rows = _collect_factor_contribution_summary_rows(args, scenarios)
    _write_summary_csv(output_dir / "validation_summary.csv", rows)
    _write_summary_csv(output_dir / "validation_monthly_summary.csv", monthly_rows)
    _write_summary_csv(output_dir / "validation_factor_health_summary.csv", factor_health_rows)
    _write_summary_csv(
        output_dir / "validation_factor_contribution_summary.csv",
        factor_contribution_rows,
    )
    summary = _validation_summary(
        args,
        years=years,
        scenarios=scenarios,
        commands=commands,
        status="completed",
        rows=rows,
        monthly_rows=monthly_rows,
        factor_health_rows=factor_health_rows,
        factor_contribution_rows=factor_contribution_rows,
        factor_risk_gate_summary=factor_risk_gate_summary,
    )
    _write_summary(output_dir, summary)
    if args.enforce_gates and summary["validation"]["overall_status"] == "fail":
        raise RuntimeError(
            "candidate policy validation gates failed; see "
            f"{output_dir / 'validation_summary.json'}"
        )
    return summary


def _prepare_factor_risk_gate(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    if not args.factor_risk_gate_feature:
        return None
    gate_output_dir = Path(args.factor_risk_gate_output_dir or output_dir / "factor_risk_gate")
    base_schedule = args.factor_risk_gate_base_schedule
    if base_schedule is None:
        base_schedule = args.policy_gross_exposure_scale_path
    original_policy_scale_path = args.policy_gross_exposure_scale_path
    generated_schedule_path = gate_output_dir / "gross_exposure_schedule.csv"
    args.policy_gross_exposure_scale_path = str(generated_schedule_path)
    params = {
        "dataset_dir": args.factor_risk_gate_dataset_dir or args.dataset_dir,
        "feature": args.factor_risk_gate_feature,
        "output_dir": str(gate_output_dir),
        "aggregate": args.factor_risk_gate_aggregate,
        "aggregate_quantile": args.factor_risk_gate_aggregate_quantile,
        "lookback_windows": args.factor_risk_gate_lookback_windows,
        "min_periods": args.factor_risk_gate_min_periods,
        "high_quantile": args.factor_risk_gate_high_quantile,
        "extreme_quantile": args.factor_risk_gate_extreme_quantile,
        "full_scale": args.factor_risk_gate_full_scale,
        "reduced_scale": args.factor_risk_gate_reduced_scale,
        "blocked_scale": args.factor_risk_gate_blocked_scale,
        "warmup_scale": args.factor_risk_gate_warmup_scale,
        "base_schedule": base_schedule,
        "combine_mode": "min",
        "partition_start": args.factor_risk_gate_partition_start,
        "partition_end": args.factor_risk_gate_partition_end,
        "max_partitions": args.factor_risk_gate_max_partitions,
    }
    if args.dry_run:
        return {
            "status": "dry_run",
            "base_policy_gross_exposure_scale_path": original_policy_scale_path,
            "effective_policy_gross_exposure_scale_path": str(generated_schedule_path),
            "params": params,
            "artifacts": {
                "schedule": str(generated_schedule_path),
                "summary": str(gate_output_dir / "summary.json"),
            },
        }
    summary = build_factor_risk_gate(argparse.Namespace(**params))
    summary["status"] = "completed"
    summary["base_policy_gross_exposure_scale_path"] = original_policy_scale_path
    summary["effective_policy_gross_exposure_scale_path"] = str(generated_schedule_path)
    return summary


def _validation_scenarios(
    args: argparse.Namespace,
    *,
    years: list[int],
) -> list[ValidationScenario]:
    if not years:
        raise ValueError("at least one validation year is required")
    first_year = min(years)
    last_year = max(years)
    scenarios = [
        ValidationScenario(
            name="full_base",
            partition_start=f"{first_year}_01",
            partition_end=f"{last_year}_12",
            start=f"{first_year}-01-01T00:00:00+08:00",
            end=f"{last_year}-12-31T23:59:59+08:00",
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            sell_stamp_tax_bps=args.sell_stamp_tax_bps,
            min_commission=args.min_commission,
            memory_estimate_gb=args.full_backtest_memory_gb,
            description="Full-window candidate policy with production-like costs.",
        )
    ]
    if args.profile in {"standard", "robust"}:
        for year in years:
            scenarios.append(
                ValidationScenario(
                    name=f"year_{year}_base",
                    partition_start=f"{year}_01",
                    partition_end=f"{year}_12",
                    start=f"{year}-01-01T00:00:00+08:00",
                    end=f"{year}-12-31T23:59:59+08:00",
                    commission_bps=args.commission_bps,
                    slippage_bps=args.slippage_bps,
                    sell_stamp_tax_bps=args.sell_stamp_tax_bps,
                    min_commission=args.min_commission,
                    memory_estimate_gb=args.yearly_backtest_memory_gb,
                    description=f"Calendar-year stability slice for {year}.",
                )
            )
        scenarios.append(
            ValidationScenario(
                name="full_high_cost",
                partition_start=f"{first_year}_01",
                partition_end=f"{last_year}_12",
                start=f"{first_year}-01-01T00:00:00+08:00",
                end=f"{last_year}-12-31T23:59:59+08:00",
                commission_bps=args.commission_bps * args.cost_stress_multiplier,
                slippage_bps=args.slippage_bps * args.cost_stress_multiplier,
                sell_stamp_tax_bps=args.sell_stamp_tax_bps * args.cost_stress_multiplier,
                min_commission=args.min_commission,
                memory_estimate_gb=args.full_backtest_memory_gb,
                description="Full-window transaction-cost stress.",
            )
        )
    if args.profile == "robust":
        scenarios.append(
            ValidationScenario(
                name="full_zero_cost",
                partition_start=f"{first_year}_01",
                partition_end=f"{last_year}_12",
                start=f"{first_year}-01-01T00:00:00+08:00",
                end=f"{last_year}-12-31T23:59:59+08:00",
                commission_bps=0.0,
                slippage_bps=0.0,
                sell_stamp_tax_bps=0.0,
                min_commission=0.0,
                memory_estimate_gb=args.full_backtest_memory_gb,
                description="Full-window zero-cost diagnostic upper bound.",
            )
        )
    return scenarios


def _scenario_command(
    args: argparse.Namespace,
    scenario: ValidationScenario,
) -> list[str]:
    command = [
        sys.executable,
        str(EXAMPLES_DIR / "run_candidate_factor_portfolios.py"),
        "--dataset-dir",
        args.dataset_dir,
        "--label-column",
        args.label_column,
        "--admission-report",
        args.admission_report,
        "--factor-correlation",
        args.factor_correlation,
        "--registry",
        args.registry,
        "--registry-statuses",
        *args.registry_statuses,
        "--output-dir",
        str(_scenario_output_dir(args, scenario)),
        "--methods",
        *args.methods,
        "--partition-start",
        scenario.partition_start,
        "--partition-end",
        scenario.partition_end,
    ]
    if args.include_features:
        command.extend(["--include-features", *args.include_features])
    command.extend(
        [
        "--run-backtests",
        "--start",
        scenario.start,
        "--end",
        scenario.end,
        "--top-n",
        str(args.top_n),
        "--score-diagnostics-top-n",
        str(args.score_diagnostics_top_n),
        "--initial-cash",
        str(args.initial_cash),
        "--commission-bps",
        str(scenario.commission_bps),
        "--slippage-bps",
        str(scenario.slippage_bps),
        "--sell-stamp-tax-bps",
        str(scenario.sell_stamp_tax_bps),
        "--min-commission",
        str(scenario.min_commission),
        "--lot-size",
        str(args.lot_size),
        "--backtest-policy-set",
        args.backtest_policy_set,
        "--trade-policy",
        args.trade_policy,
        "--rebalance-every-n-bars",
        str(args.rebalance_every_n_bars),
        "--policy-min-hold-bars",
        str(args.policy_min_hold_bars),
        "--policy-estimated-cost-bps",
        str(_resolved_policy_estimated_cost_bps(args, scenario)),
        "--policy-no-trade-weight-band",
        str(args.policy_no_trade_weight_band),
        "--policy-partial-rebalance-rate",
        str(args.policy_partial_rebalance_rate),
        "--policy-set-drop-count",
        str(args.policy_set_drop_count),
        "--policy-set-exit-rank",
        str(args.policy_set_exit_rank),
        "--policy-set-rebalance-every-n-bars",
        str(args.policy_set_rebalance_every_n_bars),
        "--policy-set-partial-rebalance-rate",
        str(args.policy_set_partial_rebalance_rate),
        "--policy-gross-exposure-scale",
        str(args.policy_gross_exposure_scale),
        "--forecast-calibration-mode",
        args.forecast_calibration_mode,
        "--forecast-calibration-lookback-windows",
        str(args.forecast_calibration_lookback_windows),
        "--forecast-calibration-min-periods",
        str(args.forecast_calibration_min_periods),
        "--forecast-calibration-label-lag-windows",
        str(args.forecast_calibration_label_lag_windows),
        "--forecast-calibration-bucket-count",
        str(args.forecast_calibration_bucket_count),
        "--forecast-calibration-risk-multiplier",
        str(args.forecast_calibration_risk_multiplier),
        "--optimizer-score-to-edge-bps",
        str(args.optimizer_score_to_edge_bps),
        "--optimizer-min-net-edge-bps",
        str(args.optimizer_min_net_edge_bps),
        "--optimizer-risk-penalty-multiplier",
        str(args.optimizer_risk_penalty_multiplier),
        "--optimizer-weighting",
        args.optimizer_weighting,
        "--min-trade-weight",
        str(args.min_trade_weight),
        "--limit-up-bps",
        str(args.limit_up_bps),
        "--limit-down-bps",
        str(args.limit_down_bps),
        "--data-access-mode",
        args.data_access_mode,
        "--streaming-chunk",
        args.streaming_chunk,
        "--streaming-chunk-padding-days",
        str(args.streaming_chunk_padding_days),
        "--backtest-workers",
        str(args.backtest_workers),
        "--backtest-memory-budget-gb",
        str(args.backtest_memory_budget_gb),
        "--backtest-memory-estimate-gb",
        str(scenario.memory_estimate_gb),
        ]
    )
    if args.backtest_policies:
        command.extend(["--backtest-policies", *args.backtest_policies])
    if args.enforce_registry:
        command.append("--enforce-registry")
    else:
        command.append("--no-enforce-registry")
    optional_ints = {
        "--hold-rank-buffer": args.hold_rank_buffer,
        "--policy-entry-rank": args.policy_entry_rank,
        "--policy-exit-rank": args.policy_exit_rank,
        "--policy-max-entries-per-rebalance": args.policy_max_entries_per_rebalance,
        "--policy-max-exits-per-rebalance": args.policy_max_exits_per_rebalance,
    }
    for option, value in optional_ints.items():
        if value is not None:
            command.extend([option, str(value)])
    optional_floats = {
        "--policy-min-expected-edge-bps": args.policy_min_expected_edge_bps,
        "--policy-max-gross-turnover-per-rebalance": (
            args.policy_max_gross_turnover_per_rebalance
        ),
        "--policy-total-gross-turnover-budget": args.policy_total_gross_turnover_budget,
        "--policy-turnover-budget-pacing": args.policy_turnover_budget_pacing,
        "--forecast-calibration-max-abs-edge-bps": (
            args.forecast_calibration_max_abs_edge_bps
        ),
        "--max-bar-turnover-participation": args.max_bar_turnover_participation,
    }
    for option, value in optional_floats.items():
        if value is not None:
            command.extend([option, str(value)])
    command.extend(["--policy-turnover-budget-period", args.policy_turnover_budget_period])
    if args.factor_max_weight is not None:
        command.extend(["--factor-max-weight", str(args.factor_max_weight)])
    if args.factor_max_contribution_share is not None:
        command.extend(
            [
                "--factor-max-contribution-share",
                str(args.factor_max_contribution_share),
            ]
        )
    if args.factor_health_mode != "off":
        command.extend(
            [
                "--factor-health-mode",
                args.factor_health_mode,
                "--factor-health-lookback-windows",
                str(args.factor_health_lookback_windows),
                "--factor-health-min-periods",
                str(args.factor_health_min_periods),
                "--factor-health-label-lag-windows",
                str(args.factor_health_label_lag_windows),
                "--factor-health-min-scale",
                str(args.factor_health_min_scale),
                "--factor-health-max-scale",
                str(args.factor_health_max_scale),
                "--factor-health-rank-ic-floor",
                str(args.factor_health_rank_ic_floor),
                "--factor-health-rank-ic-ceiling",
                str(args.factor_health_rank_ic_ceiling),
                "--factor-health-spread-floor",
                str(args.factor_health_spread_floor),
                "--factor-health-spread-ceiling",
                str(args.factor_health_spread_ceiling),
            ]
        )
    if args.optimizer_candidate_rank is not None:
        command.extend(["--optimizer-candidate-rank", str(args.optimizer_candidate_rank)])
    if args.optimizer_max_name_weight is not None:
        command.extend(["--optimizer-max-name-weight", str(args.optimizer_max_name_weight)])
    if args.optimizer_max_gross_exposure_increase_per_rebalance is not None:
        command.extend(
            [
                "--optimizer-max-gross-exposure-increase-per-rebalance",
                str(args.optimizer_max_gross_exposure_increase_per_rebalance),
            ]
        )
    if args.exclude_st:
        command.append("--exclude-st")
    else:
        command.append("--no-exclude-st")
    if args.policy_gross_exposure_scale_path:
        command.extend(
            [
                "--policy-gross-exposure-scale-path",
                args.policy_gross_exposure_scale_path,
            ]
        )
    if args.policy_drawdown_brake_threshold is not None:
        command.extend(
            [
                "--policy-drawdown-brake-threshold",
                str(args.policy_drawdown_brake_threshold),
            ]
        )
    command.extend(
        [
            "--policy-drawdown-brake-reduced-scale",
            str(args.policy_drawdown_brake_reduced_scale),
        ]
    )
    if args.resume_existing:
        command.append("--resume-existing")
    return command


def _resolved_policy_estimated_cost_bps(
    args: argparse.Namespace,
    scenario: ValidationScenario,
) -> float:
    if args.policy_estimated_cost_bps is not None:
        return float(args.policy_estimated_cost_bps)
    return _estimated_round_trip_cost_bps(
        commission_bps=scenario.commission_bps,
        slippage_bps=scenario.slippage_bps,
        sell_stamp_tax_bps=scenario.sell_stamp_tax_bps,
    )


def _estimated_round_trip_cost_bps(
    *,
    commission_bps: float,
    slippage_bps: float,
    sell_stamp_tax_bps: float,
) -> float:
    return float(2.0 * commission_bps + 2.0 * slippage_bps + sell_stamp_tax_bps)


def _scenario_output_dir(args: argparse.Namespace, scenario: ValidationScenario) -> Path:
    return Path(args.output_dir) / scenario.name


def _backtest_output_dir(
    args: argparse.Namespace,
    scenario: ValidationScenario,
    method: str,
) -> Path:
    path = _scenario_output_dir(args, scenario) / "backtests" / method
    if args.backtest_policy_set == "single":
        return path
    return path / args.policy


def _run_scenario(command: list[str], log_path: Path) -> None:
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
            f"candidate policy validation scenario failed with code "
            f"{result.returncode}: see {log_path}"
        )


def _scenario_outputs_exist(args: argparse.Namespace, scenario: ValidationScenario) -> bool:
    if not args.resume_existing:
        return False
    scenario_dir = _scenario_output_dir(args, scenario)
    return (
        (scenario_dir / "summary.json").exists()
        and (scenario_dir / "backtest_summary.csv").exists()
    )


def _run_scenarios(
    args: argparse.Namespace,
    scenarios: list[ValidationScenario],
    commands: dict[str, list[str]],
    logs_dir: Path,
) -> None:
    if not scenarios:
        return
    if args.scenario_workers == 1 or len(scenarios) == 1:
        for scenario in scenarios:
            _run_scenario(commands[scenario.name], logs_dir / f"{scenario.name}.log")
        return
    _run_scenarios_with_budget(
        scenarios,
        commands=commands,
        logs_dir=logs_dir,
        max_workers=args.scenario_workers,
        memory_budget_gb=_effective_scenario_memory_budget_gb(args),
    )


def _run_scenarios_with_budget(
    scenarios: list[ValidationScenario],
    *,
    commands: dict[str, list[str]],
    logs_dir: Path,
    max_workers: int,
    memory_budget_gb: float,
) -> None:
    pending = list(scenarios)
    running: dict[Future[None], ValidationScenario] = {}
    running_memory_gb = 0.0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while pending or running:
            while pending and len(running) < max_workers:
                scenario = pending[0]
                if running_memory_gb + scenario.memory_estimate_gb > memory_budget_gb:
                    break
                pending.pop(0)
                future = executor.submit(
                    _run_scenario,
                    commands[scenario.name],
                    logs_dir / f"{scenario.name}.log",
                )
                running[future] = scenario
                running_memory_gb += scenario.memory_estimate_gb
            if not running:
                scenario = pending[0]
                raise RuntimeError(
                    f"validation scenario {scenario.name} requires an estimated "
                    f"{scenario.memory_estimate_gb:.2f} GB, above the configured "
                    f"scenario memory budget of {memory_budget_gb:.2f} GB"
                )
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                scenario = running.pop(future)
                running_memory_gb -= scenario.memory_estimate_gb
                future.result()


def _effective_scenario_memory_budget_gb(args: argparse.Namespace) -> float:
    if args.scenario_memory_budget_gb > 0:
        return args.scenario_memory_budget_gb
    available = _available_memory_gb()
    max_estimate = max(args.full_backtest_memory_gb, args.yearly_backtest_memory_gb)
    if available is None:
        return max_estimate
    return max(min(available * 0.55, available - 4.0), max_estimate)


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


def _collect_summary_rows(
    args: argparse.Namespace,
    scenarios: list[ValidationScenario],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        path = _scenario_output_dir(args, scenario) / "backtest_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing scenario summary: {path}")
        frame = pd.read_csv(path)
        for record in frame.to_dict("records"):
            record["scenario"] = scenario.name
            record["scenario_start"] = scenario.start
            record["scenario_end"] = scenario.end
            record["commission_bps"] = scenario.commission_bps
            record["slippage_bps"] = scenario.slippage_bps
            record["sell_stamp_tax_bps"] = scenario.sell_stamp_tax_bps
            record["min_commission"] = scenario.min_commission
            record["description"] = scenario.description
            rows.append(record)
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def _collect_monthly_summary_rows(
    args: argparse.Namespace,
    scenarios: list[ValidationScenario],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for method in args.methods:
            backtest_dir = _backtest_output_dir(args, scenario, method)
            rows.extend(
                _monthly_summary_rows_for_backtest(
                    backtest_dir,
                    scenario=scenario,
                    method=method,
                    policy=args.policy,
                    initial_cash=args.initial_cash,
                )
            )
    return rows


def _collect_factor_health_summary_rows(
    args: argparse.Namespace,
    scenarios: list[ValidationScenario],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        summary_path = _scenario_output_dir(args, scenario) / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        schedule_path = payload.get("factor_health_schedule")
        if not schedule_path:
            continue
        path = Path(str(schedule_path))
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        for feature, group in frame.groupby("feature", sort=True):
            rows.append(
                {
                    "scenario": scenario.name,
                    "feature": feature,
                    "observation_count": int(len(group)),
                    "average_weight_scale": float(group["weight_scale"].mean()),
                    "min_weight_scale": float(group["weight_scale"].min()),
                    "lagged_health_shrink_count": int(
                        (group["shrink_reason"] == "lagged_health_shrink").sum()
                    ),
                    "warmup_count": int((group["shrink_reason"] == "warmup").sum()),
                }
            )
    return rows


def _collect_factor_contribution_summary_rows(
    args: argparse.Namespace,
    scenarios: list[ValidationScenario],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        summary_path = _scenario_output_dir(args, scenario) / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        methods = payload.get("methods", {})
        if not isinstance(methods, dict):
            continue
        for method, method_payload in methods.items():
            if not isinstance(method_payload, dict):
                continue
            paths = method_payload.get("factor_contribution_diagnostics", [])
            if not isinstance(paths, list) or not paths:
                continue
            frames = [
                pd.read_csv(path)
                for path in paths
                if isinstance(path, str) and Path(path).exists()
            ]
            if not frames:
                continue
            frame = pd.concat(frames, ignore_index=True)
            if frame.empty:
                continue
            rows.append(
                {
                    "scenario": scenario.name,
                    "method": method,
                    "observation_count": int(len(frame)),
                    "average_largest_abs_contribution_share": float(
                        frame["largest_abs_contribution_share"].mean()
                    ),
                    "max_largest_abs_contribution_share": float(
                        frame["largest_abs_contribution_share"].max()
                    ),
                    "average_top_two_abs_contribution_share": float(
                        frame["top_two_abs_contribution_share"].mean()
                    ),
                }
            )
    return rows


def _monthly_summary_rows_for_backtest(
    backtest_dir: Path,
    *,
    scenario: ValidationScenario,
    method: str,
    policy: str,
    initial_cash: float,
) -> list[dict[str, Any]]:
    equity_path = backtest_dir / "equity_curve.csv"
    trades_path = backtest_dir / "trades.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"missing equity curve: {equity_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"missing trades file: {trades_path}")
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"])
    trades = pd.read_csv(trades_path, parse_dates=["timestamp"])
    start = pd.Timestamp(scenario.start)
    end = pd.Timestamp(scenario.end)
    equity = equity[
        (equity["timestamp"] >= start)
        & (equity["timestamp"] <= end)
    ].copy()
    if equity.empty:
        return []
    equity["month"] = equity["timestamp"].dt.strftime("%Y-%m")
    if trades.empty:
        trades["month"] = pd.Series(dtype="object")
    else:
        trades = trades[
            (trades["timestamp"] >= start)
            & (trades["timestamp"] <= end)
        ].copy()
        trades["month"] = trades["timestamp"].dt.strftime("%Y-%m")
    month_end = equity.groupby("month").tail(1).set_index("month")
    previous_equity = float(initial_cash)
    rows: list[dict[str, Any]] = []
    for month, month_end_row in month_end.iterrows():
        month_equity = equity[equity["month"] == month]
        end_equity = float(month_end_row["equity"])
        curve = pd.concat(
            [
                pd.Series([previous_equity]),
                month_equity["equity"].reset_index(drop=True),
            ],
            ignore_index=True,
        )
        drawdown = float((curve / curve.cummax() - 1.0).min())
        month_trades = trades[trades["month"] == month]
        rows.append(
            {
                "scenario": scenario.name,
                "method": method,
                "policy": policy,
                "month": str(month),
                "return": end_equity / previous_equity - 1.0,
                "end_equity": end_equity,
                "max_drawdown": drawdown,
                "trade_count": int(len(month_trades)),
                "total_transaction_cost": (
                    float(month_trades["total_cost"].sum())
                    if not month_trades.empty
                    else 0.0
                ),
                "gross_traded_notional": (
                    float(month_trades["notional"].abs().sum())
                    if not month_trades.empty
                    else 0.0
                ),
            }
        )
        previous_equity = end_equity
    return rows


def _validation_summary(
    args: argparse.Namespace,
    *,
    years: list[int],
    scenarios: list[ValidationScenario],
    commands: dict[str, list[str]],
    status: str,
    rows: list[dict[str, Any]],
    monthly_rows: list[dict[str, Any]],
    factor_health_rows: list[dict[str, Any]],
    factor_contribution_rows: list[dict[str, Any]],
    factor_risk_gate_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "params": {
            "profile": args.profile,
            "dataset_dir": args.dataset_dir,
            "label_column": args.label_column,
            "admission_report": args.admission_report,
            "factor_correlation": args.factor_correlation,
            "registry": args.registry,
            "enforce_registry": args.enforce_registry,
            "registry_statuses": args.registry_statuses,
            "methods": args.methods,
            "primary_method": args.primary_method,
            "backtest_policy_set": args.backtest_policy_set,
            "backtest_policies": args.backtest_policies,
            "policy": args.policy,
            "years": years,
            "top_n": args.top_n,
            "commission_bps": args.commission_bps,
            "slippage_bps": args.slippage_bps,
            "sell_stamp_tax_bps": args.sell_stamp_tax_bps,
            "min_commission": args.min_commission,
            "cost_stress_multiplier": args.cost_stress_multiplier,
            "trade_policy": args.trade_policy,
            "rebalance_every_n_bars": args.rebalance_every_n_bars,
            "hold_rank_buffer": args.hold_rank_buffer,
            "policy_entry_rank": args.policy_entry_rank,
            "policy_exit_rank": args.policy_exit_rank,
            "policy_max_entries_per_rebalance": (
                args.policy_max_entries_per_rebalance
            ),
            "policy_max_exits_per_rebalance": args.policy_max_exits_per_rebalance,
            "policy_min_hold_bars": args.policy_min_hold_bars,
            "policy_min_expected_edge_bps": args.policy_min_expected_edge_bps,
            "policy_estimated_cost_bps": (
                args.policy_estimated_cost_bps
                if args.policy_estimated_cost_bps is not None
                else "auto_round_trip"
            ),
            "policy_no_trade_weight_band": args.policy_no_trade_weight_band,
            "policy_partial_rebalance_rate": args.policy_partial_rebalance_rate,
            "policy_max_gross_turnover_per_rebalance": (
                args.policy_max_gross_turnover_per_rebalance
            ),
            "policy_total_gross_turnover_budget": args.policy_total_gross_turnover_budget,
            "policy_turnover_budget_period": args.policy_turnover_budget_period,
            "policy_turnover_budget_pacing": args.policy_turnover_budget_pacing,
            "policy_set_drop_count": args.policy_set_drop_count,
            "policy_set_exit_rank": args.policy_set_exit_rank,
            "policy_set_rebalance_every_n_bars": (
                args.policy_set_rebalance_every_n_bars
            ),
            "policy_set_partial_rebalance_rate": (
                args.policy_set_partial_rebalance_rate
            ),
            "policy_gross_exposure_scale": args.policy_gross_exposure_scale,
            "policy_gross_exposure_scale_path": args.policy_gross_exposure_scale_path,
            "policy_drawdown_brake_threshold": args.policy_drawdown_brake_threshold,
            "policy_drawdown_brake_reduced_scale": (
                args.policy_drawdown_brake_reduced_scale
            ),
            "factor_risk_gate_feature": args.factor_risk_gate_feature,
            "factor_risk_gate_dataset_dir": args.factor_risk_gate_dataset_dir,
            "factor_risk_gate_output_dir": args.factor_risk_gate_output_dir,
            "factor_risk_gate_base_schedule": args.factor_risk_gate_base_schedule,
            "factor_risk_gate_aggregate": args.factor_risk_gate_aggregate,
            "factor_risk_gate_aggregate_quantile": (
                args.factor_risk_gate_aggregate_quantile
            ),
            "factor_risk_gate_lookback_windows": (
                args.factor_risk_gate_lookback_windows
            ),
            "factor_risk_gate_min_periods": args.factor_risk_gate_min_periods,
            "factor_risk_gate_high_quantile": args.factor_risk_gate_high_quantile,
            "factor_risk_gate_extreme_quantile": (
                args.factor_risk_gate_extreme_quantile
            ),
            "factor_risk_gate_full_scale": args.factor_risk_gate_full_scale,
            "factor_risk_gate_reduced_scale": args.factor_risk_gate_reduced_scale,
            "factor_risk_gate_blocked_scale": args.factor_risk_gate_blocked_scale,
            "factor_risk_gate_warmup_scale": args.factor_risk_gate_warmup_scale,
            "factor_risk_gate_partition_start": (
                args.factor_risk_gate_partition_start
            ),
            "factor_risk_gate_partition_end": args.factor_risk_gate_partition_end,
            "factor_risk_gate_max_partitions": args.factor_risk_gate_max_partitions,
            "forecast_calibration_mode": args.forecast_calibration_mode,
            "forecast_calibration_lookback_windows": (
                args.forecast_calibration_lookback_windows
            ),
            "forecast_calibration_min_periods": args.forecast_calibration_min_periods,
            "forecast_calibration_label_lag_windows": (
                args.forecast_calibration_label_lag_windows
            ),
            "forecast_calibration_bucket_count": args.forecast_calibration_bucket_count,
            "forecast_calibration_risk_multiplier": (
                args.forecast_calibration_risk_multiplier
            ),
            "forecast_calibration_max_abs_edge_bps": (
                args.forecast_calibration_max_abs_edge_bps
            ),
            "optimizer_candidate_rank": args.optimizer_candidate_rank,
            "optimizer_score_to_edge_bps": args.optimizer_score_to_edge_bps,
            "optimizer_min_net_edge_bps": args.optimizer_min_net_edge_bps,
            "optimizer_risk_penalty_multiplier": args.optimizer_risk_penalty_multiplier,
            "optimizer_weighting": args.optimizer_weighting,
            "optimizer_max_name_weight": args.optimizer_max_name_weight,
            "optimizer_max_gross_exposure_increase_per_rebalance": (
                args.optimizer_max_gross_exposure_increase_per_rebalance
            ),
            "factor_max_weight": args.factor_max_weight,
            "factor_max_contribution_share": args.factor_max_contribution_share,
            "factor_health_mode": args.factor_health_mode,
            "factor_health_lookback_windows": args.factor_health_lookback_windows,
            "factor_health_min_periods": args.factor_health_min_periods,
            "factor_health_label_lag_windows": args.factor_health_label_lag_windows,
            "factor_health_min_scale": args.factor_health_min_scale,
            "factor_health_max_scale": args.factor_health_max_scale,
            "factor_health_rank_ic_floor": args.factor_health_rank_ic_floor,
            "factor_health_rank_ic_ceiling": args.factor_health_rank_ic_ceiling,
            "factor_health_spread_floor": args.factor_health_spread_floor,
            "factor_health_spread_ceiling": args.factor_health_spread_ceiling,
            "score_diagnostics_top_n": args.score_diagnostics_top_n,
            "max_bar_turnover_participation": args.max_bar_turnover_participation,
            "scenario_workers": args.scenario_workers,
            "scenario_memory_budget_gb": args.scenario_memory_budget_gb,
        },
        "factor_risk_gate": factor_risk_gate_summary,
        "scenarios": {
            scenario.name: {
                "partition_start": scenario.partition_start,
                "partition_end": scenario.partition_end,
                "start": scenario.start,
                "end": scenario.end,
                "commission_bps": scenario.commission_bps,
                "slippage_bps": scenario.slippage_bps,
                "sell_stamp_tax_bps": scenario.sell_stamp_tax_bps,
                "min_commission": scenario.min_commission,
                "description": scenario.description,
            }
            for scenario in scenarios
        },
        "commands": commands,
        "results": rows,
        "monthly_results": monthly_rows,
        "factor_health_summary": factor_health_rows,
        "factor_contribution_summary": factor_contribution_rows,
        "validation": _validation_checks(args, rows),
    }


def _validation_checks(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {"overall_status": "pending", "checks": [], "failed_count": 0, "warning_count": 0}
    checks: list[dict[str, Any]] = []
    _add_check(
        checks,
        name="scenario_results_present",
        status="pass" if rows else "fail",
        details={"row_count": len(rows)},
    )
    for row in rows:
        scenario = str(row.get("scenario"))
        method = str(row.get("method"))
        metrics = {
            key: row.get(key)
            for key in (
                "total_return",
                "max_drawdown",
                "gross_turnover",
                "trade_count",
                "total_transaction_cost",
                "final_equity",
            )
        }
        _add_check(
            checks,
            name=f"{scenario}_{method}_metrics_finite",
            status="pass" if _metrics_are_finite(metrics) else "fail",
            details=metrics,
        )
    full_base = _find_row(rows, "full_base", args.primary_method, args.policy)
    full_high_cost = _find_row(rows, "full_high_cost", args.primary_method, args.policy)
    if full_base:
        _add_check(
            checks,
            name="primary_full_base_positive_return",
            status="pass" if _number(full_base.get("total_return")) > 0 else "fail",
            details={"total_return": full_base.get("total_return")},
        )
        _add_check(
            checks,
            name="primary_full_base_turnover_control",
            status=(
                "pass"
                if _number(full_base.get("gross_turnover")) <= args.max_full_turnover
                else "warn"
            ),
            details={
                "gross_turnover": full_base.get("gross_turnover"),
                "max_full_turnover": args.max_full_turnover,
            },
        )
    if full_high_cost:
        _add_check(
            checks,
            name="primary_full_high_cost_positive_return",
            status="pass" if _number(full_high_cost.get("total_return")) > 0 else "fail",
            details={"total_return": full_high_cost.get("total_return")},
        )
        if full_base:
            _add_check(
                checks,
                name="primary_high_cost_costs_not_lower",
                status=(
                    "pass"
                    if _number(full_high_cost.get("total_transaction_cost"))
                    >= _number(full_base.get("total_transaction_cost"))
                    else "warn"
                ),
                details={
                    "full_base_total_transaction_cost": full_base.get(
                        "total_transaction_cost"
                    ),
                    "full_high_cost_total_transaction_cost": full_high_cost.get(
                        "total_transaction_cost"
                    ),
                },
            )
    yearly_primary = [
        row
        for row in rows
        if str(row.get("scenario")).startswith("year_")
        and row.get("method") == args.primary_method
        and row.get("policy") == args.policy
    ]
    negative_years = [
        row["scenario"]
        for row in yearly_primary
        if _number(row.get("total_return")) <= 0
    ]
    if yearly_primary:
        _add_check(
            checks,
            name="primary_yearly_base_positive_returns",
            status="pass" if not negative_years else "warn",
            details={"negative_years": negative_years},
        )
    failed = [check for check in checks if check["status"] == "fail"]
    warned = [check for check in checks if check["status"] == "warn"]
    return {
        "overall_status": "fail" if failed else "warn" if warned else "pass",
        "checks": checks,
        "failed_count": len(failed),
        "warning_count": len(warned),
    }


def _find_row(
    rows: list[dict[str, Any]],
    scenario: str,
    method: str,
    policy: str,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("scenario") == scenario
            and row.get("method") == method
            and row.get("policy") == policy
        ):
            return row
    return None


def _add_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    status: str,
    details: dict[str, Any],
) -> None:
    checks.append({"name": name, "status": status, "details": details})


def _metrics_are_finite(metrics: dict[str, Any]) -> bool:
    values = [_number(value) for value in metrics.values()]
    numeric_values = [value for value in values if value is not None]
    return bool(numeric_values) and all(math.isfinite(value) for value in numeric_values)


def _number(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def _infer_full_years(dataset_dir: Path) -> list[int]:
    months_by_year: dict[int, set[int]] = {}
    for path in dataset_dir.glob("dataset_*.parquet"):
        parts = path.stem.removeprefix("dataset_").split("_")
        if len(parts) != 2:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
        except ValueError:
            continue
        months_by_year.setdefault(year, set()).add(month)
    years = sorted(
        year
        for year, months in months_by_year.items()
        if set(range(1, 13)).issubset(months)
    )
    if not years:
        raise FileNotFoundError(f"no full calendar years found under {dataset_dir}")
    return years


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="runs/framework_v1_acceptance/standard/alpha_dataset",
    )
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument(
        "--admission-report",
        default=(
            "runs/framework_v1_acceptance/standard/factor_admission/"
            "factor_admission_report.json"
        ),
    )
    parser.add_argument(
        "--factor-correlation",
        default="runs/framework_v1_acceptance/standard/factor_evaluation/feature_correlation.csv",
    )
    parser.add_argument("--registry", default="configs/factors/factor_registry.json")
    parser.add_argument(
        "--enforce-registry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only portfolio-test factors that are present in the factor registry",
    )
    parser.add_argument(
        "--registry-statuses",
        nargs="+",
        default=["candidate", "promoted"],
        help="registry statuses eligible for candidate portfolio validation",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/candidate_factor_portfolios/partial_rebalance_validation",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "standard", "robust"),
        default="standard",
        help=(
            "quick runs full_base, standard adds calendar-year base and "
            "full high-cost scenarios, robust also adds full zero-cost"
        ),
    )
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("equal", "ic_weighted", "decorrelated"),
        default=["decorrelated", "equal", "ic_weighted"],
    )
    parser.add_argument(
        "--include-features",
        nargs="+",
        default=[],
        help="optional feature allowlist passed to candidate-factor portfolio runs",
    )
    parser.add_argument("--primary-method", default="decorrelated")
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument(
        "--backtest-policies",
        nargs="+",
        help=(
            "optional subset of generated backtest policy names to run; "
            "--policy remains the primary policy used for validation checks"
        ),
    )
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--score-diagnostics-top-n",
        type=int,
        default=50,
        help="write and summarize top-N factor contribution diagnostics",
    )
    parser.add_argument("--factor-max-weight", type=float)
    parser.add_argument("--factor-max-contribution-share", type=float)
    parser.add_argument(
        "--factor-health-mode",
        choices=("off", "shrink"),
        default="off",
    )
    parser.add_argument("--factor-health-lookback-windows", type=int, default=20)
    parser.add_argument("--factor-health-min-periods", type=int, default=5)
    parser.add_argument("--factor-health-label-lag-windows", type=int)
    parser.add_argument("--factor-health-min-scale", type=float, default=0.25)
    parser.add_argument("--factor-health-max-scale", type=float, default=1.0)
    parser.add_argument("--factor-health-rank-ic-floor", type=float, default=-0.05)
    parser.add_argument("--factor-health-rank-ic-ceiling", type=float, default=0.05)
    parser.add_argument("--factor-health-spread-floor", type=float, default=-0.001)
    parser.add_argument("--factor-health-spread-ceiling", type=float, default=0.001)
    parser.add_argument(
        "--forecast-calibration-mode",
        choices=("off", "score_bucket"),
        default="off",
        help="optional lagged score-bucket forecast calibration",
    )
    parser.add_argument("--forecast-calibration-lookback-windows", type=int, default=20)
    parser.add_argument("--forecast-calibration-min-periods", type=int, default=5)
    parser.add_argument("--forecast-calibration-label-lag-windows", type=int)
    parser.add_argument("--forecast-calibration-bucket-count", type=int, default=5)
    parser.add_argument("--forecast-calibration-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--forecast-calibration-max-abs-edge-bps", type=float)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--cost-stress-multiplier", type=float, default=2.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--min-trade-weight", type=float, default=0.0005)
    parser.add_argument(
        "--exclude-st",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--limit-up-bps", type=float, default=980.0)
    parser.add_argument("--limit-down-bps", type=float, default=980.0)
    parser.add_argument("--policy-no-trade-weight-band", type=float, default=0.002)
    parser.add_argument(
        "--backtest-policy-set",
        choices=("single", "comparison"),
        default="comparison",
        help="run one configured policy or a fixed comparison policy set",
    )
    parser.add_argument(
        "--trade-policy",
        choices=("naive_top_n", "rank_buffer_drop", "cost_aware_optimizer"),
        default="rank_buffer_drop",
    )
    parser.add_argument("--rebalance-every-n-bars", type=int, default=48)
    parser.add_argument("--hold-rank-buffer", type=int)
    parser.add_argument("--policy-entry-rank", type=int)
    parser.add_argument("--policy-exit-rank", type=int)
    parser.add_argument("--policy-max-entries-per-rebalance", type=int)
    parser.add_argument("--policy-max-exits-per-rebalance", type=int)
    parser.add_argument("--policy-min-hold-bars", type=int, default=0)
    parser.add_argument("--policy-min-expected-edge-bps", type=float)
    parser.add_argument("--policy-estimated-cost-bps", type=float)
    parser.add_argument("--policy-partial-rebalance-rate", type=float, default=1.0)
    parser.add_argument("--policy-max-gross-turnover-per-rebalance", type=float)
    parser.add_argument("--policy-total-gross-turnover-budget", type=float)
    parser.add_argument(
        "--policy-turnover-budget-period",
        choices=("path", "year", "month"),
        default="path",
    )
    parser.add_argument("--policy-turnover-budget-pacing", type=float, default=0.0)
    parser.add_argument("--policy-set-drop-count", type=int, default=10)
    parser.add_argument("--policy-set-exit-rank", type=int, default=150)
    parser.add_argument("--policy-set-rebalance-every-n-bars", type=int, default=48)
    parser.add_argument("--policy-set-partial-rebalance-rate", type=float, default=0.5)
    parser.add_argument("--policy-gross-exposure-scale", type=float, default=1.0)
    parser.add_argument("--policy-gross-exposure-scale-path")
    parser.add_argument("--policy-drawdown-brake-threshold", type=float)
    parser.add_argument("--policy-drawdown-brake-reduced-scale", type=float, default=0.5)
    parser.add_argument(
        "--factor-risk-gate-feature",
        help=(
            "build a lagged factor-risk gross-exposure schedule before validation "
            "and pass it to all policy scenarios"
        ),
    )
    parser.add_argument(
        "--factor-risk-gate-dataset-dir",
        help=(
            "optional dataset directory used only for factor-risk gate construction; "
            "defaults to --dataset-dir"
        ),
    )
    parser.add_argument(
        "--factor-risk-gate-output-dir",
        help="optional output directory for the generated factor-risk gate schedule",
    )
    parser.add_argument(
        "--factor-risk-gate-base-schedule",
        help=(
            "optional base schedule to combine with the generated gate; defaults "
            "to --policy-gross-exposure-scale-path when that path is provided"
        ),
    )
    parser.add_argument(
        "--factor-risk-gate-aggregate",
        choices=("mean", "median", "quantile"),
        default="mean",
    )
    parser.add_argument("--factor-risk-gate-aggregate-quantile", type=float, default=0.75)
    parser.add_argument("--factor-risk-gate-lookback-windows", type=int, default=240)
    parser.add_argument("--factor-risk-gate-min-periods", type=int, default=48)
    parser.add_argument("--factor-risk-gate-high-quantile", type=float, default=0.80)
    parser.add_argument("--factor-risk-gate-extreme-quantile", type=float, default=0.95)
    parser.add_argument("--factor-risk-gate-full-scale", type=float, default=1.0)
    parser.add_argument("--factor-risk-gate-reduced-scale", type=float, default=0.5)
    parser.add_argument("--factor-risk-gate-blocked-scale", type=float, default=0.0)
    parser.add_argument("--factor-risk-gate-warmup-scale", type=float, default=1.0)
    parser.add_argument("--factor-risk-gate-partition-start")
    parser.add_argument("--factor-risk-gate-partition-end")
    parser.add_argument("--factor-risk-gate-max-partitions", type=int)
    parser.add_argument("--optimizer-candidate-rank", type=int)
    parser.add_argument("--optimizer-score-to-edge-bps", type=float, default=100.0)
    parser.add_argument("--optimizer-min-net-edge-bps", type=float, default=0.0)
    parser.add_argument("--optimizer-risk-penalty-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--optimizer-weighting",
        choices=("equal", "utility"),
        default="utility",
    )
    parser.add_argument("--optimizer-max-name-weight", type=float)
    parser.add_argument("--optimizer-max-gross-exposure-increase-per-rebalance", type=float)
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
    parser.add_argument("--backtest-workers", type=int, default=2)
    parser.add_argument("--backtest-memory-budget-gb", type=float, default=12.0)
    parser.add_argument("--full-backtest-memory-gb", type=float, default=5.0)
    parser.add_argument("--yearly-backtest-memory-gb", type=float, default=5.0)
    parser.add_argument(
        "--scenario-workers",
        type=int,
        default=1,
        help="maximum number of validation scenarios to run concurrently",
    )
    parser.add_argument(
        "--scenario-memory-budget-gb",
        type=float,
        default=0.0,
        help="memory budget for concurrent scenarios; 0 auto-detects available memory",
    )
    parser.add_argument("--max-bar-turnover-participation", type=float)
    parser.add_argument("--max-full-turnover", type=float, default=160.0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()
    if args.primary_method not in args.methods:
        raise ValueError("--primary-method must be included in --methods")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if not args.label_column:
        raise ValueError("--label-column must be non-empty")
    if args.score_diagnostics_top_n <= 0:
        raise ValueError("--score-diagnostics-top-n must be positive")
    if args.factor_max_weight is not None and not 0 < args.factor_max_weight <= 1:
        raise ValueError("--factor-max-weight must be in (0, 1]")
    if (
        args.factor_max_contribution_share is not None
        and not 0 < args.factor_max_contribution_share <= 1
    ):
        raise ValueError("--factor-max-contribution-share must be in (0, 1]")
    if args.factor_health_lookback_windows <= 0:
        raise ValueError("--factor-health-lookback-windows must be positive")
    if args.factor_health_min_periods <= 0:
        raise ValueError("--factor-health-min-periods must be positive")
    if args.factor_health_min_periods > args.factor_health_lookback_windows:
        raise ValueError(
            "--factor-health-min-periods must be <= --factor-health-lookback-windows"
        )
    if args.factor_health_label_lag_windows is None:
        args.factor_health_label_lag_windows = _default_label_lag_windows(
            args.label_column
        )
    if args.factor_health_label_lag_windows <= 0:
        raise ValueError("--factor-health-label-lag-windows must be positive")
    if not 0 <= args.factor_health_min_scale <= args.factor_health_max_scale <= 1:
        raise ValueError(
            "--factor-health scales must satisfy 0 <= min_scale <= max_scale <= 1"
        )
    if args.factor_health_rank_ic_floor >= args.factor_health_rank_ic_ceiling:
        raise ValueError(
            "--factor-health-rank-ic-floor must be below --factor-health-rank-ic-ceiling"
        )
    if args.factor_health_spread_floor >= args.factor_health_spread_ceiling:
        raise ValueError(
            "--factor-health-spread-floor must be below --factor-health-spread-ceiling"
        )
    if args.forecast_calibration_lookback_windows <= 0:
        raise ValueError("--forecast-calibration-lookback-windows must be positive")
    if args.forecast_calibration_min_periods <= 0:
        raise ValueError("--forecast-calibration-min-periods must be positive")
    if args.forecast_calibration_min_periods > args.forecast_calibration_lookback_windows:
        raise ValueError(
            "--forecast-calibration-min-periods must be <= "
            "--forecast-calibration-lookback-windows"
        )
    if args.forecast_calibration_label_lag_windows is None:
        args.forecast_calibration_label_lag_windows = _default_label_lag_windows(
            args.label_column
        )
    if args.forecast_calibration_label_lag_windows <= 0:
        raise ValueError("--forecast-calibration-label-lag-windows must be positive")
    if args.forecast_calibration_bucket_count <= 1:
        raise ValueError("--forecast-calibration-bucket-count must be greater than 1")
    if args.forecast_calibration_risk_multiplier < 0:
        raise ValueError("--forecast-calibration-risk-multiplier must be non-negative")
    if (
        args.forecast_calibration_max_abs_edge_bps is not None
        and args.forecast_calibration_max_abs_edge_bps <= 0
    ):
        raise ValueError("--forecast-calibration-max-abs-edge-bps must be positive")
    if args.cost_stress_multiplier <= 0:
        raise ValueError("--cost-stress-multiplier must be positive")
    if args.rebalance_every_n_bars <= 0:
        raise ValueError("--rebalance-every-n-bars must be positive")
    if args.policy_entry_rank is not None and args.policy_entry_rank <= 0:
        raise ValueError("--policy-entry-rank must be positive")
    if args.policy_exit_rank is not None and args.policy_exit_rank <= 0:
        raise ValueError("--policy-exit-rank must be positive")
    entry_rank = args.policy_entry_rank or args.top_n
    exit_rank = args.policy_exit_rank or max(args.top_n, args.policy_set_exit_rank)
    if exit_rank < entry_rank:
        raise ValueError("--policy-exit-rank must be greater than or equal to entry rank")
    if (
        args.policy_max_entries_per_rebalance is not None
        and args.policy_max_entries_per_rebalance < 0
    ):
        raise ValueError("--policy-max-entries-per-rebalance must be non-negative")
    if (
        args.policy_max_exits_per_rebalance is not None
        and args.policy_max_exits_per_rebalance < 0
    ):
        raise ValueError("--policy-max-exits-per-rebalance must be non-negative")
    if args.policy_min_hold_bars < 0:
        raise ValueError("--policy-min-hold-bars must be non-negative")
    if (
        args.policy_min_expected_edge_bps is not None
        and args.policy_min_expected_edge_bps < 0
    ):
        raise ValueError("--policy-min-expected-edge-bps must be non-negative")
    if args.policy_estimated_cost_bps is not None and args.policy_estimated_cost_bps < 0:
        raise ValueError("--policy-estimated-cost-bps must be non-negative")
    if args.policy_set_exit_rank < args.top_n:
        raise ValueError("--policy-set-exit-rank must be greater than or equal to --top-n")
    if not 0 < args.policy_partial_rebalance_rate <= 1:
        raise ValueError("--policy-partial-rebalance-rate must be in (0, 1]")
    if (
        args.policy_max_gross_turnover_per_rebalance is not None
        and args.policy_max_gross_turnover_per_rebalance < 0
    ):
        raise ValueError("--policy-max-gross-turnover-per-rebalance must be non-negative")
    if (
        args.policy_total_gross_turnover_budget is not None
        and args.policy_total_gross_turnover_budget < 0
    ):
        raise ValueError("--policy-total-gross-turnover-budget must be non-negative")
    if args.policy_turnover_budget_pacing < 0:
        raise ValueError("--policy-turnover-budget-pacing must be non-negative")
    if not 0 < args.policy_set_partial_rebalance_rate <= 1:
        raise ValueError("--policy-set-partial-rebalance-rate must be in (0, 1]")
    if not 0 <= args.policy_gross_exposure_scale <= 1:
        raise ValueError("--policy-gross-exposure-scale must be in [0, 1]")
    if (
        args.policy_drawdown_brake_threshold is not None
        and not -1 < args.policy_drawdown_brake_threshold < 0
    ):
        raise ValueError("--policy-drawdown-brake-threshold must be in (-1, 0)")
    if not 0 <= args.policy_drawdown_brake_reduced_scale <= 1:
        raise ValueError("--policy-drawdown-brake-reduced-scale must be in [0, 1]")
    if not 0 < args.factor_risk_gate_aggregate_quantile < 1:
        raise ValueError("--factor-risk-gate-aggregate-quantile must be in (0, 1)")
    if args.factor_risk_gate_lookback_windows <= 0:
        raise ValueError("--factor-risk-gate-lookback-windows must be positive")
    if args.factor_risk_gate_min_periods <= 0:
        raise ValueError("--factor-risk-gate-min-periods must be positive")
    if args.factor_risk_gate_min_periods > args.factor_risk_gate_lookback_windows:
        raise ValueError(
            "--factor-risk-gate-min-periods must be <= "
            "--factor-risk-gate-lookback-windows"
        )
    if not (
        0
        < args.factor_risk_gate_high_quantile
        < args.factor_risk_gate_extreme_quantile
        < 1
    ):
        raise ValueError(
            "--factor-risk-gate-high-quantile and "
            "--factor-risk-gate-extreme-quantile must satisfy 0 < high < extreme < 1"
        )
    for name in (
        "factor_risk_gate_full_scale",
        "factor_risk_gate_reduced_scale",
        "factor_risk_gate_blocked_scale",
        "factor_risk_gate_warmup_scale",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if (
        args.factor_risk_gate_partition_start
        and args.factor_risk_gate_partition_end
        and args.factor_risk_gate_partition_start > args.factor_risk_gate_partition_end
    ):
        raise ValueError(
            "--factor-risk-gate-partition-start must not be after "
            "--factor-risk-gate-partition-end"
        )
    if (
        args.factor_risk_gate_max_partitions is not None
        and args.factor_risk_gate_max_partitions <= 0
    ):
        raise ValueError("--factor-risk-gate-max-partitions must be positive")
    if args.optimizer_candidate_rank is not None and args.optimizer_candidate_rank <= 0:
        raise ValueError("--optimizer-candidate-rank must be positive")
    if args.optimizer_score_to_edge_bps < 0:
        raise ValueError("--optimizer-score-to-edge-bps must be non-negative")
    if args.optimizer_min_net_edge_bps < 0:
        raise ValueError("--optimizer-min-net-edge-bps must be non-negative")
    if args.optimizer_risk_penalty_multiplier < 0:
        raise ValueError("--optimizer-risk-penalty-multiplier must be non-negative")
    if (
        args.optimizer_max_name_weight is not None
        and not 0 < args.optimizer_max_name_weight <= 1
    ):
        raise ValueError("--optimizer-max-name-weight must be in (0, 1]")
    if (
        args.optimizer_max_gross_exposure_increase_per_rebalance is not None
        and args.optimizer_max_gross_exposure_increase_per_rebalance < 0
    ):
        raise ValueError(
            "--optimizer-max-gross-exposure-increase-per-rebalance must be non-negative"
        )
    if args.backtest_workers <= 0:
        raise ValueError("--backtest-workers must be positive")
    if args.backtest_memory_budget_gb < 0:
        raise ValueError("--backtest-memory-budget-gb must be non-negative")
    if args.full_backtest_memory_gb <= 0 or args.yearly_backtest_memory_gb <= 0:
        raise ValueError("backtest memory estimates must be positive")
    if args.scenario_workers <= 0:
        raise ValueError("--scenario-workers must be positive")
    if args.scenario_memory_budget_gb < 0:
        raise ValueError("--scenario-memory-budget-gb must be non-negative")
    if (
        args.max_bar_turnover_participation is not None
        and not 0 < args.max_bar_turnover_participation <= 1
    ):
        raise ValueError("--max-bar-turnover-participation must be in (0, 1]")
    if args.max_full_turnover <= 0:
        raise ValueError("--max-full-turnover must be positive")
    return args


def _default_label_lag_windows(label_column: str) -> int:
    suffix = label_column.rsplit("_", 1)[-1]
    if suffix.endswith("b") and suffix[:-1].isdigit():
        return int(suffix[:-1])
    return 48


if __name__ == "__main__":
    main()
