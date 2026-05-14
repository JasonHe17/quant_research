"""Run multi-window validation for a promoted candidate portfolio policy."""

from __future__ import annotations

import argparse
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
        )
        _write_summary(output_dir, summary)
        return summary
    for scenario in scenarios:
        scenario_dir = _scenario_output_dir(args, scenario)
        if (
            args.resume_existing
            and (scenario_dir / "summary.json").exists()
            and (scenario_dir / "backtest_summary.csv").exists()
        ):
            continue
        _run_scenario(commands[scenario.name], logs_dir / f"{scenario.name}.log")
    rows = _collect_summary_rows(args, scenarios)
    _write_summary_csv(output_dir / "validation_summary.csv", rows)
    summary = _validation_summary(
        args,
        years=years,
        scenarios=scenarios,
        commands=commands,
        status="completed",
        rows=rows,
    )
    _write_summary(output_dir, summary)
    if args.enforce_gates and summary["validation"]["overall_status"] == "fail":
        raise RuntimeError(
            "candidate policy validation gates failed; see "
            f"{output_dir / 'validation_summary.json'}"
        )
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
        "--admission-report",
        args.admission_report,
        "--factor-correlation",
        args.factor_correlation,
        "--output-dir",
        str(_scenario_output_dir(args, scenario)),
        "--methods",
        *args.methods,
        "--partition-start",
        scenario.partition_start,
        "--partition-end",
        scenario.partition_end,
        "--run-backtests",
        "--start",
        scenario.start,
        "--end",
        scenario.end,
        "--top-n",
        str(args.top_n),
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
        "comparison",
        "--backtest-policies",
        args.policy,
        "--policy-no-trade-weight-band",
        str(args.policy_no_trade_weight_band),
        "--policy-set-drop-count",
        str(args.policy_set_drop_count),
        "--policy-set-exit-rank",
        str(args.policy_set_exit_rank),
        "--policy-set-rebalance-every-n-bars",
        str(args.policy_set_rebalance_every_n_bars),
        "--policy-set-partial-rebalance-rate",
        str(args.policy_set_partial_rebalance_rate),
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
    if args.exclude_st:
        command.append("--exclude-st")
    else:
        command.append("--no-exclude-st")
    if args.resume_existing:
        command.append("--resume-existing")
    return command


def _scenario_output_dir(args: argparse.Namespace, scenario: ValidationScenario) -> Path:
    return Path(args.output_dir) / scenario.name


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


def _validation_summary(
    args: argparse.Namespace,
    *,
    years: list[int],
    scenarios: list[ValidationScenario],
    commands: dict[str, list[str]],
    status: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "params": {
            "profile": args.profile,
            "dataset_dir": args.dataset_dir,
            "admission_report": args.admission_report,
            "factor_correlation": args.factor_correlation,
            "methods": args.methods,
            "primary_method": args.primary_method,
            "policy": args.policy,
            "years": years,
            "top_n": args.top_n,
            "commission_bps": args.commission_bps,
            "slippage_bps": args.slippage_bps,
            "sell_stamp_tax_bps": args.sell_stamp_tax_bps,
            "min_commission": args.min_commission,
            "cost_stress_multiplier": args.cost_stress_multiplier,
            "policy_no_trade_weight_band": args.policy_no_trade_weight_band,
            "policy_set_drop_count": args.policy_set_drop_count,
            "policy_set_exit_rank": args.policy_set_exit_rank,
            "policy_set_rebalance_every_n_bars": (
                args.policy_set_rebalance_every_n_bars
            ),
            "policy_set_partial_rebalance_rate": (
                args.policy_set_partial_rebalance_rate
            ),
        },
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
    parser.add_argument("--primary-method", default="decorrelated")
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument("--top-n", type=int, default=50)
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
    parser.add_argument("--policy-set-drop-count", type=int, default=10)
    parser.add_argument("--policy-set-exit-rank", type=int, default=150)
    parser.add_argument("--policy-set-rebalance-every-n-bars", type=int, default=48)
    parser.add_argument("--policy-set-partial-rebalance-rate", type=float, default=0.5)
    parser.add_argument(
        "--data-access-mode",
        choices=("data_portal", "fast_parquet"),
        default="fast_parquet",
    )
    parser.add_argument(
        "--streaming-chunk",
        choices=("year", "month"),
        default="month",
    )
    parser.add_argument("--streaming-chunk-padding-days", type=int, default=10)
    parser.add_argument("--backtest-workers", type=int, default=2)
    parser.add_argument("--backtest-memory-budget-gb", type=float, default=12.0)
    parser.add_argument("--full-backtest-memory-gb", type=float, default=5.0)
    parser.add_argument("--yearly-backtest-memory-gb", type=float, default=5.0)
    parser.add_argument("--max-full-turnover", type=float, default=160.0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()
    if args.primary_method not in args.methods:
        raise ValueError("--primary-method must be included in --methods")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.cost_stress_multiplier <= 0:
        raise ValueError("--cost-stress-multiplier must be positive")
    if args.policy_set_exit_rank < args.top_n:
        raise ValueError("--policy-set-exit-rank must be greater than or equal to --top-n")
    if not 0 < args.policy_set_partial_rebalance_rate <= 1:
        raise ValueError("--policy-set-partial-rebalance-rate must be in (0, 1]")
    if args.backtest_workers <= 0:
        raise ValueError("--backtest-workers must be positive")
    if args.backtest_memory_budget_gb < 0:
        raise ValueError("--backtest-memory-budget-gb must be non-negative")
    if args.full_backtest_memory_gb <= 0 or args.yearly_backtest_memory_gb <= 0:
        raise ValueError("backtest memory estimates must be positive")
    if args.max_full_turnover <= 0:
        raise ValueError("--max-full-turnover must be positive")
    return args


if __name__ == "__main__":
    main()
