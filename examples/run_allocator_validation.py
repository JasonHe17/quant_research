"""Run or prepare validation from a governed candidate allocator definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_candidate_policy_validation import (  # noqa: E402
    _parse_args as _parse_policy_validation_args,
    run_candidate_policy_validation,
)
from quant_research.factors import load_factor_registry  # noqa: E402
from quant_research.portfolio import (  # noqa: E402
    load_allocator_registry,
    validate_allocator_registry,
)


def main() -> None:
    args = _parse_args()
    summary = run_allocator_validation(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def run_allocator_validation(args: argparse.Namespace) -> dict[str, Any]:
    allocator_registry = load_allocator_registry(args.registry)
    factor_registry = load_factor_registry(args.factor_registry)
    report = validate_allocator_registry(
        allocator_registry,
        factor_registry=factor_registry,
        project_root=PROJECT_ROOT,
    )
    if report.status == "fail":
        raise ValueError("allocator registry has validation errors")
    allocator = allocator_registry.get(args.allocator_id)
    if args.profile == "registered":
        profile = _registered_profile(allocator)
    else:
        profile = args.profile
    output_dir = Path(args.output_dir or _default_output_dir(args.allocator_id, profile))
    policy_args = _policy_validation_args_from_allocator(
        allocator.to_dict(),
        output_dir=output_dir,
        profile=profile,
        dry_run=args.dry_run,
        enforce_gates=args.enforce_gates,
        resume_existing=args.resume_existing,
        scenario_workers=args.scenario_workers,
        scenario_memory_budget_gb=args.scenario_memory_budget_gb,
        backtest_workers=args.backtest_workers,
        backtest_memory_budget_gb=args.backtest_memory_budget_gb,
    )
    summary = run_candidate_policy_validation(policy_args)
    summary["allocator"] = {
        "allocator_id": allocator.allocator_id,
        "registry": str(args.registry),
        "registry_version": allocator_registry.version,
        "profile": profile,
        "validation_status": allocator.validation.get("status"),
    }
    _write_summary(output_dir, summary)
    return summary


def _policy_validation_args_from_allocator(
    allocator: dict[str, Any],
    *,
    output_dir: Path,
    profile: str,
    dry_run: bool,
    enforce_gates: bool,
    resume_existing: bool,
    scenario_workers: int,
    scenario_memory_budget_gb: float,
    backtest_workers: int,
    backtest_memory_budget_gb: float,
) -> argparse.Namespace:
    score = allocator["score"]
    execution = allocator["execution_policy"]
    risk_controls = allocator["risk_controls"]
    cost_model = allocator["cost_model"]
    data = allocator["data"]
    health = risk_controls["factor_health"]
    event_gate = risk_controls["event_state_gate"]
    command = [
        "run_candidate_policy_validation.py",
        "--dataset-dir",
        score["dataset_dir"],
        "--label-column",
        score.get("label_column", "forward_return"),
        "--admission-report",
        score["admission_report"],
        "--factor-correlation",
        score["factor_correlation"],
        "--registry",
        "configs/factors/factor_registry.json",
        "--registry-statuses",
        "candidate",
        "promoted",
        "watchlist",
        "--admission-statuses",
        "candidate",
        "--output-dir",
        str(output_dir),
        "--profile",
        profile,
        "--methods",
        score.get("combination_method", "decorrelated"),
        "--include-features",
        *[str(feature["feature"]) for feature in score["features"]],
        "--primary-method",
        score.get("combination_method", "decorrelated"),
        "--policy",
        execution.get("policy", "partial_rebalance_daily"),
        "--backtest-policies",
        execution.get("policy", "partial_rebalance_daily"),
        "--top-n",
        str(execution["top_n"]),
        "--score-diagnostics-top-n",
        str(execution["top_n"]),
        "--factor-health-mode",
        "shrink",
        "--factor-health-lookback-windows",
        str(health["lookback_windows"]),
        "--factor-health-min-periods",
        str(health["min_periods"]),
        "--factor-health-label-lag-windows",
        str(health["label_lag_windows"]),
        "--factor-health-min-scale",
        str(health["min_scale"]),
        "--factor-health-max-scale",
        str(health["max_scale"]),
        "--factor-health-rank-ic-floor",
        str(health["rank_ic_floor"]),
        "--factor-health-rank-ic-ceiling",
        str(health["rank_ic_ceiling"]),
        "--factor-health-spread-floor",
        str(health["spread_floor"]),
        "--factor-health-spread-ceiling",
        str(health["spread_ceiling"]),
        "--forecast-calibration-mode",
        "off",
        "--initial-cash",
        "1000000.0",
        "--commission-bps",
        str(cost_model["commission_bps"]),
        "--slippage-bps",
        str(cost_model["slippage_bps"]),
        "--sell-stamp-tax-bps",
        str(cost_model["sell_stamp_tax_bps"]),
        "--min-commission",
        str(cost_model["min_commission"]),
        "--cost-stress-multiplier",
        str(cost_model.get("high_cost_multiplier", 1.5)),
        "--lot-size",
        str(execution["lot_size"]),
        "--min-trade-weight",
        str(execution["min_trade_weight"]),
        "--limit-up-bps",
        str(execution["limit_up_bps"]),
        "--limit-down-bps",
        str(execution["limit_down_bps"]),
        "--policy-no-trade-weight-band",
        str(execution["no_trade_weight_band"]),
        "--backtest-policy-set",
        "comparison",
        "--trade-policy",
        execution["trade_policy"],
        "--rebalance-every-n-bars",
        str(execution["rebalance_every_n_bars"]),
        "--policy-entry-rank",
        str(execution["entry_rank"]),
        "--policy-exit-rank",
        str(execution["exit_rank"]),
        "--policy-max-entries-per-rebalance",
        str(execution["max_entries_per_rebalance"]),
        "--policy-max-exits-per-rebalance",
        str(execution["max_exits_per_rebalance"]),
        "--policy-min-hold-bars",
        str(execution["min_hold_bars"]),
        "--policy-estimated-cost-bps",
        str(execution["estimated_cost_bps"]),
        "--policy-partial-rebalance-rate",
        str(execution["partial_rebalance_rate"]),
        "--policy-set-drop-count",
        str(execution["max_entries_per_rebalance"]),
        "--policy-set-exit-rank",
        str(execution["exit_rank"]),
        "--policy-set-rebalance-every-n-bars",
        str(execution["rebalance_every_n_bars"]),
        "--policy-set-partial-rebalance-rate",
        str(execution["partial_rebalance_rate"]),
        "--policy-gross-exposure-scale",
        str(execution["gross_exposure_scale"]),
        "--policy-gross-exposure-scale-path",
        event_gate["schedule_path"],
        "--policy-drawdown-brake-reduced-scale",
        "0.5",
        "--policy-turnover-budget-period",
        execution["turnover_budget_period"],
        "--policy-turnover-budget-pacing",
        str(execution["turnover_budget_pacing"]),
        "--optimizer-score-to-edge-bps",
        str(execution["optimizer_score_to_edge_bps"]),
        "--optimizer-min-net-edge-bps",
        str(execution["optimizer_min_net_edge_bps"]),
        "--optimizer-risk-penalty-multiplier",
        str(execution["optimizer_risk_penalty_multiplier"]),
        "--optimizer-weighting",
        execution["optimizer_weighting"],
        "--data-access-mode",
        data["data_access_mode"],
        "--streaming-chunk",
        data["streaming_chunk"],
        "--streaming-chunk-padding-days",
        str(data["streaming_chunk_padding_days"]),
        "--backtest-workers",
        str(backtest_workers),
        "--backtest-memory-budget-gb",
        str(backtest_memory_budget_gb),
        "--scenario-workers",
        str(scenario_workers),
        "--scenario-memory-budget-gb",
        str(scenario_memory_budget_gb),
        "--years",
        *[str(year) for year in _years_from_partitions(data)],
        "--max-full-turnover",
        "160.0",
    ]
    if execution.get("exclude_st", True):
        command.append("--exclude-st")
    else:
        command.append("--no-exclude-st")
    if resume_existing:
        command.append("--resume-existing")
    if dry_run:
        command.append("--dry-run")
    if enforce_gates:
        command.append("--enforce-gates")
    return _parse_policy_validation_args(command[1:])


def _registered_profile(allocator: Any) -> str:
    validation = allocator.validation
    if validation.get("robust_validation"):
        return "robust"
    if validation.get("standard_validation"):
        return "standard"
    return "quick"


def _years_from_partitions(data: dict[str, Any]) -> list[int]:
    start = str(data["partition_start"]).split("_", maxsplit=1)[0]
    end = str(data["partition_end"]).split("_", maxsplit=1)[0]
    return list(range(int(start), int(end) + 1))


def _default_output_dir(allocator_id: str, profile: str) -> Path:
    return Path("runs/candidate_allocators") / allocator_id / profile


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "allocator_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        help="factor registry used to verify allocator feature references",
    )
    parser.add_argument(
        "--allocator-id",
        default="event_limit_diffusion_complementary_health_shrink_48b",
        help="allocator identifier to run",
    )
    parser.add_argument(
        "--output-dir",
        help="output directory; defaults to runs/candidate_allocators/<allocator>/<profile>",
    )
    parser.add_argument(
        "--profile",
        choices=("registered", "quick", "standard", "robust"),
        default="registered",
        help="validation profile; registered uses the profile implied by the allocator evidence",
    )
    parser.add_argument("--scenario-workers", type=int, default=1)
    parser.add_argument("--scenario-memory-budget-gb", type=float, default=0.0)
    parser.add_argument("--backtest-workers", type=int, default=6)
    parser.add_argument("--backtest-memory-budget-gb", type=float, default=30.0)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
