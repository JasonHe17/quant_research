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
from examples.run_score_overlay_validation import (  # noqa: E402
    _parse_args as _parse_score_overlay_validation_args,
    run_score_overlay_validation,
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
    output_dir = Path(
        args.output_dir or _default_output_dir(args.allocator_id, profile)
    )
    allocator_payload = allocator.to_dict()
    if _score_construction(allocator_payload) == "score_overlay":
        validation_args = _score_overlay_validation_args_from_allocator(
            allocator_payload,
            output_dir=output_dir,
            profile=profile,
            dry_run=args.dry_run,
            resume_existing=args.resume_existing,
            job_workers=args.backtest_workers,
        )
        summary = run_score_overlay_validation(validation_args)
    else:
        policy_args = _policy_validation_args_from_allocator(
            allocator_payload,
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


def _score_construction(allocator: dict[str, Any]) -> str:
    return str((allocator.get("score") or {}).get("construction") or "factor_basket")


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


def _score_overlay_validation_args_from_allocator(
    allocator: dict[str, Any],
    *,
    output_dir: Path,
    profile: str,
    dry_run: bool,
    resume_existing: bool,
    job_workers: int,
) -> argparse.Namespace:
    score = allocator["score"]
    execution = allocator["execution_policy"]
    cost_model = allocator["cost_model"]
    data = allocator["data"]
    command = [
        "run_score_overlay_validation.py",
        "--primary-score-dir",
        score["primary_score_dir"],
        "--satellite-score-dir",
        score["satellite_score_dir"],
        "--output-dir",
        str(output_dir),
        "--method-prefix",
        score.get("method_prefix", "overlay"),
        "--overlay-weights",
        *[str(weight) for weight in score["overlay_weights"]],
        "--overlay-mode",
        score.get("overlay_mode", "blend"),
        "--downside-penalty-quantile",
        str(score.get("downside_penalty_quantile", 0.2)),
        "--decision-timing",
        score.get("decision_timing", "all"),
        "--condition-primary-mode",
        score.get("condition_primary_mode", "current"),
        "--catalog-path",
        data.get(
            "catalog_path",
            "../quant_dataset/canonical_store/catalog/quant_research.duckdb",
        ),
        "--profile",
        profile,
        "--years",
        *[str(year) for year in _years_from_partitions(data)],
        "--policy",
        execution.get("policy", "partial_rebalance_daily"),
        "--top-n",
        str(execution["top_n"]),
        "--initial-cash",
        "1000000.0",
        "--commission-bps",
        str(cost_model["commission_bps"]),
        "--slippage-bps",
        str(cost_model["slippage_bps"]),
        "--sell-stamp-tax-bps",
        str(cost_model["sell_stamp_tax_bps"]),
        "--cost-stress-multiplier",
        str(cost_model.get("high_cost_multiplier", 2.0)),
        "--min-commission",
        str(cost_model["min_commission"]),
        "--lot-size",
        str(execution["lot_size"]),
        "--trade-policy",
        execution["trade_policy"],
        "--rebalance-every-n-bars",
        str(execution["rebalance_every_n_bars"]),
        "--policy-min-hold-bars",
        str(execution["min_hold_bars"]),
        "--policy-estimated-cost-bps",
        str(execution["estimated_cost_bps"]),
        "--policy-no-trade-weight-band",
        str(execution["no_trade_weight_band"]),
        "--policy-partial-rebalance-rate",
        str(execution["partial_rebalance_rate"]),
        "--policy-gross-exposure-scale",
        str(execution["gross_exposure_scale"]),
        "--policy-entry-rank",
        str(execution["entry_rank"]),
        "--policy-exit-rank",
        str(execution["exit_rank"]),
        "--policy-max-entries-per-rebalance",
        str(execution["max_entries_per_rebalance"]),
        "--policy-max-exits-per-rebalance",
        str(execution["max_exits_per_rebalance"]),
        "--policy-turnover-budget-pacing",
        str(execution["turnover_budget_pacing"]),
        "--policy-turnover-budget-period",
        execution["turnover_budget_period"],
        "--policy-drawdown-brake-reduced-scale",
        str(execution.get("drawdown_brake_reduced_scale", 0.5)),
        "--optimizer-score-to-edge-bps",
        str(execution["optimizer_score_to_edge_bps"]),
        "--optimizer-min-net-edge-bps",
        str(execution["optimizer_min_net_edge_bps"]),
        "--optimizer-risk-penalty-multiplier",
        str(execution["optimizer_risk_penalty_multiplier"]),
        "--optimizer-target-cap-mode",
        execution.get("optimizer_target_cap_mode", "clip"),
        "--optimizer-weighting",
        execution["optimizer_weighting"],
        "--min-trade-weight",
        str(execution["min_trade_weight"]),
        "--limit-up-bps",
        str(execution["limit_up_bps"]),
        "--limit-down-bps",
        str(execution["limit_down_bps"]),
        "--data-access-mode",
        data["data_access_mode"],
        "--streaming-chunk",
        data["streaming_chunk"],
        "--streaming-chunk-padding-days",
        str(data["streaming_chunk_padding_days"]),
        "--max-full-turnover",
        str((allocator.get("validation") or {}).get("max_full_turnover", 160.0)),
        "--job-workers",
        str(job_workers),
    ]
    if bool(score.get("rank_normalize", True)):
        command.append("--rank-normalize")
    else:
        command.append("--no-rank-normalize")
    condition = score.get("condition")
    if isinstance(condition, dict):
        command.extend(
            [
                "--condition-schedule",
                condition["schedule_path"],
                "--condition-column",
                condition.get("column", "risk_state"),
                "--condition-values",
                *[str(value) for value in condition.get("values", [])],
            ]
        )
    for postprocessor in score.get("postprocessors", []) or []:
        if not isinstance(postprocessor, dict):
            continue
        if postprocessor.get("type") != "optimizer_risk_penalty_join":
            if postprocessor.get("type") != "target_weight_cap_join":
                continue
            command.extend(
                [
                    "--target-weight-cap-dir",
                    postprocessor["cap_dir"],
                    "--target-weight-cap-column",
                    postprocessor.get("cap_column", "max_target_weight"),
                ]
            )
            if postprocessor.get("fill_value") is not None:
                command.extend(
                    [
                        "--target-weight-cap-fill-value",
                        str(postprocessor["fill_value"]),
                    ]
                )
            continue
        command.extend(
            [
                "--optimizer-risk-penalty-dir",
                postprocessor["penalty_dir"],
                "--optimizer-risk-penalty-column",
                postprocessor.get("penalty_column", "optimizer_risk_penalty_bps"),
                "--optimizer-risk-penalty-fill-value",
                str(postprocessor.get("fill_value", 0.0)),
            ]
        )
    optional_float_flags = (
        (
            "max_gross_turnover_per_rebalance",
            "--policy-max-gross-turnover-per-rebalance",
        ),
        ("total_gross_turnover_budget", "--policy-total-gross-turnover-budget"),
        ("cost_pressure_threshold_bps", "--policy-cost-pressure-threshold-bps"),
        (
            "cost_pressure_max_gross_turnover_per_rebalance",
            "--policy-cost-pressure-max-gross-turnover-per-rebalance",
        ),
    )
    for key, flag in optional_float_flags:
        if execution.get(key) is not None:
            command.extend([flag, str(execution[key])])
    if execution.get("cost_pressure_reduced_scale") is not None:
        command.extend(
            [
                "--policy-cost-pressure-reduced-scale",
                str(execution["cost_pressure_reduced_scale"]),
            ]
        )
    if execution.get("gross_exposure_scale_path"):
        command.extend(
            [
                "--policy-gross-exposure-scale-path",
                execution["gross_exposure_scale_path"],
            ]
        )
    if execution.get("exclude_st", True):
        command.append("--exclude-st")
    else:
        command.append("--no-exclude-st")
    if resume_existing:
        command.append("--resume-existing")
    if dry_run:
        command.append("--dry-run")
    return _parse_score_overlay_validation_args(command[1:])


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
