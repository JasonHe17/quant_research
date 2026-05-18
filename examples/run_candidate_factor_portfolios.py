"""Build and optionally backtest candidate-factor portfolio scores."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.portfolio import (
    CandidateFactor,
    FactorHealthConfig,
    ScoreForecastCalibrationConfig,
    build_factor_health_schedule,
    factor_combination_weights,
    load_candidate_factors,
    write_score_partitions,
)
from quant_research.factors import load_factor_registry


@dataclass(frozen=True, slots=True)
class BacktestPolicySpec:
    """One score-backtest policy configuration used by portfolio experiments."""

    name: str
    trade_policy: str
    rebalance_every_n_bars: int
    hold_rank_buffer: int | None = None
    policy_entry_rank: int | None = None
    policy_exit_rank: int | None = None
    policy_max_entries_per_rebalance: int | None = None
    policy_max_exits_per_rebalance: int | None = None
    policy_min_hold_bars: int = 0
    policy_min_expected_edge_bps: float | None = None
    policy_estimated_cost_bps: float = 0.0
    policy_no_trade_weight_band: float = 0.0
    policy_partial_rebalance_rate: float = 1.0
    policy_max_gross_turnover_per_rebalance: float | None = None
    policy_total_gross_turnover_budget: float | None = None
    policy_turnover_budget_period: str = "path"
    policy_turnover_budget_pacing: float = 0.0
    policy_gross_exposure_scale: float = 1.0
    policy_gross_exposure_scale_path: str | None = None
    policy_drawdown_brake_threshold: float | None = None
    policy_drawdown_brake_reduced_scale: float = 0.5
    optimizer_candidate_rank: int | None = None
    optimizer_score_to_edge_bps: float = 100.0
    optimizer_min_net_edge_bps: float = 0.0
    optimizer_risk_penalty_multiplier: float = 1.0
    optimizer_weighting: str = "utility"
    optimizer_max_name_weight: float | None = None
    optimizer_max_gross_exposure_increase_per_rebalance: float | None = None


@dataclass(frozen=True, slots=True)
class BacktestJob:
    """One score-backtest subprocess scheduled by the portfolio runner."""

    method: str
    policy_name: str
    command: list[str]
    output_dir: Path
    summary_path: Path
    log_path: Path
    memory_estimate_gb: float


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = _dataset_paths(args)
    candidates = load_candidate_factors(
        Path(args.admission_report),
        statuses=tuple(args.statuses),
        include_features=tuple(args.include_features),
    )
    registry_filter = _registry_filter_summary(args, candidates)
    candidates = registry_filter["candidates"]
    correlation = _load_correlation(Path(args.factor_correlation))
    weights_by_method = {
        method: factor_combination_weights(
            candidates,
            method=method,
            correlation=correlation,
            ridge=args.decorrelation_ridge,
        )
        for method in args.methods
    }
    factor_health_schedule = _build_factor_health_schedule(args, dataset_paths, candidates)
    factor_health_path: Path | None = None
    if factor_health_schedule is not None:
        health_dir = output_dir / "factor_health"
        health_dir.mkdir(parents=True, exist_ok=True)
        factor_health_path = health_dir / "factor_health_schedule.csv"
        factor_health_schedule.to_csv(factor_health_path, index=False)
    scores_summary = write_score_partitions(
        dataset_paths,
        output_dir=output_dir / "scores",
        candidates=candidates,
        weights_by_method=weights_by_method,
        max_factor_weight=args.factor_max_weight,
        max_factor_contribution_share=args.factor_max_contribution_share,
        factor_health_schedule=factor_health_schedule,
        diagnostics_top_n=args.score_diagnostics_top_n,
        diagnostics_label_column=args.label_column,
        forecast_calibration_config=_forecast_calibration_config(args),
    )
    summary = {
        "params": _summary_params(args),
        "registry_filter": {
            key: value
            for key, value in registry_filter.items()
            if key != "candidates"
        },
        **scores_summary,
    }
    if factor_health_path is not None:
        summary["factor_health_schedule"] = str(factor_health_path)
    if args.run_backtests:
        backtests = _run_backtests(args, scores_summary=scores_summary)
        summary["backtests"] = backtests
        summary["backtest_summary"] = _backtest_summary_rows(backtests)
        _write_backtest_summary_csv(
            output_dir / "backtest_summary.csv",
            summary["backtest_summary"],
        )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(Path(args.dataset_dir).glob("dataset_*.parquet"))
    if args.partition_start:
        paths = [path for path in paths if _partition_name(path) >= args.partition_start]
    if args.partition_end:
        paths = [path for path in paths if _partition_name(path) <= args.partition_end]
    if args.max_partitions is not None:
        paths = paths[: args.max_partitions]
    if not paths:
        raise FileNotFoundError(f"no dataset_*.parquet files found under {args.dataset_dir}")
    return paths


def _partition_name(path: Path) -> str:
    return path.stem.removeprefix("dataset_")


def _load_correlation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def _summary_params(args: argparse.Namespace) -> dict[str, object]:
    params: dict[str, object] = {
        "dataset_dir": args.dataset_dir,
        "label_column": args.label_column,
        "admission_report": args.admission_report,
        "registry": args.registry,
        "enforce_registry": args.enforce_registry,
        "registry_statuses": args.registry_statuses,
        "factor_correlation": args.factor_correlation,
        "methods": args.methods,
        "statuses": args.statuses,
        "include_features": args.include_features,
        "max_partitions": args.max_partitions,
        "partition_start": args.partition_start,
        "partition_end": args.partition_end,
        "run_backtests": args.run_backtests,
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
        "score_diagnostics_top_n": args.score_diagnostics_top_n,
    }
    if args.run_backtests:
        params["backtest"] = {
            "catalog_path": args.catalog_path,
            "start": args.start,
            "end": args.end,
            "top_n": args.top_n,
            "initial_cash": args.initial_cash,
            "commission_bps": args.commission_bps,
            "slippage_bps": args.slippage_bps,
            "sell_stamp_tax_bps": args.sell_stamp_tax_bps,
            "min_commission": args.min_commission,
            "lot_size": args.lot_size,
            "backtest_policy_set": args.backtest_policy_set,
            "backtest_policies": args.backtest_policies,
            "trade_policy": args.trade_policy,
            "rebalance_every_n_bars": args.rebalance_every_n_bars,
            "hold_rank_buffer": args.hold_rank_buffer,
            "policy_entry_rank": args.policy_entry_rank,
            "policy_exit_rank": args.policy_exit_rank,
            "policy_max_entries_per_rebalance": args.policy_max_entries_per_rebalance,
            "policy_max_exits_per_rebalance": args.policy_max_exits_per_rebalance,
            "policy_min_hold_bars": args.policy_min_hold_bars,
            "policy_min_expected_edge_bps": args.policy_min_expected_edge_bps,
            "policy_estimated_cost_bps": args.policy_estimated_cost_bps,
            "policy_no_trade_weight_band": args.policy_no_trade_weight_band,
            "policy_partial_rebalance_rate": args.policy_partial_rebalance_rate,
            "policy_max_gross_turnover_per_rebalance": (
                args.policy_max_gross_turnover_per_rebalance
            ),
            "policy_total_gross_turnover_budget": args.policy_total_gross_turnover_budget,
            "policy_turnover_budget_period": args.policy_turnover_budget_period,
            "policy_turnover_budget_pacing": args.policy_turnover_budget_pacing,
            "policy_gross_exposure_scale": args.policy_gross_exposure_scale,
            "policy_gross_exposure_scale_path": args.policy_gross_exposure_scale_path,
            "optimizer_candidate_rank": args.optimizer_candidate_rank,
            "optimizer_score_to_edge_bps": args.optimizer_score_to_edge_bps,
            "optimizer_min_net_edge_bps": args.optimizer_min_net_edge_bps,
            "optimizer_risk_penalty_multiplier": args.optimizer_risk_penalty_multiplier,
            "optimizer_weighting": args.optimizer_weighting,
            "optimizer_max_name_weight": args.optimizer_max_name_weight,
            "optimizer_max_gross_exposure_increase_per_rebalance": (
                args.optimizer_max_gross_exposure_increase_per_rebalance
            ),
            "policy_set_drop_count": args.policy_set_drop_count,
            "policy_set_exit_rank": args.policy_set_exit_rank,
            "policy_set_rebalance_every_n_bars": args.policy_set_rebalance_every_n_bars,
            "policy_set_partial_rebalance_rate": args.policy_set_partial_rebalance_rate,
            "min_trade_weight": args.min_trade_weight,
            "exclude_st": args.exclude_st,
            "limit_up_bps": args.limit_up_bps,
            "limit_down_bps": args.limit_down_bps,
            "max_bar_turnover_participation": args.max_bar_turnover_participation,
            "data_access_mode": args.data_access_mode,
            "streaming_chunk": args.streaming_chunk,
            "streaming_chunk_padding_days": args.streaming_chunk_padding_days,
            "backtest_workers": args.backtest_workers,
            "backtest_memory_budget_gb": args.backtest_memory_budget_gb,
            "backtest_memory_estimate_gb": args.backtest_memory_estimate_gb,
            "resume_existing": args.resume_existing,
        }
    return params


def _registry_filter_summary(
    args: argparse.Namespace,
    candidates: tuple[CandidateFactor, ...],
) -> dict[str, object]:
    if not args.enforce_registry:
        return {
            "enabled": False,
            "registry": args.registry,
            "registry_statuses": args.registry_statuses,
            "input_candidate_count": len(candidates),
            "output_candidate_count": len(candidates),
            "included_features": [factor.feature for factor in candidates],
            "excluded_features": [],
            "candidates": candidates,
        }
    registry = load_factor_registry(args.registry)
    feature_status: dict[str, str] = {}
    for entry in registry.entries:
        for feature in entry.feature_columns:
            feature_status[feature] = entry.status
    allowed_statuses = set(args.registry_statuses)
    included: list[CandidateFactor] = []
    excluded: list[dict[str, str]] = []
    for factor in candidates:
        status = feature_status.get(factor.feature)
        if status in allowed_statuses:
            included.append(factor)
        else:
            excluded.append(
                {
                    "feature": factor.feature,
                    "reason": "unregistered"
                    if status is None
                    else f"registry_status={status}",
                }
            )
    if not included:
        raise ValueError(
            "no admission factors remain after registry filtering; "
            f"registry={args.registry}, allowed_statuses={sorted(allowed_statuses)}"
        )
    return {
        "enabled": True,
        "registry": args.registry,
        "registry_statuses": list(args.registry_statuses),
        "input_candidate_count": len(candidates),
        "output_candidate_count": len(included),
        "included_features": [factor.feature for factor in included],
        "excluded_features": excluded,
        "candidates": tuple(included),
    }


def _build_factor_health_schedule(
    args: argparse.Namespace,
    dataset_paths: list[Path],
    candidates: tuple[CandidateFactor, ...],
) -> pd.DataFrame | None:
    if args.factor_health_mode == "off":
        return None
    if args.factor_health_mode != "shrink":
        raise ValueError(f"unsupported factor health mode: {args.factor_health_mode}")
    return build_factor_health_schedule(
        dataset_paths,
        candidates=candidates,
        config=FactorHealthConfig(
            lookback_windows=args.factor_health_lookback_windows,
            min_periods=args.factor_health_min_periods,
            label_lag_windows=args.factor_health_label_lag_windows,
            min_scale=args.factor_health_min_scale,
            max_scale=args.factor_health_max_scale,
            rank_ic_floor=args.factor_health_rank_ic_floor,
            rank_ic_ceiling=args.factor_health_rank_ic_ceiling,
            spread_floor=args.factor_health_spread_floor,
            spread_ceiling=args.factor_health_spread_ceiling,
        ),
        top_n=args.score_diagnostics_top_n or args.top_n,
        label_column=args.label_column,
    )


def _forecast_calibration_config(
    args: argparse.Namespace,
) -> ScoreForecastCalibrationConfig | None:
    if args.forecast_calibration_mode == "off":
        return None
    if args.forecast_calibration_mode != "score_bucket":
        raise ValueError(
            f"unsupported forecast calibration mode: {args.forecast_calibration_mode}"
        )
    return ScoreForecastCalibrationConfig(
        lookback_windows=args.forecast_calibration_lookback_windows,
        min_periods=args.forecast_calibration_min_periods,
        label_lag_windows=args.forecast_calibration_label_lag_windows,
        label_column=args.label_column,
        bucket_count=args.forecast_calibration_bucket_count,
        risk_multiplier=args.forecast_calibration_risk_multiplier,
        max_abs_edge_bps=args.forecast_calibration_max_abs_edge_bps,
    )


def _run_backtests(
    args: argparse.Namespace,
    *,
    scores_summary: dict[str, object],
) -> dict[str, object]:
    if not args.start or not args.end:
        raise ValueError("--start and --end are required with --run-backtests")
    methods = scores_summary["methods"]
    if not isinstance(methods, dict):
        raise ValueError("invalid score summary methods")
    jobs = _backtest_jobs(args, scores_summary=scores_summary)
    pending_jobs = [
        job for job in jobs if not (args.resume_existing and job.summary_path.exists())
    ]
    _run_backtest_jobs(args, pending_jobs)
    return _read_backtest_summaries(args, jobs)


def _backtest_jobs(
    args: argparse.Namespace,
    *,
    scores_summary: dict[str, object],
) -> list[BacktestJob]:
    methods = scores_summary["methods"]
    if not isinstance(methods, dict):
        raise ValueError("invalid score summary methods")
    jobs: list[BacktestJob] = []
    policy_specs = _backtest_policy_specs(args)
    logs_dir = Path(args.output_dir) / "logs"
    for method, payload in methods.items():
        if not isinstance(payload, dict):
            continue
        if args.backtest_policy_set == "single":
            spec = policy_specs[0]
            backtest_dir = Path(args.output_dir) / "backtests" / method
            jobs.append(
                BacktestJob(
                    method=method,
                    policy_name=spec.name,
                    command=_backtest_command(args, str(payload["path"]), backtest_dir, spec),
                    output_dir=backtest_dir,
                    summary_path=backtest_dir / "summary.json",
                    log_path=logs_dir / f"backtest_{method}.log",
                    memory_estimate_gb=args.backtest_memory_estimate_gb,
                )
            )
            continue
        for spec in policy_specs:
            backtest_dir = Path(args.output_dir) / "backtests" / method / spec.name
            jobs.append(
                BacktestJob(
                    method=method,
                    policy_name=spec.name,
                    command=_backtest_command(args, str(payload["path"]), backtest_dir, spec),
                    output_dir=backtest_dir,
                    summary_path=backtest_dir / "summary.json",
                    log_path=logs_dir / f"backtest_{method}_{spec.name}.log",
                    memory_estimate_gb=args.backtest_memory_estimate_gb,
                )
            )
    return jobs


def _run_backtest_jobs(args: argparse.Namespace, jobs: list[BacktestJob]) -> None:
    if not jobs:
        return
    if args.backtest_workers == 1 or len(jobs) == 1:
        for job in jobs:
            _run_backtest_job(job)
        return
    _run_backtest_jobs_with_budget(
        jobs,
        max_workers=args.backtest_workers,
        memory_budget_gb=_effective_backtest_memory_budget_gb(args),
    )


def _run_backtest_job(job: BacktestJob) -> None:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(job.command) + "\n\n")
        log.flush()
        result = subprocess.run(
            job.command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"candidate portfolio backtest failed with code {result.returncode}: "
            f"{job.method}/{job.policy_name} (see {job.log_path})"
        )


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
                if running_memory_gb + job.memory_estimate_gb > memory_budget_gb:
                    break
                pending.pop(0)
                future = executor.submit(_run_backtest_job, job)
                running[future] = job
                running_memory_gb += job.memory_estimate_gb
            if not running:
                job = pending[0]
                raise RuntimeError(
                    f"backtest job {job.method}/{job.policy_name} requires an "
                    f"estimated {job.memory_estimate_gb:.2f} GB, above the "
                    f"configured budget of {memory_budget_gb:.2f} GB"
                )
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                running_memory_gb -= job.memory_estimate_gb
                future.result()


def _effective_backtest_memory_budget_gb(args: argparse.Namespace) -> float:
    if args.backtest_memory_budget_gb > 0:
        return args.backtest_memory_budget_gb
    available = _available_memory_gb()
    if available is None:
        return args.backtest_memory_estimate_gb
    return max(
        min(available * 0.60, available - 2.0),
        args.backtest_memory_estimate_gb,
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


def _read_backtest_summaries(
    args: argparse.Namespace,
    jobs: list[BacktestJob],
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for job in jobs:
        if not job.summary_path.exists():
            raise FileNotFoundError(
                f"missing backtest summary for {job.method}/{job.policy_name}: "
                f"{job.summary_path}"
            )
        payload = json.loads(job.summary_path.read_text(encoding="utf-8"))
        if args.backtest_policy_set == "single":
            rows[job.method] = payload
        else:
            method_rows = rows.setdefault(job.method, {})
            if not isinstance(method_rows, dict):
                raise ValueError(f"invalid backtest rows for method: {job.method}")
            method_rows[job.policy_name] = payload
    return rows


def _backtest_summary_rows(backtests: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, payload in backtests.items():
        if not isinstance(payload, dict):
            continue
        if "metrics" in payload:
            rows.append(_backtest_summary_row(method, "single", payload))
            continue
        for policy_name, policy_payload in payload.items():
            if isinstance(policy_payload, dict):
                rows.append(_backtest_summary_row(method, str(policy_name), policy_payload))
    return rows


def _backtest_summary_row(
    method: str,
    policy_name: str,
    payload: dict[str, object],
) -> dict[str, object]:
    metrics = payload.get("metrics", {})
    params = payload.get("params", {})
    diagnostics = payload.get("policy_diagnostics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(params, dict):
        params = {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return {
        "method": method,
        "policy": policy_name,
        "trade_policy": params.get("trade_policy"),
        "rebalance_every_n_bars": params.get("rebalance_every_n_bars"),
        "policy_entry_rank": params.get("policy_entry_rank"),
        "policy_exit_rank": params.get("policy_exit_rank"),
        "policy_max_entries_per_rebalance": params.get("policy_max_entries_per_rebalance"),
        "policy_max_exits_per_rebalance": params.get("policy_max_exits_per_rebalance"),
        "policy_no_trade_weight_band": params.get("policy_no_trade_weight_band"),
        "policy_partial_rebalance_rate": params.get("policy_partial_rebalance_rate"),
        "policy_total_gross_turnover_budget": params.get(
            "policy_total_gross_turnover_budget"
        ),
        "policy_turnover_budget_pacing": params.get("policy_turnover_budget_pacing"),
        "policy_turnover_budget_period": params.get("policy_turnover_budget_period"),
        "policy_gross_exposure_scale": params.get("policy_gross_exposure_scale"),
        "policy_gross_exposure_scale_path": params.get("policy_gross_exposure_scale_path"),
        "optimizer_candidate_rank": params.get("optimizer_candidate_rank"),
        "optimizer_score_to_edge_bps": params.get("optimizer_score_to_edge_bps"),
        "optimizer_min_net_edge_bps": params.get("optimizer_min_net_edge_bps"),
        "optimizer_risk_penalty_multiplier": params.get(
            "optimizer_risk_penalty_multiplier"
        ),
        "optimizer_weighting": params.get("optimizer_weighting"),
        "optimizer_max_name_weight": params.get("optimizer_max_name_weight"),
        "optimizer_max_gross_exposure_increase_per_rebalance": params.get(
            "optimizer_max_gross_exposure_increase_per_rebalance"
        ),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "gross_turnover": metrics.get("gross_turnover"),
        "trade_count": metrics.get("trade_count"),
        "total_transaction_cost": metrics.get("total_transaction_cost"),
        "final_equity": metrics.get("final_equity"),
        "signal_count": payload.get("signal_count"),
        "execution_row_count": payload.get("execution_row_count"),
        "planned_gross_turnover": diagnostics.get("planned_gross_turnover"),
        "average_target_gross_exposure": diagnostics.get("average_target_gross_exposure"),
        "average_dynamic_turnover_cap": diagnostics.get("average_dynamic_turnover_cap"),
        "turnover_budget_period_count": diagnostics.get("turnover_budget_period_count"),
        "turnover_path_budget_remaining": diagnostics.get(
            "turnover_path_budget_remaining"
        ),
        "gross_exposure_scaled_count": diagnostics.get("gross_exposure_scaled_count"),
        "risk_reduction_count": diagnostics.get("risk_reduction_count"),
        "order_intent_count": diagnostics.get("order_intent_count"),
        "entry_count": diagnostics.get("entry_count"),
        "exit_count": diagnostics.get("exit_count"),
        "hold_count": diagnostics.get("hold_count"),
        "no_trade_count": diagnostics.get("no_trade_count"),
    }


def _write_backtest_summary_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _backtest_policy_specs(args: argparse.Namespace) -> list[BacktestPolicySpec]:
    if args.backtest_policy_set == "single":
        specs = [
            BacktestPolicySpec(
                name="single",
                trade_policy=args.trade_policy,
                rebalance_every_n_bars=args.rebalance_every_n_bars,
                hold_rank_buffer=args.hold_rank_buffer,
                policy_entry_rank=args.policy_entry_rank,
                policy_exit_rank=args.policy_exit_rank,
                policy_max_entries_per_rebalance=args.policy_max_entries_per_rebalance,
                policy_max_exits_per_rebalance=args.policy_max_exits_per_rebalance,
                policy_min_hold_bars=args.policy_min_hold_bars,
                policy_min_expected_edge_bps=args.policy_min_expected_edge_bps,
                policy_estimated_cost_bps=args.policy_estimated_cost_bps,
                policy_no_trade_weight_band=args.policy_no_trade_weight_band,
                policy_partial_rebalance_rate=args.policy_partial_rebalance_rate,
                policy_max_gross_turnover_per_rebalance=(
                    args.policy_max_gross_turnover_per_rebalance
                ),
                policy_total_gross_turnover_budget=args.policy_total_gross_turnover_budget,
                policy_turnover_budget_period=args.policy_turnover_budget_period,
                policy_turnover_budget_pacing=args.policy_turnover_budget_pacing,
                policy_gross_exposure_scale=args.policy_gross_exposure_scale,
                policy_gross_exposure_scale_path=args.policy_gross_exposure_scale_path,
                policy_drawdown_brake_threshold=args.policy_drawdown_brake_threshold,
                policy_drawdown_brake_reduced_scale=(
                    args.policy_drawdown_brake_reduced_scale
                ),
                optimizer_candidate_rank=args.optimizer_candidate_rank,
                optimizer_score_to_edge_bps=args.optimizer_score_to_edge_bps,
                optimizer_min_net_edge_bps=args.optimizer_min_net_edge_bps,
                optimizer_risk_penalty_multiplier=args.optimizer_risk_penalty_multiplier,
                optimizer_weighting=args.optimizer_weighting,
                optimizer_max_name_weight=args.optimizer_max_name_weight,
                optimizer_max_gross_exposure_increase_per_rebalance=(
                    args.optimizer_max_gross_exposure_increase_per_rebalance
                ),
            )
        ]
        return _filter_backtest_policy_specs(specs, args.backtest_policies)
    if args.backtest_policy_set != "comparison":
        raise ValueError(f"unsupported backtest policy set: {args.backtest_policy_set}")
    exit_rank = args.policy_set_exit_rank or args.top_n * 3
    drop_count = args.policy_set_drop_count
    daily_rebalance = args.policy_set_rebalance_every_n_bars
    base = {
        "trade_policy": "rank_buffer_drop",
        "policy_entry_rank": args.top_n,
        "policy_exit_rank": exit_rank,
        "policy_max_entries_per_rebalance": drop_count,
        "policy_max_exits_per_rebalance": drop_count,
        "policy_min_hold_bars": args.policy_min_hold_bars,
        "policy_min_expected_edge_bps": args.policy_min_expected_edge_bps,
        "policy_estimated_cost_bps": args.policy_estimated_cost_bps,
        "policy_no_trade_weight_band": args.policy_no_trade_weight_band,
        "policy_max_gross_turnover_per_rebalance": (
            args.policy_max_gross_turnover_per_rebalance
        ),
        "policy_total_gross_turnover_budget": args.policy_total_gross_turnover_budget,
        "policy_turnover_budget_period": args.policy_turnover_budget_period,
        "policy_turnover_budget_pacing": args.policy_turnover_budget_pacing,
        "policy_gross_exposure_scale": args.policy_gross_exposure_scale,
        "policy_gross_exposure_scale_path": args.policy_gross_exposure_scale_path,
        "policy_drawdown_brake_threshold": args.policy_drawdown_brake_threshold,
        "policy_drawdown_brake_reduced_scale": args.policy_drawdown_brake_reduced_scale,
        "optimizer_candidate_rank": args.optimizer_candidate_rank,
        "optimizer_score_to_edge_bps": args.optimizer_score_to_edge_bps,
        "optimizer_min_net_edge_bps": args.optimizer_min_net_edge_bps,
        "optimizer_risk_penalty_multiplier": args.optimizer_risk_penalty_multiplier,
        "optimizer_weighting": args.optimizer_weighting,
        "optimizer_max_name_weight": args.optimizer_max_name_weight,
        "optimizer_max_gross_exposure_increase_per_rebalance": (
            args.optimizer_max_gross_exposure_increase_per_rebalance
        ),
    }
    specs = [
        BacktestPolicySpec(
            name="naive_top_n_every_bar",
            trade_policy="naive_top_n",
            rebalance_every_n_bars=1,
        ),
        BacktestPolicySpec(
            name="top_k_drop_daily",
            rebalance_every_n_bars=daily_rebalance,
            policy_exit_rank=args.top_n,
            policy_partial_rebalance_rate=1.0,
            **{key: value for key, value in base.items() if key != "policy_exit_rank"},
        ),
        BacktestPolicySpec(
            name="entry_exit_buffer_every_bar",
            rebalance_every_n_bars=1,
            policy_partial_rebalance_rate=1.0,
            **base,
        ),
        BacktestPolicySpec(
            name="entry_exit_buffer_daily",
            rebalance_every_n_bars=daily_rebalance,
            policy_partial_rebalance_rate=1.0,
            **base,
        ),
        BacktestPolicySpec(
            name="partial_rebalance_daily",
            rebalance_every_n_bars=daily_rebalance,
            policy_partial_rebalance_rate=args.policy_set_partial_rebalance_rate,
            **base,
        ),
    ]
    return _filter_backtest_policy_specs(specs, args.backtest_policies)


def _filter_backtest_policy_specs(
    specs: list[BacktestPolicySpec],
    selected: list[str] | None,
) -> list[BacktestPolicySpec]:
    if not selected:
        return specs
    known = {spec.name for spec in specs}
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(
            "unknown backtest policies: "
            + ", ".join(unknown)
            + f"; available policies: {', '.join(sorted(known))}"
        )
    selected_names = set(selected)
    return [spec for spec in specs if spec.name in selected_names]


def _backtest_command(
    args: argparse.Namespace,
    predictions_path: str,
    backtest_dir: Path,
    spec: BacktestPolicySpec,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "examples" / "run_tree_score_backtest.py"),
        "--predictions-path",
        predictions_path,
        "--catalog-path",
        args.catalog_path,
        "--start",
        args.start,
        "--end",
        args.end,
        "--top-n",
        str(args.top_n),
        "--initial-cash",
        str(args.initial_cash),
        "--commission-bps",
        str(args.commission_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--sell-stamp-tax-bps",
        str(args.sell_stamp_tax_bps),
        "--min-commission",
        str(args.min_commission),
        "--lot-size",
        str(args.lot_size),
        "--trade-policy",
        spec.trade_policy,
        "--rebalance-every-n-bars",
        str(spec.rebalance_every_n_bars),
        "--policy-min-hold-bars",
        str(spec.policy_min_hold_bars),
        "--policy-estimated-cost-bps",
        str(spec.policy_estimated_cost_bps),
        "--policy-no-trade-weight-band",
        str(spec.policy_no_trade_weight_band),
        "--policy-partial-rebalance-rate",
        str(spec.policy_partial_rebalance_rate),
        "--policy-gross-exposure-scale",
        str(spec.policy_gross_exposure_scale),
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
        "--output-dir",
        str(backtest_dir),
    ]
    optional_ints = {
        "--hold-rank-buffer": spec.hold_rank_buffer,
        "--policy-entry-rank": spec.policy_entry_rank,
        "--policy-exit-rank": spec.policy_exit_rank,
        "--policy-max-entries-per-rebalance": spec.policy_max_entries_per_rebalance,
        "--policy-max-exits-per-rebalance": spec.policy_max_exits_per_rebalance,
    }
    for option, value in optional_ints.items():
        if value is not None:
            command.extend([option, str(value)])
    optional_floats = {
        "--policy-min-expected-edge-bps": spec.policy_min_expected_edge_bps,
        "--policy-max-gross-turnover-per-rebalance": (
            spec.policy_max_gross_turnover_per_rebalance
        ),
        "--policy-total-gross-turnover-budget": args.policy_total_gross_turnover_budget,
        "--policy-turnover-budget-pacing": args.policy_turnover_budget_pacing,
        "--max-bar-turnover-participation": args.max_bar_turnover_participation,
    }
    for option, value in optional_floats.items():
        if value is not None:
            command.extend([option, str(value)])
    if spec.policy_gross_exposure_scale_path is not None:
        command.extend(
            [
                "--policy-gross-exposure-scale-path",
                spec.policy_gross_exposure_scale_path,
            ]
        )
    if spec.policy_drawdown_brake_threshold is not None:
        command.extend(
            [
                "--policy-drawdown-brake-threshold",
                str(spec.policy_drawdown_brake_threshold),
            ]
        )
    command.extend(
        [
            "--policy-drawdown-brake-reduced-scale",
            str(spec.policy_drawdown_brake_reduced_scale),
        ]
    )
    command.extend(["--policy-turnover-budget-period", args.policy_turnover_budget_period])
    if spec.optimizer_candidate_rank is not None:
        command.extend(["--optimizer-candidate-rank", str(spec.optimizer_candidate_rank)])
    command.extend(["--optimizer-score-to-edge-bps", str(spec.optimizer_score_to_edge_bps)])
    command.extend(["--optimizer-min-net-edge-bps", str(spec.optimizer_min_net_edge_bps)])
    command.extend(
        [
            "--optimizer-risk-penalty-multiplier",
            str(spec.optimizer_risk_penalty_multiplier),
        ]
    )
    command.extend(["--optimizer-weighting", spec.optimizer_weighting])
    if spec.optimizer_max_name_weight is not None:
        command.extend(["--optimizer-max-name-weight", str(spec.optimizer_max_name_weight)])
    if spec.optimizer_max_gross_exposure_increase_per_rebalance is not None:
        command.extend(
            [
                "--optimizer-max-gross-exposure-increase-per-rebalance",
                str(spec.optimizer_max_gross_exposure_increase_per_rebalance),
            ]
        )
    if args.exclude_st:
        command.append("--exclude-st")
    return command


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
        "--registry",
        default="configs/factors/factor_registry.json",
        help="factor registry used to filter admission candidates before portfolio testing",
    )
    parser.add_argument(
        "--enforce-registry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require admission candidates to be registered with an allowed lifecycle status",
    )
    parser.add_argument(
        "--registry-statuses",
        nargs="+",
        default=["candidate", "promoted"],
        help="registry statuses eligible for portfolio-level testing",
    )
    parser.add_argument(
        "--factor-correlation",
        default="runs/framework_v1_acceptance/standard/factor_evaluation/feature_correlation.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/candidate_factor_portfolios",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("equal", "ic_weighted", "decorrelated"),
        default=["equal", "ic_weighted", "decorrelated"],
    )
    parser.add_argument("--statuses", nargs="+", default=["candidate"])
    parser.add_argument(
        "--include-features",
        nargs="+",
        default=[],
        help="optional factor feature allowlist for shared admission reports",
    )
    parser.add_argument("--max-partitions", type=int)
    parser.add_argument(
        "--partition-start",
        help="first dataset partition to include, for example 2023_04",
    )
    parser.add_argument(
        "--partition-end",
        help="last dataset partition to include, for example 2023_06",
    )
    parser.add_argument("--decorrelation-ridge", type=float, default=0.05)
    parser.add_argument(
        "--factor-max-weight",
        type=float,
        help="cap each normalized factor weight before composite score construction",
    )
    parser.add_argument(
        "--factor-max-contribution-share",
        type=float,
        help="cap each row-level factor contribution as a share of absolute contribution",
    )
    parser.add_argument(
        "--factor-health-mode",
        choices=("off", "shrink"),
        default="off",
        help="optional lagged rolling factor-leg health shrinkage",
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
    parser.add_argument(
        "--score-diagnostics-top-n",
        type=int,
        help="write per-partition top-N factor contribution diagnostics",
    )
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument(
        "--catalog-path",
        default="../quant_dataset/canonical_store/catalog/quant_research.duckdb",
    )
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument(
        "--trade-policy",
        choices=("naive_top_n", "rank_buffer_drop", "cost_aware_optimizer"),
        default="naive_top_n",
    )
    parser.add_argument("--rebalance-every-n-bars", type=int, default=1)
    parser.add_argument("--hold-rank-buffer", type=int)
    parser.add_argument("--policy-entry-rank", type=int)
    parser.add_argument("--policy-exit-rank", type=int)
    parser.add_argument("--policy-max-entries-per-rebalance", type=int)
    parser.add_argument("--policy-max-exits-per-rebalance", type=int)
    parser.add_argument("--policy-min-hold-bars", type=int, default=0)
    parser.add_argument("--policy-min-expected-edge-bps", type=float)
    parser.add_argument("--policy-estimated-cost-bps", type=float, default=0.0)
    parser.add_argument("--policy-no-trade-weight-band", type=float, default=0.0)
    parser.add_argument("--policy-partial-rebalance-rate", type=float, default=1.0)
    parser.add_argument("--policy-max-gross-turnover-per-rebalance", type=float)
    parser.add_argument("--policy-total-gross-turnover-budget", type=float)
    parser.add_argument(
        "--policy-turnover-budget-period",
        choices=("path", "year", "month"),
        default="path",
    )
    parser.add_argument("--policy-turnover-budget-pacing", type=float, default=0.0)
    parser.add_argument("--policy-gross-exposure-scale", type=float, default=1.0)
    parser.add_argument("--policy-gross-exposure-scale-path")
    parser.add_argument("--policy-drawdown-brake-threshold", type=float)
    parser.add_argument("--policy-drawdown-brake-reduced-scale", type=float, default=0.5)
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
        "--backtest-policy-set",
        choices=("single", "comparison"),
        default="single",
        help=(
            "Backtest one configured policy or a fixed comparison set covering "
            "naive top-N, top-k-drop, entry/exit buffer, daily rebalance, and "
            "partial rebalance."
        ),
    )
    parser.add_argument(
        "--backtest-policies",
        nargs="+",
        help=(
            "optional subset of generated policy names to backtest, for example "
            "top_k_drop_daily partial_rebalance_daily"
        ),
    )
    parser.add_argument("--policy-set-drop-count", type=int, default=10)
    parser.add_argument("--policy-set-exit-rank", type=int)
    parser.add_argument("--policy-set-rebalance-every-n-bars", type=int, default=48)
    parser.add_argument("--policy-set-partial-rebalance-rate", type=float, default=0.5)
    parser.add_argument(
        "--backtest-workers",
        type=int,
        default=1,
        help="maximum number of score-backtest subprocesses to run concurrently",
    )
    parser.add_argument(
        "--backtest-memory-budget-gb",
        type=float,
        default=0.0,
        help="memory budget for concurrent backtests; 0 auto-detects available memory",
    )
    parser.add_argument(
        "--backtest-memory-estimate-gb",
        type=float,
        default=5.0,
        help="estimated memory footprint for each score-backtest subprocess",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="skip backtests whose summary.json already exists",
    )
    parser.add_argument("--min-trade-weight", type=float, default=0.0005)
    parser.add_argument("--exclude-st", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-up-bps", type=float, default=980.0)
    parser.add_argument("--limit-down-bps", type=float, default=980.0)
    parser.add_argument("--max-bar-turnover-participation", type=float)
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
    args = parser.parse_args()
    if args.max_partitions is not None and args.max_partitions <= 0:
        raise ValueError("--max-partitions must be positive")
    if not args.label_column:
        raise ValueError("--label-column must be non-empty")
    if args.partition_start and args.partition_end and args.partition_start > args.partition_end:
        raise ValueError("--partition-start must not be after --partition-end")
    if args.decorrelation_ridge < 0:
        raise ValueError("--decorrelation-ridge must be non-negative")
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
    if args.score_diagnostics_top_n is not None and args.score_diagnostics_top_n <= 0:
        raise ValueError("--score-diagnostics-top-n must be positive")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.rebalance_every_n_bars <= 0:
        raise ValueError("--rebalance-every-n-bars must be positive")
    if args.hold_rank_buffer is not None and args.hold_rank_buffer < args.top_n:
        raise ValueError("--hold-rank-buffer must be greater than or equal to --top-n")
    if args.policy_entry_rank is not None and args.policy_entry_rank <= 0:
        raise ValueError("--policy-entry-rank must be positive")
    if args.policy_exit_rank is not None and args.policy_exit_rank <= 0:
        raise ValueError("--policy-exit-rank must be positive")
    entry_rank = args.policy_entry_rank or args.top_n
    exit_rank = args.policy_exit_rank or max(args.top_n, args.hold_rank_buffer or args.top_n)
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
    if args.policy_estimated_cost_bps < 0:
        raise ValueError("--policy-estimated-cost-bps must be non-negative")
    if args.policy_no_trade_weight_band < 0:
        raise ValueError("--policy-no-trade-weight-band must be non-negative")
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
    if not 0 <= args.policy_gross_exposure_scale <= 1:
        raise ValueError("--policy-gross-exposure-scale must be in [0, 1]")
    if (
        args.policy_drawdown_brake_threshold is not None
        and not -1 < args.policy_drawdown_brake_threshold < 0
    ):
        raise ValueError("--policy-drawdown-brake-threshold must be in (-1, 0)")
    if not 0 <= args.policy_drawdown_brake_reduced_scale <= 1:
        raise ValueError("--policy-drawdown-brake-reduced-scale must be in [0, 1]")
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
    if args.policy_set_drop_count <= 0:
        raise ValueError("--policy-set-drop-count must be positive")
    if args.policy_set_exit_rank is not None and args.policy_set_exit_rank < args.top_n:
        raise ValueError("--policy-set-exit-rank must be greater than or equal to --top-n")
    if args.policy_set_rebalance_every_n_bars <= 0:
        raise ValueError("--policy-set-rebalance-every-n-bars must be positive")
    if not 0 < args.policy_set_partial_rebalance_rate <= 1:
        raise ValueError("--policy-set-partial-rebalance-rate must be in (0, 1]")
    if args.backtest_workers <= 0:
        raise ValueError("--backtest-workers must be positive")
    if args.backtest_memory_budget_gb < 0:
        raise ValueError("--backtest-memory-budget-gb must be non-negative")
    if args.backtest_memory_estimate_gb <= 0:
        raise ValueError("--backtest-memory-estimate-gb must be positive")
    if args.commission_bps < 0:
        raise ValueError("--commission-bps must be non-negative")
    if args.slippage_bps < 0:
        raise ValueError("--slippage-bps must be non-negative")
    if args.sell_stamp_tax_bps < 0:
        raise ValueError("--sell-stamp-tax-bps must be non-negative")
    if args.min_commission < 0:
        raise ValueError("--min-commission must be non-negative")
    if args.lot_size <= 0:
        raise ValueError("--lot-size must be positive")
    if not 0 <= args.min_trade_weight <= 1:
        raise ValueError("--min-trade-weight must be in [0, 1]")
    if args.limit_up_bps is not None and args.limit_up_bps <= 0:
        raise ValueError("--limit-up-bps must be positive")
    if args.limit_down_bps is not None and args.limit_down_bps <= 0:
        raise ValueError("--limit-down-bps must be positive")
    if (
        args.max_bar_turnover_participation is not None
        and not 0 < args.max_bar_turnover_participation <= 1
    ):
        raise ValueError("--max-bar-turnover-participation must be in (0, 1]")
    if args.streaming_chunk_padding_days < 0:
        raise ValueError("--streaming-chunk-padding-days must be non-negative")
    return args


def _default_label_lag_windows(label_column: str) -> int:
    suffix = label_column.rsplit("_", 1)[-1]
    if suffix.endswith("b") and suffix[:-1].isdigit():
        return int(suffix[:-1])
    return 48


if __name__ == "__main__":
    main()
