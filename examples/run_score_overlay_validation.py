"""Validate a controlled overlay between two prebuilt score streams."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    start: str
    end: str
    partition_glob: str
    commission_bps: float
    slippage_bps: float
    sell_stamp_tax_bps: float
    estimated_cost_bps: float
    description: str


@dataclass(frozen=True, slots=True)
class BacktestJob:
    scenario: Scenario
    method: str
    command: list[str]
    output_dir: Path
    summary_path: Path
    log_path: Path


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = _build_overlay_scores(args, output_dir=output_dir)
    scenarios = _scenarios(args)
    commands = _run_backtests(args, output_dir=output_dir, methods=methods, scenarios=scenarios)
    rows = _collect_rows(
        methods=methods,
        scenarios=scenarios,
        output_dir=output_dir,
        policy=args.policy,
    )
    summary = {
        "params": _summary_params(args),
        "methods": methods,
        "scenarios": {scenario.name: asdict(scenario) for scenario in scenarios},
        "commands": commands,
        "results": rows,
        "policy_leaderboard": _leaderboard(rows, policy=args.policy),
        "validation": _validation(rows, policy=args.policy, max_full_turnover=args.max_full_turnover),
        "status": "completed",
    }
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def _build_overlay_scores(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    score_root = output_dir / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    primary_paths = _score_paths(Path(args.primary_score_dir))
    satellite_paths = _score_paths(Path(args.satellite_score_dir))
    condition = _condition_lookup(args)
    methods: dict[str, Any] = {}
    for weight in args.overlay_weights:
        if args.overlay_mode == "optimizer_risk_penalty":
            if weight < 0:
                raise ValueError("optimizer_risk_penalty weights must be non-negative")
        elif not 0 <= weight <= 1:
            raise ValueError("overlay weights must be in [0, 1]")
        method = f"{args.method_prefix}_{_weight_label(weight)}"
        method_dir = score_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume_existing:
            for old_path in method_dir.glob("score_*.parquet"):
                old_path.unlink()
        row_count = 0
        partition_count = 0
        for partition, primary_path in primary_paths.items():
            output_path = method_dir / f"score_{partition}.parquet"
            if args.resume_existing and output_path.exists():
                existing = pd.read_parquet(output_path, columns=["score"])
                row_count += int(len(existing))
                partition_count += 1
                continue
            satellite_path = satellite_paths.get(partition)
            if satellite_path is None:
                raise FileNotFoundError(f"missing satellite score partition: {partition}")
            scores = _overlay_partition(
                primary_path,
                satellite_path,
                overlay_weight=weight,
                condition=condition,
                rank_normalize=args.rank_normalize,
                decision_timing=args.decision_timing,
                condition_primary_mode=args.condition_primary_mode,
                overlay_mode=args.overlay_mode,
                downside_penalty_quantile=args.downside_penalty_quantile,
            )
            scores.to_parquet(output_path, index=False)
            row_count += int(len(scores))
            partition_count += 1
            del scores
        methods[method] = {
            "path": str(method_dir / "*.parquet"),
            "partition_count": partition_count,
            "row_count": row_count,
            "overlay_weight": weight,
            "primary_weight": 1.0
            if args.overlay_mode in {"entry_exclusion", "optimizer_risk_penalty"}
            else 1.0 - weight,
            "overlay_mode": args.overlay_mode,
            "entry_exclusion_quantile": weight
            if args.overlay_mode == "entry_exclusion" and weight > 0
            else None,
            "risk_penalty_bps_scale": weight
            if args.overlay_mode == "optimizer_risk_penalty"
            else None,
            "risk_penalty_quantile": args.downside_penalty_quantile
            if args.overlay_mode == "optimizer_risk_penalty"
            else None,
            "condition": _condition_summary(args),
        }
    return methods


def _score_paths(directory: Path) -> dict[str, Path]:
    paths = sorted(directory.glob("score_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no score_*.parquet files found under {directory}")
    return {path.stem.removeprefix("score_"): path for path in paths}


def _condition_lookup(args: argparse.Namespace) -> dict[str, bool] | None:
    if not args.condition_schedule:
        return None
    frame = pd.read_csv(args.condition_schedule)
    missing = [column for column in ("timestamp", args.condition_column) if column not in frame]
    if missing:
        raise ValueError(f"condition schedule missing columns: {missing}")
    active_values = set(args.condition_values)
    active = frame[args.condition_column].astype(str).isin(active_values)
    return dict(zip(frame["timestamp"].astype(str), active.astype(bool), strict=False))


def _condition_summary(args: argparse.Namespace) -> dict[str, Any]:
    if not args.condition_schedule:
        return {"enabled": False}
    return {
        "enabled": True,
        "schedule": args.condition_schedule,
        "column": args.condition_column,
        "active_values": args.condition_values,
    }


def _overlay_partition(
    primary_path: Path,
    satellite_path: Path,
    *,
    overlay_weight: float,
    condition: dict[str, bool] | None,
    rank_normalize: bool,
    decision_timing: str,
    condition_primary_mode: str = "current",
    overlay_mode: str = "blend",
    downside_penalty_quantile: float = 0.2,
) -> pd.DataFrame:
    primary = pd.read_parquet(primary_path, columns=["timestamp", "instrument_id", "score"])
    satellite = pd.read_parquet(
        satellite_path,
        columns=["timestamp", "instrument_id", "score"],
    ).rename(columns={"score": "satellite_score"})
    frame = primary.rename(columns={"score": "primary_score"}).merge(
        satellite,
        on=["timestamp", "instrument_id"],
        how="left",
    )
    if rank_normalize:
        frame["primary_component"] = (
            frame.groupby("timestamp", sort=False)["primary_score"]
            .rank(method="average", pct=True)
            .sub(0.5)
        )
        frame["satellite_component"] = (
            frame.groupby("timestamp", sort=False)["satellite_score"]
            .rank(method="average", pct=True)
            .sub(0.5)
        )
    else:
        frame["primary_component"] = pd.to_numeric(
            frame["primary_score"],
            errors="coerce",
        )
        frame["satellite_component"] = pd.to_numeric(
            frame["satellite_score"],
            errors="coerce",
        )
    frame["satellite_component"] = frame["satellite_component"].fillna(0.0)
    if condition_primary_mode == "daily_first":
        frame = _use_daily_first_primary_component_for_condition(
            frame,
            condition=condition,
        )
    elif condition_primary_mode != "current":
        raise ValueError(f"unsupported condition primary mode: {condition_primary_mode}")
    if condition is None:
        effective_weight = pd.Series(float(overlay_weight), index=frame.index)
    else:
        active = frame["timestamp"].astype(str).map(condition).fillna(False).astype(bool)
        effective_weight = pd.Series(0.0, index=frame.index)
        effective_weight.loc[active] = float(overlay_weight)
    frame["score"] = _combine_overlay_components(
        frame,
        effective_weight=effective_weight,
        overlay_mode=overlay_mode,
        downside_penalty_quantile=downside_penalty_quantile,
    )
    output = frame.loc[:, ["timestamp", "instrument_id", "score"]].copy()
    if overlay_mode == "entry_exclusion":
        if overlay_weight == 0:
            output["entry_eligible"] = True
        else:
            output["entry_eligible"] = _entry_exclusion_mask(
                frame,
                effective_weight=effective_weight,
                entry_exclusion_quantile=overlay_weight,
            )
    elif overlay_mode == "optimizer_risk_penalty":
        output["risk_penalty_bps"] = _optimizer_risk_penalty_bps(
            frame,
            effective_weight=effective_weight,
            risk_penalty_quantile=downside_penalty_quantile,
        )
    if decision_timing == "daily_first_plus_condition":
        output = _daily_first_plus_condition_decisions(output, condition=condition)
    elif decision_timing != "all":
        raise ValueError(f"unsupported decision timing: {decision_timing}")
    return output.sort_values(
        ["timestamp", "score", "instrument_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _combine_overlay_components(
    frame: pd.DataFrame,
    *,
    effective_weight: pd.Series,
    overlay_mode: str,
    downside_penalty_quantile: float,
) -> pd.Series:
    if overlay_mode == "blend":
        return (
            (1.0 - effective_weight) * frame["primary_component"]
            + effective_weight * frame["satellite_component"]
        )
    if overlay_mode == "entry_exclusion":
        return frame["primary_component"]
    if overlay_mode == "optimizer_risk_penalty":
        return frame["primary_component"]
    if overlay_mode != "downside_penalty":
        raise ValueError(f"unsupported overlay mode: {overlay_mode}")
    if not 0.0 < downside_penalty_quantile < 1.0:
        raise ValueError("downside_penalty_quantile must be in (0, 1)")
    threshold = frame.groupby("timestamp", sort=False)["satellite_component"].transform(
        lambda values: values.quantile(downside_penalty_quantile)
    )
    penalty = (threshold - frame["satellite_component"]).clip(lower=0.0)
    return frame["primary_component"] - effective_weight * penalty


def _entry_exclusion_mask(
    frame: pd.DataFrame,
    *,
    effective_weight: pd.Series,
    entry_exclusion_quantile: float,
) -> pd.Series:
    if not 0.0 < entry_exclusion_quantile < 1.0:
        raise ValueError("entry_exclusion_quantile must be in (0, 1)")
    threshold = frame.groupby("timestamp", sort=False)["satellite_component"].transform(
        lambda values: values.quantile(entry_exclusion_quantile)
    )
    active = effective_weight.astype(float).gt(0.0)
    return ~(active & frame["satellite_component"].lt(threshold))


def _optimizer_risk_penalty_bps(
    frame: pd.DataFrame,
    *,
    effective_weight: pd.Series,
    risk_penalty_quantile: float,
) -> pd.Series:
    if not 0.0 < risk_penalty_quantile < 1.0:
        raise ValueError("risk_penalty_quantile must be in (0, 1)")
    threshold = frame.groupby("timestamp", sort=False)["satellite_component"].transform(
        lambda values: values.quantile(risk_penalty_quantile)
    )
    penalty_depth = (threshold - frame["satellite_component"]).clip(lower=0.0)
    return effective_weight.astype(float) * penalty_depth


def _use_daily_first_primary_component_for_condition(
    frame: pd.DataFrame,
    *,
    condition: dict[str, bool] | None,
) -> pd.DataFrame:
    if condition is None:
        raise ValueError("daily_first condition primary mode requires --condition-schedule")
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce", format="ISO8601")
    if parsed.isna().any():
        raise ValueError("daily_first condition primary mode requires parseable timestamps")
    output = frame.copy()
    output["_parsed_timestamp"] = parsed
    output["_session_date"] = parsed.dt.strftime("%Y-%m-%d")
    timestamp_frame = (
        output.loc[:, ["timestamp", "_session_date", "_parsed_timestamp"]]
        .drop_duplicates()
        .sort_values(["_session_date", "_parsed_timestamp"], kind="mergesort")
    )
    first_timestamps = set(
        timestamp_frame.drop_duplicates("_session_date", keep="first")["timestamp"]
    )
    anchors = output.loc[
        output["timestamp"].isin(first_timestamps),
        ["_session_date", "instrument_id", "primary_component"],
    ].rename(columns={"primary_component": "_daily_first_primary_component"})
    output = output.merge(
        anchors,
        on=["_session_date", "instrument_id"],
        how="left",
    )
    active = output["timestamp"].astype(str).map(condition).fillna(False).astype(bool)
    anchored = output["_daily_first_primary_component"].notna()
    replace = active & anchored
    output.loc[replace, "primary_component"] = output.loc[
        replace,
        "_daily_first_primary_component",
    ]
    return output.drop(
        columns=["_parsed_timestamp", "_session_date", "_daily_first_primary_component"],
    )


def _daily_first_plus_condition_decisions(
    scores: pd.DataFrame,
    *,
    condition: dict[str, bool] | None,
) -> pd.DataFrame:
    if condition is None:
        raise ValueError("daily_first_plus_condition requires --condition-schedule")
    timestamps = scores.loc[:, ["timestamp"]].drop_duplicates().copy()
    parsed = pd.to_datetime(timestamps["timestamp"], errors="coerce", format="ISO8601")
    if parsed.isna().any():
        raise ValueError("daily_first_plus_condition requires parseable timestamps")
    timestamps["_parsed_timestamp"] = parsed
    timestamps["_session_date"] = parsed.dt.strftime("%Y-%m-%d")
    timestamps = timestamps.sort_values(
        ["_session_date", "_parsed_timestamp"],
        kind="mergesort",
    )
    first_daily = timestamps.groupby("_session_date", sort=False).cumcount() == 0
    condition_active = (
        timestamps["timestamp"].astype(str).map(condition).fillna(False).astype(bool)
    )
    keep_timestamps = set(
        timestamps.loc[first_daily | condition_active, "timestamp"].astype(str)
    )
    return scores.loc[scores["timestamp"].astype(str).isin(keep_timestamps)].copy()


def _scenarios(args: argparse.Namespace) -> list[Scenario]:
    years = args.years or _infer_years(Path(args.primary_score_dir))
    first_year = min(years)
    last_year = max(years)
    scenarios = [
        Scenario(
            name="full_base",
            start=f"{first_year}-01-01T00:00:00+08:00",
            end=f"{last_year}-12-31T23:59:59+08:00",
            partition_glob="score_*.parquet",
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            sell_stamp_tax_bps=args.sell_stamp_tax_bps,
            estimated_cost_bps=args.policy_estimated_cost_bps,
            description="Full-window candidate policy with production-like costs.",
        )
    ]
    if args.profile in {"standard", "robust"}:
        for year in years:
            scenarios.append(
                Scenario(
                    name=f"year_{year}_base",
                    start=f"{year}-01-01T00:00:00+08:00",
                    end=f"{year}-12-31T23:59:59+08:00",
                    partition_glob=f"score_{year}_*.parquet",
                    commission_bps=args.commission_bps,
                    slippage_bps=args.slippage_bps,
                    sell_stamp_tax_bps=args.sell_stamp_tax_bps,
                    estimated_cost_bps=args.policy_estimated_cost_bps,
                    description=f"Calendar-year stability slice for {year}.",
                )
            )
        scenarios.append(
            Scenario(
                name="full_high_cost",
                start=f"{first_year}-01-01T00:00:00+08:00",
                end=f"{last_year}-12-31T23:59:59+08:00",
                partition_glob="score_*.parquet",
                commission_bps=args.commission_bps * args.cost_stress_multiplier,
                slippage_bps=args.slippage_bps * args.cost_stress_multiplier,
                sell_stamp_tax_bps=args.sell_stamp_tax_bps * args.cost_stress_multiplier,
                estimated_cost_bps=args.policy_estimated_cost_bps
                * args.cost_stress_multiplier,
                description="Full-window transaction-cost stress.",
            )
        )
    if args.profile == "robust":
        scenarios.append(
            Scenario(
                name="full_zero_cost",
                start=f"{first_year}-01-01T00:00:00+08:00",
                end=f"{last_year}-12-31T23:59:59+08:00",
                partition_glob="score_*.parquet",
                commission_bps=0.0,
                slippage_bps=0.0,
                sell_stamp_tax_bps=0.0,
                estimated_cost_bps=0.0,
                description="Full-window zero-cost diagnostic.",
            )
        )
    return scenarios


def _infer_years(score_dir: Path) -> list[int]:
    years = sorted(
        {
            int(path.stem.removeprefix("score_").split("_", maxsplit=1)[0])
            for path in score_dir.glob("score_*.parquet")
        }
    )
    if not years:
        raise FileNotFoundError(f"no score years found under {score_dir}")
    return years


def _run_backtests(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    methods: dict[str, Any],
    scenarios: list[Scenario],
) -> dict[str, list[str]]:
    jobs = [
        _backtest_job(args, output_dir=output_dir, method=method, scenario=scenario)
        for method in methods
        for scenario in scenarios
    ]
    commands = {f"{job.method}:{job.scenario.name}": job.command for job in jobs}
    command_path = output_dir / "commands.json"
    command_path.write_text(
        json.dumps(commands, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pending_jobs = [
        job
        for job in jobs
        if not (args.resume_existing and job.summary_path.exists())
    ]
    if not pending_jobs:
        return commands
    with ProcessPoolExecutor(max_workers=args.job_workers) as executor:
        futures: dict[Future[None], BacktestJob] = {}
        for job in pending_jobs:
            futures[executor.submit(_run_backtest_job, job)] = job
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"backtest failed for {job.method} {job.scenario.name}; "
                        f"log={job.log_path}"
                    ) from exc
    return commands


def _backtest_job(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    method: str,
    scenario: Scenario,
) -> BacktestJob:
    score_dir = output_dir / "scores" / method
    scenario_dir = output_dir / scenario.name
    backtest_dir = scenario_dir / "backtests" / method / args.policy
    log_dir = scenario_dir / "logs"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    predictions = score_dir / scenario.partition_glob
    command = [
        sys.executable,
        str(PROJECT_ROOT / "examples" / "run_tree_score_backtest.py"),
        "--predictions-path",
        str(predictions),
        "--catalog-path",
        args.catalog_path,
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
        str(args.min_commission),
        "--lot-size",
        str(args.lot_size),
        "--trade-policy",
        args.trade_policy,
        "--rebalance-every-n-bars",
        str(args.rebalance_every_n_bars),
        "--policy-min-hold-bars",
        str(args.policy_min_hold_bars),
        "--policy-estimated-cost-bps",
        str(scenario.estimated_cost_bps),
        "--policy-no-trade-weight-band",
        str(args.policy_no_trade_weight_band),
        "--policy-partial-rebalance-rate",
        str(args.policy_partial_rebalance_rate),
        "--policy-gross-exposure-scale",
        str(args.policy_gross_exposure_scale),
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
        "--policy-entry-rank",
        str(args.policy_entry_rank),
        "--policy-exit-rank",
        str(args.policy_exit_rank),
        "--policy-max-entries-per-rebalance",
        str(args.policy_max_entries_per_rebalance),
        "--policy-max-exits-per-rebalance",
        str(args.policy_max_exits_per_rebalance),
        "--policy-turnover-budget-pacing",
        str(args.policy_turnover_budget_pacing),
        "--policy-drawdown-brake-reduced-scale",
        str(args.policy_drawdown_brake_reduced_scale),
        "--policy-turnover-budget-period",
        args.policy_turnover_budget_period,
        "--optimizer-score-to-edge-bps",
        str(args.optimizer_score_to_edge_bps),
        "--optimizer-min-net-edge-bps",
        str(args.optimizer_min_net_edge_bps),
        "--optimizer-risk-penalty-multiplier",
        str(args.optimizer_risk_penalty_multiplier),
        "--optimizer-weighting",
        args.optimizer_weighting,
    ]
    if args.policy_max_gross_turnover_per_rebalance is not None:
        command.extend(
            [
                "--policy-max-gross-turnover-per-rebalance",
                str(args.policy_max_gross_turnover_per_rebalance),
            ]
        )
    if args.policy_total_gross_turnover_budget is not None:
        command.extend(
            [
                "--policy-total-gross-turnover-budget",
                str(args.policy_total_gross_turnover_budget),
            ]
        )
    if args.policy_cost_pressure_threshold_bps is not None:
        command.extend(
            [
                "--policy-cost-pressure-threshold-bps",
                str(args.policy_cost_pressure_threshold_bps),
            ]
        )
    if args.policy_cost_pressure_reduced_scale != 0.7:
        command.extend(
            [
                "--policy-cost-pressure-reduced-scale",
                str(args.policy_cost_pressure_reduced_scale),
            ]
        )
    if args.policy_cost_pressure_max_gross_turnover_per_rebalance is not None:
        command.extend(
            [
                "--policy-cost-pressure-max-gross-turnover-per-rebalance",
                str(args.policy_cost_pressure_max_gross_turnover_per_rebalance),
            ]
        )
    if args.policy_gross_exposure_scale_path:
        command.extend(
            ["--policy-gross-exposure-scale-path", args.policy_gross_exposure_scale_path]
        )
    if args.exclude_st:
        command.append("--exclude-st")
    return BacktestJob(
        scenario=scenario,
        method=method,
        command=command,
        output_dir=backtest_dir,
        summary_path=backtest_dir / "summary.json",
        log_path=log_dir / f"backtest_{method}_{args.policy}.log",
    )


def _run_backtest_job(job: BacktestJob) -> None:
    with job.log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(job.command) + "\n")
        log_file.flush()
        result = subprocess.run(
            job.command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, job.command)


def _collect_rows(
    *,
    methods: dict[str, Any],
    scenarios: list[Scenario],
    output_dir: Path,
    policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        for scenario in scenarios:
            summary_path = (
                output_dir
                / scenario.name
                / "backtests"
                / method
                / policy
                / "summary.json"
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", payload)
            rows.append(
                {
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "method": method,
                    "policy": policy,
                    "total_return": _number(metrics.get("total_return")),
                    "final_equity": _number(metrics.get("final_equity")),
                    "gross_turnover": _number(metrics.get("gross_turnover")),
                    "max_drawdown": _number(metrics.get("max_drawdown")),
                    "total_transaction_cost": _number(
                        metrics.get("total_transaction_cost")
                    ),
                    "trade_count": _number(metrics.get("trade_count")),
                    "signal_count": _number(payload.get("signal_count")),
                    "commission_bps": scenario.commission_bps,
                    "slippage_bps": scenario.slippage_bps,
                    "sell_stamp_tax_bps": scenario.sell_stamp_tax_bps,
                    "scenario_start": scenario.start,
                    "scenario_end": scenario.end,
                }
            )
    return rows


def _leaderboard(rows: list[dict[str, Any]], *, policy: str) -> list[dict[str, Any]]:
    methods = sorted({str(row["method"]) for row in rows if row.get("policy") == policy})
    output: list[dict[str, Any]] = []
    for method in methods:
        current = [row for row in rows if row.get("method") == method and row.get("policy") == policy]
        by_scenario = {str(row["scenario"]): row for row in current}
        returns = [_number(row.get("total_return")) for row in current]
        turnovers = [_number(row.get("gross_turnover")) for row in current]
        drawdowns = [_number(row.get("max_drawdown")) for row in current]
        costs = [_number(row.get("total_transaction_cost")) for row in current]
        output.append(
            {
                "method": method,
                "policy": policy,
                "scenario_count": len(current),
                "full_base_return": _number(
                    by_scenario.get("full_base", {}).get("total_return")
                ),
                "full_high_cost_return": _number(
                    by_scenario.get("full_high_cost", {}).get("total_return")
                ),
                "mean_return": _mean(returns),
                "worst_return": _min(returns),
                "mean_gross_turnover": _mean(turnovers),
                "worst_drawdown": _min(drawdowns),
                "total_transaction_cost": _sum(costs),
            }
        )
    return sorted(output, key=lambda row: _number(row["full_base_return"]), reverse=True)


def _validation(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    max_full_turnover: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows if row.get("policy") == policy}):
        by_scenario = {
            str(row["scenario"]): row
            for row in rows
            if row.get("method") == method and row.get("policy") == policy
        }
        full_base = by_scenario.get("full_base", {})
        full_high_cost = by_scenario.get("full_high_cost", {})
        checks.append(
            {
                "name": f"{method}_full_base_positive_return",
                "status": "pass" if _number(full_base.get("total_return")) > 0 else "fail",
                "details": {"total_return": full_base.get("total_return")},
            }
        )
        checks.append(
            {
                "name": f"{method}_full_base_turnover_control",
                "status": (
                    "pass"
                    if _number(full_base.get("gross_turnover")) <= max_full_turnover
                    else "warn"
                ),
                "details": {
                    "gross_turnover": full_base.get("gross_turnover"),
                    "max_full_turnover": max_full_turnover,
                },
            }
        )
        if full_high_cost:
            checks.append(
                {
                    "name": f"{method}_full_high_cost_positive_return",
                    "status": (
                        "pass"
                        if _number(full_high_cost.get("total_return")) > 0
                        else "warn"
                    ),
                    "details": {"total_return": full_high_cost.get("total_return")},
                }
            )
            checks.append(
                {
                    "name": f"{method}_high_cost_costs_not_lower",
                    "status": (
                        "pass"
                        if _number(full_high_cost.get("total_transaction_cost"))
                        >= _number(full_base.get("total_transaction_cost"))
                        else "warn"
                    ),
                    "details": {
                        "full_base_total_transaction_cost": full_base.get(
                            "total_transaction_cost"
                        ),
                        "full_high_cost_total_transaction_cost": full_high_cost.get(
                            "total_transaction_cost"
                        ),
                    },
                }
            )
        negative_years = [
            scenario
            for scenario, row in by_scenario.items()
            if scenario.startswith("year_") and _number(row.get("total_return")) <= 0
        ]
        if any(scenario.startswith("year_") for scenario in by_scenario):
            checks.append(
                {
                    "name": f"{method}_yearly_base_positive_returns",
                    "status": "pass" if not negative_years else "warn",
                    "details": {"negative_years": negative_years},
                }
            )
    failed = [check for check in checks if check["status"] == "fail"]
    warned = [check for check in checks if check["status"] == "warn"]
    return {
        "overall_status": "fail" if failed else "warn" if warned else "pass",
        "checks": checks,
        "failed_count": len(failed),
        "warning_count": len(warned),
    }


def _summary_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "primary_score_dir": args.primary_score_dir,
        "satellite_score_dir": args.satellite_score_dir,
        "overlay_weights": args.overlay_weights,
        "overlay_mode": args.overlay_mode,
        "downside_penalty_quantile": args.downside_penalty_quantile,
        "rank_normalize": args.rank_normalize,
        "condition_primary_mode": args.condition_primary_mode,
        "policy_gross_exposure_scale_path": args.policy_gross_exposure_scale_path,
        "policy_max_gross_turnover_per_rebalance": (
            args.policy_max_gross_turnover_per_rebalance
        ),
        "policy_total_gross_turnover_budget": args.policy_total_gross_turnover_budget,
        "policy_turnover_budget_period": args.policy_turnover_budget_period,
        "policy_turnover_budget_pacing": args.policy_turnover_budget_pacing,
        "policy_cost_pressure_threshold_bps": (
            args.policy_cost_pressure_threshold_bps
        ),
        "policy_cost_pressure_reduced_scale": (
            args.policy_cost_pressure_reduced_scale
        ),
        "policy_cost_pressure_max_gross_turnover_per_rebalance": (
            args.policy_cost_pressure_max_gross_turnover_per_rebalance
        ),
        "condition": _condition_summary(args),
        "decision_timing": args.decision_timing,
        "profile": args.profile,
        "years": args.years,
        "job_workers": args.job_workers,
    }


def _weight_label(weight: float) -> str:
    if weight >= 1.0:
        if float(weight).is_integer():
            return f"w{int(weight):02d}"
        return f"w{str(weight).replace('.', 'p')}"
    tenths_of_percent = int(round(weight * 1000))
    if tenths_of_percent % 10 == 0:
        return f"w{tenths_of_percent // 10:02d}"
    return f"w{tenths_of_percent:03d}"


def _number(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else None


def _min(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(min(finite)) if finite else None


def _sum(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite)) if finite else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-score-dir", required=True)
    parser.add_argument("--satellite-score-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-prefix", default="overlay")
    parser.add_argument(
        "--overlay-weights",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
    )
    parser.add_argument(
        "--rank-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rank-normalize each input score within timestamp before blending",
    )
    parser.add_argument(
        "--overlay-mode",
        choices=(
            "blend",
            "downside_penalty",
            "entry_exclusion",
            "optimizer_risk_penalty",
        ),
        default="blend",
        help=(
            "blend linearly combines primary and satellite ranks; "
            "downside_penalty keeps the primary score and only penalizes names in "
            "the weak satellite tail; entry_exclusion keeps the primary score and "
            "uses each positive overlay weight as the excluded lower-tail satellite "
            "quantile; optimizer_risk_penalty keeps the primary score and writes "
            "risk_penalty_bps for weak satellite-tail names"
        ),
    )
    parser.add_argument(
        "--downside-penalty-quantile",
        type=float,
        default=0.2,
        help="satellite rank quantile below which downside_penalty subtracts score",
    )
    parser.add_argument(
        "--decision-timing",
        choices=("all", "daily_first_plus_condition"),
        default="all",
        help=(
            "which score timestamps to expose to the backtest; "
            "daily_first_plus_condition keeps the first timestamp of each session "
            "plus condition-active timestamps"
        ),
    )
    parser.add_argument(
        "--condition-primary-mode",
        choices=("current", "daily_first"),
        default="current",
        help=(
            "primary score used when condition is active; daily_first anchors "
            "condition-active rows to the session's first primary ranking"
        ),
    )
    parser.add_argument("--condition-schedule")
    parser.add_argument("--condition-column", default="risk_state")
    parser.add_argument("--condition-values", nargs="+", default=["reduced", "blocked"])
    parser.add_argument("--catalog-path", default="../quant_dataset/canonical_store/catalog/quant_research.duckdb")
    parser.add_argument("--profile", choices=("quick", "standard", "robust"), default="standard")
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--cost-stress-multiplier", type=float, default=2.0)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--trade-policy", default="rank_buffer_drop")
    parser.add_argument("--rebalance-every-n-bars", type=int, default=48)
    parser.add_argument("--policy-min-hold-bars", type=int, default=0)
    parser.add_argument("--policy-estimated-cost-bps", type=float, default=13.0)
    parser.add_argument("--policy-no-trade-weight-band", type=float, default=0.002)
    parser.add_argument("--policy-partial-rebalance-rate", type=float, default=0.5)
    parser.add_argument("--policy-gross-exposure-scale", type=float, default=1.0)
    parser.add_argument("--policy-gross-exposure-scale-path")
    parser.add_argument("--policy-entry-rank", type=int, default=50)
    parser.add_argument("--policy-exit-rank", type=int, default=150)
    parser.add_argument("--policy-max-entries-per-rebalance", type=int, default=10)
    parser.add_argument("--policy-max-exits-per-rebalance", type=int, default=10)
    parser.add_argument("--policy-max-gross-turnover-per-rebalance", type=float)
    parser.add_argument("--policy-total-gross-turnover-budget", type=float)
    parser.add_argument("--policy-turnover-budget-pacing", type=float, default=0.0)
    parser.add_argument("--policy-turnover-budget-period", default="path")
    parser.add_argument("--policy-drawdown-brake-reduced-scale", type=float, default=0.5)
    parser.add_argument("--policy-cost-pressure-threshold-bps", type=float)
    parser.add_argument("--policy-cost-pressure-reduced-scale", type=float, default=0.7)
    parser.add_argument(
        "--policy-cost-pressure-max-gross-turnover-per-rebalance",
        type=float,
    )
    parser.add_argument("--min-trade-weight", type=float, default=0.0005)
    parser.add_argument("--limit-up-bps", type=float, default=980.0)
    parser.add_argument("--limit-down-bps", type=float, default=980.0)
    parser.add_argument("--data-access-mode", default="fast_parquet")
    parser.add_argument("--streaming-chunk", default="month")
    parser.add_argument("--streaming-chunk-padding-days", type=int, default=10)
    parser.add_argument("--optimizer-score-to-edge-bps", type=float, default=100.0)
    parser.add_argument("--optimizer-min-net-edge-bps", type=float, default=0.0)
    parser.add_argument("--optimizer-risk-penalty-multiplier", type=float, default=1.0)
    parser.add_argument("--optimizer-weighting", default="utility")
    parser.add_argument("--exclude-st", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-full-turnover", type=float, default=160.0)
    parser.add_argument("--job-workers", type=int, default=2)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.downside_penalty_quantile < 1.0:
        raise ValueError("--downside-penalty-quantile must be in (0, 1)")
    if args.overlay_mode == "entry_exclusion":
        invalid_weights = [
            weight for weight in args.overlay_weights if weight != 0 and not 0.0 < weight < 1.0
        ]
        if invalid_weights:
            raise ValueError(
                "entry_exclusion overlay weights must be 0 for control or quantiles in (0, 1)"
            )
    if args.overlay_mode == "optimizer_risk_penalty":
        invalid_weights = [weight for weight in args.overlay_weights if weight < 0]
        if invalid_weights:
            raise ValueError("optimizer_risk_penalty weights must be non-negative")
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
    if (
        args.policy_cost_pressure_threshold_bps is not None
        and args.policy_cost_pressure_threshold_bps < 0
    ):
        raise ValueError("--policy-cost-pressure-threshold-bps must be non-negative")
    if not 0 <= args.policy_cost_pressure_reduced_scale <= 1:
        raise ValueError("--policy-cost-pressure-reduced-scale must be in [0, 1]")
    if (
        args.policy_cost_pressure_max_gross_turnover_per_rebalance is not None
        and args.policy_cost_pressure_max_gross_turnover_per_rebalance < 0
    ):
        raise ValueError(
            "--policy-cost-pressure-max-gross-turnover-per-rebalance must be non-negative"
        )
    return args


if __name__ == "__main__":
    main()
