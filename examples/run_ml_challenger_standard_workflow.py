"""Plan or run the standard no-leak ML challenger workflow.

The workflow is intentionally opinionated.  It trains purged walk-forward
LightGBM primary-pool rerank scores, builds primary/ML blends, backtests each
blend on the walk-forward span and live window, then builds and backtests
adaptive source switches with source-transition exits enabled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


EXAMPLES_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class WorkflowStage:
    """One ordered subprocess stage in the standard workflow."""

    name: str
    command: list[str]
    log_path: Path


def main() -> None:
    args = _parse_args()
    stages = build_standard_workflow_plan(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "standard_workflow_plan.json"
    _write_plan(plan_path, args=args, stages=stages)
    if args.execute:
        _run_stages(stages)
    print(
        json.dumps(
            {
                "status": "executed" if args.execute else "planned",
                "plan_path": str(plan_path),
                "stage_count": len(stages),
                "stages": [stage.name for stage in stages],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


def build_standard_workflow_plan(args: argparse.Namespace) -> list[WorkflowStage]:
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    ml_dir = output_dir / "ml_challenger"
    blend_dir = output_dir / "primary_pool_blends"
    backtest_dir = output_dir / "backtests"
    adaptive_dir = output_dir / "adaptive_switch"
    method_name = args.method_name
    stages: list[WorkflowStage] = []

    stages.append(
        _stage(
            logs_dir,
            "01_train_walk_forward_ml_challenger",
            [
                sys.executable,
                str(EXAMPLES_DIR / "run_ml_factor_challenger.py"),
                "--dataset-dir",
                args.dataset_dir,
                "--admission-report",
                args.admission_report,
                "--output-dir",
                str(ml_dir),
                "--label-column",
                args.label_column,
                "--statuses",
                *args.statuses,
                "--evaluation-roles",
                *args.evaluation_roles,
                "--include-features",
                *args.include_features,
                "--score-transform",
                args.score_transform,
                "--score-mode",
                "primary_pool_rerank",
                "--primary-score-dir",
                args.primary_score_dir,
                "--primary-pool-rank",
                str(args.primary_pool_rank),
                "--primary-blend-weight",
                "0.0",
                "--sample-weight-mode",
                "top_bottom",
                "--sample-weight-top-quantile",
                str(args.sample_weight_top_quantile),
                "--sample-weight-multiplier",
                str(args.sample_weight_multiplier),
                "--max-train-rows",
                str(args.max_train_rows),
                "--max-valid-rows",
                str(args.max_valid_rows),
                "--redundancy-sample-rows",
                str(args.redundancy_sample_rows),
                "--num-threads",
                str(args.num_threads),
                "--fold",
                _fold_spec(
                    "history",
                    train_start=args.train_start,
                    train_end=args.history_train_end,
                    test_start=args.history_test_start,
                    test_end=args.history_test_end,
                ),
                "--fold",
                _fold_spec(
                    "live",
                    train_start=args.train_start,
                    train_end=args.live_train_end,
                    test_start=args.live_start,
                    test_end=args.live_end,
                ),
                "--backtest-start",
                args.history_test_start,
                "--backtest-end",
                args.live_end,
            ],
        )
    )

    stages.append(
        _stage(
            logs_dir,
            "02_build_primary_pool_blends",
            [
                sys.executable,
                str(EXAMPLES_DIR / "build_primary_pool_score_blends.py"),
                "--primary-score-dir",
                args.primary_score_dir,
                "--ml-pool-score-dir",
                str(ml_dir / "scores" / "lightgbm"),
                "--output-dir",
                str(blend_dir),
                "--primary-blend-weights",
                *[str(weight) for weight in args.primary_blend_weights],
            ],
        )
    )

    candidate_score_args: list[str] = []
    candidate_backtest_args: list[str] = []
    for weight in args.primary_blend_weights:
        label = _weight_label(weight)
        score_dir = blend_dir / "scores" / label
        history_output = backtest_dir / f"history_{label}"
        live_output = backtest_dir / f"live_{label}"
        candidate_name = f"expanded_{label}"
        candidate_score_args.extend(["--candidate-score", f"{candidate_name}={score_dir}"])
        candidate_backtest_args.extend(
            ["--candidate-backtest", f"{candidate_name}={history_output}"]
        )
        stages.append(
            _stage(
                logs_dir,
                f"03_backtest_history_{label}",
                _tree_score_backtest_command(
                    predictions_path=score_dir / "score_*.parquet",
                    start=args.history_test_start,
                    end=args.live_end,
                    output_dir=history_output,
                    args=args,
                    source_transition=False,
                ),
            )
        )
        stages.append(
            _stage(
                logs_dir,
                f"04_backtest_live_{label}",
                _tree_score_backtest_command(
                    predictions_path=score_dir / "score_*.parquet",
                    start=args.live_start,
                    end=args.live_end,
                    output_dir=live_output,
                    args=args,
                    source_transition=False,
                ),
            )
        )

    for lookback_days in args.selection_lookback_days:
        label = f"lb{lookback_days:03d}"
        selector_output = adaptive_dir / label
        stages.append(
            _stage(
                logs_dir,
                f"05_build_adaptive_selector_{label}",
                [
                    sys.executable,
                    str(EXAMPLES_DIR / "build_backtest_adaptive_state_switch.py"),
                    "--baseline-score-dir",
                    args.baseline_score_dir,
                    "--baseline-backtest-dir",
                    args.baseline_backtest_dir,
                    *candidate_score_args,
                    *candidate_backtest_args,
                    "--output-dir",
                    str(selector_output),
                    "--method-name",
                    method_name,
                    "--start",
                    args.live_start,
                    "--end",
                    args.live_end,
                    "--selection-lookback-days",
                    str(lookback_days),
                    "--selection-embargo-days",
                    str(args.selection_embargo_days),
                    "--selection-min-equity-points",
                    str(args.selection_min_equity_points),
                    "--switch-penalty",
                    str(args.switch_penalty),
                ],
            )
        )
        stages.append(
            _stage(
                logs_dir,
                f"06_backtest_live_adaptive_{label}",
                _tree_score_backtest_command(
                    predictions_path=selector_output / "scores" / method_name / "score_*.parquet",
                    start=args.live_start,
                    end=args.live_end,
                    output_dir=backtest_dir / f"live_adaptive_{label}",
                    args=args,
                    source_transition=True,
                ),
            )
        )
    return stages


def _tree_score_backtest_command(
    *,
    predictions_path: Path,
    start: str,
    end: str,
    output_dir: Path,
    args: argparse.Namespace,
    source_transition: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(EXAMPLES_DIR / "run_tree_score_backtest.py"),
        "--predictions-path",
        str(predictions_path),
        "--start",
        start,
        "--end",
        end,
        "--top-n",
        str(args.top_n),
        "--trade-policy",
        "rank_buffer_drop",
        "--rebalance-every-n-bars",
        str(args.rebalance_every_n_bars),
        "--policy-entry-rank",
        str(args.policy_entry_rank),
        "--policy-exit-rank",
        str(args.policy_exit_rank),
        "--policy-max-entries-per-rebalance",
        str(args.policy_max_entries_per_rebalance),
        "--policy-max-exits-per-rebalance",
        str(args.policy_max_exits_per_rebalance),
        "--policy-no-trade-weight-band",
        str(args.policy_no_trade_weight_band),
        "--policy-partial-rebalance-rate",
        str(args.policy_partial_rebalance_rate),
        "--commission-bps",
        str(args.commission_bps),
        "--slippage-bps",
        str(args.slippage_bps),
        "--sell-stamp-tax-bps",
        str(args.sell_stamp_tax_bps),
        "--min-commission",
        str(args.min_commission),
        "--min-trade-weight",
        str(args.min_trade_weight),
        "--exclude-st",
        "--limit-up-bps",
        str(args.limit_up_bps),
        "--limit-down-bps",
        str(args.limit_down_bps),
        "--data-access-mode",
        "fast_parquet",
        "--streaming-chunk",
        "month",
        "--output-dir",
        str(output_dir),
    ]
    if source_transition:
        command.extend(
            [
                "--policy-force-source-transition-exits",
                "--policy-source-transition-exit-rate",
                str(args.source_transition_exit_rate),
            ]
        )
    return command


def _stage(logs_dir: Path, name: str, command: list[str]) -> WorkflowStage:
    return WorkflowStage(name=name, command=command, log_path=logs_dir / f"{name}.log")


def _fold_spec(
    name: str,
    *,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> str:
    return (
        f"{name}:train_start={train_start},train_end={train_end},"
        f"test_start={test_start},test_end={test_end}"
    )


def _weight_label(weight: float) -> str:
    return f"primary_w{int(round(weight * 100)):03d}"


def _write_plan(
    path: Path,
    *,
    args: argparse.Namespace,
    stages: list[WorkflowStage],
) -> None:
    payload = {
        "status": "planned",
        "strict_rules": [
            "standard workflow never passes --allow-label-derived-features",
            "run_ml_factor_challenger.py enforces label-derived feature filtering",
            "run_ml_factor_challenger.py validates required columns for every partition",
            "two explicit walk-forward folds are required: history and live",
            "adaptive live backtests use source-transition exits",
        ],
        "params": {
            key: _jsonable(value)
            for key, value in vars(args).items()
            if key != "execute"
        },
        "stages": [
            {
                "name": stage.name,
                "command": stage.command,
                "log_path": str(stage.log_path),
            }
            for stage in stages
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_stages(stages: list[WorkflowStage]) -> None:
    for stage in stages:
        stage.log_path.parent.mkdir(parents=True, exist_ok=True)
        with stage.log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("$ " + " ".join(stage.command) + "\n")
            log_file.flush()
            result = subprocess.run(
                stage.command,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"workflow stage failed: {stage.name}; see {stage.log_path}"
            )


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--admission-report", required=True)
    parser.add_argument("--primary-score-dir", required=True)
    parser.add_argument("--baseline-score-dir")
    parser.add_argument("--baseline-backtest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-features", nargs="+", required=True)
    parser.add_argument("--label-column", default="forward_return_48b")
    parser.add_argument("--statuses", nargs="+", default=["candidate", "watchlist"])
    parser.add_argument("--evaluation-roles", nargs="+", default=["alpha_rank"])
    parser.add_argument("--score-transform", choices=("rank", "zscore"), default="rank")
    parser.add_argument("--primary-pool-rank", type=int, default=150)
    parser.add_argument(
        "--primary-blend-weights",
        nargs="+",
        type=float,
        default=[0.50, 0.75],
    )
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--history-train-end", required=True)
    parser.add_argument("--history-test-start", required=True)
    parser.add_argument("--history-test-end", required=True)
    parser.add_argument("--live-train-end", required=True)
    parser.add_argument("--live-start", required=True)
    parser.add_argument("--live-end", required=True)
    parser.add_argument("--max-train-rows", type=int, default=5_000_000)
    parser.add_argument("--max-valid-rows", type=int, default=500_000)
    parser.add_argument("--redundancy-sample-rows", type=int, default=1_000_000)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--sample-weight-top-quantile", type=float, default=0.20)
    parser.add_argument("--sample-weight-multiplier", type=float, default=3.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--rebalance-every-n-bars", type=int, default=48)
    parser.add_argument("--policy-entry-rank", type=int, default=50)
    parser.add_argument("--policy-exit-rank", type=int, default=150)
    parser.add_argument("--policy-max-entries-per-rebalance", type=int, default=10)
    parser.add_argument("--policy-max-exits-per-rebalance", type=int, default=10)
    parser.add_argument("--policy-no-trade-weight-band", type=float, default=0.002)
    parser.add_argument("--policy-partial-rebalance-rate", type=float, default=0.5)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    parser.add_argument("--sell-stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--min-commission", type=float, default=5.0)
    parser.add_argument("--min-trade-weight", type=float, default=0.0005)
    parser.add_argument("--limit-up-bps", type=float, default=980.0)
    parser.add_argument("--limit-down-bps", type=float, default=980.0)
    parser.add_argument(
        "--selection-lookback-days",
        nargs="+",
        type=int,
        default=[63, 126, 252],
    )
    parser.add_argument("--selection-embargo-days", type=int, default=3)
    parser.add_argument("--selection-min-equity-points", type=int, default=40)
    parser.add_argument("--switch-penalty", type=float, default=0.02)
    parser.add_argument("--source-transition-exit-rate", type=float, default=1.0)
    parser.add_argument("--method-name", default="ml_challenger_adaptive_switch")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the planned stages; without this flag only write the plan",
    )
    args = parser.parse_args(argv)
    _validate_args(args)
    if args.baseline_score_dir is None:
        args.baseline_score_dir = args.primary_score_dir
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "dataset_dir",
        "admission_report",
        "primary_score_dir",
        "baseline_backtest_dir",
    ):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if args.baseline_score_dir is not None and not Path(args.baseline_score_dir).exists():
        raise FileNotFoundError(f"baseline_score_dir not found: {args.baseline_score_dir}")
    if not args.include_features:
        raise ValueError("--include-features is required for standard workflow")
    if any(not 0 < weight < 1 for weight in args.primary_blend_weights):
        raise ValueError("--primary-blend-weights must be strictly inside (0, 1)")
    positive_int_fields = (
        "primary_pool_rank",
        "max_train_rows",
        "max_valid_rows",
        "redundancy_sample_rows",
        "num_threads",
        "top_n",
        "rebalance_every_n_bars",
        "policy_entry_rank",
        "policy_exit_rank",
        "policy_max_entries_per_rebalance",
        "policy_max_exits_per_rebalance",
        "selection_min_equity_points",
    )
    for name in positive_int_fields:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if any(value <= 0 for value in args.selection_lookback_days):
        raise ValueError("--selection-lookback-days values must be positive")
    if args.selection_embargo_days < 0:
        raise ValueError("--selection-embargo-days must be non-negative")
    if args.switch_penalty < 0:
        raise ValueError("--switch-penalty must be non-negative")
    if not 0 < args.source_transition_exit_rate <= 1:
        raise ValueError("--source-transition-exit-rate must be in (0, 1]")
    if not 0 < args.sample_weight_top_quantile < 0.5:
        raise ValueError("--sample-weight-top-quantile must be in (0, 0.5)")
    if args.sample_weight_multiplier <= 1:
        raise ValueError("--sample-weight-multiplier must be greater than 1")
    if not 0 <= args.policy_no_trade_weight_band <= 1:
        raise ValueError("--policy-no-trade-weight-band must be in [0, 1]")
    if not 0 < args.policy_partial_rebalance_rate <= 1:
        raise ValueError("--policy-partial-rebalance-rate must be in (0, 1]")
    for name in (
        "commission_bps",
        "slippage_bps",
        "sell_stamp_tax_bps",
        "min_commission",
        "min_trade_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.limit_up_bps <= 0 or args.limit_down_bps <= 0:
        raise ValueError("--limit-up-bps and --limit-down-bps must be positive")


if __name__ == "__main__":
    main()
