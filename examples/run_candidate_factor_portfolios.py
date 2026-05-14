"""Build and optionally backtest candidate-factor portfolio scores."""

from __future__ import annotations

import argparse
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
    factor_combination_weights,
    load_candidate_factors,
    write_score_partitions,
)


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


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = _dataset_paths(args)
    candidates = load_candidate_factors(
        Path(args.admission_report),
        statuses=tuple(args.statuses),
    )
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
    scores_summary = write_score_partitions(
        dataset_paths,
        output_dir=output_dir / "scores",
        candidates=candidates,
        weights_by_method=weights_by_method,
    )
    summary = {
        "params": _summary_params(args),
        **scores_summary,
    }
    if args.run_backtests:
        summary["backtests"] = _run_backtests(args, scores_summary=scores_summary)
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
        "admission_report": args.admission_report,
        "factor_correlation": args.factor_correlation,
        "methods": args.methods,
        "statuses": args.statuses,
        "max_partitions": args.max_partitions,
        "partition_start": args.partition_start,
        "partition_end": args.partition_end,
        "run_backtests": args.run_backtests,
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
        }
    return params


def _run_backtests(
    args: argparse.Namespace,
    *,
    scores_summary: dict[str, object],
) -> dict[str, object]:
    if not args.start or not args.end:
        raise ValueError("--start and --end are required with --run-backtests")
    rows = {}
    methods = scores_summary["methods"]
    if not isinstance(methods, dict):
        raise ValueError("invalid score summary methods")
    policy_specs = _backtest_policy_specs(args)
    for method, payload in methods.items():
        if not isinstance(payload, dict):
            continue
        if args.backtest_policy_set == "single":
            spec = policy_specs[0]
            backtest_dir = Path(args.output_dir) / "backtests" / method
            command = _backtest_command(args, str(payload["path"]), backtest_dir, spec)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            rows[method] = json.loads(
                (backtest_dir / "summary.json").read_text(encoding="utf-8")
            )
            continue
        policy_rows = {}
        for spec in policy_specs:
            backtest_dir = Path(args.output_dir) / "backtests" / method / spec.name
            command = _backtest_command(args, str(payload["path"]), backtest_dir, spec)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            policy_rows[spec.name] = json.loads(
                (backtest_dir / "summary.json").read_text(encoding="utf-8")
            )
        rows[method] = policy_rows
    return rows


def _backtest_policy_specs(args: argparse.Namespace) -> list[BacktestPolicySpec]:
    if args.backtest_policy_set == "single":
        return [
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
            )
        ]
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
    }
    return [
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
        "--max-bar-turnover-participation": args.max_bar_turnover_participation,
    }
    for option, value in optional_floats.items():
        if value is not None:
            command.extend([option, str(value)])
    if args.exclude_st:
        command.append("--exclude-st")
    return command


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
        default="runs/candidate_factor_portfolios",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("equal", "ic_weighted", "decorrelated"),
        default=["equal", "ic_weighted", "decorrelated"],
    )
    parser.add_argument("--statuses", nargs="+", default=["candidate"])
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
        choices=("naive_top_n", "rank_buffer_drop"),
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
    parser.add_argument("--policy-set-drop-count", type=int, default=10)
    parser.add_argument("--policy-set-exit-rank", type=int)
    parser.add_argument("--policy-set-rebalance-every-n-bars", type=int, default=48)
    parser.add_argument("--policy-set-partial-rebalance-rate", type=float, default=0.5)
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
        choices=("year", "month"),
        default="month",
    )
    parser.add_argument("--streaming-chunk-padding-days", type=int, default=10)
    args = parser.parse_args()
    if args.max_partitions is not None and args.max_partitions <= 0:
        raise ValueError("--max-partitions must be positive")
    if args.partition_start and args.partition_end and args.partition_start > args.partition_end:
        raise ValueError("--partition-start must not be after --partition-end")
    if args.decorrelation_ridge < 0:
        raise ValueError("--decorrelation-ridge must be non-negative")
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
    if args.policy_set_drop_count <= 0:
        raise ValueError("--policy-set-drop-count must be positive")
    if args.policy_set_exit_rank is not None and args.policy_set_exit_rank < args.top_n:
        raise ValueError("--policy-set-exit-rank must be greater than or equal to --top-n")
    if args.policy_set_rebalance_every_n_bars <= 0:
        raise ValueError("--policy-set-rebalance-every-n-bars must be positive")
    if not 0 < args.policy_set_partial_rebalance_rate <= 1:
        raise ValueError("--policy-set-partial-rebalance-rate must be in (0, 1]")
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


if __name__ == "__main__":
    main()
