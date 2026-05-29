"""Analyze monthly and regime attribution for ML challenger score overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_STATE_COLUMNS = (
    "market_state_return_5m",
    "market_state_downside_mean_5m_w48",
    "market_state_weak_breadth_mean_5m_w48",
    "market_state_limit_pressure_mean_5m_w48",
)


def main() -> None:
    args = _parse_args()
    summary = analyze_ml_challenger_attribution(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def analyze_ml_challenger_attribution(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = _timestamp(args.start)
    end = _timestamp(args.end)
    baseline_perf = _monthly_performance(
        Path(args.baseline_backtest_dir),
        method="baseline",
        start=start,
        end=end,
        initial_cash=args.initial_cash,
    )
    challenger_perf = _monthly_performance(
        Path(args.challenger_backtest_dir),
        method="challenger",
        start=start,
        end=end,
        initial_cash=args.initial_cash,
    )
    monthly_performance = _performance_delta(baseline_perf, challenger_perf)
    monthly_states = _monthly_state_features(
        Path(args.dataset_dir),
        start=start,
        end=end,
        state_columns=tuple(args.state_columns),
    )
    monthly_score_attribution = _monthly_score_attribution(
        baseline_score_dir=Path(args.baseline_score_dir),
        challenger_score_dir=Path(args.challenger_score_dir),
        dataset_dir=Path(args.dataset_dir),
        start=start,
        end=end,
        top_n=args.top_n,
        label_column=args.label_column,
    )
    monthly_attribution = monthly_performance.merge(
        monthly_states,
        on="month",
        how="left",
    ).merge(
        monthly_score_attribution,
        on="month",
        how="left",
    )
    monthly_switch_rules = _monthly_switch_rules(monthly_attribution)
    monthly_performance.to_csv(output_dir / "monthly_performance.csv", index=False)
    monthly_states.to_csv(output_dir / "monthly_states.csv", index=False)
    monthly_score_attribution.to_csv(
        output_dir / "monthly_score_attribution.csv",
        index=False,
    )
    monthly_attribution.to_csv(output_dir / "monthly_attribution.csv", index=False)
    monthly_switch_rules.to_csv(output_dir / "monthly_switch_rules.csv", index=False)
    summary = {
        "status": "completed",
        "params": {
            "baseline_backtest_dir": args.baseline_backtest_dir,
            "challenger_backtest_dir": args.challenger_backtest_dir,
            "baseline_score_dir": args.baseline_score_dir,
            "challenger_score_dir": args.challenger_score_dir,
            "dataset_dir": args.dataset_dir,
            "start": args.start,
            "end": args.end,
            "top_n": args.top_n,
            "label_column": args.label_column,
            "state_columns": list(args.state_columns),
        },
        "performance": _performance_summary(monthly_attribution),
        "outputs": {
            "monthly_performance": str(output_dir / "monthly_performance.csv"),
            "monthly_states": str(output_dir / "monthly_states.csv"),
            "monthly_score_attribution": str(
                output_dir / "monthly_score_attribution.csv"
            ),
            "monthly_attribution": str(output_dir / "monthly_attribution.csv"),
            "monthly_switch_rules": str(output_dir / "monthly_switch_rules.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _monthly_performance(
    backtest_dir: Path,
    *,
    method: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_cash: float,
) -> pd.DataFrame:
    equity_path = backtest_dir / "equity_curve.csv"
    trades_path = backtest_dir / "trades.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"missing equity curve: {equity_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"missing trades file: {trades_path}")
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"])
    trades = pd.read_csv(trades_path, parse_dates=["timestamp"])
    equity = equity.loc[
        (equity["timestamp"] >= start) & (equity["timestamp"] <= end)
    ].copy()
    trades = trades.loc[
        (trades["timestamp"] >= start) & (trades["timestamp"] <= end)
    ].copy()
    if equity.empty:
        return pd.DataFrame(
            columns=[
                "month",
                f"{method}_return",
                f"{method}_max_drawdown",
                f"{method}_trade_count",
                f"{method}_total_transaction_cost",
                f"{method}_gross_traded_notional",
            ]
        )
    equity["month"] = equity["timestamp"].dt.strftime("%Y-%m")
    trades["month"] = (
        trades["timestamp"].dt.strftime("%Y-%m")
        if not trades.empty
        else pd.Series(dtype="object")
    )
    previous_equity = float(initial_cash)
    rows: list[dict[str, Any]] = []
    for month, month_end_row in equity.groupby("month", sort=True).tail(1).set_index(
        "month"
    ).iterrows():
        month_equity = equity.loc[equity["month"] == month, "equity"].reset_index(
            drop=True
        )
        end_equity = float(month_end_row["equity"])
        curve = pd.concat(
            [pd.Series([previous_equity]), month_equity],
            ignore_index=True,
        )
        drawdown = float((curve / curve.cummax() - 1.0).min())
        month_trades = trades.loc[trades["month"] == month]
        rows.append(
            {
                "month": str(month),
                f"{method}_return": end_equity / previous_equity - 1.0,
                f"{method}_max_drawdown": drawdown,
                f"{method}_trade_count": int(len(month_trades)),
                f"{method}_total_transaction_cost": (
                    float(month_trades["total_cost"].sum())
                    if not month_trades.empty
                    else 0.0
                ),
                f"{method}_gross_traded_notional": (
                    float(month_trades["notional"].abs().sum())
                    if not month_trades.empty
                    else 0.0
                ),
            }
        )
        previous_equity = end_equity
    return pd.DataFrame(rows)


def _performance_delta(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
) -> pd.DataFrame:
    frame = baseline.merge(challenger, on="month", how="outer", sort=True)
    frame["return_delta"] = frame["challenger_return"] - frame["baseline_return"]
    frame["max_drawdown_delta"] = (
        frame["challenger_max_drawdown"] - frame["baseline_max_drawdown"]
    )
    frame["trade_count_delta"] = (
        frame["challenger_trade_count"] - frame["baseline_trade_count"]
    )
    frame["transaction_cost_delta"] = (
        frame["challenger_total_transaction_cost"]
        - frame["baseline_total_transaction_cost"]
    )
    return frame


def _monthly_state_features(
    dataset_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    state_columns: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in _dataset_paths(dataset_dir, start=start, end=end):
        available = _available_columns(path)
        columns = ["timestamp", *(column for column in state_columns if column in available)]
        if len(columns) <= 1:
            continue
        frame = pd.read_parquet(path, columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.loc[
            (frame["timestamp"] >= start) & (frame["timestamp"] <= end)
        ].copy()
        if frame.empty:
            continue
        by_timestamp = frame.groupby("timestamp", sort=True).mean(numeric_only=True)
        by_timestamp["month"] = by_timestamp.index.strftime("%Y-%m")
        rows.append(by_timestamp.reset_index(drop=True))
    if not rows:
        return pd.DataFrame(columns=["month"])
    state = pd.concat(rows, ignore_index=True)
    aggregations: dict[str, tuple[str, str]] = {
        "state_timestamp_count": ("month", "size"),
    }
    for column in state_columns:
        if column in state.columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            if column.endswith("return_5m"):
                aggregations[f"{column}_sum"] = (column, "sum")
    return state.groupby("month", sort=True).agg(**aggregations).reset_index()


def _monthly_score_attribution(
    *,
    baseline_score_dir: Path,
    challenger_score_dir: Path,
    dataset_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int,
    label_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_path in _dataset_paths(dataset_dir, start=start, end=end):
        partition = dataset_path.stem.removeprefix("dataset_")
        baseline_path = baseline_score_dir / f"score_{partition}.parquet"
        challenger_path = challenger_score_dir / f"score_{partition}.parquet"
        if not baseline_path.exists() or not challenger_path.exists():
            continue
        labels = pd.read_parquet(
            dataset_path,
            columns=["timestamp", "instrument_id", label_column],
        )
        labels["timestamp"] = _score_timestamp_strings(labels["timestamp"])
        labels = labels.loc[
            labels[label_column].notna(),
            ["timestamp", "instrument_id", label_column],
        ]
        baseline_top = _top_n_scores(
            pd.read_parquet(baseline_path),
            top_n=top_n,
        ).rename(columns={"score": "baseline_score"})
        challenger_top = _top_n_scores(
            pd.read_parquet(challenger_path),
            top_n=top_n,
        ).rename(columns={"score": "challenger_score"})
        joined = baseline_top.merge(
            challenger_top,
            on=["timestamp", "instrument_id"],
            how="outer",
            indicator=True,
        ).merge(labels, on=["timestamp", "instrument_id"], how="left")
        joined["month"] = pd.to_datetime(joined["timestamp"], utc=True).dt.strftime(
            "%Y-%m"
        )
        for month, group in joined.groupby("month", sort=True):
            both = group["_merge"] == "both"
            baseline_only = group["_merge"] == "left_only"
            challenger_only = group["_merge"] == "right_only"
            rows.append(
                {
                    "month": str(month),
                    "top_union_count": int(len(group)),
                    "top_overlap_count": int(both.sum()),
                    "top_overlap_share": (
                        float(both.sum() / max(len(group), 1)) if len(group) else 0.0
                    ),
                    "baseline_only_count": int(baseline_only.sum()),
                    "challenger_only_count": int(challenger_only.sum()),
                    "baseline_top_label_mean": _mean_label(
                        group.loc[baseline_only | both, label_column]
                    ),
                    "challenger_top_label_mean": _mean_label(
                        group.loc[challenger_only | both, label_column]
                    ),
                    "baseline_only_label_mean": _mean_label(
                        group.loc[baseline_only, label_column]
                    ),
                    "challenger_only_label_mean": _mean_label(
                        group.loc[challenger_only, label_column]
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["month"])
    output = pd.DataFrame(rows)
    grouped = output.groupby("month", sort=True).agg(
        top_union_count=("top_union_count", "sum"),
        top_overlap_count=("top_overlap_count", "sum"),
        baseline_only_count=("baseline_only_count", "sum"),
        challenger_only_count=("challenger_only_count", "sum"),
        baseline_top_label_mean=("baseline_top_label_mean", "mean"),
        challenger_top_label_mean=("challenger_top_label_mean", "mean"),
        baseline_only_label_mean=("baseline_only_label_mean", "mean"),
        challenger_only_label_mean=("challenger_only_label_mean", "mean"),
    )
    grouped["top_overlap_share"] = (
        grouped["top_overlap_count"] / grouped["top_union_count"].replace(0, pd.NA)
    )
    grouped["top_label_delta"] = (
        grouped["challenger_top_label_mean"] - grouped["baseline_top_label_mean"]
    )
    grouped["replacement_label_delta"] = (
        grouped["challenger_only_label_mean"] - grouped["baseline_only_label_mean"]
    )
    return grouped.reset_index()


def _top_n_scores(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    _require_columns(frame, ("timestamp", "instrument_id", "score"))
    output = frame.loc[:, ["timestamp", "instrument_id", "score"]].copy()
    output["timestamp"] = _score_timestamp_strings(output["timestamp"])
    output = output.loc[output["score"].notna()].sort_values(
        ["timestamp", "score", "instrument_id"],
        ascending=[True, False, True],
    )
    output["rank"] = output.groupby("timestamp", sort=False).cumcount() + 1
    return output.loc[
        output["rank"] <= top_n,
        ["timestamp", "instrument_id", "score"],
    ]


def _performance_summary(monthly_attribution: pd.DataFrame) -> dict[str, Any]:
    if monthly_attribution.empty:
        return {"month_count": 0}
    positive = monthly_attribution.loc[monthly_attribution["return_delta"] > 0]
    negative = monthly_attribution.loc[monthly_attribution["return_delta"] <= 0]
    return {
        "month_count": int(len(monthly_attribution)),
        "positive_delta_month_count": int(len(positive)),
        "negative_delta_month_count": int(len(negative)),
        "mean_monthly_return_delta": _nullable_float(
            monthly_attribution["return_delta"].mean()
        ),
        "mean_positive_month_state": _state_summary(positive),
        "mean_negative_month_state": _state_summary(negative),
        "worst_delta_months": _month_records(
            monthly_attribution.sort_values("return_delta").head(5)
        ),
        "best_delta_months": _month_records(
            monthly_attribution.sort_values("return_delta", ascending=False).head(5)
        ),
    }


def _monthly_switch_rules(monthly_attribution: pd.DataFrame) -> pd.DataFrame:
    if monthly_attribution.empty:
        return pd.DataFrame(
            columns=[
                "rule",
                "threshold",
                "ml_month_count",
                "positive_ml_month_count",
                "compound_return",
            ]
        )
    rows: list[dict[str, Any]] = []
    baseline_return = _compound_return(monthly_attribution["baseline_return"])
    challenger_return = _compound_return(monthly_attribution["challenger_return"])
    rows.append(
        {
            "rule": "always_baseline",
            "threshold": None,
            "ml_month_count": 0,
            "positive_ml_month_count": 0,
            "compound_return": baseline_return,
        }
    )
    rows.append(
        {
            "rule": "always_challenger",
            "threshold": None,
            "ml_month_count": int(len(monthly_attribution)),
            "positive_ml_month_count": int((monthly_attribution["return_delta"] > 0).sum()),
            "compound_return": challenger_return,
        }
    )
    state_columns = [
        column
        for column in monthly_attribution.columns
        if column.startswith("market_state_")
        and (column.endswith("_mean") or column.endswith("_sum"))
    ]
    for column in state_columns:
        lagged = monthly_attribution[column].shift(1)
        valid = lagged.notna()
        if not valid.any():
            continue
        for quantile in (0.33, 0.50, 0.67):
            threshold = float(lagged.loc[valid].quantile(quantile))
            for operator in ("<=", ">="):
                use_ml = (
                    (lagged <= threshold)
                    if operator == "<="
                    else (lagged >= threshold)
                ).fillna(False)
                returns = monthly_attribution["baseline_return"].where(
                    ~use_ml,
                    monthly_attribution["challenger_return"],
                )
                rows.append(
                    {
                        "rule": f"lag1_{column}{operator}q{quantile:.2f}",
                        "threshold": threshold,
                        "ml_month_count": int(use_ml.sum()),
                        "positive_ml_month_count": int(
                            ((monthly_attribution["return_delta"] > 0) & use_ml).sum()
                        ),
                        "compound_return": _compound_return(returns),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "compound_return",
        ascending=False,
    )


def _compound_return(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() - 1.0)


def _state_summary(frame: pd.DataFrame) -> dict[str, float | None]:
    state_columns = [
        column
        for column in frame.columns
        if column.startswith("market_state_") and column.endswith("_mean")
    ]
    return {column: _nullable_float(frame[column].mean()) for column in state_columns}


def _month_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "month",
        "baseline_return",
        "challenger_return",
        "return_delta",
        "top_label_delta",
        "replacement_label_delta",
    ]
    return [
        {
            field: (
                _nullable_float(row[field])
                if field != "month" and field in row
                else row.get(field)
            )
            for field in fields
            if field in row
        }
        for row in frame.to_dict("records")
    ]


def _dataset_paths(
    dataset_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    start_partition = start.tz_convert("Asia/Shanghai").strftime("%Y_%m")
    end_partition = end.tz_convert("Asia/Shanghai").strftime("%Y_%m")
    return [
        path
        for path in sorted(dataset_dir.glob("dataset_*.parquet"))
        if start_partition <= path.stem.removeprefix("dataset_") <= end_partition
    ]


def _available_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


def _score_timestamp_strings(timestamp: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(timestamp, utc=True, errors="coerce")
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
        + "+08:00"
    )


def _mean_label(values: pd.Series) -> float | None:
    if values.empty:
        return None
    value = values.mean()
    return None if pd.isna(value) else float(value)


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-backtest-dir", required=True)
    parser.add_argument("--challenger-backtest-dir", required=True)
    parser.add_argument("--baseline-score-dir", required=True)
    parser.add_argument("--challenger-score-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--label-column", default="forward_return_48b")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--state-columns", nargs="+", default=list(DEFAULT_STATE_COLUMNS))
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "baseline_backtest_dir",
        "challenger_backtest_dir",
        "baseline_score_dir",
        "challenger_score_dir",
        "dataset_dir",
    ):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.initial_cash <= 0:
        raise ValueError("--initial-cash must be positive")


if __name__ == "__main__":
    main()
