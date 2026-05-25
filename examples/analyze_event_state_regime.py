"""Analyze event-state regimes for event-shock proxy factor runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_EVENT_FEATURES = (
    "intraday_event_sync_down_resilience_5m_w48",
    "intraday_event_limit_diffusion_resilience_5m_w48",
    "intraday_event_turnover_dislocation_recovery_5m_w48",
    "intraday_event_open_jump_recovery_quality_5m_w48",
)


def main() -> None:
    args = _parse_args()
    summary = analyze_event_state_regime(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def analyze_event_state_regime(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_features = tuple(args.event_feature_columns)
    timestamp_parts = [
        _partition_timestamp_metrics(
            dataset_path,
            score_path=_score_path(args, dataset_path),
            event_features=event_features,
            label_column=args.label_column,
            top_n=args.top_n,
        )
        for dataset_path in _dataset_paths(args)
    ]
    if not timestamp_parts:
        raise ValueError("no dataset partitions selected")
    timestamp_metrics = pd.concat(timestamp_parts, ignore_index=True)
    timestamp_metrics = _add_event_states(
        timestamp_metrics,
        lookback_windows=args.lookback_windows,
        min_periods=args.min_periods,
        high_z=args.high_z,
        extreme_z=args.extreme_z,
        max_z_score=args.max_z_score,
        stabilization_windows=args.stabilization_windows,
    )
    monthly_pnl = _load_monthly_pnl(
        Path(args.validation_dir) if args.validation_dir else None,
        scenario=args.scenario,
        method=args.method,
        policy=args.policy,
    )
    state_summary = _state_summary(timestamp_metrics)
    monthly_state_summary = _monthly_state_summary(timestamp_metrics)
    monthly_summary = _monthly_summary(timestamp_metrics, monthly_pnl)

    timestamp_path = output_dir / "timestamp_event_states.csv"
    state_path = output_dir / "event_state_performance.csv"
    monthly_state_path = output_dir / "monthly_event_state_performance.csv"
    monthly_path = output_dir / "monthly_event_state_summary.csv"
    report_path = output_dir / "event_state_regime_report.md"
    timestamp_metrics.to_csv(timestamp_path, index=False)
    state_summary.to_csv(state_path, index=False)
    monthly_state_summary.to_csv(monthly_state_path, index=False)
    monthly_summary.to_csv(monthly_path, index=False)
    report_path.write_text(
        _render_report(
            args,
            state_summary=state_summary,
            monthly_summary=monthly_summary,
            monthly_state_summary=monthly_state_summary,
        ),
        encoding="utf-8",
    )
    summary = {
        "params": {
            "dataset_dir": args.dataset_dir,
            "validation_dir": args.validation_dir,
            "scenario": args.scenario,
            "method": args.method,
            "policy": args.policy,
            "label_column": args.label_column,
            "top_n": args.top_n,
            "event_feature_columns": list(event_features),
            "lookback_windows": args.lookback_windows,
            "min_periods": args.min_periods,
            "high_z": args.high_z,
            "extreme_z": args.extreme_z,
            "max_z_score": args.max_z_score,
            "stabilization_windows": args.stabilization_windows,
            "months": args.months,
        },
        "artifacts": {
            "timestamp_event_states": str(timestamp_path),
            "event_state_performance": str(state_path),
            "monthly_event_state_performance": str(monthly_state_path),
            "monthly_event_state_summary": str(monthly_path),
            "report": str(report_path),
        },
        "timestamp_count": int(len(timestamp_metrics)),
        "state_counts": {
            str(key): int(value)
            for key, value in timestamp_metrics["event_state"].value_counts().items()
        },
        "worst_months": _json_records(
            monthly_summary.sort_values("portfolio_return", na_position="last").head(
                args.report_months
            )
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    dataset_dir = Path(args.dataset_dir)
    if args.months:
        paths = [dataset_dir / f"dataset_{month}.parquet" for month in args.months]
    else:
        paths = sorted(dataset_dir.glob("dataset_*.parquet"))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dataset partitions: {missing}")
    if not paths:
        raise FileNotFoundError(f"no dataset_*.parquet files found under {dataset_dir}")
    return paths


def _score_path(args: argparse.Namespace, dataset_path: Path) -> Path:
    partition = dataset_path.stem.removeprefix("dataset_")
    if args.score_dir:
        path = Path(args.score_dir) / f"score_{partition}.parquet"
    else:
        path = (
            Path(args.validation_dir)
            / args.scenario
            / "scores"
            / args.method
            / f"score_{partition}.parquet"
        )
    if not path.exists():
        raise FileNotFoundError(f"missing score partition: {path}")
    return path


def _partition_timestamp_metrics(
    dataset_path: Path,
    *,
    score_path: Path,
    event_features: tuple[str, ...],
    label_column: str,
    top_n: int,
) -> pd.DataFrame:
    required_columns = [
        "timestamp",
        "instrument_id",
        label_column,
        "entry_tradable_bar",
        "entry_limit_up_open",
        "entry_limit_down_open",
        *event_features,
    ]
    dataset = pd.read_parquet(dataset_path, columns=required_columns)
    scores = pd.read_parquet(score_path, columns=["timestamp", "instrument_id", "score"])
    frame = dataset.merge(scores, on=["timestamp", "instrument_id"], how="inner")
    if frame.empty:
        raise ValueError(f"empty dataset-score join for {dataset_path}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    for column in ("entry_tradable_bar", "entry_limit_up_open", "entry_limit_down_open"):
        frame[column] = frame[column].astype(float)
    for feature in event_features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    valid = frame.dropna(subset=["score", label_column]).copy()
    if valid.empty:
        raise ValueError(f"no valid score/label rows for {dataset_path}")
    grouped = valid.groupby("timestamp", sort=True)
    valid["score_rank"] = grouped["score"].rank(method="average", pct=True)
    valid["label_rank"] = grouped[label_column].rank(method="average", pct=True)
    valid["score_desc_rank"] = grouped["score"].rank(method="first", ascending=False)
    valid["score_asc_rank"] = grouped["score"].rank(method="first", ascending=True)
    top = valid[valid["score_desc_rank"] <= top_n]
    bottom = valid[valid["score_asc_rank"] <= top_n]

    agg_spec: dict[str, tuple[str, str]] = {
        "sample_count": ("instrument_id", "size"),
        "market_mean_label": (label_column, "mean"),
        "market_median_label": (label_column, "median"),
        "market_label_std": (label_column, "std"),
        "entry_tradable_rate": ("entry_tradable_bar", "mean"),
        "entry_limit_up_open_rate": ("entry_limit_up_open", "mean"),
        "entry_limit_down_open_rate": ("entry_limit_down_open", "mean"),
    }
    for feature in event_features:
        prefix = _feature_prefix(feature)
        agg_spec[f"{prefix}_mean"] = (feature, "mean")
        agg_spec[f"{prefix}_std"] = (feature, "std")
        agg_spec[f"{prefix}_p10"] = (feature, lambda values: values.quantile(0.10))
        agg_spec[f"{prefix}_p90"] = (feature, lambda values: values.quantile(0.90))
    metrics = grouped.agg(**agg_spec).reset_index()
    metrics["month"] = metrics["timestamp"].dt.strftime("%Y-%m")
    rank_ic = _grouped_corr(valid, group_column="timestamp", left="score_rank", right="label_rank")
    metrics = metrics.merge(rank_ic, on="timestamp", how="left")
    top_metrics = _top_bottom_metrics(
        top,
        bottom,
        event_features=event_features,
        label_column=label_column,
    )
    metrics = metrics.merge(top_metrics, on="timestamp", how="left")
    metrics["score_top_minus_universe_label"] = (
        metrics["score_top_n_mean_label"] - metrics["market_mean_label"]
    )
    return metrics.sort_values("timestamp").reset_index(drop=True)


def _top_bottom_metrics(
    top: pd.DataFrame,
    bottom: pd.DataFrame,
    *,
    event_features: tuple[str, ...],
    label_column: str,
) -> pd.DataFrame:
    top_spec: dict[str, tuple[str, str]] = {
        "score_top_n_count": ("instrument_id", "size"),
        "score_top_n_mean_label": (label_column, "mean"),
    }
    bottom_spec: dict[str, tuple[str, str]] = {
        "score_bottom_n_count": ("instrument_id", "size"),
        "score_bottom_n_mean_label": (label_column, "mean"),
    }
    for feature in event_features:
        top_spec[f"top_score_{_feature_prefix(feature)}_mean"] = (feature, "mean")
    top_metrics = top.groupby("timestamp", sort=True).agg(**top_spec).reset_index()
    bottom_metrics = bottom.groupby("timestamp", sort=True).agg(**bottom_spec).reset_index()
    metrics = top_metrics.merge(bottom_metrics, on="timestamp", how="outer")
    metrics["score_top_minus_bottom_label"] = (
        metrics["score_top_n_mean_label"] - metrics["score_bottom_n_mean_label"]
    )
    return metrics


def _grouped_corr(
    frame: pd.DataFrame,
    *,
    group_column: str,
    left: str,
    right: str,
) -> pd.DataFrame:
    values = frame.loc[:, [group_column, left, right]].dropna().copy()
    if values.empty:
        return pd.DataFrame(columns=[group_column, "score_rank_ic"])
    values["left_sq"] = values[left] * values[left]
    values["right_sq"] = values[right] * values[right]
    values["left_right"] = values[left] * values[right]
    grouped = values.groupby(group_column, sort=True)
    stats = grouped.agg(
        n=(left, "size"),
        left_mean=(left, "mean"),
        right_mean=(right, "mean"),
        left_sq_mean=("left_sq", "mean"),
        right_sq_mean=("right_sq", "mean"),
        left_right_mean=("left_right", "mean"),
    ).reset_index()
    covariance = stats["left_right_mean"] - stats["left_mean"] * stats["right_mean"]
    left_var = stats["left_sq_mean"] - stats["left_mean"] * stats["left_mean"]
    right_var = stats["right_sq_mean"] - stats["right_mean"] * stats["right_mean"]
    denominator = (left_var.clip(lower=0.0) * right_var.clip(lower=0.0)).pow(0.5)
    stats["score_rank_ic"] = covariance / denominator.where(denominator != 0.0)
    stats.loc[stats["n"] < 2, "score_rank_ic"] = np.nan
    return stats.loc[:, [group_column, "score_rank_ic"]]


def _add_event_states(
    timestamp_metrics: pd.DataFrame,
    *,
    lookback_windows: int,
    min_periods: int,
    high_z: float,
    extreme_z: float,
    max_z_score: float,
    stabilization_windows: int,
) -> pd.DataFrame:
    output = timestamp_metrics.sort_values("timestamp").reset_index(drop=True).copy()
    output["limit_pressure_rate"] = (
        output["entry_limit_up_open_rate"] + output["entry_limit_down_open_rate"]
    )
    output["limit_down_imbalance"] = (
        output["entry_limit_down_open_rate"] - output["entry_limit_up_open_rate"]
    )
    stress_sources = [
        "limit_pressure_rate",
        "limit_down_imbalance",
        "sync_down_resilience_std",
        "limit_diffusion_resilience_std",
        "turnover_dislocation_recovery_std",
        "open_jump_recovery_quality_std",
    ]
    z_columns: list[str] = []
    for column in stress_sources:
        if column not in output.columns:
            continue
        z_column = f"{column}_rolling_z"
        output[z_column] = _lagged_rolling_z(
            output[column],
            lookback_windows=lookback_windows,
            min_periods=min_periods,
            max_abs_z=max_z_score,
        )
        z_columns.append(z_column)
    if not z_columns:
        raise ValueError("no event-state stress columns were available")
    output["event_intensity_score"] = (
        output[z_columns].clip(lower=0.0, upper=max_z_score).mean(axis=1).fillna(0.0)
    )
    rolling_observation_count = (
        output["event_intensity_score"]
        .shift(1)
        .rolling(lookback_windows, min_periods=1)
        .count()
        .astype(int)
    )
    output["event_state_observation_count"] = rolling_observation_count
    prior_intensity_max = (
        output["event_intensity_score"]
        .shift(1)
        .rolling(stabilization_windows, min_periods=1)
        .max()
    )
    output["prior_event_intensity_max"] = prior_intensity_max
    limit_pressure_z = output.get(
        "limit_pressure_rate_rolling_z",
        pd.Series(np.nan, index=output.index),
    )
    limit_down_z = output.get(
        "limit_down_imbalance_rolling_z",
        pd.Series(np.nan, index=output.index),
    )
    output["event_state"] = "calm"
    warmup = rolling_observation_count < min_periods
    limit_extreme = limit_pressure_z.ge(extreme_z) | limit_down_z.ge(extreme_z)
    limit_elevated = limit_pressure_z.ge(high_z) | limit_down_z.ge(high_z)
    shock_extreme = output["event_intensity_score"].ge(extreme_z)
    shock_elevated = output["event_intensity_score"].ge(high_z)
    stabilization = prior_intensity_max.ge(high_z) & ~shock_elevated & ~limit_elevated
    output.loc[stabilization, "event_state"] = "post_shock_stabilization"
    output.loc[shock_elevated, "event_state"] = "shock_elevated"
    output.loc[limit_elevated, "event_state"] = "limit_diffusion"
    output.loc[shock_extreme, "event_state"] = "shock_extreme"
    output.loc[limit_extreme, "event_state"] = "limit_diffusion_extreme"
    output.loc[warmup, "event_state"] = "warmup"
    return output


def _lagged_rolling_z(
    series: pd.Series,
    *,
    lookback_windows: int,
    min_periods: int,
    max_abs_z: float,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    historical = numeric.shift(1)
    rolling = historical.rolling(lookback_windows, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std()
    z_score = (numeric - mean) / std.where(std != 0.0)
    return z_score.clip(lower=-max_abs_z, upper=max_abs_z)


def _state_summary(timestamp_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = timestamp_metrics.groupby("event_state", sort=True)
    summary = grouped.agg(
        timestamp_count=("timestamp", "size"),
        market_mean_label=("market_mean_label", "mean"),
        score_rank_ic_mean=("score_rank_ic", "mean"),
        score_top_n_mean_label=("score_top_n_mean_label", "mean"),
        score_top_minus_universe_label=("score_top_minus_universe_label", "mean"),
        score_top_minus_bottom_label=("score_top_minus_bottom_label", "mean"),
        event_intensity_mean=("event_intensity_score", "mean"),
        limit_pressure_rate_mean=("limit_pressure_rate", "mean"),
        limit_down_imbalance_mean=("limit_down_imbalance", "mean"),
    ).reset_index()
    summary["timestamp_share"] = summary["timestamp_count"] / max(
        int(summary["timestamp_count"].sum()),
        1,
    )
    return summary.sort_values("score_top_n_mean_label").reset_index(drop=True)


def _monthly_state_summary(timestamp_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = timestamp_metrics.groupby(["month", "event_state"], sort=True)
    summary = grouped.agg(
        timestamp_count=("timestamp", "size"),
        market_mean_label=("market_mean_label", "mean"),
        score_rank_ic_mean=("score_rank_ic", "mean"),
        score_top_n_mean_label=("score_top_n_mean_label", "mean"),
        score_top_minus_universe_label=("score_top_minus_universe_label", "mean"),
        score_top_minus_bottom_label=("score_top_minus_bottom_label", "mean"),
        event_intensity_mean=("event_intensity_score", "mean"),
        limit_pressure_rate_mean=("limit_pressure_rate", "mean"),
    ).reset_index()
    month_counts = summary.groupby("month")["timestamp_count"].transform("sum")
    summary["month_state_share"] = summary["timestamp_count"] / month_counts.where(
        month_counts != 0
    )
    return summary


def _monthly_summary(
    timestamp_metrics: pd.DataFrame,
    monthly_pnl: pd.DataFrame,
) -> pd.DataFrame:
    monthly = timestamp_metrics.groupby("month", sort=True).agg(
        timestamp_count=("timestamp", "size"),
        market_mean_label=("market_mean_label", "mean"),
        score_rank_ic_mean=("score_rank_ic", "mean"),
        score_top_n_mean_label=("score_top_n_mean_label", "mean"),
        score_top_minus_universe_label=("score_top_minus_universe_label", "mean"),
        score_top_minus_bottom_label=("score_top_minus_bottom_label", "mean"),
        event_intensity_mean=("event_intensity_score", "mean"),
        limit_pressure_rate_mean=("limit_pressure_rate", "mean"),
    ).reset_index()
    state_counts = (
        timestamp_metrics.pivot_table(
            index="month",
            columns="event_state",
            values="timestamp",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    state_columns = [column for column in state_counts.columns if column != "month"]
    total = state_counts[state_columns].sum(axis=1).where(
        state_counts[state_columns].sum(axis=1) != 0
    )
    for column in state_columns:
        state_counts[f"state_share_{_slug(column)}"] = state_counts[column] / total
    state_share = state_counts.loc[
        :,
        ["month", *[f"state_share_{_slug(column)}" for column in state_columns]],
    ]
    monthly = monthly.merge(state_share, on="month", how="left")
    if not monthly_pnl.empty:
        monthly = monthly.merge(monthly_pnl, on="month", how="left")
    else:
        monthly["portfolio_return"] = np.nan
        monthly["portfolio_max_drawdown"] = np.nan
    return monthly


def _load_monthly_pnl(
    validation_dir: Path | None,
    *,
    scenario: str,
    method: str,
    policy: str,
) -> pd.DataFrame:
    if validation_dir is None:
        return pd.DataFrame(columns=["month", "portfolio_return"])
    path = validation_dir / "validation_monthly_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["month", "portfolio_return"])
    frame = pd.read_csv(path)
    frame = frame[
        (frame["scenario"] == scenario)
        & (frame["method"] == method)
        & (frame["policy"] == policy)
    ].copy()
    if frame.empty:
        return pd.DataFrame(columns=["month", "portfolio_return"])
    return frame.rename(
        columns={
            "return": "portfolio_return",
            "max_drawdown": "portfolio_max_drawdown",
            "trade_count": "portfolio_trade_count",
            "total_transaction_cost": "portfolio_transaction_cost",
        }
    )[
        [
            "month",
            "portfolio_return",
            "portfolio_max_drawdown",
            "portfolio_trade_count",
            "portfolio_transaction_cost",
        ]
    ]


def _render_report(
    args: argparse.Namespace,
    *,
    state_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    monthly_state_summary: pd.DataFrame,
) -> str:
    worst = monthly_summary.sort_values("portfolio_return", na_position="last").head(
        args.report_months
    )
    best = monthly_summary.sort_values("portfolio_return", ascending=False).head(
        args.report_months
    )
    lines = [
        "# Event-State Regime Diagnostic",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Validation: `{args.validation_dir}`",
        f"- Scenario: `{args.scenario}`",
        f"- Method: `{args.method}`",
        f"- Policy: `{args.policy}`",
        f"- Top N: `{args.top_n}`",
        f"- Max z-score: `{args.max_z_score}`",
        "",
        "## Event-State Performance",
        "",
        "| state | timestamps | share | top label | top-universe | score IC | intensity | limit pressure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in state_summary.to_dict("records"):
        lines.append(
            "| {state} | {count} | {share} | {top} | {edge} | {ic} | {intensity} | {limit} |".format(
                state=row["event_state"],
                count=int(row["timestamp_count"]),
                share=_format_pct(row.get("timestamp_share")),
                top=_format_pct(row.get("score_top_n_mean_label")),
                edge=_format_pct(row.get("score_top_minus_universe_label")),
                ic=_format_number(row.get("score_rank_ic_mean")),
                intensity=_format_number(row.get("event_intensity_mean")),
                limit=_format_pct(row.get("limit_pressure_rate_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Worst Months",
            "",
            "| month | portfolio return | top label | top-universe | intensity | dominant state |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in worst.to_dict("records"):
        lines.append(_month_row(row, monthly_state_summary))
    lines.extend(
        [
            "",
            "## Best Months",
            "",
            "| month | portfolio return | top label | top-universe | intensity | dominant state |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in best.to_dict("records"):
        lines.append(_month_row(row, monthly_state_summary))
    lines.append("")
    return "\n".join(lines)


def _month_row(row: dict[str, Any], monthly_state_summary: pd.DataFrame) -> str:
    month = str(row["month"])
    month_states = monthly_state_summary[monthly_state_summary["month"].astype(str) == month]
    dominant = "-"
    if not month_states.empty:
        state_row = month_states.sort_values("month_state_share", ascending=False).iloc[0]
        dominant = f"{state_row['event_state']} ({_format_pct(state_row['month_state_share'])})"
    return "| {month} | {ret} | {top} | {edge} | {intensity} | {state} |".format(
        month=month,
        ret=_format_pct(row.get("portfolio_return")),
        top=_format_pct(row.get("score_top_n_mean_label")),
        edge=_format_pct(row.get("score_top_minus_universe_label")),
        intensity=_format_number(row.get("event_intensity_mean")),
        state=dominant,
    )


def _feature_prefix(feature: str) -> str:
    mapping = {
        "intraday_event_sync_down_resilience_5m_w48": "sync_down_resilience",
        "intraday_event_limit_diffusion_resilience_5m_w48": "limit_diffusion_resilience",
        "intraday_event_turnover_dislocation_recovery_5m_w48": "turnover_dislocation_recovery",
        "intraday_event_open_jump_recovery_quality_5m_w48": "open_jump_recovery_quality",
    }
    return mapping.get(feature, _slug(feature))


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.replace({np.nan: None}).to_dict("records")
    return [{str(key): _json_value(value) for key, value in record.items()} for record in records]


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _format_pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _format_number(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--validation-dir",
        help=(
            "candidate policy validation directory containing scenario scores and "
            "validation_monthly_summary.csv"
        ),
    )
    parser.add_argument(
        "--score-dir",
        help=(
            "optional direct score partition directory; defaults to "
            "<validation-dir>/<scenario>/scores/<method>"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario", default="full_base")
    parser.add_argument("--method", default="decorrelated")
    parser.add_argument("--policy", default="partial_rebalance_daily")
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument(
        "--event-feature-columns",
        nargs="+",
        default=list(DEFAULT_EVENT_FEATURES),
    )
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--lookback-windows", type=int, default=240)
    parser.add_argument("--min-periods", type=int, default=48)
    parser.add_argument("--high-z", type=float, default=0.75)
    parser.add_argument("--extreme-z", type=float, default=1.50)
    parser.add_argument("--max-z-score", type=float, default=6.0)
    parser.add_argument("--stabilization-windows", type=int, default=12)
    parser.add_argument("--report-months", type=int, default=6)
    args = parser.parse_args()
    if args.validation_dir is None and args.score_dir is None:
        raise ValueError("either --validation-dir or --score-dir is required")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.lookback_windows <= 0:
        raise ValueError("--lookback-windows must be positive")
    if args.min_periods <= 0:
        raise ValueError("--min-periods must be positive")
    if args.min_periods > args.lookback_windows:
        raise ValueError("--min-periods must be <= --lookback-windows")
    if args.stabilization_windows <= 0:
        raise ValueError("--stabilization-windows must be positive")
    if args.high_z >= args.extreme_z:
        raise ValueError("--high-z must be below --extreme-z")
    if args.max_z_score <= 0.0:
        raise ValueError("--max-z-score must be positive")
    if args.max_z_score < args.extreme_z:
        raise ValueError("--max-z-score must be >= --extreme-z")
    if args.report_months <= 0:
        raise ValueError("--report-months must be positive")
    return args


if __name__ == "__main__":
    main()
