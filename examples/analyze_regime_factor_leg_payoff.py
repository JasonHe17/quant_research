"""Analyze candidate factor-leg payoff by observable regime buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.portfolio import CandidateFactor, load_candidate_factors  # noqa: E402


def main() -> None:
    args = _parse_args()
    summary = analyze_regime_factor_leg_payoff(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def analyze_regime_factor_leg_payoff(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidate_factors(
        Path(args.admission_report),
        statuses=tuple(args.statuses),
        include_features=tuple(args.include_features),
    )
    if not candidates:
        raise ValueError("no candidate factors selected")
    weights = _load_method_weights(
        Path(args.validation_dir),
        scenario=args.scenario,
        method=args.method,
    )
    health = _load_factor_health_schedule(args.factor_health_schedule)
    state = _load_state_table(
        Path(args.stress_schedule),
        event_state_path=Path(args.event_state_table) if args.event_state_table else None,
    )
    timestamp_rows: list[dict[str, Any]] = []
    for dataset_path in _dataset_paths(args):
        timestamp_rows.extend(
            _analyze_partition(
                dataset_path,
                candidates=candidates,
                weights=weights,
                factor_health=health,
                state=state,
                label_column=args.label_column,
                top_n=args.top_n,
            )
        )
    timestamp_frame = pd.DataFrame(timestamp_rows)
    if timestamp_frame.empty:
        raise ValueError("no timestamp diagnostics were produced")
    timestamp_path = output_dir / "factor_leg_payoff_by_timestamp.csv"
    bucket_path = output_dir / "factor_leg_payoff_by_bucket.csv"
    month_path = output_dir / "factor_leg_payoff_by_month.csv"
    timestamp_frame.to_csv(timestamp_path, index=False)
    bucket = _summarize(
        timestamp_frame,
        keys=["regime_bucket", "lag_event_state", "feature"],
    )
    month = _summarize(
        timestamp_frame,
        keys=["month", "regime_bucket", "feature"],
    )
    bucket.to_csv(bucket_path, index=False)
    month.to_csv(month_path, index=False)
    summary = {
        "params": {
            "dataset_dir": args.dataset_dir,
            "validation_dir": args.validation_dir,
            "scenario": args.scenario,
            "method": args.method,
            "label_column": args.label_column,
            "top_n": args.top_n,
            "months": args.months,
            "include_features": args.include_features,
            "stress_schedule": args.stress_schedule,
            "event_state_table": args.event_state_table,
            "factor_health_schedule": args.factor_health_schedule,
        },
        "artifacts": {
            "timestamp": str(timestamp_path),
            "bucket": str(bucket_path),
            "month": str(month_path),
        },
        "bucket_rows": int(len(bucket)),
        "timestamp_rows": int(len(timestamp_frame)),
        "best_buckets": (
            bucket.sort_values("top_minus_bottom_label_mean", ascending=False)
            .head(args.report_rows)
            .to_dict("records")
        ),
        "worst_buckets": (
            bucket.sort_values("top_minus_bottom_label_mean", ascending=True)
            .head(args.report_rows)
            .to_dict("records")
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    dataset_dir = Path(args.dataset_dir)
    months = args.months
    if not months:
        paths = sorted(dataset_dir.glob("dataset_*.parquet"))
    else:
        paths = [dataset_dir / f"dataset_{month}.parquet" for month in months]
    if args.partition_start:
        paths = [path for path in paths if _partition_name(path) >= args.partition_start]
    if args.partition_end:
        paths = [path for path in paths if _partition_name(path) <= args.partition_end]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dataset partitions: {missing}")
    if not paths:
        raise FileNotFoundError(f"no dataset partitions found under {dataset_dir}")
    return paths


def _partition_name(path: Path) -> str:
    return path.stem.removeprefix("dataset_")


def _load_method_weights(
    validation_dir: Path,
    *,
    scenario: str,
    method: str,
) -> dict[str, float]:
    summary_path = validation_dir / scenario / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    method_payload = payload.get("methods", {}).get(method)
    if not isinstance(method_payload, dict):
        raise KeyError(f"method {method!r} not found in {summary_path}")
    weights = method_payload.get("weights", {})
    return {str(key): float(value) for key, value in weights.items()}


def _load_factor_health_schedule(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["timestamp", "feature", "weight_scale"])
    schedule = pd.read_csv(path)
    missing = {"timestamp", "feature", "weight_scale"} - set(schedule.columns)
    if missing:
        raise ValueError(f"factor health schedule missing columns: {sorted(missing)}")
    output = schedule.loc[:, ["timestamp", "feature", "weight_scale"]].copy()
    output["feature"] = output["feature"].astype(str)
    output["weight_scale"] = pd.to_numeric(output["weight_scale"], errors="coerce")
    if output["weight_scale"].isna().any():
        raise ValueError("factor health schedule contains invalid weight_scale")
    return output


def _load_state_table(
    stress_schedule_path: Path,
    *,
    event_state_path: Path | None,
) -> pd.DataFrame:
    stress = pd.read_csv(stress_schedule_path)
    required = {"timestamp", "risk_value", "gross_exposure_scale", "risk_state"}
    missing = required - set(stress.columns)
    if missing:
        raise ValueError(f"stress schedule missing columns: {sorted(missing)}")
    state = stress.loc[
        :,
        ["timestamp", "risk_value", "gross_exposure_scale", "risk_state"],
    ].copy()
    state["dt"] = pd.to_datetime(state["timestamp"])
    state["risk_value"] = pd.to_numeric(state["risk_value"], errors="coerce")
    state["gross_exposure_scale"] = pd.to_numeric(
        state["gross_exposure_scale"],
        errors="coerce",
    ).fillna(1.0)
    if event_state_path is not None:
        event = pd.read_csv(
            event_state_path,
            usecols=[
                "timestamp",
                "event_state",
                "limit_pressure_rate",
                "event_intensity_score",
                "prior_event_intensity_max",
            ],
        )
        event["dt"] = pd.to_datetime(event["timestamp"])
        event = event.sort_values("dt").reset_index(drop=True)
        for column in [
            "event_state",
            "limit_pressure_rate",
            "event_intensity_score",
            "prior_event_intensity_max",
        ]:
            event[f"lag_{column}"] = event[column].shift(1)
        state = state.merge(
            event[
                [
                    "dt",
                    "lag_event_state",
                    "lag_limit_pressure_rate",
                    "lag_event_intensity_score",
                    "lag_prior_event_intensity_max",
                ]
            ],
            on="dt",
            how="left",
        )
    for column in [
        "lag_limit_pressure_rate",
        "lag_event_intensity_score",
        "lag_prior_event_intensity_max",
    ]:
        if column not in state.columns:
            state[column] = 0.0
        state[column] = pd.to_numeric(state[column], errors="coerce").fillna(0.0)
    if "lag_event_state" not in state.columns:
        state["lag_event_state"] = "missing"
    state["lag_event_state"] = state["lag_event_state"].fillna("missing").astype(str)
    state["regime_bucket"] = [
        _regime_bucket(
            scale=scale,
            risk_value=risk,
            event_state=event_state,
            limit_pressure=limit_pressure,
        )
        for scale, risk, event_state, limit_pressure in zip(
            state["gross_exposure_scale"],
            state["risk_value"],
            state["lag_event_state"],
            state["lag_limit_pressure_rate"],
            strict=True,
        )
    ]
    return state.drop(columns=["timestamp"]).drop_duplicates("dt", keep="last")


def _regime_bucket(
    *,
    scale: float,
    risk_value: float,
    event_state: str,
    limit_pressure: float,
) -> str:
    if scale >= 0.999999:
        return "calm"
    if event_state == "shock_extreme":
        return "stress_shock_extreme"
    if limit_pressure >= 0.003:
        return "stress_high_limit_pressure"
    if risk_value >= 0.17:
        return "stress_high_risk"
    if event_state in {"limit_diffusion", "limit_diffusion_extreme"}:
        return "stress_limit_diffusion_low_pressure"
    return "stress_weak_tape"


def _analyze_partition(
    dataset_path: Path,
    *,
    candidates: tuple[CandidateFactor, ...],
    weights: dict[str, float],
    factor_health: pd.DataFrame,
    state: pd.DataFrame,
    label_column: str,
    top_n: int,
) -> list[dict[str, Any]]:
    features = [candidate.feature for candidate in candidates]
    frame = pd.read_parquet(
        dataset_path,
        columns=["timestamp", "instrument_id", label_column, *features],
    )
    frame["dt"] = pd.to_datetime(frame["timestamp"])
    frame = frame.merge(state, on="dt", how="left")
    month = _partition_name(dataset_path).replace("_", "-")
    rows: list[dict[str, Any]] = []
    health_lookup = _health_lookup(factor_health)
    for timestamp, group in frame.groupby("timestamp", sort=True):
        rows.extend(
            _timestamp_rows(
                group,
                timestamp=str(timestamp),
                month=month,
                candidates=candidates,
                weights=weights,
                health_lookup=health_lookup,
                label_column=label_column,
                top_n=top_n,
            )
        )
    return rows


def _health_lookup(schedule: pd.DataFrame) -> dict[str, pd.Series]:
    if schedule.empty:
        return {}
    output: dict[str, pd.Series] = {}
    for feature, group in schedule.groupby("feature", sort=False):
        output[str(feature)] = group.drop_duplicates("timestamp", keep="last").set_index(
            "timestamp"
        )["weight_scale"].astype(float)
    return output


def _timestamp_rows(
    group: pd.DataFrame,
    *,
    timestamp: str,
    month: str,
    candidates: tuple[CandidateFactor, ...],
    weights: dict[str, float],
    health_lookup: dict[str, pd.Series],
    label_column: str,
    top_n: int,
) -> list[dict[str, Any]]:
    state_row = group.iloc[0]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        feature = candidate.feature
        valid = group.dropna(subset=[feature, label_column])
        if valid.empty:
            continue
        raw_rank = valid[feature].rank(method="average", pct=True)
        oriented_rank = raw_rank if candidate.direction > 0 else 1.0 - raw_rank
        n = min(top_n, len(valid))
        top = valid.loc[oriented_rank.nlargest(n).index]
        bottom = valid.loc[oriented_rank.nsmallest(n).index]
        base_weight = float(weights.get(feature, 0.0))
        weight_scale = _weight_scale(health_lookup, feature=feature, timestamp=timestamp)
        contribution = (oriented_rank - 0.5) * base_weight * weight_scale
        rows.append(
            {
                "timestamp": timestamp,
                "month": month,
                "feature": feature,
                "direction": int(candidate.direction),
                "sample_count": int(len(valid)),
                "regime_bucket": str(state_row.get("regime_bucket", "missing")),
                "risk_state": str(state_row.get("risk_state", "missing")),
                "lag_event_state": str(state_row.get("lag_event_state", "missing")),
                "risk_value": _float_or_none(state_row.get("risk_value")),
                "gross_exposure_scale": _float_or_none(
                    state_row.get("gross_exposure_scale")
                ),
                "lag_limit_pressure_rate": _float_or_none(
                    state_row.get("lag_limit_pressure_rate")
                ),
                "base_weight": base_weight,
                "weight_scale": weight_scale,
                "effective_weight": base_weight * weight_scale,
                "directional_rank_ic": _correlation(oriented_rank, valid[label_column]),
                "contribution_rank_ic": _correlation(contribution, valid[label_column]),
                "top_n_mean_label": _mean(top[label_column]),
                "bottom_n_mean_label": _mean(bottom[label_column]),
                "top_minus_bottom_label": _mean(top[label_column])
                - _mean(bottom[label_column]),
                "mean_abs_contribution": float(contribution.abs().mean()),
            }
        )
    return rows


def _weight_scale(
    health_lookup: dict[str, pd.Series],
    *,
    feature: str,
    timestamp: str,
) -> float:
    series = health_lookup.get(feature)
    if series is None:
        return 1.0
    value = series.get(timestamp, 1.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _summarize(frame: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False, sort=True)
    rows: list[dict[str, Any]] = []
    for values, group in grouped:
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values, strict=True))
        row.update(
            {
                "timestamp_count": int(len(group)),
                "sample_count": int(group["sample_count"].sum()),
                "directional_rank_ic_mean": _mean(group["directional_rank_ic"]),
                "directional_rank_ic_positive_rate": _positive_rate(
                    group["directional_rank_ic"]
                ),
                "contribution_rank_ic_mean": _mean(group["contribution_rank_ic"]),
                "top_n_mean_label_mean": _mean(group["top_n_mean_label"]),
                "bottom_n_mean_label_mean": _mean(group["bottom_n_mean_label"]),
                "top_minus_bottom_label_mean": _mean(group["top_minus_bottom_label"]),
                "weight_scale_mean": _mean(group["weight_scale"]),
                "effective_weight_mean": _mean(group["effective_weight"]),
                "mean_abs_contribution": _mean(group["mean_abs_contribution"]),
                "risk_value_mean": _mean(group["risk_value"]),
                "limit_pressure_rate_mean": _mean(group["lag_limit_pressure_rate"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _mean(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").mean())


def _positive_rate(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float((numeric > 0).mean())


def _correlation(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 2:
        return float("nan")
    return float(pair["left"].corr(pair["right"], method="spearman"))


def _float_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


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
        "--validation-dir",
        default="runs/candidate_factor_portfolios/partial_rebalance_validation_standard",
    )
    parser.add_argument("--scenario", default="full_base")
    parser.add_argument("--method", default="decorrelated")
    parser.add_argument("--factor-health-schedule")
    parser.add_argument(
        "--stress-schedule",
        required=True,
        help="gross exposure stress schedule with risk_value and risk_state columns",
    )
    parser.add_argument("--event-state-table")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--partition-start")
    parser.add_argument("--partition-end")
    parser.add_argument("--statuses", nargs="+", default=["candidate"])
    parser.add_argument("--include-features", nargs="+", default=[])
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--report-rows", type=int, default=10)
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.report_rows <= 0:
        raise ValueError("--report-rows must be positive")
    if not args.label_column:
        raise ValueError("--label-column must be non-empty")
    if args.partition_start and args.partition_end and args.partition_start > args.partition_end:
        raise ValueError("--partition-start must not be after --partition-end")
    return args


if __name__ == "__main__":
    main()
