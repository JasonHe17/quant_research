"""Train walk-forward LightGBM factor challenger scores.

The script is intentionally score-only: it writes OOS prediction parquet files
that can be consumed by ``examples/run_tree_score_backtest.py``.  Trading policy
validation remains in the existing backtest layer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.models import (  # noqa: E402
    TreeBaselineConfig,
    evaluate_cross_sectional_predictions,
    train_lightgbm_regressor,
)
from quant_research.portfolio import CandidateFactor, load_candidate_factors  # noqa: E402

SAMPLE_WEIGHT_COLUMN = "sample_weight"


@dataclass(frozen=True, slots=True)
class WalkForwardSpec:
    """One explicit purged walk-forward train/validation/test window."""

    name: str
    train_start: str | None
    train_end: str
    valid_start: str | None
    valid_end: str | None
    test_start: str
    test_end: str | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fold name must be non-empty")
        if not self.train_end:
            raise ValueError("fold train_end must be non-empty")
        if not self.test_start:
            raise ValueError("fold test_start must be non-empty")


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    """Prepared supervised matrix plus raw identifiers needed for scoring."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]


def main() -> None:
    args = _parse_args()
    summary = run_ml_factor_challenger(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def run_ml_factor_challenger(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir = output_dir / "scores" / "lightgbm"
    score_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_existing:
        for path in score_dir.glob("score_*.parquet"):
            path.unlink()

    raw_candidates = load_candidate_factors(
        Path(args.admission_report),
        statuses=tuple(args.statuses),
        evaluation_roles=tuple(args.evaluation_roles),
        include_features=tuple(args.include_features),
    )
    candidates, excluded_label_derived_features = _filter_label_derived_candidates(
        raw_candidates,
        label_column=args.label_column,
        include_features=tuple(args.include_features),
        allow_label_derived_features=args.allow_label_derived_features,
    )
    feature_columns = tuple(factor.feature for factor in candidates)
    directions = {factor.feature: factor.direction for factor in candidates}
    label_end_column = _label_exit_timestamp_column(args.label_column)
    dataset_paths = _dataset_paths(args)
    _validate_dataset_columns(
        dataset_paths,
        required_columns=(
            "timestamp",
            "instrument_id",
            args.label_column,
            label_end_column,
            *feature_columns,
        ),
    )
    folds = tuple(_parse_fold(value) for value in args.folds) or _default_folds(args)
    commands = {
        "backtest_example": _backtest_example_command(args, score_dir),
    }

    redundancy = _write_redundancy_diagnostics(
        dataset_paths,
        output_dir=output_dir / "redundancy",
        feature_columns=feature_columns,
        directions=directions,
        label_column=args.label_column,
        score_transform=args.score_transform,
        sample_rows=args.redundancy_sample_rows,
        random_seed=args.seed,
        correlation_threshold=args.correlation_threshold,
    )

    fold_summaries: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    oos_observations: list[pd.DataFrame] = []
    total_prediction_count = 0
    for fold in folds:
        fold_result = _run_fold(
            fold,
            args,
            dataset_paths=dataset_paths,
            feature_columns=feature_columns,
            directions=directions,
            score_dir=score_dir,
        )
        fold_summaries.append(fold_result["summary"])
        total_prediction_count += int(fold_result["prediction_count"])
        if not fold_result["by_timestamp"].empty:
            oos_observations.append(fold_result["by_timestamp"])
        if not fold_result["feature_importance"].empty:
            importance_frames.append(fold_result["feature_importance"])

    by_timestamp = (
        pd.concat(oos_observations, ignore_index=True)
        if oos_observations
        else _empty_oos_by_timestamp()
    )
    oos_metrics = _metrics_from_by_timestamp(
        by_timestamp,
        sample_count=total_prediction_count,
    )
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    by_timestamp.to_csv(diagnostics_dir / "oos_by_timestamp.csv", index=False)
    feature_importance = _aggregate_importance(importance_frames)
    feature_importance.to_csv(diagnostics_dir / "feature_importance.csv", index=False)
    drop_suggestions = _drop_suggestions(
        feature_importance,
        redundancy.get("high_correlation_pairs", []),
        low_importance_quantile=args.low_importance_quantile,
    )
    pd.DataFrame(drop_suggestions).to_csv(
        diagnostics_dir / "drop_suggestions.csv",
        index=False,
    )

    summary = {
        "status": "completed",
        "params": _summary_params(args),
        "candidate_features": list(feature_columns),
        "excluded_label_derived_features": excluded_label_derived_features,
        "folds": fold_summaries,
        "oos_metrics": oos_metrics,
        "redundancy": {
            key: value
            for key, value in redundancy.items()
            if key != "high_correlation_pairs"
        },
        "high_correlation_pair_count": len(redundancy.get("high_correlation_pairs", [])),
        "drop_suggestion_count": len(drop_suggestions),
        "scores": {
            "method": "lightgbm",
            "path": str(score_dir / "score_*.parquet"),
            "partition_count": len(list(score_dir.glob("score_*.parquet"))),
            "row_count": total_prediction_count,
        },
        "diagnostics": {
            "oos_by_timestamp": str(diagnostics_dir / "oos_by_timestamp.csv"),
            "feature_importance": str(diagnostics_dir / "feature_importance.csv"),
            "drop_suggestions": str(diagnostics_dir / "drop_suggestions.csv"),
            "redundancy_dir": str(output_dir / "redundancy"),
        },
        "commands": commands,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_fold(
    fold: WalkForwardSpec,
    args: argparse.Namespace,
    *,
    dataset_paths: list[Path],
    feature_columns: tuple[str, ...],
    directions: dict[str, int],
    score_dir: Path,
) -> dict[str, Any]:
    label_end_column = _label_exit_timestamp_column(args.label_column)
    train = _load_window(
        dataset_paths,
        start=fold.train_start,
        end=fold.train_end,
        columns=(
            "timestamp",
            "instrument_id",
            args.label_column,
            label_end_column,
            *feature_columns,
        ),
        max_rows=args.max_train_rows,
        seed=args.seed,
    )
    eval_start = _timestamp(fold.valid_start or fold.test_start)
    train = _purge_train(
        train,
        eval_start=eval_start,
        embargo=args.embargo,
        label_end_column=label_end_column,
    )
    if args.max_train_rows is not None:
        train = _sample_rows(train, max_rows=args.max_train_rows, seed=args.seed)
    valid = (
        _load_window(
            dataset_paths,
            start=fold.valid_start,
            end=fold.valid_end,
            columns=(
                "timestamp",
                "instrument_id",
                args.label_column,
                label_end_column,
                *feature_columns,
            ),
            max_rows=args.max_valid_rows,
            seed=args.seed + 17,
        )
        if fold.valid_start
        else pd.DataFrame(columns=train.columns)
    )
    if args.max_valid_rows is not None and not valid.empty:
        valid = _sample_rows(valid, max_rows=args.max_valid_rows, seed=args.seed + 17)
    train_prepared = _prepare_supervised_frame(
        train,
        feature_columns=feature_columns,
        directions=directions,
        label_column=args.label_column,
        score_transform=args.score_transform,
        drop_missing_label=True,
    )
    valid_prepared = _prepare_supervised_frame(
        valid,
        feature_columns=feature_columns,
        directions=directions,
        label_column=args.label_column,
        score_transform=args.score_transform,
        drop_missing_label=True,
    )
    train_frame = _apply_sample_weights(
        train_prepared.frame,
        label_column=args.label_column,
        mode=args.sample_weight_mode,
        top_quantile=args.sample_weight_top_quantile,
        multiplier=args.sample_weight_multiplier,
    )
    valid_frame = _apply_sample_weights(
        valid_prepared.frame,
        label_column=args.label_column,
        mode=args.sample_weight_mode,
        top_quantile=args.sample_weight_top_quantile,
        multiplier=args.sample_weight_multiplier,
    )
    config = TreeBaselineConfig(
        label_column=args.label_column,
        feature_columns=train_prepared.feature_columns,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        num_threads=args.num_threads,
        sample_weight_column=(
            SAMPLE_WEIGHT_COLUMN if args.sample_weight_mode != "off" else None
        ),
        extra_params=_extra_lgbm_params(args),
    )
    model = train_lightgbm_regressor(
        train_frame,
        valid_frame,
        config,
    )
    importance = _feature_importance(model, fold.name, train_prepared.feature_columns)
    prediction_result = _write_fold_predictions(
        model,
        fold,
        args,
        dataset_paths=dataset_paths,
        feature_columns=feature_columns,
        directions=directions,
        model_feature_columns=train_prepared.feature_columns,
        score_dir=score_dir,
    )
    fold_dir = Path(args.output_dir) / "folds" / fold.name
    fold_dir.mkdir(parents=True, exist_ok=True)
    prediction_result["by_timestamp"].to_csv(
        fold_dir / "oos_by_timestamp.csv",
        index=False,
    )
    importance.to_csv(fold_dir / "feature_importance.csv", index=False)
    summary = {
        "name": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "train_row_count": int(len(train_prepared.frame)),
        "valid_row_count": int(len(valid_prepared.frame)),
        "raw_test_prediction_count": int(prediction_result["raw_prediction_count"]),
        "test_prediction_count": int(prediction_result["prediction_count"]),
        "metrics": prediction_result["metrics"],
        "feature_importance": str(fold_dir / "feature_importance.csv"),
        "oos_by_timestamp": str(fold_dir / "oos_by_timestamp.csv"),
    }
    return {
        "summary": summary,
        "prediction_count": prediction_result["prediction_count"],
        "by_timestamp": prediction_result["by_timestamp"],
        "feature_importance": importance,
    }


def _write_redundancy_diagnostics(
    dataset_paths: list[Path],
    *,
    output_dir: Path,
    feature_columns: tuple[str, ...],
    directions: dict[str, int],
    label_column: str,
    score_transform: str,
    sample_rows: int | None,
    random_seed: int,
    correlation_threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = _load_sample(
        dataset_paths,
        columns=("timestamp", "instrument_id", label_column, *feature_columns),
        max_rows=sample_rows,
        seed=random_seed,
    )
    prepared = _prepare_supervised_frame(
        sample,
        feature_columns=feature_columns,
        directions=directions,
        label_column=label_column,
        score_transform=score_transform,
        drop_missing_label=False,
    )
    features = prepared.frame.loc[:, prepared.feature_columns].astype(float)
    correlation = features.corr(method="spearman").fillna(0.0)
    correlation.to_csv(output_dir / "feature_correlation.csv")
    pairs = _high_correlation_pairs(
        correlation,
        threshold=correlation_threshold,
    )
    pd.DataFrame(pairs).to_csv(output_dir / "high_correlation_pairs.csv", index=False)
    singular_values = _svd_diagnostics(features, output_dir=output_dir)
    return {
        "sample_rows": int(len(prepared.frame)),
        "feature_count": len(prepared.feature_columns),
        "correlation_path": str(output_dir / "feature_correlation.csv"),
        "high_correlation_pairs_path": str(output_dir / "high_correlation_pairs.csv"),
        "svd_singular_values_path": str(output_dir / "svd_singular_values.csv"),
        "svd_loadings_path": str(output_dir / "svd_loadings.csv"),
        "svd_effective_rank_95": _effective_rank(singular_values, threshold=0.95),
        "high_correlation_pairs": pairs,
    }


def _prepare_supervised_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    directions: dict[str, int],
    label_column: str,
    score_transform: str,
    drop_missing_label: bool,
) -> PreparedFrame:
    if frame.empty:
        return PreparedFrame(
            frame=pd.DataFrame(columns=["timestamp", "instrument_id", label_column, *feature_columns]),
            feature_columns=feature_columns,
        )
    _require_columns(frame, ("timestamp", "instrument_id", *feature_columns))
    output = frame.loc[:, ["timestamp", "instrument_id", label_column, *feature_columns]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    for feature in feature_columns:
        transformed = _cross_sectional_transform(
            output,
            feature=feature,
            score_transform=score_transform,
        )
        output[feature] = transformed * float(directions.get(feature, 1))
    if drop_missing_label:
        _require_columns(output, (label_column,))
        output = output.loc[output[label_column].notna()].copy()
    return PreparedFrame(
        frame=output.sort_values(["timestamp", "instrument_id"]).reset_index(drop=True),
        feature_columns=feature_columns,
    )


def _filter_label_derived_candidates(
    candidates: tuple[CandidateFactor, ...],
    *,
    label_column: str,
    include_features: tuple[str, ...],
    allow_label_derived_features: bool,
) -> tuple[tuple[CandidateFactor, ...], list[str]]:
    unsafe = [
        candidate.feature
        for candidate in candidates
        if _is_label_derived_feature(candidate.feature, label_column=label_column)
    ]
    if allow_label_derived_features or not unsafe:
        return candidates, []
    explicit_unsafe = sorted(set(unsafe) & set(include_features))
    if explicit_unsafe:
        raise ValueError(
            "explicit include_features contains label-derived columns: "
            f"{explicit_unsafe}; rerun with --allow-label-derived-features only "
            "for controlled leakage diagnostics"
        )
    unsafe_set = set(unsafe)
    filtered = tuple(
        candidate for candidate in candidates if candidate.feature not in unsafe_set
    )
    if not filtered:
        raise ValueError(
            "all candidate features were excluded as label-derived columns: "
            f"{sorted(unsafe)}"
        )
    return filtered, sorted(unsafe)


def _is_label_derived_feature(feature: str, *, label_column: str) -> bool:
    if not feature:
        return True
    blocked_exact = {
        label_column,
        f"{label_column}_rank",
        "forward_return",
        "forward_return_rank",
        "entry_timestamp",
        "entry_price",
        "exit_timestamp",
        "exit_price",
    }
    if feature in blocked_exact:
        return True
    blocked_prefixes = (
        f"{label_column}_",
        "forward_return_",
        "entry_",
        "exit_",
    )
    if feature.startswith(blocked_prefixes):
        return True
    return feature.endswith(("_exit_timestamp", "_exit_price", "_rank"))


def _cross_sectional_transform(
    frame: pd.DataFrame,
    *,
    feature: str,
    score_transform: str,
) -> pd.Series:
    values = frame[feature].astype(float)
    grouped = values.groupby(frame["timestamp"], sort=False)
    if score_transform == "rank":
        return grouped.rank(pct=True, method="average") - 0.5
    if score_transform == "zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        return (values - mean) / std
    raise ValueError("score_transform must be rank or zscore")


def _apply_sample_weights(
    frame: pd.DataFrame,
    *,
    label_column: str,
    mode: str,
    top_quantile: float,
    multiplier: float,
) -> pd.DataFrame:
    if mode == "off" or frame.empty:
        return frame
    if mode != "top_bottom":
        raise ValueError("sample weight mode must be off or top_bottom")
    _require_columns(frame, ("timestamp", label_column))
    output = frame.copy()
    labels = pd.to_numeric(output[label_column], errors="coerce")
    ranks = labels.groupby(output["timestamp"], sort=False).rank(
        pct=True,
        method="average",
    )
    emphasized = (ranks <= top_quantile) | (ranks > 1.0 - top_quantile)
    weights = pd.Series(1.0, index=output.index, dtype=float)
    weights.loc[emphasized & labels.notna()] = float(multiplier)
    output[SAMPLE_WEIGHT_COLUMN] = weights
    return output


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


def _load_window(
    dataset_paths: list[Path],
    *,
    start: str | None,
    end: str | None,
    columns: tuple[str, ...],
    max_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    frames = []
    start_at = _timestamp(start) if start else None
    end_at = _timestamp(end) if end else None
    per_path_cap = (
        max(1, int(np.ceil(max_rows / max(len(dataset_paths), 1) * 2.0)))
        if max_rows is not None
        else None
    )
    for path in dataset_paths:
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
        timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        mask = pd.Series(True, index=frame.index)
        if start_at is not None:
            mask = mask & (timestamp >= start_at)
        if end_at is not None:
            mask = mask & (timestamp <= end_at)
        if mask.any():
            current = frame.loc[mask].copy()
            if per_path_cap is not None and len(current) > per_path_cap:
                current = current.sample(
                    n=per_path_cap,
                    random_state=seed + len(frames),
                )
            frames.append(current)
    if not frames:
        return pd.DataFrame(columns=columns)
    return _sample_rows(
        pd.concat(frames, ignore_index=True),
        max_rows=max_rows,
        seed=seed,
    )


def _validate_dataset_columns(
    dataset_paths: list[Path],
    *,
    required_columns: tuple[str, ...],
) -> None:
    required = tuple(dict.fromkeys(required_columns))
    missing_by_path: list[dict[str, object]] = []
    for path in dataset_paths:
        columns = set(pq.read_schema(path).names)
        missing = [column for column in required if column not in columns]
        if missing:
            missing_by_path.append(
                {
                    "path": str(path),
                    "missing_columns": missing,
                }
            )
    if not missing_by_path:
        return
    preview = missing_by_path[:5]
    suffix = (
        f"; additional_bad_partitions={len(missing_by_path) - len(preview)}"
        if len(missing_by_path) > len(preview)
        else ""
    )
    raise ValueError(
        "dataset partitions are missing required model columns: "
        f"{preview}{suffix}"
    )


def _load_sample(
    dataset_paths: list[Path],
    *,
    columns: tuple[str, ...],
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    return _load_window(
        dataset_paths,
        start=None,
        end=None,
        columns=columns,
        max_rows=max_rows,
        seed=seed,
    )


def _sample_rows(frame: pd.DataFrame, *, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _purge_train(
    frame: pd.DataFrame,
    *,
    eval_start: pd.Timestamp,
    embargo: str | None,
    label_end_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    _require_columns(frame, ("timestamp", label_end_column))
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    label_end = pd.to_datetime(frame[label_end_column], utc=True, errors="coerce")
    cutoff = eval_start - (pd.Timedelta(embargo) if embargo else pd.Timedelta(0))
    mask = (timestamp < eval_start) & (label_end < cutoff)
    return frame.loc[mask].copy().reset_index(drop=True)


def _write_prediction_partitions(predictions: pd.DataFrame, *, score_dir: Path) -> None:
    if predictions.empty:
        return
    output = predictions.loc[:, ["timestamp", "instrument_id", "score"]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output["partition"] = output["timestamp"].dt.strftime("%Y_%m")
    for partition, group in output.groupby("partition", sort=True):
        path = score_dir / f"score_{partition}.parquet"
        current = group.loc[:, ["timestamp", "instrument_id", "score"]].copy()
        current["timestamp"] = _score_timestamp_strings(current["timestamp"])
        current.sort_values(
            ["timestamp", "score", "instrument_id"],
            ascending=[True, False, True],
        ).to_parquet(path, index=False)


def _score_timestamp_strings(timestamp: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(timestamp, utc=True, errors="coerce")
        .dt.tz_convert("Asia/Shanghai")
        .dt.strftime("%Y-%m-%dT%H:%M:%S")
        + "+08:00"
    )


def _write_fold_predictions(
    model: Any,
    fold: WalkForwardSpec,
    args: argparse.Namespace,
    *,
    dataset_paths: list[Path],
    feature_columns: tuple[str, ...],
    directions: dict[str, int],
    model_feature_columns: tuple[str, ...],
    score_dir: Path,
) -> dict[str, Any]:
    observations: list[pd.DataFrame] = []
    raw_prediction_count = 0
    prediction_count = 0
    start_at = _timestamp(fold.test_start)
    end_at = _timestamp(fold.test_end) if fold.test_end else None
    for dataset_path in dataset_paths:
        frame = pd.read_parquet(
            dataset_path,
            columns=["timestamp", "instrument_id", args.label_column, *feature_columns],
        )
        timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        mask = timestamp >= start_at
        if end_at is not None:
            mask = mask & (timestamp <= end_at)
        if not mask.any():
            continue
        prepared = _prepare_supervised_frame(
            frame.loc[mask].copy(),
            feature_columns=feature_columns,
            directions=directions,
            label_column=args.label_column,
            score_transform=args.score_transform,
            drop_missing_label=False,
        )
        predictions = prepared.frame.loc[
            :, ["timestamp", "instrument_id", args.label_column]
        ].copy()
        predictions["score"] = model.predict(
            prepared.frame.loc[:, model_feature_columns],
            num_iteration=getattr(model, "best_iteration", None),
        )
        predictions = predictions.loc[predictions["score"].notna()].copy()
        if predictions.empty:
            continue
        raw_prediction_count += int(len(predictions))
        if args.score_mode == "primary_pool_rerank":
            predictions = _apply_primary_pool_rerank(
                predictions,
                Path(args.primary_score_dir) / f"score_{_partition_name(dataset_path)}.parquet",
                label_column=args.label_column,
                pool_rank=args.primary_pool_rank,
                primary_score_column=args.primary_score_column,
                primary_blend_weight=args.primary_blend_weight,
            )
            if predictions.empty:
                continue
        _write_prediction_partitions(predictions, score_dir=score_dir)
        prediction_count += int(len(predictions))
        _, by_timestamp = evaluate_cross_sectional_predictions(
            predictions,
            label_column=args.label_column,
            score_column="score",
            top_n=args.top_n,
        )
        if not by_timestamp.empty:
            observations.append(by_timestamp)
        del frame, prepared, predictions
    by_timestamp = (
        pd.concat(observations, ignore_index=True)
        if observations
        else _empty_oos_by_timestamp()
    )
    return {
        "raw_prediction_count": raw_prediction_count,
        "prediction_count": prediction_count,
        "by_timestamp": by_timestamp,
        "metrics": _metrics_from_by_timestamp(
            by_timestamp,
            sample_count=prediction_count,
        ),
    }


def _apply_primary_pool_rerank(
    predictions: pd.DataFrame,
    primary_score_path: Path,
    *,
    label_column: str,
    pool_rank: int,
    primary_score_column: str,
    primary_blend_weight: float = 0.0,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    if not primary_score_path.exists():
        raise FileNotFoundError(f"primary score partition not found: {primary_score_path}")
    _require_columns(predictions, ("timestamp", "instrument_id", label_column, "score"))
    primary = pd.read_parquet(
        primary_score_path,
        columns=["timestamp", "instrument_id", primary_score_column],
    )
    _require_columns(primary, ("timestamp", "instrument_id", primary_score_column))
    primary = primary.rename(columns={primary_score_column: "_primary_score"})
    primary = primary.loc[primary["_primary_score"].notna()].copy()
    if primary.empty:
        return _empty_predictions(label_column)
    primary["timestamp_key"] = _score_timestamp_strings(primary["timestamp"])
    primary = primary.sort_values(
        ["timestamp_key", "_primary_score", "instrument_id"],
        ascending=[True, False, True],
    )
    primary["primary_rank"] = primary.groupby("timestamp_key", sort=False).cumcount() + 1
    pool = primary.loc[
        primary["primary_rank"] <= pool_rank,
        ["timestamp_key", "instrument_id", "_primary_score"],
    ].drop_duplicates()
    output = predictions.copy()
    output["timestamp_key"] = _score_timestamp_strings(output["timestamp"])
    output = output.merge(
        pool,
        on=["timestamp_key", "instrument_id"],
        how="inner",
        sort=False,
    )
    if primary_blend_weight > 0:
        primary_rank_score = output.groupby("timestamp_key", sort=False)[
            "_primary_score"
        ].rank(pct=True, method="average")
        ml_rank_score = output.groupby("timestamp_key", sort=False)["score"].rank(
            pct=True,
            method="average",
        )
        output["score"] = (
            primary_blend_weight * primary_rank_score
            + (1.0 - primary_blend_weight) * ml_rank_score
        )
    return output.loc[:, ["timestamp", "instrument_id", label_column, "score"]]


def _feature_importance(
    model: Any,
    fold_name: str,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    def importance(kind: str) -> list[float]:
        try:
            return [float(value) for value in model.feature_importance(importance_type=kind)]
        except TypeError:
            return [float(value) for value in model.feature_importance(kind)]

    gain = importance("gain")
    split = importance("split")
    return pd.DataFrame(
        {
            "fold": fold_name,
            "feature": feature_columns,
            "gain_importance": gain,
            "split_importance": split,
        }
    )


def _aggregate_importance(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=[
                "feature",
                "fold_count",
                "mean_gain_importance",
                "std_gain_importance",
                "mean_split_importance",
                "nonzero_gain_fold_count",
            ]
        )
    frame = pd.concat(frames, ignore_index=True)
    grouped = frame.groupby("feature", sort=True)
    return grouped.agg(
        fold_count=("fold", "nunique"),
        mean_gain_importance=("gain_importance", "mean"),
        std_gain_importance=("gain_importance", "std"),
        mean_split_importance=("split_importance", "mean"),
        nonzero_gain_fold_count=("gain_importance", lambda values: int((values > 0).sum())),
    ).reset_index()


def _high_correlation_pairs(
    correlation: pd.DataFrame,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = list(correlation.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = float(correlation.loc[left, right])
            if abs(value) >= threshold:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "correlation": value,
                        "abs_correlation": abs(value),
                    }
                )
    return sorted(rows, key=lambda row: row["abs_correlation"], reverse=True)


def _svd_diagnostics(features: pd.DataFrame, *, output_dir: Path) -> np.ndarray:
    matrix = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    matrix = matrix / std
    if matrix.size == 0:
        singular_values = np.array([], dtype=float)
        loadings = pd.DataFrame(columns=["component", "feature", "loading", "abs_loading"])
    else:
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        loadings = pd.DataFrame(
            [
                {
                    "component": component_index + 1,
                    "feature": feature,
                    "loading": float(loading),
                    "abs_loading": abs(float(loading)),
                }
                for component_index, row in enumerate(vt)
                for feature, loading in zip(features.columns, row, strict=True)
            ]
        ).sort_values(["component", "abs_loading"], ascending=[True, False])
    explained = singular_values**2
    total = float(explained.sum())
    ratios = explained / total if total > 0 else np.zeros_like(explained)
    cumulative = np.cumsum(ratios)
    pd.DataFrame(
        {
            "component": np.arange(1, len(singular_values) + 1),
            "singular_value": singular_values,
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance_ratio": cumulative,
        }
    ).to_csv(output_dir / "svd_singular_values.csv", index=False)
    loadings.to_csv(output_dir / "svd_loadings.csv", index=False)
    return singular_values


def _effective_rank(singular_values: np.ndarray, *, threshold: float) -> int:
    if singular_values.size == 0:
        return 0
    variance = singular_values**2
    total = float(variance.sum())
    if total <= 0:
        return 0
    cumulative = np.cumsum(variance / total)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def _drop_suggestions(
    importance: pd.DataFrame,
    high_correlation_pairs: list[dict[str, Any]],
    *,
    low_importance_quantile: float,
) -> list[dict[str, Any]]:
    if importance.empty:
        return []
    gain_by_feature = {
        str(row.feature): float(row.mean_gain_importance or 0.0)
        for row in importance.itertuples(index=False)
    }
    threshold = float(importance["mean_gain_importance"].quantile(low_importance_quantile))
    suggestions: dict[str, dict[str, Any]] = {}
    for row in importance.itertuples(index=False):
        gain = float(row.mean_gain_importance or 0.0)
        if gain <= threshold:
            suggestions[str(row.feature)] = {
                "feature": str(row.feature),
                "reason": "low_mean_gain_importance",
                "mean_gain_importance": gain,
                "reference_feature": None,
                "abs_correlation": None,
            }
    for pair in high_correlation_pairs:
        left = str(pair["feature_left"])
        right = str(pair["feature_right"])
        drop, keep = (
            (left, right)
            if gain_by_feature.get(left, 0.0) <= gain_by_feature.get(right, 0.0)
            else (right, left)
        )
        suggestions[drop] = {
            "feature": drop,
            "reason": "redundant_lower_gain_pair",
            "mean_gain_importance": gain_by_feature.get(drop, 0.0),
            "reference_feature": keep,
            "abs_correlation": float(pair["abs_correlation"]),
        }
    return sorted(
        suggestions.values(),
        key=lambda row: (str(row["reason"]), -float(row["mean_gain_importance"])),
    )


def _parse_fold(value: str) -> WalkForwardSpec:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise ValueError("--fold must be name:key=value,...")
    name, payload = parts
    values: dict[str, str | None] = {
        "train_start": None,
        "valid_start": None,
        "valid_end": None,
        "test_end": None,
    }
    for item in payload.split(","):
        if not item:
            continue
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"invalid fold item: {item}")
        values[key] = raw or None
    return WalkForwardSpec(
        name=name,
        train_start=values.get("train_start"),
        train_end=str(values["train_end"]),
        valid_start=values.get("valid_start"),
        valid_end=values.get("valid_end"),
        test_start=str(values["test_start"]),
        test_end=values.get("test_end"),
    )


def _default_folds(args: argparse.Namespace) -> tuple[WalkForwardSpec, ...]:
    return (
        WalkForwardSpec(
            name="test_2024",
            train_start=args.default_train_start,
            train_end="2023-12-31T23:59:59+08:00",
            valid_start=None,
            valid_end=None,
            test_start="2024-01-01T00:00:00+08:00",
            test_end="2024-12-31T23:59:59+08:00",
        ),
        WalkForwardSpec(
            name="test_2025",
            train_start=args.default_train_start,
            train_end="2024-12-31T23:59:59+08:00",
            valid_start="2024-01-01T00:00:00+08:00",
            valid_end="2024-12-31T23:59:59+08:00",
            test_start="2025-01-01T00:00:00+08:00",
            test_end="2025-12-31T23:59:59+08:00",
        ),
    )


def _extra_lgbm_params(args: argparse.Namespace) -> dict[str, object]:
    params: dict[str, object] = {
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
    }
    if args.objective:
        params["objective"] = args.objective
    if args.metric:
        params["metric"] = args.metric
    return params


def _backtest_example_command(args: argparse.Namespace, score_dir: Path) -> list[str]:
    return [
        sys.executable,
        "examples/run_tree_score_backtest.py",
        "--predictions-path",
        str(score_dir / "score_*.parquet"),
        "--start",
        args.backtest_start,
        "--end",
        args.backtest_end,
        "--top-n",
        str(args.top_n),
        "--trade-policy",
        "rank_buffer_drop",
        "--rebalance-every-n-bars",
        "48",
        "--policy-entry-rank",
        str(args.top_n),
        "--policy-exit-rank",
        "150",
        "--policy-max-entries-per-rebalance",
        "10",
        "--policy-max-exits-per-rebalance",
        "10",
        "--policy-no-trade-weight-band",
        "0.002",
        "--policy-partial-rebalance-rate",
        "0.5",
        "--data-access-mode",
        "fast_parquet",
        "--streaming-chunk",
        "month",
        "--output-dir",
        str(Path(args.output_dir) / "backtests" / "lightgbm" / "partial_rebalance_daily"),
    ]


def _summary_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_dir": args.dataset_dir,
        "admission_report": args.admission_report,
        "statuses": args.statuses,
        "evaluation_roles": args.evaluation_roles,
        "include_features": args.include_features,
        "allow_label_derived_features": args.allow_label_derived_features,
        "label_column": args.label_column,
        "label_exit_timestamp_column": _label_exit_timestamp_column(args.label_column),
        "score_transform": args.score_transform,
        "score_mode": args.score_mode,
        "primary_score_dir": args.primary_score_dir,
        "primary_pool_rank": args.primary_pool_rank,
        "primary_score_column": args.primary_score_column,
        "primary_blend_weight": args.primary_blend_weight,
        "sample_weight_mode": args.sample_weight_mode,
        "sample_weight_top_quantile": args.sample_weight_top_quantile,
        "sample_weight_multiplier": args.sample_weight_multiplier,
        "embargo": args.embargo,
        "max_train_rows": args.max_train_rows,
        "max_valid_rows": args.max_valid_rows,
        "redundancy_sample_rows": args.redundancy_sample_rows,
        "correlation_threshold": args.correlation_threshold,
        "lightgbm": {
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "num_boost_round": args.num_boost_round,
            "early_stopping_rounds": args.early_stopping_rounds,
            "num_threads": args.num_threads,
            "seed": args.seed,
        },
    }


def _empty_predictions(label_column: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "instrument_id", label_column, "score"])


def _empty_oos_by_timestamp() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "sample_count",
            "pearson_ic",
            "spearman_rank_ic",
            "all_mean_label",
            "top_n_mean_label",
            "bottom_n_mean_label",
            "top_minus_bottom_label",
        ]
    )


def _metrics_from_by_timestamp(
    by_timestamp: pd.DataFrame,
    *,
    sample_count: int,
) -> dict[str, object]:
    if by_timestamp.empty:
        return {
            "timestamp_count": 0,
            "sample_count": sample_count,
        }
    return {
        "timestamp_count": int(len(by_timestamp)),
        "sample_count": int(sample_count),
        "pearson_ic_mean": _nullable_float(by_timestamp["pearson_ic"].mean()),
        "spearman_rank_ic_mean": _nullable_float(
            by_timestamp["spearman_rank_ic"].mean()
        ),
        "all_mean_label": _nullable_float(by_timestamp["all_mean_label"].mean()),
        "top_n_mean_label": _nullable_float(by_timestamp["top_n_mean_label"].mean()),
        "bottom_n_mean_label": _nullable_float(
            by_timestamp["bottom_n_mean_label"].mean()
        ),
        "top_minus_bottom_label": _nullable_float(
            by_timestamp["top_minus_bottom_label"].mean()
        ),
    }


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _label_exit_timestamp_column(label_column: str) -> str:
    if label_column == "forward_return":
        return "exit_timestamp"
    return f"{label_column}_exit_timestamp"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--admission-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="forward_return_48b")
    parser.add_argument("--statuses", nargs="+", default=["candidate"])
    parser.add_argument("--evaluation-roles", nargs="+", default=["alpha_rank"])
    parser.add_argument("--include-features", nargs="+", default=[])
    parser.add_argument(
        "--allow-label-derived-features",
        action="store_true",
        help=(
            "allow columns derived from labels or execution metadata; intended "
            "only for controlled leakage diagnostics"
        ),
    )
    parser.add_argument("--score-transform", choices=("rank", "zscore"), default="rank")
    parser.add_argument(
        "--score-mode",
        choices=("standalone", "primary_pool_rerank"),
        default="standalone",
    )
    parser.add_argument("--primary-score-dir")
    parser.add_argument("--primary-pool-rank", type=int, default=150)
    parser.add_argument("--primary-score-column", default="score")
    parser.add_argument("--primary-blend-weight", type=float, default=0.0)
    parser.add_argument(
        "--sample-weight-mode",
        choices=("off", "top_bottom"),
        default="off",
    )
    parser.add_argument("--sample-weight-top-quantile", type=float, default=0.20)
    parser.add_argument("--sample-weight-multiplier", type=float, default=3.0)
    parser.add_argument("--fold", dest="folds", action="append", default=[])
    parser.add_argument("--default-train-start")
    parser.add_argument("--partition-start")
    parser.add_argument("--partition-end")
    parser.add_argument("--max-partitions", type=int)
    parser.add_argument("--embargo", default="2D")
    parser.add_argument("--max-train-rows", type=int, default=2_000_000)
    parser.add_argument("--max-valid-rows", type=int, default=500_000)
    parser.add_argument("--redundancy-sample-rows", type=int, default=1_000_000)
    parser.add_argument("--correlation-threshold", type=float, default=0.90)
    parser.add_argument("--low-importance-quantile", type=float, default=0.20)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=500)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--objective")
    parser.add_argument("--metric")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--backtest-start", default="2024-01-01T00:00:00+08:00")
    parser.add_argument("--backtest-end", default="2025-12-31T23:59:59+08:00")
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_partitions is not None and args.max_partitions <= 0:
        raise ValueError("--max-partitions must be positive")
    for name in (
        "max_train_rows",
        "max_valid_rows",
        "redundancy_sample_rows",
        "top_n",
        "primary_pool_rank",
        "num_leaves",
        "min_data_in_leaf",
        "num_boost_round",
        "num_threads",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.correlation_threshold <= 1:
        raise ValueError("--correlation-threshold must be in (0, 1]")
    if not 0 <= args.low_importance_quantile <= 1:
        raise ValueError("--low-importance-quantile must be in [0, 1]")
    if not 0 <= args.primary_blend_weight <= 1:
        raise ValueError("--primary-blend-weight must be in [0, 1]")
    if not 0 < args.sample_weight_top_quantile <= 0.5:
        raise ValueError("--sample-weight-top-quantile must be in (0, 0.5]")
    if args.sample_weight_multiplier <= 0:
        raise ValueError("--sample-weight-multiplier must be positive")
    if args.score_mode == "primary_pool_rerank":
        if not args.primary_score_dir:
            raise ValueError("--primary-score-dir is required for primary_pool_rerank")
        if not Path(args.primary_score_dir).exists():
            raise FileNotFoundError(f"primary score dir not found: {args.primary_score_dir}")
    for name in ("learning_rate", "feature_fraction", "bagging_fraction"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds must be non-negative")


if __name__ == "__main__":
    main()
