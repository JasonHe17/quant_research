"""Evaluate single alpha features in supervised alpha datasets."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.factors import (
    SingleFactorEvaluationConfig,
    SingleFactorEvaluationResult,
    evaluate_single_factors,
)
from quant_research.models import infer_feature_columns


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = _dataset_paths(args)
    config = SingleFactorEvaluationConfig(
        label_column=args.label_column,
        horizon_label_columns=tuple(args.horizon_label_columns or ()),
        feature_columns=tuple(args.feature_columns or ()),
        top_n=args.top_n,
        quantiles=args.quantiles,
        correlation_method=args.correlation_method,
    )
    result = _evaluate_dataset_paths(
        dataset_paths,
        config,
        output_dir=output_dir,
        workers=args.workers,
        backend=args.backend,
        skip_feature_correlation=args.skip_feature_correlation,
        correlation_sample_rows=args.correlation_sample_rows or None,
    )
    summary_path = output_dir / "single_factor_summary.csv"
    by_timestamp_path = output_dir / "single_factor_by_timestamp.csv"
    quantile_path = output_dir / "single_factor_quantiles.csv"
    correlation_path = output_dir / "feature_correlation.csv"
    result.summary.to_csv(summary_path, index=False)
    result.by_timestamp.to_csv(by_timestamp_path, index=False)
    result.quantile_returns.to_csv(quantile_path, index=False)
    result.feature_correlation.to_csv(correlation_path)
    payload = {
        "params": {
            "dataset_paths": [str(path) for path in dataset_paths],
            "label_column": args.label_column,
            "horizon_label_columns": args.horizon_label_columns,
            "feature_columns": args.feature_columns,
            "top_n": args.top_n,
            "quantiles": args.quantiles,
            "correlation_method": args.correlation_method,
            "skip_feature_correlation": args.skip_feature_correlation,
            "correlation_sample_rows": args.correlation_sample_rows,
            "workers": args.workers,
            "backend": args.backend,
        },
        "artifacts": {
            "summary": str(summary_path),
            "by_timestamp": str(by_timestamp_path),
            "quantiles": str(quantile_path),
            "feature_correlation": str(correlation_path),
            "partition_artifacts": str(output_dir / "_partitions"),
        },
        "summary": result.summary.to_dict("records"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result.summary.to_string(index=False))


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.dataset_paths:
        return [Path(path) for path in args.dataset_paths]
    dataset_dir = Path(args.dataset_dir)
    paths = sorted(dataset_dir.glob("dataset_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no dataset_*.parquet files found under {dataset_dir}")
    return paths


def _evaluate_dataset_paths(
    dataset_paths: list[Path],
    config: SingleFactorEvaluationConfig,
    *,
    output_dir: Path,
    workers: int,
    backend: str,
    skip_feature_correlation: bool,
    correlation_sample_rows: int | None,
) -> SingleFactorEvaluationResult:
    feature_columns = config.feature_columns or _infer_feature_columns_from_path(
        dataset_paths[0],
        label_column=config.label_column,
        horizon_label_columns=config.horizon_label_columns,
    )
    partition_config = replace(
        config,
        feature_columns=feature_columns,
        include_feature_correlation=False,
    )
    partition_dir = output_dir / "_partitions"
    partition_dir.mkdir(parents=True, exist_ok=True)
    _clear_partition_artifacts(partition_dir)
    total_rows = 0
    partition_artifacts: list[_PartitionEvaluation] = []
    corr_stats: _CorrelationStats | None = None
    if workers == 1:
        partition_results = (
            _evaluate_dataset_path(
                path,
                partition_config,
                partition_dir=partition_dir,
                skip_feature_correlation=skip_feature_correlation,
                correlation_sample_rows=correlation_sample_rows,
            )
            for path in dataset_paths
        )
        for partition in partition_results:
            total_rows += partition.row_count
            partition_artifacts.append(partition)
            if partition.correlation is not None:
                corr_stats = _merge_correlation_stats(corr_stats, partition.correlation)
    else:
        if backend != "process":
            raise ValueError("only process backend is supported")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_dataset_path,
                    path,
                    partition_config,
                    partition_dir=partition_dir,
                    skip_feature_correlation=skip_feature_correlation,
                    correlation_sample_rows=correlation_sample_rows,
                )
                for path in dataset_paths
            ]
            for future in as_completed(futures):
                partition = future.result()
                total_rows += partition.row_count
                partition_artifacts.append(partition)
                if partition.correlation is not None:
                    corr_stats = _merge_correlation_stats(
                        corr_stats,
                        partition.correlation,
                    )
    if not partition_artifacts:
        raise ValueError("no dataset rows loaded")
    partition_artifacts.sort(key=lambda item: item.name)
    by_timestamp_frames = [
        pd.read_parquet(artifact.by_timestamp_path)
        for artifact in partition_artifacts
    ]
    quantile_frames = [
        pd.read_parquet(artifact.quantile_by_timestamp_path)
        for artifact in partition_artifacts
    ]
    by_timestamp = pd.concat(by_timestamp_frames, ignore_index=True)
    quantile_by_timestamp = pd.concat(quantile_frames, ignore_index=True)
    feature_correlation = (
        corr_stats.to_frame()
        if corr_stats is not None
        else pd.DataFrame(index=feature_columns, columns=feature_columns)
    )
    return SingleFactorEvaluationResult(
        summary=_summarize_by_timestamp(by_timestamp, total_rows=total_rows),
        by_timestamp=by_timestamp,
        quantile_by_timestamp=quantile_by_timestamp,
        quantile_returns=_summarize_quantiles(quantile_by_timestamp),
        feature_correlation=feature_correlation,
        decay_by_label=pd.DataFrame(),
        group_summary=pd.DataFrame(),
        multiple_testing=pd.DataFrame(),
    )


@dataclass(frozen=True, slots=True)
class _PartitionEvaluation:
    name: str
    row_count: int
    by_timestamp_path: Path
    quantile_by_timestamp_path: Path
    correlation: "_CorrelationStats | None"


def _infer_feature_columns_from_path(
    path: Path,
    *,
    label_column: str,
    horizon_label_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    frame = pd.read_parquet(path)
    try:
        exclude_columns: list[str] = []
        for column in (label_column, *horizon_label_columns):
            exclude_columns.extend(
                [
                    column,
                    f"{column}_rank",
                    f"{column}_entry_timestamp",
                    f"{column}_entry_price",
                    f"{column}_exit_timestamp",
                    f"{column}_exit_price",
                    f"{column}_exit_tradable_bar",
                    f"{column}_exit_limit_up_open",
                    f"{column}_exit_limit_down_open",
                ]
            )
        return infer_feature_columns(
            frame,
            label_column=label_column,
            exclude_columns=tuple(exclude_columns),
        )
    finally:
        del frame


def _evaluate_dataset_path(
    path: Path,
    config: SingleFactorEvaluationConfig,
    *,
    partition_dir: Path,
    skip_feature_correlation: bool,
    correlation_sample_rows: int | None = None,
) -> _PartitionEvaluation:
    frame = pd.read_parquet(path)
    print(f"evaluating {path.name}: rows={len(frame)}", flush=True)
    result = evaluate_single_factors(frame, config)
    by_timestamp_path = partition_dir / f"{path.stem}_by_timestamp.parquet"
    quantile_by_timestamp_path = partition_dir / f"{path.stem}_quantiles.parquet"
    result.by_timestamp.to_parquet(by_timestamp_path, index=False)
    result.quantile_by_timestamp.to_parquet(quantile_by_timestamp_path, index=False)
    corr_stats = None
    if not skip_feature_correlation:
        correlation_frame = frame
        weight_scale = 1.0
        if correlation_sample_rows is not None and len(frame) > correlation_sample_rows:
            correlation_frame = frame.sample(
                n=correlation_sample_rows,
                random_state=0,
            )
            weight_scale = len(frame) / correlation_sample_rows
        corr_stats = _CorrelationStats(
            config.feature_columns,
            method=config.correlation_method,
        )
        corr_stats.update(correlation_frame, weight_scale=weight_scale)
    row_count = len(frame)
    del frame
    del result
    return _PartitionEvaluation(
        name=path.stem,
        row_count=row_count,
        by_timestamp_path=by_timestamp_path,
        quantile_by_timestamp_path=quantile_by_timestamp_path,
        correlation=corr_stats,
    )


def _clear_partition_artifacts(partition_dir: Path) -> None:
    for path in partition_dir.glob("*.parquet"):
        path.unlink()


def _summarize_by_timestamp(
    by_timestamp: pd.DataFrame,
    *,
    total_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature, group in by_timestamp.groupby("feature", sort=True):
        sample_count = int(group["sample_count"].sum())
        rows.append(
            {
                "feature": feature,
                "sample_count": sample_count,
                "coverage": sample_count / total_rows if total_rows else 0.0,
                "timestamp_count": int(group["timestamp"].nunique()),
                "pearson_ic_mean": _nullable_float(group["pearson_ic"].mean()),
                "spearman_rank_ic_mean": _nullable_float(
                    group["spearman_rank_ic"].mean()
                ),
                "top_n_mean_label": _nullable_float(group["top_n_mean_label"].mean()),
                "bottom_n_mean_label": _nullable_float(
                    group["bottom_n_mean_label"].mean()
                ),
                "top_minus_bottom_label": _nullable_float(
                    group["top_minus_bottom_label"].mean()
                ),
                "top_n_turnover": _nullable_float(group["top_n_turnover"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "spearman_rank_ic_mean",
        ascending=False,
        na_position="last",
    )


def _summarize_quantiles(quantile_by_timestamp: pd.DataFrame) -> pd.DataFrame:
    if quantile_by_timestamp.empty:
        return pd.DataFrame()
    return (
        quantile_by_timestamp.groupby(["feature", "quantile"], as_index=False)
        .agg(
            timestamp_count=("timestamp", "nunique"),
            sample_count=("sample_count", "sum"),
            mean_label=("mean_label", "mean"),
        )
        .sort_values(["feature", "quantile"])
        .reset_index(drop=True)
    )


class _CorrelationStats:
    def __init__(self, feature_columns: tuple[str, ...], *, method: str) -> None:
        self.feature_columns = feature_columns
        self.method = method
        self.values: dict[tuple[str, str], dict[str, float]] = {}
        for left in feature_columns:
            for right in feature_columns:
                self.values[(left, right)] = {
                    "count": 0.0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_x2": 0.0,
                    "sum_y2": 0.0,
                    "sum_xy": 0.0,
                    "weighted_corr": 0.0,
                    "weight": 0.0,
                }

    def update(self, frame: pd.DataFrame, *, weight_scale: float = 1.0) -> None:
        if self.method == "spearman":
            self._update_spearman(frame, weight_scale=weight_scale)
            return
        for left_index, left in enumerate(self.feature_columns):
            for right in self.feature_columns[left_index:]:
                pair = frame.loc[:, [left, right]].dropna()
                if pair.empty:
                    continue
                x = pair.iloc[:, 0].astype(float)
                y = pair.iloc[:, 1].astype(float)
                self._add_symmetric(
                    left,
                    right,
                    {
                        "count": float(len(pair)),
                        "sum_x": float(x.sum()),
                        "sum_y": float(y.sum()),
                        "sum_x2": float((x * x).sum()),
                        "sum_y2": float((y * y).sum()),
                        "sum_xy": float((x * y).sum()),
                    },
                    weight_scale=weight_scale,
                )

    def _update_spearman(self, frame: pd.DataFrame, *, weight_scale: float) -> None:
        data = frame.loc[:, list(self.feature_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        valid = data.notna().astype("int64")
        counts = valid.T.dot(valid)
        correlations = data.corr(method="spearman")
        for left in self.feature_columns:
            for right in self.feature_columns:
                weight = float(counts.loc[left, right]) * weight_scale
                if weight == 0:
                    continue
                corr = 1.0 if left == right else correlations.loc[left, right]
                if pd.isna(corr):
                    continue
                self.values[(left, right)]["weighted_corr"] += float(corr) * weight
                self.values[(left, right)]["weight"] += float(weight)

    def _add_symmetric(
        self,
        left: str,
        right: str,
        updates: dict[str, float],
        *,
        weight_scale: float = 1.0,
    ) -> None:
        for key in ((left, right),) if left == right else ((left, right), (right, left)):
            stats = self.values[key]
            for name, value in updates.items():
                stats[name] += value * weight_scale

    def merge(self, other: "_CorrelationStats") -> None:
        if self.feature_columns != other.feature_columns:
            raise ValueError("cannot merge correlation stats with different features")
        if self.method != other.method:
            raise ValueError("cannot merge correlation stats with different methods")
        for key, stats in self.values.items():
            for name, value in other.values[key].items():
                stats[name] += value

    def to_frame(self) -> pd.DataFrame:
        rows: list[list[float | None]] = []
        for left in self.feature_columns:
            row: list[float | None] = []
            for right in self.feature_columns:
                row.append(_correlation_from_stats(self.values[(left, right)], self.method))
            rows.append(row)
        return pd.DataFrame(rows, index=self.feature_columns, columns=self.feature_columns)


def _merge_correlation_stats(
    stats: _CorrelationStats | None,
    other: _CorrelationStats,
) -> _CorrelationStats:
    if stats is None:
        return other
    stats.merge(other)
    return stats


def _correlation_from_stats(stats: dict[str, float], method: str) -> float | None:
    if method == "spearman":
        weight = stats["weight"]
        if weight == 0:
            return None
        return stats["weighted_corr"] / weight
    count = stats["count"]
    if count <= 1:
        return None
    numerator = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / count
    variance_x = stats["sum_x2"] - stats["sum_x"] ** 2 / count
    variance_y = stats["sum_y2"] - stats["sum_y"] ** 2 / count
    denominator = (variance_x * variance_y) ** 0.5
    if denominator == 0:
        return None
    return numerator / denominator


def _nullable_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir")
    parser.add_argument("--dataset-paths", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--horizon-label-columns", nargs="+")
    parser.add_argument("--feature-columns", nargs="+")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument(
        "--correlation-method",
        choices=("pearson", "spearman"),
        default="spearman",
    )
    parser.add_argument("--skip-feature-correlation", action="store_true")
    parser.add_argument(
        "--correlation-sample-rows",
        type=int,
        default=0,
        help=(
            "maximum rows per dataset partition used for feature-correlation "
            "estimation; 0 uses all rows"
        ),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--backend",
        choices=("process",),
        default="process",
        help="parallel execution backend used when --workers is greater than 1",
    )
    args = parser.parse_args()
    if bool(args.dataset_dir) == bool(args.dataset_paths):
        raise ValueError("provide exactly one of --dataset-dir or --dataset-paths")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.correlation_sample_rows < 0:
        raise ValueError("--correlation-sample-rows must be non-negative")
    return args


if __name__ == "__main__":
    main()
