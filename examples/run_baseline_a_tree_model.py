"""Train a LightGBM baseline on Baseline A alpha/label datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_research.models import (
    TreeBaselineConfig,
    evaluate_cross_sectional_predictions,
    infer_feature_columns,
    train_lightgbm_regressor,
)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = _dataset_paths(args)
    feature_columns = (
        tuple(args.feature_columns)
        if args.feature_columns
        else _infer_feature_columns_from_path(
            dataset_paths[0],
            label_column=args.label_column,
        )
    )
    splits = _load_time_splits(
        dataset_paths,
        label_column=args.label_column,
        feature_columns=feature_columns,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    config = TreeBaselineConfig(
        label_column=args.label_column,
        feature_columns=feature_columns,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        num_threads=args.num_threads,
    )
    booster = train_lightgbm_regressor(splits["train"], splits["valid"], config)
    split_rows = {name: int(len(split)) for name, split in splits.items()}
    del splits["train"]
    del splits["valid"]
    test = splits.pop("test")
    del splits
    if test.empty:
        raise ValueError("test split is empty")
    scores = booster.predict(test.loc[:, feature_columns])
    predictions = test.loc[:, ["timestamp", "instrument_id", args.label_column]].copy()
    predictions["score"] = scores
    del test
    predictions_path = output_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    metrics, by_timestamp = evaluate_cross_sectional_predictions(
        predictions,
        label_column=args.label_column,
        top_n=args.top_n,
    )
    by_timestamp_path = output_dir / "metrics_by_timestamp.csv"
    by_timestamp.to_csv(by_timestamp_path, index=False)
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance_path = output_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    payload = {
        "params": {
            "dataset_paths": [str(path) for path in dataset_paths],
            "label_column": args.label_column,
            "feature_columns": list(feature_columns),
            "train_end": args.train_end,
            "valid_start": args.valid_start,
            "valid_end": args.valid_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "top_n": args.top_n,
            "num_boost_round": args.num_boost_round,
            "best_iteration": booster.best_iteration,
        },
        "split_rows": split_rows,
        "metrics": metrics,
        "artifacts": {
            "predictions": str(predictions_path),
            "metrics_by_timestamp": str(by_timestamp_path),
            "feature_importance": str(importance_path),
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.dataset_paths:
        return [Path(path) for path in args.dataset_paths]
    dataset_dir = Path(args.dataset_dir)
    paths = sorted(dataset_dir.glob("dataset_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no dataset_*.parquet files found under {dataset_dir}")
    return paths


def _infer_feature_columns_from_path(
    path: Path,
    *,
    label_column: str,
) -> tuple[str, ...]:
    frame = pd.read_parquet(path)
    try:
        return infer_feature_columns(frame, label_column=label_column)
    finally:
        del frame


def _load_time_splits(
    paths: list[Path],
    *,
    label_column: str,
    feature_columns: tuple[str, ...],
    train_end: str,
    valid_start: str | None,
    valid_end: str | None,
    test_start: str,
    test_end: str | None,
) -> dict[str, pd.DataFrame]:
    columns = ["timestamp", "instrument_id", label_column, *feature_columns]
    train_frames: list[pd.DataFrame] = []
    valid_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        _downcast_numeric(frame, columns=(label_column, *feature_columns))
        train_mask = frame["timestamp"] <= train_end
        if train_mask.any():
            train_frames.append(
                frame.loc[train_mask, [label_column, *feature_columns]]
                .reset_index(drop=True)
            )
        if valid_start is not None:
            valid_mask = frame["timestamp"] >= valid_start
            if valid_end is not None:
                valid_mask = valid_mask & (frame["timestamp"] <= valid_end)
            if valid_mask.any():
                valid_frames.append(
                    frame.loc[valid_mask, [label_column, *feature_columns]]
                    .reset_index(drop=True)
                )
        test_mask = frame["timestamp"] >= test_start
        if test_end is not None:
            test_mask = test_mask & (frame["timestamp"] <= test_end)
        if test_mask.any():
            test_frames.append(frame.loc[test_mask].reset_index(drop=True))
        del frame
    return {
        "train": _concat_or_empty(train_frames, columns=[label_column, *feature_columns]),
        "valid": _concat_or_empty(valid_frames, columns=[label_column, *feature_columns]),
        "test": _concat_or_empty(test_frames, columns=columns),
    }


def _downcast_numeric(frame: pd.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], downcast="float")


def _concat_or_empty(frames: list[pd.DataFrame], *, columns: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir")
    parser.add_argument("--dataset-paths", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="forward_return")
    parser.add_argument("--feature-columns", nargs="+")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--valid-start")
    parser.add_argument("--valid-end")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    if bool(args.dataset_dir) == bool(args.dataset_paths):
        raise ValueError("provide exactly one of --dataset-dir or --dataset-paths")
    return args


if __name__ == "__main__":
    main()
