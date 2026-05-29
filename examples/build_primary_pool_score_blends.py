"""Build primary-pool score blends from existing primary and ML pool scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    args = _parse_args()
    summary = build_primary_pool_score_blends(args)
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


def build_primary_pool_score_blends(args: argparse.Namespace) -> dict[str, Any]:
    primary_dir = Path(args.primary_score_dir)
    ml_pool_dir = Path(args.ml_pool_score_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = tuple(float(weight) for weight in args.primary_blend_weights)
    partition_summaries: dict[str, dict[str, int]] = {
        _weight_label(weight): {"partition_count": 0, "row_count": 0}
        for weight in weights
    }
    for ml_path in sorted(ml_pool_dir.glob("score_*.parquet")):
        partition = ml_path.stem.removeprefix("score_")
        primary_path = primary_dir / f"score_{partition}.parquet"
        blended_by_weight = _blend_partition(
            primary_path,
            ml_path,
            primary_blend_weights=weights,
            primary_pool_rank=args.primary_pool_rank,
        )
        for weight, blended in blended_by_weight.items():
            label = _weight_label(weight)
            score_dir = output_dir / "scores" / label
            score_dir.mkdir(parents=True, exist_ok=True)
            blended.to_parquet(score_dir / f"score_{partition}.parquet", index=False)
            partition_summaries[label]["partition_count"] += 1
            partition_summaries[label]["row_count"] += int(len(blended))
    summary = {
        "status": "completed",
        "params": {
            "primary_score_dir": str(primary_dir),
            "ml_pool_score_dir": str(ml_pool_dir),
            "output_dir": str(output_dir),
            "primary_blend_weights": list(weights),
            "primary_pool_rank": args.primary_pool_rank,
        },
        "methods": {
            label: {
                "primary_blend_weight": weight,
                "path": str(output_dir / "scores" / label / "score_*.parquet"),
                **partition_summaries[label],
            }
            for weight in weights
            for label in [_weight_label(weight)]
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _blend_partition(
    primary_path: Path,
    ml_pool_path: Path,
    *,
    primary_blend_weights: tuple[float, ...],
    primary_pool_rank: int | None = None,
) -> dict[float, pd.DataFrame]:
    if not primary_path.exists():
        raise FileNotFoundError(f"primary score partition not found: {primary_path}")
    primary = pd.read_parquet(
        primary_path,
        columns=["timestamp", "instrument_id", "score"],
    ).rename(columns={"score": "primary_score"})
    ml_pool = pd.read_parquet(
        ml_pool_path,
        columns=["timestamp", "instrument_id", "score"],
    ).rename(columns={"score": "ml_score"})
    _require_columns(primary, ("timestamp", "instrument_id", "primary_score"))
    _require_columns(ml_pool, ("timestamp", "instrument_id", "ml_score"))
    frame = ml_pool.merge(
        primary,
        on=["timestamp", "instrument_id"],
        how="left",
        sort=False,
    )
    if frame["primary_score"].isna().any():
        missing = int(frame["primary_score"].isna().sum())
        raise ValueError(f"{ml_pool_path} has {missing} rows missing primary scores")
    if primary_pool_rank is not None:
        frame = _filter_primary_pool_rank(frame, primary_pool_rank=primary_pool_rank)
    frame["primary_rank_score"] = frame.groupby("timestamp", sort=False)[
        "primary_score"
    ].rank(pct=True, method="average")
    frame["ml_rank_score"] = frame.groupby("timestamp", sort=False)["ml_score"].rank(
        pct=True,
        method="average",
    )
    outputs: dict[float, pd.DataFrame] = {}
    for weight in primary_blend_weights:
        output = frame.loc[:, ["timestamp", "instrument_id"]].copy()
        output["score"] = (
            weight * frame["primary_rank_score"]
            + (1.0 - weight) * frame["ml_rank_score"]
        )
        outputs[weight] = output.sort_values(
            ["timestamp", "score", "instrument_id"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    return outputs


def _filter_primary_pool_rank(
    frame: pd.DataFrame,
    *,
    primary_pool_rank: int,
) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["timestamp", "primary_score", "instrument_id"],
        ascending=[True, False, True],
    ).copy()
    ranked["primary_rank"] = ranked.groupby("timestamp", sort=False).cumcount() + 1
    return ranked.loc[ranked["primary_rank"] <= primary_pool_rank].drop(
        columns=["primary_rank"]
    )


def _weight_label(weight: float) -> str:
    basis_points = int(round(weight * 100))
    return f"primary_w{basis_points:03d}"


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-score-dir", required=True)
    parser.add_argument("--ml-pool-score-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--primary-blend-weights",
        nargs="+",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--primary-pool-rank",
        type=int,
        help=(
            "optional stricter primary rank cutoff applied inside the existing "
            "ML pool; useful for deriving rank100 from rank150 ML pool scores"
        ),
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not Path(args.primary_score_dir).exists():
        raise FileNotFoundError(f"primary score dir not found: {args.primary_score_dir}")
    if not Path(args.ml_pool_score_dir).exists():
        raise FileNotFoundError(f"ML pool score dir not found: {args.ml_pool_score_dir}")
    if not args.primary_blend_weights:
        raise ValueError("--primary-blend-weights must be non-empty")
    for weight in args.primary_blend_weights:
        if not 0 <= weight <= 1:
            raise ValueError("--primary-blend-weights values must be in [0, 1]")
    if args.primary_pool_rank is not None and args.primary_pool_rank <= 0:
        raise ValueError("--primary-pool-rank must be positive")


if __name__ == "__main__":
    main()
