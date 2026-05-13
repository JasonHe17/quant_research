"""Candidate-factor portfolio scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CandidateFactor:
    """One admitted factor and its preferred direction."""

    feature: str
    direction: int
    rank_ic_mean: float

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError("feature must be non-empty")
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")


def load_candidate_factors(
    admission_report_path: Path,
    *,
    statuses: tuple[str, ...] = ("candidate",),
) -> tuple[CandidateFactor, ...]:
    """Load candidate factors from a factor admission report."""

    report = json.loads(admission_report_path.read_text(encoding="utf-8"))
    factors = []
    for row in report.get("factors", []):
        if row.get("admission_status") not in statuses:
            continue
        factors.append(
            CandidateFactor(
                feature=str(row["feature"]),
                direction=-1 if row.get("direction") == "invert" else 1,
                rank_ic_mean=float(row.get("spearman_rank_ic_mean") or 0.0),
            )
        )
    if not factors:
        raise ValueError(f"no factors found for statuses: {statuses}")
    return tuple(factors)


def factor_combination_weights(
    candidates: tuple[CandidateFactor, ...],
    *,
    method: str,
    correlation: pd.DataFrame | None = None,
    ridge: float = 0.05,
) -> dict[str, float]:
    """Compute non-negative combination weights for candidate factors."""

    if method == "equal":
        return _normalize({factor.feature: 1.0 for factor in candidates})
    base = {
        factor.feature: max(abs(float(factor.rank_ic_mean)), 1e-12)
        for factor in candidates
    }
    if method == "ic_weighted":
        return _normalize(base)
    if method != "decorrelated":
        raise ValueError("method must be equal, ic_weighted, or decorrelated")
    if correlation is None or correlation.empty:
        return _normalize(base)
    features = [factor.feature for factor in candidates]
    directions = np.array([factor.direction for factor in candidates], dtype=float)
    matrix = correlation.reindex(index=features, columns=features).astype(float)
    matrix = matrix.fillna(0.0)
    oriented = matrix.to_numpy(dtype=float) * np.outer(directions, directions)
    oriented = np.nan_to_num(oriented, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(oriented, 1.0)
    system = oriented + np.eye(len(features)) * ridge
    target = np.array([base[feature] for feature in features], dtype=float)
    try:
        raw = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        raw = target
    raw = np.clip(raw, 0.0, None)
    if float(raw.sum()) <= 0:
        raw = target
    return _normalize(dict(zip(features, raw.tolist(), strict=True)))


def build_composite_scores(
    frame: pd.DataFrame,
    *,
    candidates: tuple[CandidateFactor, ...],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Build timestamp-level composite scores from candidate factor columns."""

    features = tuple(factor.feature for factor in candidates)
    _require_columns(frame, ("timestamp", "instrument_id", *features))
    output = frame.loc[:, ["timestamp", "instrument_id"]].copy()
    weighted = pd.Series(0.0, index=frame.index)
    available_weight = pd.Series(0.0, index=frame.index)
    for factor in candidates:
        weight = float(weights.get(factor.feature, 0.0))
        if weight <= 0:
            continue
        ranks = frame.groupby("timestamp", sort=False)[factor.feature].rank(
            method="average",
            pct=True,
        )
        oriented = (ranks - 0.5) * factor.direction
        valid = oriented.notna()
        weighted.loc[valid] += oriented.loc[valid] * weight
        available_weight.loc[valid] += weight
    output["score"] = weighted.where(available_weight <= 0, weighted / available_weight)
    output = output.loc[available_weight > 0].copy()
    return output.sort_values(["timestamp", "score", "instrument_id"], ascending=[True, False, True]).reset_index(drop=True)


def write_score_partitions(
    dataset_paths: list[Path],
    *,
    output_dir: Path,
    candidates: tuple[CandidateFactor, ...],
    weights_by_method: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Write composite score parquet partitions for each method."""

    output_dir.mkdir(parents=True, exist_ok=True)
    features = [factor.feature for factor in candidates]
    methods: dict[str, dict[str, Any]] = {}
    for method, weights in weights_by_method.items():
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for old_path in method_dir.glob("score_*.parquet"):
            old_path.unlink()
        row_count = 0
        partition_count = 0
        for dataset_path in dataset_paths:
            frame = pd.read_parquet(
                dataset_path,
                columns=["timestamp", "instrument_id", *features],
            )
            scores = build_composite_scores(
                frame,
                candidates=candidates,
                weights=weights,
            )
            score_path = method_dir / f"score_{dataset_path.stem.removeprefix('dataset_')}.parquet"
            scores.to_parquet(score_path, index=False)
            row_count += len(scores)
            partition_count += 1
            del frame, scores
        methods[method] = {
            "path": str(method_dir / "*.parquet"),
            "weights": weights,
            "row_count": row_count,
            "partition_count": partition_count,
        }
    return {
        "candidate_features": [factor.feature for factor in candidates],
        "methods": methods,
    }


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in values.values())
    if total <= 0:
        raise ValueError("cannot normalize zero weights")
    return {key: max(float(value), 0.0) / total for key, value in values.items()}


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
