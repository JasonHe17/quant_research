"""Portfolio construction scaffold."""

from __future__ import annotations

import pandas as pd

from quant_research.artifacts import ArtifactStore
from quant_research.portfolio.models import (
    PortfolioConfig,
    PortfolioConstructionResult,
)


class PortfolioConstructor:
    """Transforms signals into target portfolios."""

    def __init__(self, *, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store

    def build(
        self,
        signals: pd.DataFrame,
        config: PortfolioConfig,
        *,
        current_positions: pd.DataFrame | None = None,
        persist: bool = False,
    ) -> PortfolioConstructionResult:
        _require_columns(signals, ("timestamp", "instrument_id", "signal"))
        target_weights = _target_weights(signals, config=config)
        rebalance_orders = _rebalance_orders(
            target_weights,
            current_positions=current_positions,
        )
        diagnostics = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "instrument_count": len(group),
                    "gross_weight": float(group["target_weight"].abs().sum()),
                }
                for timestamp, group in target_weights.groupby("timestamp", sort=True)
            ]
        )
        result = PortfolioConstructionResult(
            config=config,
            target_weights=target_weights,
            rebalance_orders=rebalance_orders,
            diagnostics=diagnostics,
        )
        if persist:
            if self.artifact_store is None:
                raise ValueError("artifact_store is required when persist=True")
            return result.with_artifacts(self.artifact_store.write_portfolio(result))
        return result


def _target_weights(
    signals: pd.DataFrame, *, config: PortfolioConfig
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for timestamp, group in signals.groupby("timestamp", sort=True):
        ordered = group.sort_values("instrument_id").copy()
        if config.weighting == "equal":
            weights = pd.Series(1.0 / len(ordered), index=ordered.index)
        else:
            signal_sum = ordered["signal"].abs().sum()
            if signal_sum == 0:
                weights = pd.Series(0.0, index=ordered.index)
            else:
                weights = ordered["signal"] / signal_sum
        if config.max_weight is not None:
            weights = weights.clip(lower=-config.max_weight, upper=config.max_weight)
        output = ordered.loc[:, ["timestamp", "instrument_id"]].copy()
        output["target_weight"] = weights.astype(float)
        rows.append(output)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "instrument_id", "target_weight"])
    return pd.concat(rows, ignore_index=True)


def _rebalance_orders(
    target_weights: pd.DataFrame,
    *,
    current_positions: pd.DataFrame | None,
) -> pd.DataFrame:
    current = (
        current_positions.copy()
        if current_positions is not None
        else pd.DataFrame(columns=["instrument_id", "current_weight"])
    )
    if not current.empty:
        _require_columns(current, ("instrument_id", "current_weight"))
    frames: list[pd.DataFrame] = []
    for timestamp, group in target_weights.groupby("timestamp", sort=True):
        merged = group.merge(current, on="instrument_id", how="left")
        merged["current_weight"] = merged["current_weight"].fillna(0.0)
        merged["delta_weight"] = merged["target_weight"] - merged["current_weight"]
        frames.append(
            merged.loc[
                :,
                [
                    "timestamp",
                    "instrument_id",
                    "current_weight",
                    "target_weight",
                    "delta_weight",
                ],
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "instrument_id",
                "current_weight",
                "target_weight",
                "delta_weight",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
