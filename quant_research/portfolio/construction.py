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
        diagnostics = _diagnostics(target_weights)
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
    if signals.empty:
        return pd.DataFrame(columns=["timestamp", "instrument_id", "target_weight"])
    ordered = signals.sort_values(["timestamp", "instrument_id"]).reset_index(drop=True)
    output = ordered.loc[:, ["timestamp", "instrument_id"]].copy()
    if config.weighting == "equal":
        counts = ordered.groupby("timestamp", sort=False)["instrument_id"].transform("size")
        weights = 1.0 / counts.astype(float)
    else:
        denominator = ordered["signal"].abs().groupby(ordered["timestamp"], sort=False).transform("sum")
        weights = ordered["signal"].astype(float).div(denominator).fillna(0.0)
    if config.max_weight is not None:
        weights = weights.clip(lower=-config.max_weight, upper=config.max_weight)
    output["target_weight"] = weights.astype(float)
    return output


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
    if target_weights.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "instrument_id",
                "current_weight",
                "target_weight",
                "delta_weight",
            ]
        )
    if current.empty:
        output = target_weights.copy()
        output["current_weight"] = 0.0
    else:
        current = current.loc[:, ["instrument_id", "current_weight"]].copy()
        expanded_current = _expanded_current_positions(
            target_weights["timestamp"].drop_duplicates().tolist(),
            current,
        )
        output = target_weights.merge(
            expanded_current,
            on=["timestamp", "instrument_id"],
            how="outer",
            sort=False,
        )
        output["target_weight"] = output["target_weight"].fillna(0.0)
        output["current_weight"] = output["current_weight"].fillna(0.0)
    output["delta_weight"] = output["target_weight"] - output["current_weight"]
    return output.loc[
        :,
        [
            "timestamp",
            "instrument_id",
            "current_weight",
            "target_weight",
            "delta_weight",
        ],
    ]


def _diagnostics(target_weights: pd.DataFrame) -> pd.DataFrame:
    if target_weights.empty:
        return pd.DataFrame(columns=["timestamp", "instrument_count", "gross_weight"])
    diagnostics = (
        target_weights.assign(abs_weight=target_weights["target_weight"].abs())
        .groupby("timestamp", sort=True)
        .agg(
            instrument_count=("instrument_id", "size"),
            gross_weight=("abs_weight", "sum"),
        )
        .reset_index()
    )
    diagnostics["gross_weight"] = diagnostics["gross_weight"].astype(float)
    return diagnostics


def _expanded_current_positions(
    timestamps: list[object],
    current: pd.DataFrame,
) -> pd.DataFrame:
    if not timestamps or current.empty:
        return pd.DataFrame(columns=["timestamp", "instrument_id", "current_weight"])
    timestamp_frame = pd.DataFrame({"timestamp": timestamps})
    return timestamp_frame.merge(current, how="cross")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
