"""Configurable 5-minute cross-sectional A-share strategy shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from quant_research.portfolio import (
    PortfolioConfig,
    PortfolioConstructor,
    apply_cn_t1_constraints,
)

Weighting = Literal["equal", "signal"]
ExecutionRule = Literal["cn_t1"]


@dataclass(frozen=True, slots=True)
class FiveMinuteCrossSectionalConfig:
    """Parameters for a 5-minute cross-sectional target portfolio."""

    name: str
    top_n: int
    weighting: Weighting
    max_weight: float | None = None
    min_signal: float | None = None
    rebalance_frequency: str = "5m"
    execution_rule: ExecutionRule = "cn_t1"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("strategy name is required")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.weighting not in {"equal", "signal"}:
            raise ValueError("weighting must be 'equal' or 'signal'")
        if self.max_weight is not None and not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.rebalance_frequency != "5m":
            raise ValueError("FiveMinuteCrossSectionalConfig requires 5m frequency")
        if self.execution_rule != "cn_t1":
            raise ValueError("execution_rule must be 'cn_t1'")


@dataclass(frozen=True, slots=True)
class FiveMinuteCrossSectionalResult:
    """Output tables produced by the strategy shell."""

    config: FiveMinuteCrossSectionalConfig
    selected_signals: pd.DataFrame
    target_weights: pd.DataFrame
    rebalance_orders: pd.DataFrame
    constrained_orders: pd.DataFrame
    diagnostics: pd.DataFrame


class FiveMinuteCrossSectionalStrategy:
    """Selects top-ranked 5-minute signals and applies A-share T+1 constraints."""

    def __init__(
        self, *, portfolio_constructor: PortfolioConstructor | None = None
    ) -> None:
        self._portfolio_constructor = portfolio_constructor or PortfolioConstructor()

    def build(
        self,
        signals: pd.DataFrame,
        config: FiveMinuteCrossSectionalConfig,
        *,
        current_positions: pd.DataFrame | None,
    ) -> FiveMinuteCrossSectionalResult:
        _require_columns(signals, ("timestamp", "instrument_id", "signal"))
        selected = _select_top_signals(signals, config=config)
        if config.weighting == "signal" and not selected.empty:
            negative = selected["signal"].astype(float) < 0
            if negative.any():
                raise ValueError(
                    "signal weighting requires non-negative selected signals for long-only A-share portfolios"
                )
        portfolio = self._portfolio_constructor.build(
            selected,
            PortfolioConfig(
                name=config.name,
                rebalance_frequency=config.rebalance_frequency,
                weighting=config.weighting,
                max_weight=config.max_weight,
                parameters=dict(config.parameters),
            ),
            current_positions=current_positions,
        )
        constrained = apply_cn_t1_constraints(
            portfolio.rebalance_orders,
            current_positions=current_positions,
        )
        diagnostics = _diagnostics(
            selected_signals=selected,
            constrained_orders=constrained,
        )
        return FiveMinuteCrossSectionalResult(
            config=config,
            selected_signals=selected,
            target_weights=portfolio.target_weights,
            rebalance_orders=portfolio.rebalance_orders,
            constrained_orders=constrained,
            diagnostics=diagnostics,
        )


def _select_top_signals(
    signals: pd.DataFrame, *, config: FiveMinuteCrossSectionalConfig
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame(
            columns=["timestamp", "instrument_id", "signal", "rank"]
        )
    eligible = signals
    if config.min_signal is not None:
        eligible = eligible.loc[
            eligible["signal"].astype(float) >= config.min_signal
        ]
    if eligible.empty:
        return pd.DataFrame(
            columns=["timestamp", "instrument_id", "signal", "rank"]
        )
    selected = (
        eligible.sort_values(
            ["timestamp", "signal", "instrument_id"],
            ascending=[True, False, True],
        )
        .groupby("timestamp", sort=False)
        .head(config.top_n)
        .loc[:, ["timestamp", "instrument_id", "signal"]]
        .reset_index(drop=True)
    )
    selected["rank"] = selected.groupby("timestamp", sort=False).cumcount() + 1
    return selected


def _diagnostics(
    *,
    selected_signals: pd.DataFrame,
    constrained_orders: pd.DataFrame,
) -> pd.DataFrame:
    selected_counts = (
        selected_signals.groupby("timestamp", sort=True)
        .size()
        .rename("selected_count")
        .reset_index()
    )
    blocked = (
        constrained_orders.groupby("timestamp", sort=True)
        .agg(
            blocked_sell_weight=("blocked_sell_weight", "sum"),
            executable_turnover=("executable_delta_weight", lambda s: s.abs().sum()),
            desired_turnover=("desired_delta_weight", lambda s: s.abs().sum()),
        )
        .reset_index()
    )
    if selected_counts.empty:
        return blocked
    return selected_counts.merge(blocked, on="timestamp", how="outer").fillna(0.0)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
