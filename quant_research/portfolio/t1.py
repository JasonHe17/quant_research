"""A-share T+1 portfolio order constraints."""

from __future__ import annotations

import pandas as pd


def apply_cn_t1_constraints(
    rebalance_orders: pd.DataFrame,
    *,
    current_positions: pd.DataFrame | None,
) -> pd.DataFrame:
    """Split desired rebalance weights into T+1-executable and blocked amounts.

    ``current_positions`` must include ``sellable_weight`` for any existing
    position. This keeps the caller responsible for maintaining the A-share
    distinction between total position and shares that can actually be sold.
    """

    _require_columns(
        rebalance_orders,
        (
            "timestamp",
            "instrument_id",
            "current_weight",
            "target_weight",
            "delta_weight",
        ),
        name="rebalance_orders",
    )
    positions = _positions_frame(current_positions)
    merged = rebalance_orders.copy()
    merged = merged.merge(positions, on="instrument_id", how="left")
    merged["sellable_weight"] = merged["sellable_weight"].fillna(0.0).astype(float)
    desired_delta = merged["delta_weight"].astype(float)
    sellable_weight = merged["sellable_weight"].clip(lower=0.0)
    executable_delta = desired_delta.where(
        desired_delta >= 0.0,
        desired_delta.clip(lower=-sellable_weight),
    )
    blocked_sell_weight = (executable_delta - desired_delta).clip(lower=0.0)
    output = merged.copy()
    output["desired_delta_weight"] = desired_delta
    output["executable_delta_weight"] = executable_delta.astype(float)
    output["blocked_sell_weight"] = blocked_sell_weight.astype(float)
    output["t1_blocked"] = output["blocked_sell_weight"] > 0.0
    output["trade_action"] = output["executable_delta_weight"].apply(_trade_action)
    return output.loc[
        :,
        [
            "timestamp",
            "instrument_id",
            "current_weight",
            "sellable_weight",
            "target_weight",
            "desired_delta_weight",
            "executable_delta_weight",
            "blocked_sell_weight",
            "t1_blocked",
            "trade_action",
        ],
    ]


def _positions_frame(current_positions: pd.DataFrame | None) -> pd.DataFrame:
    if current_positions is None:
        return pd.DataFrame(columns=["instrument_id", "sellable_weight"])
    if current_positions.empty:
        return pd.DataFrame(columns=["instrument_id", "sellable_weight"])
    _require_columns(
        current_positions,
        ("instrument_id", "sellable_weight"),
        name="current_positions",
    )
    positions = current_positions.loc[:, ["instrument_id", "sellable_weight"]].copy()
    positions["sellable_weight"] = positions["sellable_weight"].astype(float)
    return positions


def _trade_action(delta_weight: float) -> str:
    if delta_weight > 0:
        return "buy"
    if delta_weight < 0:
        return "sell"
    return "hold"


def _require_columns(
    frame: pd.DataFrame, columns: tuple[str, ...], *, name: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
