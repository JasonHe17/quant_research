"""Stateful strategy policy contracts and rank-buffer policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd


DecisionAction = Literal["entry", "exit", "hold", "resize_up", "resize_down", "no_trade"]
WeightingMethod = Literal["equal", "score"]

REASON_CODES = (
    "entry_rank",
    "hold_buffer",
    "exit_rank",
    "resize_up",
    "resize_down",
    "below_edge",
    "below_weight_band",
    "min_hold_blocked",
    "t1_sell_blocked",
    "limit_up_buy_blocked",
    "limit_down_sell_blocked",
    "capacity_capped",
    "risk_reduction",
    "cash_limited",
    "universe_removed",
)

PORTFOLIO_STATE_COLUMNS = (
    "instrument_id",
    "current_weight",
    "sellable_weight",
    "holding_bars",
)
PORTFOLIO_INTENT_COLUMNS = (
    "timestamp",
    "instrument_id",
    "current_weight",
    "aim_weight",
    "policy_target_weight",
    "rank",
    "score",
    "expected_edge_bps",
    "estimated_cost_bps",
    "reason",
    "constraint_flags",
)
TRADE_DECISION_COLUMNS = (
    "timestamp",
    "instrument_id",
    "action",
    "current_weight",
    "aim_weight",
    "target_weight",
    "delta_weight",
    "expected_edge_bps",
    "estimated_cost_bps",
    "priority",
    "decision_reason",
    "constraint_flags",
)
ORDER_INTENT_COLUMNS = (
    "timestamp",
    "instrument_id",
    "side",
    "quantity",
    "target_weight",
    "delta_weight",
    "price_style",
    "limit_price",
    "time_in_force",
    "participation_limit",
    "expire_time",
    "priority",
    "client_order_id",
    "decision_reason",
)


@dataclass(frozen=True, slots=True)
class StrategyPolicyResult:
    """Standard result emitted by a strategy policy decision."""

    portfolio_intent: pd.DataFrame
    trade_decisions: pd.DataFrame
    order_intents: pd.DataFrame
    diagnostics: pd.DataFrame
    policy_state: pd.DataFrame = field(default_factory=lambda: empty_portfolio_state())


class StrategyPolicy(Protocol):
    """Converts forecasts plus portfolio state into trade intent."""

    def decide(
        self,
        forecasts: pd.DataFrame,
        portfolio_state: pd.DataFrame | None = None,
    ) -> StrategyPolicyResult:
        """Return the policy decision for one timestamp."""


@dataclass(frozen=True, slots=True)
class RankBufferDropConfig:
    """Configuration for a stateful rank-buffer top-k-drop policy."""

    target_count: int
    entry_rank: int
    exit_rank: int
    policy_id: str = "rank_buffer_drop"
    policy_version: str = "v1"
    max_entries_per_rebalance: int | None = None
    max_exits_per_rebalance: int | None = None
    min_hold_bars: int = 0
    weighting: WeightingMethod = "equal"
    max_name_weight: float | None = None
    min_expected_edge_bps: float | None = None
    estimated_cost_bps: float = 0.0
    no_trade_weight_band: float = 0.0
    partial_rebalance_rate: float = 1.0
    max_gross_turnover_per_rebalance: float | None = None
    participation_limit: float | None = None
    order_price_style: str = "market"
    time_in_force: str = "day"

    def __post_init__(self) -> None:
        if self.target_count <= 0:
            raise ValueError("target_count must be positive")
        if self.entry_rank <= 0:
            raise ValueError("entry_rank must be positive")
        if self.exit_rank < self.entry_rank:
            raise ValueError("exit_rank must be greater than or equal to entry_rank")
        if self.max_entries_per_rebalance is not None and self.max_entries_per_rebalance < 0:
            raise ValueError("max_entries_per_rebalance must be non-negative")
        if self.max_exits_per_rebalance is not None and self.max_exits_per_rebalance < 0:
            raise ValueError("max_exits_per_rebalance must be non-negative")
        if self.min_hold_bars < 0:
            raise ValueError("min_hold_bars must be non-negative")
        if self.weighting not in {"equal", "score"}:
            raise ValueError("weighting must be equal or score")
        if self.max_name_weight is not None and not 0 < self.max_name_weight <= 1:
            raise ValueError("max_name_weight must be in (0, 1]")
        if self.min_expected_edge_bps is not None and self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps must be non-negative")
        if self.estimated_cost_bps < 0:
            raise ValueError("estimated_cost_bps must be non-negative")
        if self.no_trade_weight_band < 0:
            raise ValueError("no_trade_weight_band must be non-negative")
        if not 0 < self.partial_rebalance_rate <= 1:
            raise ValueError("partial_rebalance_rate must be in (0, 1]")
        if (
            self.max_gross_turnover_per_rebalance is not None
            and self.max_gross_turnover_per_rebalance < 0
        ):
            raise ValueError("max_gross_turnover_per_rebalance must be non-negative")
        if self.participation_limit is not None and not 0 < self.participation_limit <= 1:
            raise ValueError("participation_limit must be in (0, 1]")
        if not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")


class RankBufferDropPolicy:
    """Stateful rank-buffer policy with replacement caps and no-trade bands."""

    def __init__(self, config: RankBufferDropConfig) -> None:
        self.config = config

    def decide(
        self,
        forecasts: pd.DataFrame,
        portfolio_state: pd.DataFrame | None = None,
    ) -> StrategyPolicyResult:
        _require_columns(forecasts, ("timestamp", "instrument_id", "score"), name="forecasts")
        if forecasts.empty:
            raise ValueError("forecasts must contain one timestamp")
        timestamp = _single_timestamp(forecasts)
        ranked = _rank_forecasts(forecasts)
        state = _prepare_portfolio_state(portfolio_state)
        rows = self._decision_rows(timestamp, ranked, state)
        rows = self._apply_turnover_cap(rows)
        intent = _rows_to_frame(rows, PORTFOLIO_INTENT_COLUMNS, _intent_row)
        decisions = _rows_to_frame(rows, TRADE_DECISION_COLUMNS, _trade_decision_row)
        orders = _order_intents_from_rows(rows, self.config)
        diagnostics = _diagnostics_from_rows(timestamp, rows, self.config)
        next_state = _next_policy_state(rows, state)
        return StrategyPolicyResult(
            portfolio_intent=intent,
            trade_decisions=decisions,
            order_intents=orders,
            diagnostics=diagnostics,
            policy_state=next_state,
        )

    def _decision_rows(
        self,
        timestamp: object,
        ranked: pd.DataFrame,
        state: pd.DataFrame,
    ) -> list[dict[str, object]]:
        cfg = self.config
        forecast_by_id = _forecast_mapping(ranked)
        state_by_id = _state_mapping(state)
        held = [
            instrument_id
            for instrument_id, payload in state_by_id.items()
            if float(payload["current_weight"]) > 0
        ]
        retained: list[str] = []
        exit_candidates: list[str] = []
        reason_by_id: dict[str, str] = {}
        flags_by_id: dict[str, set[str]] = {}

        for instrument_id in held:
            rank = _rank_for(instrument_id, forecast_by_id)
            holding_bars = int(state_by_id[instrument_id].get("holding_bars") or 0)
            if rank is not None and rank <= cfg.exit_rank:
                retained.append(instrument_id)
                reason_by_id[instrument_id] = "hold_buffer"
                continue
            if holding_bars < cfg.min_hold_bars:
                retained.append(instrument_id)
                reason_by_id[instrument_id] = "min_hold_blocked"
                continue
            exit_candidates.append(instrument_id)

        selected_exits = _limited_exits(
            exit_candidates,
            forecast_by_id=forecast_by_id,
            state_by_id=state_by_id,
            max_exits=cfg.max_exits_per_rebalance,
        )
        selected_exit_set = set(selected_exits)
        for instrument_id in exit_candidates:
            if instrument_id in selected_exit_set:
                reason_by_id[instrument_id] = (
                    "exit_rank"
                    if instrument_id in forecast_by_id
                    else "universe_removed"
                )
            else:
                retained.append(instrument_id)
                reason_by_id[instrument_id] = "hold_buffer"
                flags_by_id.setdefault(instrument_id, set()).add("exit_budget_deferred")

        active = list(dict.fromkeys(retained))
        for instrument_id in active:
            if instrument_id in state_by_id:
                reason_by_id.setdefault(instrument_id, "hold_buffer")
        slots = max(0, cfg.target_count - len(active))
        entry_budget = (
            slots
            if cfg.max_entries_per_rebalance is None
            else min(slots, cfg.max_entries_per_rebalance)
        )
        entries: list[str] = []
        rejected_edge: list[str] = []
        if entry_budget > 0:
            for row in ranked.itertuples(index=False):
                instrument_id = str(row.instrument_id)
                if instrument_id in active or instrument_id in state_by_id:
                    continue
                if int(row.rank) > cfg.entry_rank:
                    break
                edge = _expected_edge_bps(row)
                if cfg.min_expected_edge_bps is not None and edge < (
                    cfg.min_expected_edge_bps + cfg.estimated_cost_bps
                ):
                    rejected_edge.append(instrument_id)
                    continue
                entries.append(instrument_id)
                if len(entries) >= entry_budget:
                    break
        active.extend(entries)
        aim_weights = _aim_weights(active, forecast_by_id=forecast_by_id, config=cfg)
        output_ids = list(dict.fromkeys([*active, *selected_exits, *rejected_edge]))
        rows: list[dict[str, object]] = []
        for priority, instrument_id in enumerate(output_ids, start=1):
            forecast = forecast_by_id.get(instrument_id, {})
            state_row = state_by_id.get(instrument_id, {})
            current_weight = float(state_row.get("current_weight") or 0.0)
            aim_weight = float(aim_weights.get(instrument_id, 0.0))
            reason = reason_by_id.get(instrument_id)
            if instrument_id in entries:
                reason = "entry_rank"
            elif instrument_id in rejected_edge:
                reason = "below_edge"
            elif instrument_id in selected_exit_set:
                reason = reason or "exit_rank"
            elif reason is None:
                reason = "hold_buffer" if current_weight > 0 else "entry_rank"
            target_weight, reason, flags = _target_weight_for_row(
                current_weight=current_weight,
                aim_weight=aim_weight,
                reason=reason,
                state_row=state_row,
                config=cfg,
                existing_flags=flags_by_id.get(instrument_id, set()),
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "instrument_id": instrument_id,
                    "current_weight": current_weight,
                    "aim_weight": aim_weight,
                    "target_weight": target_weight,
                    "rank": forecast.get("rank"),
                    "score": forecast.get("score"),
                    "expected_edge_bps": forecast.get("expected_edge_bps"),
                    "estimated_cost_bps": cfg.estimated_cost_bps,
                    "priority": priority,
                    "decision_reason": reason,
                    "constraint_flags": ",".join(sorted(flags)),
                }
            )
        return rows

    def _apply_turnover_cap(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        cap = self.config.max_gross_turnover_per_rebalance
        if cap is None or not rows:
            return rows
        gross_turnover = sum(
            abs(float(row["target_weight"]) - float(row["current_weight"]))
            for row in rows
        )
        if gross_turnover <= cap or gross_turnover <= 0:
            return rows
        scale = cap / gross_turnover
        for row in rows:
            current = float(row["current_weight"])
            target = float(row["target_weight"])
            row["target_weight"] = current + (target - current) * scale
            flags = set(str(row.get("constraint_flags") or "").split(","))
            flags.discard("")
            flags.add("turnover_scaled")
            row["constraint_flags"] = ",".join(sorted(flags))
        return rows


def empty_portfolio_state() -> pd.DataFrame:
    """Return an empty strategy portfolio-state frame."""

    return pd.DataFrame(columns=PORTFOLIO_STATE_COLUMNS)


def empty_portfolio_intent() -> pd.DataFrame:
    """Return an empty portfolio-intent frame."""

    return pd.DataFrame(columns=PORTFOLIO_INTENT_COLUMNS)


def empty_trade_decisions() -> pd.DataFrame:
    """Return an empty trade-decision frame."""

    return pd.DataFrame(columns=TRADE_DECISION_COLUMNS)


def empty_order_intents() -> pd.DataFrame:
    """Return an empty order-intent frame."""

    return pd.DataFrame(columns=ORDER_INTENT_COLUMNS)


def _single_timestamp(frame: pd.DataFrame) -> object:
    timestamps = frame["timestamp"].drop_duplicates().tolist()
    if len(timestamps) != 1:
        raise ValueError("forecasts must contain exactly one timestamp")
    return timestamps[0]


def _rank_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    ranked = forecasts.copy()
    if "rank" not in ranked.columns:
        ranked = ranked.sort_values(["score", "instrument_id"], ascending=[False, True])
        ranked["rank"] = range(1, len(ranked) + 1)
    ranked["rank"] = ranked["rank"].astype(int)
    return ranked.sort_values(["rank", "instrument_id"]).reset_index(drop=True)


def _prepare_portfolio_state(portfolio_state: pd.DataFrame | None) -> pd.DataFrame:
    if portfolio_state is None or portfolio_state.empty:
        return empty_portfolio_state()
    _require_columns(portfolio_state, ("instrument_id", "current_weight"), name="portfolio_state")
    state = portfolio_state.copy()
    if "sellable_weight" not in state.columns:
        state["sellable_weight"] = pd.NA
    if "holding_bars" not in state.columns:
        state["holding_bars"] = 0
    return state.loc[:, PORTFOLIO_STATE_COLUMNS]


def _forecast_mapping(ranked: pd.DataFrame) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    for row in ranked.itertuples(index=False):
        expected_edge = _expected_edge_bps(row)
        mapping[str(row.instrument_id)] = {
            "rank": int(row.rank),
            "score": float(row.score),
            "expected_edge_bps": expected_edge,
        }
    return mapping


def _state_mapping(state: pd.DataFrame) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    for row in state.itertuples(index=False):
        sellable = getattr(row, "sellable_weight", pd.NA)
        mapping[str(row.instrument_id)] = {
            "current_weight": float(row.current_weight),
            "sellable_weight": sellable,
            "holding_bars": int(getattr(row, "holding_bars", 0) or 0),
        }
    return mapping


def _rank_for(
    instrument_id: str,
    forecast_by_id: dict[str, dict[str, object]],
) -> int | None:
    forecast = forecast_by_id.get(instrument_id)
    if not forecast:
        return None
    return int(forecast["rank"])


def _expected_edge_bps(row: object) -> float:
    if hasattr(row, "expected_edge_bps"):
        value = getattr(row, "expected_edge_bps")
    elif hasattr(row, "expected_return_bps"):
        value = getattr(row, "expected_return_bps")
    else:
        value = None
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _limited_exits(
    exit_candidates: list[str],
    *,
    forecast_by_id: dict[str, dict[str, object]],
    state_by_id: dict[str, dict[str, object]],
    max_exits: int | None,
) -> list[str]:
    ordered = sorted(
        exit_candidates,
        key=lambda instrument_id: (
            _rank_for(instrument_id, forecast_by_id) is None,
            -(_rank_for(instrument_id, forecast_by_id) or 10**9),
            float(forecast_by_id.get(instrument_id, {}).get("score") or -10**9),
            -float(state_by_id.get(instrument_id, {}).get("current_weight") or 0.0),
            instrument_id,
        ),
    )
    if max_exits is None:
        return ordered
    return ordered[:max_exits]


def _aim_weights(
    active: list[str],
    *,
    forecast_by_id: dict[str, dict[str, object]],
    config: RankBufferDropConfig,
) -> dict[str, float]:
    if not active:
        return {}
    if config.weighting == "equal":
        raw = {instrument_id: 1.0 for instrument_id in active}
    else:
        raw = {
            instrument_id: max(float(forecast_by_id.get(instrument_id, {}).get("score") or 0.0), 0.0)
            for instrument_id in active
        }
        if sum(raw.values()) <= 0:
            raw = {instrument_id: 1.0 for instrument_id in active}
    total = sum(raw.values())
    weights = {instrument_id: value / total for instrument_id, value in raw.items()}
    if config.max_name_weight is not None:
        weights = {
            instrument_id: min(weight, config.max_name_weight)
            for instrument_id, weight in weights.items()
        }
    return weights


def _target_weight_for_row(
    *,
    current_weight: float,
    aim_weight: float,
    reason: str,
    state_row: dict[str, object],
    config: RankBufferDropConfig,
    existing_flags: set[str],
) -> tuple[float, str, set[str]]:
    flags = set(existing_flags)
    if reason == "below_edge":
        return current_weight, reason, flags
    delta = aim_weight - current_weight
    if abs(delta) <= 1e-12:
        return current_weight, reason, flags
    if abs(delta) < config.no_trade_weight_band:
        return current_weight, "below_weight_band", flags
    target = current_weight + config.partial_rebalance_rate * delta
    sellable = state_row.get("sellable_weight")
    if target < current_weight and sellable is not None and not pd.isna(sellable):
        sellable_weight = max(float(sellable), 0.0)
        minimum_target = max(0.0, current_weight - sellable_weight)
        if target < minimum_target:
            target = minimum_target
            reason = "t1_sell_blocked"
            flags.add("sellable_weight_limited")
    return max(float(target), 0.0), reason, flags


def _rows_to_frame(
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
    mapper: object,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    mapped = [mapper(row) for row in rows]  # type: ignore[misc]
    return pd.DataFrame(mapped).loc[:, columns]


def _intent_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "timestamp": row["timestamp"],
        "instrument_id": row["instrument_id"],
        "current_weight": row["current_weight"],
        "aim_weight": row["aim_weight"],
        "policy_target_weight": row["target_weight"],
        "rank": row["rank"],
        "score": row["score"],
        "expected_edge_bps": row["expected_edge_bps"],
        "estimated_cost_bps": row["estimated_cost_bps"],
        "reason": row["decision_reason"],
        "constraint_flags": row["constraint_flags"],
    }


def _trade_decision_row(row: dict[str, object]) -> dict[str, object]:
    current = float(row["current_weight"])
    target = float(row["target_weight"])
    delta = target - current
    return {
        "timestamp": row["timestamp"],
        "instrument_id": row["instrument_id"],
        "action": _action(current=current, target=target, reason=str(row["decision_reason"])),
        "current_weight": current,
        "aim_weight": row["aim_weight"],
        "target_weight": target,
        "delta_weight": delta,
        "expected_edge_bps": row["expected_edge_bps"],
        "estimated_cost_bps": row["estimated_cost_bps"],
        "priority": row["priority"],
        "decision_reason": row["decision_reason"],
        "constraint_flags": row["constraint_flags"],
    }


def _action(*, current: float, target: float, reason: str) -> DecisionAction:
    delta = target - current
    if abs(delta) <= 1e-12:
        return "no_trade" if reason in {"below_edge", "below_weight_band"} else "hold"
    if current <= 0 and target > 0:
        return "entry"
    if current > 0 and target <= 0:
        return "exit"
    if delta > 0:
        return "resize_up"
    return "resize_down"


def _order_intents_from_rows(
    rows: list[dict[str, object]],
    config: RankBufferDropConfig,
) -> pd.DataFrame:
    order_rows = []
    for row in rows:
        current = float(row["current_weight"])
        target = float(row["target_weight"])
        delta = target - current
        if abs(delta) <= 1e-12:
            continue
        side = "buy" if delta > 0 else "sell"
        order_rows.append(
            {
                "timestamp": row["timestamp"],
                "instrument_id": row["instrument_id"],
                "side": side,
                "quantity": pd.NA,
                "target_weight": target,
                "delta_weight": delta,
                "price_style": config.order_price_style,
                "limit_price": pd.NA,
                "time_in_force": config.time_in_force,
                "participation_limit": config.participation_limit,
                "expire_time": pd.NA,
                "priority": row["priority"],
                "client_order_id": _client_order_id(config, row, side),
                "decision_reason": row["decision_reason"],
            }
        )
    if not order_rows:
        return empty_order_intents()
    return pd.DataFrame(order_rows).loc[:, ORDER_INTENT_COLUMNS]


def _client_order_id(
    config: RankBufferDropConfig,
    row: dict[str, object],
    side: str,
) -> str:
    timestamp = str(row["timestamp"]).replace(":", "").replace("+", "p")
    return f"{config.policy_id}:{config.policy_version}:{timestamp}:{row['instrument_id']}:{side}"


def _diagnostics_from_rows(
    timestamp: object,
    rows: list[dict[str, object]],
    config: RankBufferDropConfig,
) -> pd.DataFrame:
    decisions = [_trade_decision_row(row) for row in rows]
    planned_gross_turnover = sum(abs(float(row["delta_weight"])) for row in decisions)
    action_counts = pd.Series([row["action"] for row in decisions]).value_counts()
    reason_counts = pd.Series([row["decision_reason"] for row in decisions]).value_counts()
    flags = ",".join(str(row.get("constraint_flags") or "") for row in rows)
    diagnostic = {
        "timestamp": timestamp,
        "policy_id": config.policy_id,
        "policy_version": config.policy_version,
        "configured_target_count": config.target_count,
        "active_count": sum(float(row["target_weight"]) > 0 for row in rows),
        "decision_count": len(rows),
        "order_intent_count": sum(abs(float(row["delta_weight"])) > 1e-12 for row in decisions),
        "planned_gross_turnover": planned_gross_turnover,
        "entry_count": int(action_counts.get("entry", 0)),
        "exit_count": int(action_counts.get("exit", 0)),
        "resize_count": int(action_counts.get("resize_up", 0) + action_counts.get("resize_down", 0)),
        "hold_count": int(action_counts.get("hold", 0)),
        "no_trade_count": int(action_counts.get("no_trade", 0)),
        "below_edge_count": int(reason_counts.get("below_edge", 0)),
        "below_weight_band_count": int(reason_counts.get("below_weight_band", 0)),
        "t1_sell_blocked_count": int(reason_counts.get("t1_sell_blocked", 0)),
        "turnover_scaled_count": flags.count("turnover_scaled"),
    }
    return pd.DataFrame([diagnostic])


def _next_policy_state(
    rows: list[dict[str, object]],
    previous_state: pd.DataFrame,
) -> pd.DataFrame:
    previous = _state_mapping(previous_state)
    state_rows = []
    for row in rows:
        target = float(row["target_weight"])
        if target <= 1e-12:
            continue
        instrument_id = str(row["instrument_id"])
        current = float(row["current_weight"])
        previous_holding = int(previous.get(instrument_id, {}).get("holding_bars") or 0)
        state_rows.append(
            {
                "instrument_id": instrument_id,
                "current_weight": target,
                "sellable_weight": pd.NA,
                "holding_bars": previous_holding + 1 if current > 0 else 0,
            }
        )
    if not state_rows:
        return empty_portfolio_state()
    return pd.DataFrame(state_rows).loc[:, PORTFOLIO_STATE_COLUMNS]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
