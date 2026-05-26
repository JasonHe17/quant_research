"""Stateful strategy policy contracts and rank-buffer policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd


DecisionAction = Literal["entry", "exit", "hold", "resize_up", "resize_down", "no_trade"]
WeightingMethod = Literal["equal", "score"]
OptimizerWeightingMethod = Literal["equal", "utility"]

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
    gross_exposure_scale: float = 1.0
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
        if not 0 <= self.gross_exposure_scale <= 1:
            raise ValueError("gross_exposure_scale must be in [0, 1]")
        if self.participation_limit is not None and not 0 < self.participation_limit <= 1:
            raise ValueError("participation_limit must be in (0, 1]")
        if not self.policy_id:
            raise ValueError("policy_id must be non-empty")
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")


@dataclass(frozen=True, slots=True)
class CostAwareOptimizerConfig:
    """Configuration for a cost-aware single-period optimizer policy MVP."""

    target_count: int
    candidate_rank: int
    policy_id: str = "cost_aware_optimizer"
    policy_version: str = "v1"
    min_hold_bars: int = 0
    max_entries_per_rebalance: int | None = None
    max_exits_per_rebalance: int | None = None
    weighting: OptimizerWeightingMethod = "utility"
    max_name_weight: float | None = None
    score_to_edge_bps: float = 100.0
    min_net_edge_bps: float = 0.0
    estimated_cost_bps: float = 0.0
    risk_penalty_multiplier: float = 1.0
    no_trade_weight_band: float = 0.0
    partial_rebalance_rate: float = 1.0
    max_gross_turnover_per_rebalance: float | None = None
    max_gross_exposure_increase_per_rebalance: float | None = None
    gross_exposure_scale: float = 1.0
    participation_limit: float | None = None
    order_price_style: str = "market"
    time_in_force: str = "day"

    def __post_init__(self) -> None:
        if self.target_count <= 0:
            raise ValueError("target_count must be positive")
        if self.candidate_rank <= 0:
            raise ValueError("candidate_rank must be positive")
        if self.candidate_rank < self.target_count:
            raise ValueError("candidate_rank must be greater than or equal to target_count")
        if self.min_hold_bars < 0:
            raise ValueError("min_hold_bars must be non-negative")
        if self.max_entries_per_rebalance is not None and self.max_entries_per_rebalance < 0:
            raise ValueError("max_entries_per_rebalance must be non-negative")
        if self.max_exits_per_rebalance is not None and self.max_exits_per_rebalance < 0:
            raise ValueError("max_exits_per_rebalance must be non-negative")
        if self.weighting not in {"equal", "utility"}:
            raise ValueError("weighting must be equal or utility")
        if self.max_name_weight is not None and not 0 < self.max_name_weight <= 1:
            raise ValueError("max_name_weight must be in (0, 1]")
        if self.score_to_edge_bps < 0:
            raise ValueError("score_to_edge_bps must be non-negative")
        if self.min_net_edge_bps < 0:
            raise ValueError("min_net_edge_bps must be non-negative")
        if self.estimated_cost_bps < 0:
            raise ValueError("estimated_cost_bps must be non-negative")
        if self.risk_penalty_multiplier < 0:
            raise ValueError("risk_penalty_multiplier must be non-negative")
        if self.no_trade_weight_band < 0:
            raise ValueError("no_trade_weight_band must be non-negative")
        if not 0 < self.partial_rebalance_rate <= 1:
            raise ValueError("partial_rebalance_rate must be in (0, 1]")
        if (
            self.max_gross_turnover_per_rebalance is not None
            and self.max_gross_turnover_per_rebalance < 0
        ):
            raise ValueError("max_gross_turnover_per_rebalance must be non-negative")
        if (
            self.max_gross_exposure_increase_per_rebalance is not None
            and self.max_gross_exposure_increase_per_rebalance < 0
        ):
            raise ValueError(
                "max_gross_exposure_increase_per_rebalance must be non-negative"
            )
        if not 0 <= self.gross_exposure_scale <= 1:
            raise ValueError("gross_exposure_scale must be in [0, 1]")
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
        has_entry_eligibility = "entry_eligible" in ranked.columns
        eligible_entry_rank = 0
        if cfg.gross_exposure_scale <= 0:
            entry_budget = 0
        if entry_budget > 0:
            for row in ranked.itertuples(index=False):
                instrument_id = str(row.instrument_id)
                if instrument_id in active or instrument_id in state_by_id:
                    continue
                if has_entry_eligibility and not _entry_eligible_from_row(row):
                    continue
                if has_entry_eligibility:
                    eligible_entry_rank += 1
                    if eligible_entry_rank > cfg.entry_rank:
                        break
                elif int(row.rank) > cfg.entry_rank:
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
        unscaled_aim_weights = _aim_weights(active, forecast_by_id=forecast_by_id, config=cfg)
        aim_weights = _scale_aim_weights(unscaled_aim_weights, cfg.gross_exposure_scale)
        output_ids = list(dict.fromkeys([*active, *selected_exits, *rejected_edge]))
        rows: list[dict[str, object]] = []
        for priority, instrument_id in enumerate(output_ids, start=1):
            forecast = forecast_by_id.get(instrument_id, {})
            state_row = state_by_id.get(instrument_id, {})
            current_weight = float(state_row.get("current_weight") or 0.0)
            aim_weight = float(aim_weights.get(instrument_id, 0.0))
            unscaled_aim_weight = float(unscaled_aim_weights.get(instrument_id, 0.0))
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
            if unscaled_aim_weight > aim_weight + 1e-12:
                flags.add("gross_exposure_scaled")
                if target_weight < current_weight - 1e-12 and reason in {
                    "hold_buffer",
                    "min_hold_blocked",
                    "resize_down",
                }:
                    reason = "risk_reduction"
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
                    "net_edge_bps": forecast.get("net_edge_bps"),
                    "estimated_cost_bps": cfg.estimated_cost_bps,
                    "priority": priority,
                    "decision_reason": reason,
                    "constraint_flags": ",".join(sorted(flags)),
                }
            )
        return rows

    def _apply_turnover_cap(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return _apply_turnover_cap(
            rows,
            self.config.max_gross_turnover_per_rebalance,
            no_trade_weight_band=self.config.no_trade_weight_band,
            gross_exposure_scale=self.config.gross_exposure_scale,
        )


class CostAwareOptimizerPolicy:
    """Single-period optimizer-style policy with cost and risk penalties."""

    def __init__(self, config: CostAwareOptimizerConfig) -> None:
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
        rows = _apply_optimizer_turnover_budget(rows, self.config)
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
        forecast_by_id = _optimizer_forecast_mapping(ranked, cfg)
        state_by_id = _state_mapping(state)
        candidate_ids = _optimizer_candidate_ids(ranked, cfg.candidate_rank)
        held_ids = [
            instrument_id
            for instrument_id, payload in state_by_id.items()
            if float(payload["current_weight"]) > 0
        ]
        scored_ids = list(dict.fromkeys([*candidate_ids, *held_ids]))
        eligible_ids = []
        forced_hold_ids = []
        for instrument_id in scored_ids:
            current_weight = float(state_by_id.get(instrument_id, {}).get("current_weight") or 0.0)
            holding_bars = int(state_by_id.get(instrument_id, {}).get("holding_bars") or 0)
            forecast = forecast_by_id.get(instrument_id, {})
            net_edge = float(forecast.get("net_edge_bps") or -10**9)
            if current_weight > 0 and holding_bars < cfg.min_hold_bars:
                forced_hold_ids.append(instrument_id)
                continue
            if net_edge >= cfg.min_net_edge_bps and instrument_id in forecast_by_id:
                eligible_ids.append(instrument_id)
        selected = list(dict.fromkeys(forced_hold_ids))[: cfg.target_count]
        entry_count = 0
        for instrument_id in _ordered_optimizer_candidates(
            [
                instrument_id
                for instrument_id in eligible_ids
                if instrument_id not in forced_hold_ids
            ],
            forecast_by_id=forecast_by_id,
            state_by_id=state_by_id,
            config=cfg,
        ):
            if len(selected) >= cfg.target_count:
                break
            current_weight = float(
                state_by_id.get(instrument_id, {}).get("current_weight") or 0.0
            )
            if current_weight <= 0:
                if (
                    cfg.max_entries_per_rebalance is not None
                    and entry_count >= cfg.max_entries_per_rebalance
                ):
                    continue
                entry_count += 1
            selected.append(instrument_id)
        if cfg.max_exits_per_rebalance is not None:
            exit_candidates = [
                instrument_id
                for instrument_id in held_ids
                if instrument_id not in selected and instrument_id not in forced_hold_ids
            ]
            allowed_exits = set(
                _ordered_optimizer_exit_candidates(
                    exit_candidates,
                    forecast_by_id=forecast_by_id,
                    state_by_id=state_by_id,
                )[: cfg.max_exits_per_rebalance]
            )
            selected = list(
                dict.fromkeys(
                    [
                        *selected,
                        *[
                            instrument_id
                            for instrument_id in exit_candidates
                            if instrument_id not in allowed_exits
                        ],
                    ]
                )
            )
        aim_weights = _optimizer_aim_weights(
            selected,
            forecast_by_id=forecast_by_id,
            config=cfg,
        )
        output_ids = list(dict.fromkeys([*selected, *held_ids, *candidate_ids]))
        rows: list[dict[str, object]] = []
        for priority, instrument_id in enumerate(output_ids, start=1):
            forecast = forecast_by_id.get(instrument_id, {})
            state_row = state_by_id.get(instrument_id, {})
            current_weight = float(state_row.get("current_weight") or 0.0)
            aim_weight = float(aim_weights.get(instrument_id, 0.0))
            reason = _optimizer_reason(
                instrument_id,
                selected=set(selected),
                forecast=forecast,
                current_weight=current_weight,
                min_net_edge_bps=cfg.min_net_edge_bps,
            )
            target_weight, reason, flags = _target_weight_for_optimizer_row(
                current_weight=current_weight,
                aim_weight=aim_weight,
                reason=reason,
                selected=instrument_id in selected,
                state_row=state_row,
                config=cfg,
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
                    "net_edge_bps": forecast.get("net_edge_bps"),
                    "estimated_cost_bps": cfg.estimated_cost_bps,
                    "priority": priority,
                    "decision_reason": reason,
                    "constraint_flags": ",".join(sorted(flags)),
                }
            )
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
    instrument_values = ranked["instrument_id"].to_numpy(copy=False)
    rank_values = ranked["rank"].to_numpy(copy=False)
    score_values = ranked["score"].to_numpy(copy=False)
    edge_values = _expected_edge_values(ranked)
    for index, instrument_id in enumerate(instrument_values):
        expected_edge = (
            _coerce_optional_float(edge_values[index])
            if edge_values is not None
            else 0.0
        )
        mapping[str(instrument_id)] = {
            "rank": int(rank_values[index]),
            "score": float(score_values[index]),
            "expected_edge_bps": expected_edge,
        }
        if "entry_eligible" in ranked.columns:
            mapping[str(instrument_id)]["entry_eligible"] = _entry_eligible_value(
                ranked["entry_eligible"].iloc[index]
            )
    return mapping


def _optimizer_forecast_mapping(
    ranked: pd.DataFrame,
    config: CostAwareOptimizerConfig,
) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    instrument_values = ranked["instrument_id"].to_numpy(copy=False)
    rank_values = ranked["rank"].to_numpy(copy=False)
    score_values = ranked["score"].to_numpy(copy=False)
    edge_values = _expected_edge_values(ranked)
    risk_penalty_values = _risk_penalty_value_arrays(ranked)
    for index, instrument_id in enumerate(instrument_values):
        score = float(score_values[index])
        expected_edge = (
            _coerce_optional_float(edge_values[index])
            if edge_values is not None
            else 0.0
        )
        if expected_edge == 0.0:
            expected_edge = max(score, 0.0) * config.score_to_edge_bps
        risk_penalty = _risk_penalty_value_at(risk_penalty_values, index)
        risk_penalty = max(risk_penalty, 0.0) * config.risk_penalty_multiplier
        net_edge = expected_edge - config.estimated_cost_bps - risk_penalty
        mapping[str(instrument_id)] = {
            "rank": int(rank_values[index]),
            "score": score,
            "expected_edge_bps": expected_edge,
            "risk_penalty_bps": risk_penalty,
            "net_edge_bps": net_edge,
        }
        if "entry_eligible" in ranked.columns:
            mapping[str(instrument_id)]["entry_eligible"] = _entry_eligible_value(
                ranked["entry_eligible"].iloc[index]
            )
    return mapping


def _optimizer_candidate_ids(ranked: pd.DataFrame, candidate_rank: int) -> list[str]:
    has_entry_eligibility = "entry_eligible" in ranked.columns
    candidates: list[str] = []
    eligible_rank = 0
    for row in ranked.itertuples(index=False):
        if has_entry_eligibility:
            if not _entry_eligible_from_row(row):
                continue
            eligible_rank += 1
            if eligible_rank > candidate_rank:
                break
        elif int(row.rank) > candidate_rank:
            break
        candidates.append(str(row.instrument_id))
    return candidates


def _state_mapping(state: pd.DataFrame) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    if state.empty:
        return mapping
    instrument_values = state["instrument_id"].to_numpy(copy=False)
    current_values = state["current_weight"].to_numpy(copy=False)
    sellable_values = (
        state["sellable_weight"].to_numpy(copy=False)
        if "sellable_weight" in state.columns
        else None
    )
    holding_values = (
        state["holding_bars"].to_numpy(copy=False)
        if "holding_bars" in state.columns
        else None
    )
    for index, instrument_id in enumerate(instrument_values):
        holding_bars = holding_values[index] if holding_values is not None else 0
        mapping[str(instrument_id)] = {
            "current_weight": float(current_values[index]),
            "sellable_weight": (
                sellable_values[index] if sellable_values is not None else pd.NA
            ),
            "holding_bars": int(holding_bars or 0),
        }
    return mapping


def _expected_edge_values(frame: pd.DataFrame) -> object | None:
    if "expected_edge_bps" in frame.columns:
        return frame["expected_edge_bps"].to_numpy(copy=False)
    if "expected_return_bps" in frame.columns:
        return frame["expected_return_bps"].to_numpy(copy=False)
    return None


def _risk_penalty_value_arrays(frame: pd.DataFrame) -> list[object]:
    return [
        frame[name].to_numpy(copy=False)
        for name in ("risk_penalty_bps", "health_risk_bps", "optimizer_risk_penalty_bps")
        if name in frame.columns
    ]


def _risk_penalty_value_at(values: list[object], index: int) -> float:
    for column_values in values:
        value = column_values[index]  # type: ignore[index]
        if value is not None and not pd.isna(value):
            return float(value)
    return 0.0


def _coerce_optional_float(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


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


def _entry_eligible_from_row(row: object) -> bool:
    return _entry_eligible_value(getattr(row, "entry_eligible", True))


def _entry_eligible_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "n", "off"}
    return bool(value)


def _optimizer_selection_score(
    instrument_id: str,
    *,
    forecast: dict[str, object],
    current_weight: float,
    config: CostAwareOptimizerConfig,
) -> float:
    del instrument_id
    score = float(forecast["net_edge_bps"])
    if current_weight > 0:
        score += config.estimated_cost_bps
    return score


def _ordered_optimizer_candidates(
    instrument_ids: list[str],
    *,
    forecast_by_id: dict[str, dict[str, object]],
    state_by_id: dict[str, dict[str, object]],
    config: CostAwareOptimizerConfig,
) -> list[str]:
    return sorted(
        instrument_ids,
        key=lambda instrument_id: (
            -_optimizer_selection_score(
                instrument_id,
                forecast=forecast_by_id[instrument_id],
                current_weight=float(
                    state_by_id.get(instrument_id, {}).get("current_weight") or 0.0
                ),
                config=config,
            ),
            int(forecast_by_id[instrument_id]["rank"]),
            instrument_id,
        ),
    )


def _ordered_optimizer_exit_candidates(
    instrument_ids: list[str],
    *,
    forecast_by_id: dict[str, dict[str, object]],
    state_by_id: dict[str, dict[str, object]],
) -> list[str]:
    return sorted(
        instrument_ids,
        key=lambda instrument_id: (
            instrument_id in forecast_by_id,
            float(forecast_by_id.get(instrument_id, {}).get("net_edge_bps") or -10**9),
            -float(state_by_id.get(instrument_id, {}).get("current_weight") or 0.0),
            instrument_id,
        ),
    )


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


def _scale_aim_weights(
    weights: dict[str, float],
    gross_exposure_scale: float,
) -> dict[str, float]:
    if gross_exposure_scale == 1:
        return weights
    return {
        instrument_id: weight * gross_exposure_scale
        for instrument_id, weight in weights.items()
    }


def _optimizer_aim_weights(
    selected: list[str],
    *,
    forecast_by_id: dict[str, dict[str, object]],
    config: CostAwareOptimizerConfig,
) -> dict[str, float]:
    if not selected or config.gross_exposure_scale <= 0:
        return {}
    if config.weighting == "equal":
        raw = {instrument_id: 1.0 for instrument_id in selected}
    else:
        raw = {
            instrument_id: max(float(forecast_by_id.get(instrument_id, {}).get("net_edge_bps") or 0.0), 0.0)
            for instrument_id in selected
        }
        if sum(raw.values()) <= 0:
            raw = {instrument_id: 1.0 for instrument_id in selected}
    total = sum(raw.values())
    weights = {instrument_id: value / total for instrument_id, value in raw.items()}
    if config.max_name_weight is not None:
        weights = {
            instrument_id: min(weight, config.max_name_weight)
            for instrument_id, weight in weights.items()
        }
    return _scale_aim_weights(weights, config.gross_exposure_scale)


def _optimizer_reason(
    instrument_id: str,
    *,
    selected: set[str],
    forecast: dict[str, object],
    current_weight: float,
    min_net_edge_bps: float,
) -> str:
    if instrument_id in selected:
        return "hold_buffer" if current_weight > 0 else "entry_rank"
    if forecast and float(forecast.get("net_edge_bps") or 0.0) < min_net_edge_bps:
        return "below_edge"
    if current_weight > 0:
        return "exit_rank"
    return "below_edge"


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


def _target_weight_for_optimizer_row(
    *,
    current_weight: float,
    aim_weight: float,
    reason: str,
    selected: bool,
    state_row: dict[str, object],
    config: CostAwareOptimizerConfig,
) -> tuple[float, str, set[str]]:
    flags: set[str] = set()
    if not selected:
        if current_weight > 0:
            target = 0.0
            reason = "exit_rank"
            sellable = state_row.get("sellable_weight")
            if sellable is not None and not pd.isna(sellable):
                sellable_weight = max(float(sellable), 0.0)
                minimum_target = max(0.0, current_weight - sellable_weight)
                if target < minimum_target:
                    target = minimum_target
                    reason = "t1_sell_blocked"
                    flags.add("sellable_weight_limited")
            return target, reason, flags
        return 0.0, reason, flags
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


def _apply_turnover_cap(
    rows: list[dict[str, object]],
    cap: float | None,
    *,
    no_trade_weight_band: float = 0.0,
    gross_exposure_scale: float = 1.0,
) -> list[dict[str, object]]:
    if not rows:
        return rows
    if cap is not None:
        gross_turnover = sum(
            abs(float(row["target_weight"]) - float(row["current_weight"]))
            for row in rows
        )
        if gross_turnover > cap and gross_turnover > 0:
            scale = cap / gross_turnover
            for row in rows:
                current = float(row["current_weight"])
                target = float(row["target_weight"])
                row["target_weight"] = current + (target - current) * scale
                flags = set(str(row.get("constraint_flags") or "").split(","))
                flags.discard("")
                flags.add("turnover_scaled")
                row["constraint_flags"] = ",".join(sorted(flags))
    if no_trade_weight_band > 0:
        _filter_small_scaled_deltas(
            rows,
            no_trade_weight_band=no_trade_weight_band,
            gross_exposure_scale=gross_exposure_scale,
        )
    _enforce_incremental_gross_cap(rows, gross_exposure_scale=gross_exposure_scale)
    return rows


def _apply_optimizer_turnover_budget(
    rows: list[dict[str, object]],
    config: CostAwareOptimizerConfig,
) -> list[dict[str, object]]:
    if not rows:
        return rows
    cap = config.max_gross_turnover_per_rebalance
    if cap is None:
        if config.no_trade_weight_band > 0:
            _filter_small_scaled_deltas(
                rows,
                no_trade_weight_band=config.no_trade_weight_band,
                gross_exposure_scale=config.gross_exposure_scale,
            )
        _enforce_incremental_gross_cap(rows, gross_exposure_scale=config.gross_exposure_scale)
        return rows
    desired_targets = {
        str(row["instrument_id"]): float(row["target_weight"])
        for row in rows
    }
    for row in rows:
        row["target_weight"] = float(row["current_weight"])
    current_gross = sum(float(row["current_weight"]) for row in rows)
    sells = _optimizer_budget_rows(rows, desired_targets, side="sell")
    buys = _optimizer_budget_rows(rows, desired_targets, side="buy")
    exposure_gap = max(config.gross_exposure_scale - current_gross, 0.0)
    if config.max_gross_exposure_increase_per_rebalance is None:
        remaining = cap + exposure_gap
        if current_gross > config.gross_exposure_scale + 1e-12 or not buys:
            sell_budget = remaining
        else:
            sell_budget = remaining / 2
        remaining -= _spend_turnover_budget(
            sells,
            desired_targets,
            budget=sell_budget,
            no_trade_weight_band=config.no_trade_weight_band,
        )
        target_gross = sum(float(row["target_weight"]) for row in rows)
        buy_budget = min(remaining, max(config.gross_exposure_scale - target_gross, 0.0))
        remaining -= _spend_turnover_budget(
            buys,
            desired_targets,
            budget=buy_budget,
            no_trade_weight_band=config.no_trade_weight_band,
        )
        if remaining > 1e-12 and current_gross > config.gross_exposure_scale + 1e-12:
            _spend_turnover_budget(
                sells,
                desired_targets,
                budget=remaining,
                no_trade_weight_band=config.no_trade_weight_band,
            )
    else:
        exposure_increase_budget = min(
            exposure_gap,
            config.max_gross_exposure_increase_per_rebalance,
        )
        _spend_turnover_budget(
            sells,
            desired_targets,
            budget=cap,
            no_trade_weight_band=config.no_trade_weight_band,
        )
        target_gross = sum(float(row["target_weight"]) for row in rows)
        max_target_gross = min(
            config.gross_exposure_scale,
            current_gross + exposure_increase_budget,
        )
        buy_budget = max(max_target_gross - target_gross, 0.0)
        _spend_turnover_budget(
            buys,
            desired_targets,
            budget=buy_budget,
            no_trade_weight_band=config.no_trade_weight_band,
        )
    _mark_turnover_budget_limited(rows, desired_targets)
    _enforce_incremental_gross_cap(rows, gross_exposure_scale=config.gross_exposure_scale)
    return rows


def _optimizer_budget_rows(
    rows: list[dict[str, object]],
    desired_targets: dict[str, float],
    *,
    side: Literal["buy", "sell"],
) -> list[dict[str, object]]:
    candidates = []
    for row in rows:
        current = float(row["current_weight"])
        desired = desired_targets[str(row["instrument_id"])]
        delta = desired - current
        if side == "buy" and delta > 1e-12:
            candidates.append(row)
        elif side == "sell" and delta < -1e-12:
            candidates.append(row)
    if side == "buy":
        return sorted(
            candidates,
            key=lambda row: (
                -float(row.get("net_edge_bps") or 0.0),
                int(row.get("priority") or 10**9),
                str(row["instrument_id"]),
            ),
        )
    return sorted(
        candidates,
        key=lambda row: (
            float(row.get("net_edge_bps") or -10**9),
            int(row.get("priority") or 10**9),
            str(row["instrument_id"]),
        ),
    )


def _spend_turnover_budget(
    rows: list[dict[str, object]],
    desired_targets: dict[str, float],
    *,
    budget: float,
    no_trade_weight_band: float,
) -> float:
    spent = 0.0
    remaining = max(float(budget), 0.0)
    for row in rows:
        if remaining <= 1e-12:
            break
        current_target = float(row["target_weight"])
        desired = desired_targets[str(row["instrument_id"])]
        delta = desired - current_target
        if abs(delta) <= 1e-12:
            continue
        step = min(abs(delta), remaining)
        if step < no_trade_weight_band:
            continue
        row["target_weight"] = current_target + step * (1 if delta > 0 else -1)
        remaining -= step
        spent += step
    return spent


def _mark_turnover_budget_limited(
    rows: list[dict[str, object]],
    desired_targets: dict[str, float],
) -> None:
    for row in rows:
        desired = desired_targets[str(row["instrument_id"])]
        target = float(row["target_weight"])
        current = float(row["current_weight"])
        if abs(desired - target) <= 1e-12:
            continue
        flags = set(str(row.get("constraint_flags") or "").split(","))
        flags.discard("")
        flags.add("turnover_budget_limited")
        row["constraint_flags"] = ",".join(sorted(flags))
        if abs(target - current) <= 1e-12:
            row["decision_reason"] = "turnover_budget_limited"


def _filter_small_scaled_deltas(
    rows: list[dict[str, object]],
    *,
    no_trade_weight_band: float,
    gross_exposure_scale: float,
) -> None:
    target_gross = sum(float(row["target_weight"]) for row in rows)
    for row in rows:
        current = float(row["current_weight"])
        target = float(row["target_weight"])
        delta = target - current
        if abs(delta) >= no_trade_weight_band:
            continue
        proposed_gross = target_gross + current - target
        if delta < 0 and proposed_gross > gross_exposure_scale + 1e-12:
            continue
        row["target_weight"] = current
        row["decision_reason"] = "below_weight_band"
        target_gross = proposed_gross


def _enforce_incremental_gross_cap(
    rows: list[dict[str, object]],
    *,
    gross_exposure_scale: float,
) -> None:
    excess = sum(float(row["target_weight"]) for row in rows) - gross_exposure_scale
    if excess <= 1e-12:
        return
    buy_rows = sorted(
        [
            row
            for row in rows
            if float(row["target_weight"]) > float(row["current_weight"]) + 1e-12
        ],
        key=lambda row: (
            int(row.get("priority") or 10**9),
            -float(row["target_weight"]) + float(row["current_weight"]),
        ),
        reverse=True,
    )
    for row in buy_rows:
        if excess <= 1e-12:
            break
        current = float(row["current_weight"])
        target = float(row["target_weight"])
        reducible = target - current
        reduction = min(reducible, excess)
        row["target_weight"] = target - reduction
        excess -= reduction
        flags = set(str(row.get("constraint_flags") or "").split(","))
        flags.discard("")
        flags.add("gross_exposure_scaled")
        row["constraint_flags"] = ",".join(sorted(flags))
        if float(row["target_weight"]) <= current + 1e-12:
            row["decision_reason"] = "risk_reduction"


def _rows_to_frame(
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
    mapper: object,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    mapped = [mapper(row) for row in rows]  # type: ignore[misc]
    return pd.DataFrame.from_records(mapped, columns=columns)


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
    config: RankBufferDropConfig | CostAwareOptimizerConfig,
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
    return pd.DataFrame.from_records(order_rows, columns=ORDER_INTENT_COLUMNS)


def _client_order_id(
    config: RankBufferDropConfig | CostAwareOptimizerConfig,
    row: dict[str, object],
    side: str,
) -> str:
    timestamp = str(row["timestamp"]).replace(":", "").replace("+", "p")
    return f"{config.policy_id}:{config.policy_version}:{timestamp}:{row['instrument_id']}:{side}"


def _diagnostics_from_rows(
    timestamp: object,
    rows: list[dict[str, object]],
    config: RankBufferDropConfig | CostAwareOptimizerConfig,
) -> pd.DataFrame:
    action_counts = {
        "entry": 0,
        "exit": 0,
        "resize_up": 0,
        "resize_down": 0,
        "hold": 0,
        "no_trade": 0,
    }
    reason_counts = {
        "below_edge": 0,
        "below_weight_band": 0,
        "t1_sell_blocked": 0,
        "risk_reduction": 0,
        "turnover_budget_limited": 0,
    }
    planned_gross_turnover = 0.0
    order_intent_count = 0
    target_gross_exposure = 0.0
    gross_exposure_scaled_count = 0
    turnover_scaled_count = 0
    turnover_budget_limited_flag_count = 0
    for row in rows:
        current = float(row["current_weight"])
        target = float(row["target_weight"])
        delta = target - current
        action = _action(
            current=current,
            target=target,
            reason=str(row["decision_reason"]),
        )
        action_counts[action] = action_counts.get(action, 0) + 1
        reason = str(row["decision_reason"])
        if reason in reason_counts:
            reason_counts[reason] += 1
        planned_gross_turnover += abs(delta)
        if abs(delta) > 1e-12:
            order_intent_count += 1
        target_gross_exposure += target
        constraint_flags = str(row.get("constraint_flags") or "")
        gross_exposure_scaled_count += constraint_flags.count("gross_exposure_scaled")
        turnover_scaled_count += constraint_flags.count("turnover_scaled")
        turnover_budget_limited_flag_count += constraint_flags.count(
            "turnover_budget_limited"
        )
    diagnostic = {
        "timestamp": timestamp,
        "policy_id": config.policy_id,
        "policy_version": config.policy_version,
        "configured_target_count": config.target_count,
        "configured_gross_exposure_scale": config.gross_exposure_scale,
        "target_gross_exposure": target_gross_exposure,
        "active_count": sum(float(row["target_weight"]) > 0 for row in rows),
        "decision_count": len(rows),
        "order_intent_count": order_intent_count,
        "planned_gross_turnover": planned_gross_turnover,
        "entry_count": action_counts["entry"],
        "exit_count": action_counts["exit"],
        "resize_count": action_counts["resize_up"] + action_counts["resize_down"],
        "hold_count": action_counts["hold"],
        "no_trade_count": action_counts["no_trade"],
        "below_edge_count": reason_counts["below_edge"],
        "below_weight_band_count": reason_counts["below_weight_band"],
        "t1_sell_blocked_count": reason_counts["t1_sell_blocked"],
        "risk_reduction_count": reason_counts["risk_reduction"],
        "turnover_budget_limited_count": reason_counts["turnover_budget_limited"],
        "gross_exposure_scaled_count": gross_exposure_scaled_count,
        "turnover_scaled_count": turnover_scaled_count,
        "turnover_budget_limited_flag_count": turnover_budget_limited_flag_count,
    }
    return pd.DataFrame.from_records([diagnostic])


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
    return pd.DataFrame.from_records(state_rows, columns=PORTFOLIO_STATE_COLUMNS)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
