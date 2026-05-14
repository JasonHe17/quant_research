# Production Strategy Framework and Roadmap

## Status

Active design document for the strategy layer.

The current every-bar top-N baseline is retained only as a regression benchmark
and a negative control. It must not be used as proof that a factor is tradable.
The next development line is to replace it with a stateful, cost-aware strategy
decision layer that can later connect to paper trading and live trading adapters.

## Scope

This document owns the path from model scores to auditable trade intent:

```text
alpha forecasts
  -> strategy policy
  -> portfolio intent
  -> trade decision
  -> order intent
  -> execution simulator or live adapter
  -> fills, ledger, diagnostics, reconciliation
```

The strategy layer must be usable in both historical replay and real-time
automatic trading. Backtests are not a separate research shortcut; they are
offline replays of the same decision contracts used by live trading.

## External References

The design is based on these research and engineering references:

- Garleanu and Pedersen, "Dynamic Trading with Predictable Returns and
  Transaction Costs": optimal trading with costs should trade partially toward
  an aim portfolio, and slower-decaying predictors deserve more trading weight.
  https://www.nber.org/papers/w15205
- Boyd et al., "Multi-Period Trading via Convex Optimization": trades should be
  chosen by balancing expected return, risk, transaction costs, and holding
  costs; the multi-period version plans a sequence of trades and executes the
  first one. https://web.stanford.edu/~boyd/papers/cvx_portfolio.html
- Cvxportfolio: transaction costs, holding costs, constraints, and soft
  constraints are first-class policy objects, not post-processing patches.
  https://www.cvxportfolio.com/en/1.3.1/costs.html
  https://www.cvxportfolio.com/en/stable/constraints.html
- Microsoft Qlib: `BaseStrategy` generates trade decisions per trading bar;
  `WeightStrategyBase` separates target-position generation from order-list
  generation; `TopkDropoutStrategy` limits daily replacements instead of fully
  replacing a top-K book. https://github.com/microsoft/qlib/blob/main/docs/component/strategy.rst
- QuantConnect LEAN Algorithm Framework: Universe, Alpha, Portfolio
  Construction, Execution, and Risk are separate modules; alpha insights include
  direction, magnitude, confidence, and period; portfolio rebalance triggers are
  explicitly configurable. https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/overview
  https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/portfolio-construction/key-concepts
- Shanghai Stock Exchange trading mechanism: A-share order size, tick size, and
  price-limit assumptions must be explicit market rules. The exchange states
  that stock buy orders through auction trading are in multiples of 100 shares,
  residual odd-lot sells are handled as one order, A-share tick size is RMB
  0.01, and Main Board A/B shares have a 10% daily price limit unless otherwise
  specified. https://english.sse.com.cn/start/trading/mechanism/
- Mainland A-share day trading constraints: A-shares purchased through Stock
  Connect can only be sold on T+1 or later. This is a useful public reference
  for the same practical no-day-trading constraint that the simulator already
  models. https://www.ifec.org.hk/web/en/investment/investment-products/stock/stock-trading/stock-settlement.page

## Why The Baseline Fails

The current baseline has a simple data path:

```text
score at timestamp
  -> rank all names
  -> select top-N
  -> assign equal target weights
  -> missing names get target_weight = 0
  -> execution simulator trades the full target delta
```

This is useful for smoke testing, but it is not a valid production strategy
logic.

Problems:

- Boundary churn: rank 50 versus rank 51 can trigger a full exit and full new
  entry even if the scores are nearly identical.
- No cost threshold: a trade can be triggered without expected edge exceeding
  commission, stamp tax, slippage, spread, market impact, or opportunity cost.
- No alpha horizon: a 5-minute score has no explicit forecast period, decay, or
  confidence, so every new bar is treated as a complete replacement signal.
- No no-trade region: current holdings have no inertia. Small target changes
  are traded unless blocked by execution constraints.
- No order planning: the strategy creates target weights, not order intents with
  priority, participation budget, expiry, price policy, or reason codes.
- No live-state input: pending orders, partial fills, sellable quantity, cash
  reservation, and reconciliation state are not part of the policy contract.

Therefore the baseline cannot answer whether a factor has tradable net alpha.
It can only answer whether a naive every-bar top-N portfolio survives a
particular simulator configuration.

## Target Architecture

The production path must keep research, policy, risk, execution, and accounting
separate while using shared contracts.

```text
DataSnapshot / RealtimeMarketData
  -> UniverseService
  -> FeaturePipeline
  -> AlphaForecastService
  -> PortfolioStateService
  -> StrategyPolicy
  -> RiskGate
  -> OrderScheduler
  -> BrokerAdapter or BacktestExecutionAdapter
  -> FillLedger
  -> Metrics, Diagnostics, Reconciliation
```

Module responsibilities:

- `UniverseService`: defines tradable instruments as of decision time. It must
  exclude suspended, non-tradable, ST-filtered, and unsupported-board names
  according to policy configuration.
- `AlphaForecastService`: emits forecasts with horizon, confidence, and version
  metadata. It does not create trades.
- `PortfolioStateService`: provides current positions, cash, sellable shares,
  pending orders, last known prices, and previous policy state.
- `StrategyPolicy`: converts forecasts and state into portfolio intent and trade
  decisions. This is where entry, exit, hold, resize, and no-trade rules live.
- `RiskGate`: applies hard constraints before orders reach execution.
- `OrderScheduler`: converts approved trade decisions into order intents with
  priority, quantity, price policy, participation limits, and expiry.
- `BrokerAdapter`: submits, amends, cancels, and reconciles real orders. It must
  be replaceable by the backtest execution adapter.
- `FillLedger`: is the source of truth for positions, cash, realized costs, and
  replayable fills.

## Standard Contracts

All timestamps must be timezone-aware. All emitted artifacts must include
`policy_id`, `policy_version`, `model_version`, and `data_snapshot` or live
market-data sequence identifiers.

Alpha forecasts:

```text
timestamp, instrument_id, score, expected_return_bps, horizon_bars,
confidence, decay_half_life_bars, model_version, data_snapshot
```

Portfolio state:

```text
timestamp, instrument_id, shares, sellable_shares, market_value,
current_weight, last_price, average_cost, pending_buy_shares,
pending_sell_shares
```

Portfolio intent:

```text
timestamp, instrument_id, current_weight, aim_weight, policy_target_weight,
rank, score, expected_edge_bps, estimated_cost_bps, reason
```

Trade decisions:

```text
timestamp, instrument_id, action, current_weight, aim_weight, target_weight,
delta_weight, expected_edge_bps, estimated_cost_bps, priority,
decision_reason, constraint_flags
```

Order intents:

```text
timestamp, instrument_id, side, quantity, price_style, limit_price,
time_in_force, participation_limit, expire_time, priority, client_order_id,
decision_reason
```

Execution events:

```text
timestamp, client_order_id, broker_order_id, instrument_id, event_type,
filled_quantity, average_price, commission, taxes, slippage_estimate,
remaining_quantity, raw_status
```

## Policy Design

### MVP Policy: RankBufferDropPolicy

The first production-oriented policy should be simple, explainable, and
stateful. It should combine Qlib-style top-k-drop behavior with A-share
constraints and no-trade bands.

Required parameters:

```text
entry_rank
exit_rank
target_count
max_entries_per_rebalance
max_exits_per_rebalance
rebalance_every_n_bars
min_hold_bars
weighting
max_name_weight
min_expected_edge_bps
no_trade_weight_band
partial_rebalance_rate
max_gross_turnover_per_rebalance
max_daily_gross_turnover
```

Decision rules:

1. Rank valid forecasts at the decision timestamp.
2. Existing holdings remain candidates while `rank <= exit_rank`, the name is
   tradable, and no hard risk rule is breached.
3. New entries are allowed only when `rank <= entry_rank` and expected edge
   exceeds estimated cost plus a policy margin.
4. Exits are prioritized by risk breach, delisting or non-tradable state,
   invalid forecast, rank beyond exit threshold, and then weakest expected edge.
5. Replacement count is capped by `max_exits_per_rebalance` and
   `max_entries_per_rebalance`.
6. Target weights are computed for the retained plus new book, then clipped by
   name, cash, and gross exposure limits.
7. If `abs(target_weight - current_weight) < no_trade_weight_band`, do not
   resize.
8. If resizing is required, move only partially toward the aim weight:
   `target = current + partial_rebalance_rate * (aim - current)`.
9. T+1 sellability, lot size, price limits, and liquidity capacity are checked
   before order intents are created.
10. Every decision must carry a stable reason code.

Required reason codes:

```text
entry_rank
hold_buffer
exit_rank
resize_up
resize_down
below_edge
below_weight_band
min_hold_blocked
t1_sell_blocked
limit_up_buy_blocked
limit_down_sell_blocked
capacity_capped
risk_reduction
cash_limited
universe_removed
```

### Optimizer Policy: CostAwareMpcPolicy

After the MVP policy is stable, add an optimizer-backed policy. It should use
the same contracts and reason-code diagnostics.

Single-period objective shape:

```text
maximize
    expected_alpha(w)
  - risk_penalty(w)
  - transaction_cost(w - current_w)
  - holding_cost(w)
  - soft_constraint_penalties(w)
```

Core constraints:

```text
long_only
cash_budget
max_name_weight
max_sector_weight
max_gross_exposure
max_turnover
max_participation
sellable_quantity
board_lot_quantity
price_limit_buy_sell_blocks
```

The multi-period version should run in model-predictive-control style: plan a
sequence of trades using forecast decay and liquidity estimates, execute the
first slice, then re-plan on the next event.

## A-Share Requirements

The default market profile is China A-share cash equity. These rules must be
configuration-driven because board rules and broker support differ.

Required market constraints:

- Long-only by default. Short selling and margin financing require explicit
  adapter support and separate risk controls.
- No day trading for cash A-share holdings: shares bought today are not sellable
  until the next trading day.
- Buy quantities use 100-share board lots by default. Odd-lot residual sells
  must be handled explicitly.
- A-share tick size is RMB 0.01 by default.
- Price limits are board-specific and state-specific: main-board, ST, STAR,
  ChiNext, IPO early days, relisting, and delisting transition must not share one
  hard-coded threshold.
- Suspensions, zero-volume bars, zero-turnover bars, and missing prices block
  trading.
- Capacity checks must use ex-ante liquidity estimates by default. Same-bar
  turnover can be used only when the experiment explicitly declares it as a
  bar-volume execution assumption.
- The order scheduler must understand call auction, continuous auction, closing
  auction, and broker-specific order types before live trading is enabled.

## Live Trading Readiness

The strategy layer must be deterministic and replayable before connecting to a
real broker.

Live requirements:

- Event-driven loop with explicit input sequence numbers.
- Idempotent decision and order IDs.
- Persistent policy state and pending-order state.
- Broker reconciliation before every trading session and after every reconnect.
- Kill switch for market-wide, strategy-level, and symbol-level blocks.
- Dry-run mode that emits order intents without submission.
- Paper-trading adapter using the same order and fill contracts as live trading.
- Strict separation between desired portfolio intent and submitted order state.
- Complete audit trail from forecast row to order, fill, position, and PnL.

## Acceptance Requirements

Behavioral tests:

- A name just outside `entry_rank` must not be bought.
- A held name inside `exit_rank` must not be sold only because it left top-N.
- A held name beyond `exit_rank` is eligible for exit, but exits are capped by
  replacement and turnover budgets.
- A same-day bought A-share cannot be sold before T+1.
- A target change within the no-trade band must not create an order.
- Partial rebalance must move toward the aim portfolio without overshooting.
- Capacity caps must reduce order size and record `capacity_capped`.
- Every emitted trade decision must have exactly one primary reason code.

Backtest acceptance:

- Compare naive top-N, RankBufferDropPolicy, and at least one lower-frequency
  rebalance policy on the same score files.
- Report gross turnover, one-way turnover, trade count, transaction costs,
  cost-adjusted return, max drawdown, hit rate, capacity caps, blocked sells,
  and reason-code counts.
- Run zero-cost, base-cost, and stressed-cost variants to separate alpha decay
  from execution drag.
- Do not promote a factor or model unless its conclusions are stable under the
  improved policy and at least one cost stress.

Live-readiness acceptance:

- Historical replay and streaming replay must produce identical decisions for
  the same event sequence.
- Restarting from persisted state must not duplicate orders.
- Broker reconciliation mismatches must block new orders until resolved.
- All order intents must be traceable back to forecast, state, policy version,
  and risk-gate decision.

## Technical Roadmap

- [x] Diagnose the every-bar top-N baseline as research-only.
- [x] Add this production strategy framework design document.
- [x] Add strategy domain models for forecasts, portfolio state, portfolio
  intent, trade decisions, order intents, and reason codes.
- [x] Add a `StrategyPolicy` interface with deterministic `decide()` semantics.
- [x] Implement `RankBufferDropPolicy`.
- [x] Integrate the policy into `run_tree_score_backtest.py` without removing
  the old top-N baseline path.
- [x] Add reason-code and planned-turnover diagnostics to backtest outputs.
- [x] Add unit tests for rank buffer, replacement caps, no-trade bands, partial
  rebalance, and T+1 sellability interaction.
- [x] Use sparse streaming execution frames for score backtests so larger
  policy grids do not materialize inactive full-market rows.
- [x] Add candidate-factor policy comparison runs: naive top-N, top-k-drop,
  entry/exit buffer, daily rebalance, and partial rebalance.
- [x] Add controlled concurrent candidate policy sweeps with resumable
  backtests and flat comparison summaries.
- [ ] Define promotion gates for policy-level acceptance before any new factor
  research resumes.
- [ ] Add a cost-aware optimizer policy behind the same strategy contracts.
- [ ] Add paper-trading contracts for broker adapters, order IDs, fills, and
  reconciliation.

## Development Rule

Do not expand candidate-factor research as the main line until the strategy
decision layer can explain:

```text
why trade,
why not trade,
how much to trade,
why this order priority,
what cost and constraint assumptions were used.
```
