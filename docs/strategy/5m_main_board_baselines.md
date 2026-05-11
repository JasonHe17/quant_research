# 5-Minute Main-Board Baseline Notes

## Scope

- Universe: CN main-board A shares only.
- Frequency: 5-minute bars.
- Trading rule: A-share T+1.
- Goal: cross-sectional long-only research baseline.

## Why this direction

The intraday literature points more strongly to short-term reversal / liquidity
provision than to naive intraday momentum in stocks:

- Heston, Korajczyk, and Sadka, *Intraday Patterns in the Cross-Section of Stock Returns*:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1107590
- Dai, Medhat, Novy-Marx, and Rizova, *Reversals and the Returns to Liquidity Provision*:
  https://www.nber.org/papers/w30917
- Qiu, Deschamps, Huang, and Jiang, *Intensity of Intraday Reversals and Future Stock Returns: The Role of Retail Investors*:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4710419
- Baltussen, Da, and Soebhag, *End-of-Day Reversal*:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5039009

Qlib is a useful reference for how the community structures baselines and
high-frequency workflows:

- Alpha158 / Alpha360 documentation:
  https://qlib.readthedocs.io/en/latest/component/data.html
- High-frequency trading support:
  https://github.com/microsoft/qlib/blob/main/docs/component/highfreq.rst

## Candidate Baselines

### Baseline A: 5-minute short-term reversal

Idea:

- rank stocks by recent 5-minute return
- buy the weakest names in the cross-section
- optionally require volume/liquidity confirmation

Implemented factor:

```text
lookback_return = close / close.shift(lookback_bars) - 1
factor_value = -lookback_return
```

Signal ranking note:

- `SignalGenerator(method="rank")` should use `ascending=True` for this factor.
- The strategy shell selects larger `signal` values first.
- Therefore recent losers receive larger `factor_value`, larger rank signal, and
  higher selection priority.

Pros:

- aligns with short-run reversal literature
- simple to implement and test
- naturally fits T+1 if only target weights are generated

Cons:

- may be sensitive to transaction costs
- may perform poorly in strong trend regimes

### Baseline B: 5-minute momentum with liquidity filter

Idea:

- rank stocks by recent 5-minute return
- buy the strongest names only when liquidity and spread proxies are acceptable

Pros:

- intuitive and easy to interpret
- useful as a control baseline

Cons:

- weaker fit to intraday reversal literature
- can be more fragile under T+1 if used for fast turnover

### Baseline C: end-of-session reversal bias

Idea:

- emphasize the last 30 minutes of the session
- score intraday losers more favorably near the close

Pros:

- matches end-of-day reversal evidence
- easier to align with daily overnight holding

Cons:

- narrower than a full-session strategy
- may underuse earlier intraday information

## Practical Research Preference

For the first iteration, the least assumption-heavy path is:

1. main-board universe only
2. 5-minute bars
3. long-only
4. T+1 constrained execution
5. explicit turnover and sellability limits

The exact alpha choice should be selected before implementation.
