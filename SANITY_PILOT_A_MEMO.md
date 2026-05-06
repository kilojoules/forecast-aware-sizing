# Pilot (a) consolidated memo: forecast-uncertainty angle

**Date:** 2026-05-06
**Question:** Does forecast uncertainty open a real gap that RL trained
on (forecast, realized) pairs could close vs LP-with-forecast?
**Verdict:** **NO** at hourly arbitrage with linear cycling cost. Killing
this angle.

## Setup

Same environment as `SANITY_DAY1_MEMO.md`: T=168 h, B_E=2 MWh, B_P=2 MW,
diurnal AR(1) prices, single deterministic week.

## Day 1 result: forecast-vs-oracle gap

| Forecast AR(1) noise std | Realized R / oracle | Gap |
|---|---|---|
| 1 (very good) | 1581 / 1585 | 0.2% |
| 3 (typical good DA forecast) | 1547 / 1585 | 2.4% |
| 5 (typical hourly forecast) | 1466 / 1585 | 7.5% |
| 8 (poor) | 1349 / 1585 | 14.9% |
| 12 (very poor) | 1122 / 1585 | 29.2% |
| 18 (terrible) | 942 / 1585 | 40.5% |
| 25 (catastrophic) | 802 / 1585 | 49.4% |

S-curve. Real-world day-ahead price forecast MAEs are typically 5-10% of
mean, corresponding to noise=2-5 here. **Realistic forecast quality →
2-8% gap from oracle.** This is the upside ceiling for any non-LP
forecast-robust policy at this resolution.

Lit review (Burlando 2025, Hancock 2026, Krishnamurthy 2018) cited 11-16%
gaps at intraday/day-ahead arbitrage; my numbers track those if I assume
their forecasts were closer to noise=5-8 (poor end of realistic).

## Day 2 result: shrinkage policy

Shrinkage LP: solve LP on `w*forecast + (1-w)*mean(forecast)` for
w ∈ {0, 0.25, 0.5, 0.75, 1.0}. Score on realized.

| w | Revenue (averaged over 8 fc-seeds, 6 noise levels) |
|---|---|
| 0.0 | 66 (collapse: no signal) |
| 0.25 | identical to w=1.0 |
| 0.50 | identical to w=1.0 |
| 0.75 | identical to w=1.0 |
| 1.0 | raw LP-with-forecast baseline |

**LP linear-cost rank-invariance.** Shrinkage by any factor preserves
relative rank order of prices. Linear-cost LP only uses rank, so
shrinkage cannot change decisions. The ONLY way shrinkage could matter
is at w=0 where the rank is destroyed, and that's a collapse. RL given
the same forecast input has the same information budget; it cannot
recover information from preserved rank that LP already extracts.

Realized-rolling-threshold (forecast-free, uses past 48 h realized
percentiles): R=694 across all noise levels. Beats LP only at
noise≥25 (LP=738), and by 6%. Not a viable RL angle — going
forecast-free barely matches catastrophic-forecast LP.

## Why pilot (a) dies

The Hancock 2026 finding (Kendall tau is the binding metric, not MAE)
is the explanation. As long as forecast preserves rank order well enough,
deterministic LP captures most of the perfect-foresight value. Forecast
noise only hurts when it scrambles rank — which my AR(1) noise does
gradually but my linear LP is robust to within rank-correlation
thresholds.

For RL given the same forecast to *exceed* LP, RL would need access to
information LP doesn't have:
- A learned conditional distribution of (realized | forecast) — but that
  only helps if the objective is nonlinear in price, which a linear-cost
  LP isn't.
- Online state RL can use during execution — but rolling-horizon LP
  already has this and gives identical results to full-horizon at 24h
  lookahead.

Information-theoretically, RL has no edge over LP at this problem.

## What survives

The argument fails for **linear-cost** LP at hourly arbitrage. It does
not formally fail for:

1. **Nonlinear cost LP** (rainflow-as-objective via convex relaxation,
   or QP with quadratic cycling). Jensen's inequality breaks the
   deterministic-substitution argument. Stochastic-LP / RL would beat
   deterministic LP. **Untested.**
2. **Multi-stage stochastic problems** with state evolution beyond SoC
   (e.g., capacity reservation decisions interacting with realized
   outcomes). **Not in this setup.**
3. **Long-horizon settings where LP is computationally intractable.** RL
   amortizes inference cost. Not a quality argument, an efficiency one.

(1) is the one path where the forecast-uncertainty angle could survive.
Cost: ~3-5 days to build a stochastic LP baseline. Probability of
finding a meaningful gap: moderate (Jensen gap depends on cycling cost
nonlinearity, which is small at hourly per Day 1).

## Implication for project rework

Combined with `SANITY_DAY1_MEMO.md`:

- Day 1 falsified: in-loop rainflow doesn't open a gap at perfect foresight.
- Pilot (a) day 1+2 falsified: forecast uncertainty doesn't open a gap
  at linear cost.

The **rework as currently framed is dead.** Two residual paths:

**(i)** Combined nonlinear-cost + stochastic-forecast experiment. Tests
whether Jensen's gap on degradation cost matters. ~5 days. Moderate
probability of meaningful result. Defensible if it works; another null
result if not.

**(ii)** Stop pivoting. Keep `battery_gym` standalone with its current
honest framing (Srinivasa repro + heterogeneous-fleet RL). Use lit review
as the conceptual contribution. Use it directly in the BMS application as
"thoughtful engineering project + literature awareness."

## Recommendation

(ii). The repo is already credible. Two pilots have killed the proposed
extension. A third pilot might survive but the prior is now low. Time
better spent on the BMS application or elsewhere.

If the user wants the third pilot, scope is clear and bounded: 5 days,
go/no-go on Jensen's-gap-from-degradation.

## Files

- `sanity_forecast.py` — day 1 runner.
- `fig_forecast_ceiling.png` — value-of-perfect-foresight S-curve.
- `sanity_robust.py` — day 2 runner.
- `fig_forecast_robust.png` — shrinkage demonstration.
