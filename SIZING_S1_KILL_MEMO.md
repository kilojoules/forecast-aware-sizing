# Pilot S1 result: sizing-shift hypothesis falsified

**Date:** 2026-05-08
**Verdict:** KILL the sizing-shift research direction. The optimal
battery capacity is **invariant** to dispatch fidelity in the regime
tested. NPV at the optimum changes meaningfully (+1.6% → +55% as noise
grows), but `b_E*` does not.

## Setup

`pilot_s1.py`. T=168 h synthetic diurnal week, projected to 15-yr lifetime
NPV with replacement scheduling. CAPEX = 100 kEUR/MWh + 75 kEUR/MW (PCS).
Discount rate 7%. LoH budget 20%. Fixed `b_P` = 1 MW, sweep
`b_E ∈ {0.25, 0.5, 1, 2, 4, 8, 16}` MWh.

Two dispatch regimes:
1. **Deterministic**: single-forecast QP-quadratic.
2. **Stochastic**: K=4 ensemble QP-quadratic.

Both scored on realized prices. 8 outer forecast seeds for variance reduction.

## Result table — single noise level (8.0)

| `b_E` | Reps det | Reps sto | NPV det | NPV sto | NPV diff |
|---|---|---|---|---|---|
| 0.25 | 7 | 6 | -123k | -97k | +26k |
| 1.0 | 7 | 6 | -269k | -167k | +102k |
| 2.0 | 3 | 3 | +14k | +72k | +58k |
| 4.0 | 2 | 2 | +276k | +340k | +64k |
| **8.0** | **1** | **1** | **+503k** | **+549k** | **+46k** |
| 16.0 | 0 | 0 | +440k | +489k | +49k |

`b_E*` = 8 MWh in **both** regimes. 0% shift.

## Result — noise sweep

| Noise std | `b_E*_det` | `b_E*_sto` | Shift | NPV uplift at optimum |
|---|---|---|---|---|
| 3 | 8 | 8 | 0% | **+1.6%** |
| 8 | 8 | 8 | 0% | **+9.3%** |
| 18 | 8 | 8 | 0% | **+55.2%** |

Sizing decision is robust across the entire noise spectrum tested.

## Interpretation

**The Pareto curve is shifted up by stochastic dispatch but its shape is preserved.** Both regimes pick the same optimum. The mechanism:

- Stochastic dispatch reduces phantom-cycling at small `b_E` (saves 1
  replacement when `b_E ≤ 1` MWh in this run).
- For `b_E ≥ 2` MWh, replacement counts coincide.
- Revenue advantage of stochastic over deterministic is roughly
  proportional, not super-proportional, in `b_E` — so the relative NPV
  curve doesn't tilt enough to move the argmax.

What this means for the SIZING_PILOTS.md thesis:

> "stochastic-aware-inner-dispatch-changes-sizing-decisions" — **falsified**

What this means for the operational claim:

> "stochastic dispatch raises lifetime NPV at the chosen capacity" —
> **confirmed**, +1.6 to +55% depending on forecast quality

The operational claim is the Cao 2020 / Jensen-gap result restated in a
sizing wrapper. Not novel.

## What would need to be true to revive the sizing thesis

1. Stochastic-vs-deterministic dispatch advantage must be
   **super-proportional in `b_E`** at some range, so the NPV curve tilts
   under stochastic dispatch.
2. Replacement-count savings must concentrate at `b_E` regions where the
   capacity decision is otherwise marginal — not just at small `b_E`
   where capacity is structurally too small.
3. Multi-day price patterns (weekend dips, weather-driven volatility)
   that bigger batteries can exploit but smaller can't, where forecast
   errors specifically hurt the inter-day arbitrage strategy.

None of (1)-(3) are guaranteed in real data either. Worth one more
quick test on multi-week traces with weekend pattern, but the prior
should be low.

## Per the pilot tree

`SIZING_PILOTS.md` said:

> Go: shift ≥ 10%. Null: shift < 5% → **kill, sizing robust**.

We got 0%. Per the pre-committed criterion, **kill**. Skip S2-S4.

## Forward path

The dispatch + degradation + cost infrastructure built for S1 is reusable
for any future operational study, but the sizing-shift thesis is dead.

Three options for the project as a whole:

1. **Stop the rework completely.** Keep the current repo (Srinivasa
   reproduction + heterogeneous-fleet RL + lit review + clean negative-
   result memos for 4 pilots). Polish for BMS application portfolio
   piece. Defensible and clean.

2. **Polish the operational Jensen-gap result for a workshop paper.**
   Frame: "Forecast-uncertainty matters for operational NPV but not
   sizing — comment on academic-vs-production gap." Workshop venue:
   RLEM @ BuildSys, Climate Change AI @ NeurIPS. ~1 week of writeup.

3. **Drop battery_gym entirely** and refocus on FAR.AI / MATS / Anthropic
   Fellows track per memory's strategic priorities. Time-budget honest.

(1) is the safe play; (2) is the maximum extraction; (3) is the
opportunity-cost honest play.

## Files

- `pilot_s1.py` — runner.
- `fig_pilot_s1.png` — NPV vs `b_E`, replacement count, weekly revenue
  for both regimes.
- `SIZING_PILOTS.md` — original scope (S2-S4 cancelled).
