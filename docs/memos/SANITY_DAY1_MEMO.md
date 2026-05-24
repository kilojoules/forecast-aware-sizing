# Day-1 sanity check: in-loop rainflow vs LP-linear-penalty

**Date:** 2026-05-06
**Question:** At hourly arbitrage, does in-loop rainflow open a Pareto gap
vs LP with linear cycling penalty?
**Verdict:** **NO** — at perfect foresight, LP linear-penalty traces out
the rainflow-optimal Pareto frontier. Heuristics fall on or above the
LP curve.

## Setup

- T=168 h, B_E=2 MWh, B_P=2 MW (1-h battery — DoD per cycle reaches 100%).
- Synthetic diurnal AR(1) prices, mean 48.5, std 28.4, range [-18, 99].
- `seed=42`. Single deterministic week.
- Three policies: LP (sweep μ ∈ [0, 150], 31 points), sparse-deep heuristic
  (sweep k ∈ [1, 30]), threshold heuristic (4×4 low/high pct grid).
- All have perfect foresight.

## Result

See `fig_arbitrage_pareto.png`.

| Policy | R range | D range | vs LP at matched R |
|---|---|---|---|
| LP linear-penalty | 0 .. 1585 | 0 .. 0.00183 | (baseline) |
| Sparse-deep | 112 .. 847 | 0 .. 0.00027 | falls on LP curve |
| Threshold | 453 .. 1096 | 0.00033 .. 0.00054 | strictly worse than LP |

Median (heuristic D − LP D) / LP D at matched revenue: **+1.7%**.
LP wins or ties at every R.

## Interpretation

The hypothesis from the lit review (Cao 2020: NN-DDQN beats MILP by 58.5%
with rainflow-aware reward) does not transfer to this setup. Three
candidate reasons:

1. **Perfect foresight removes RL's main lever.** My LP gets the entire
   168-h price trace as known input. Cao 2020's MILP comparison may have
   used noisier forecasts, in which case the 58% comes from RL's
   distributional robustness, not in-loop rainflow.

2. **Single-week horizon doesn't accumulate enough degradation curvature.**
   Total LP D maxes out at 0.0018. The stress function `f(δ) = (k1·δ^k2 + k3)^-1`
   is steeply nonlinear at deep DoD but nearly linear in the operating
   region this trace explores. Multi-year / multi-replacement horizons
   may see different behavior.

3. **The convex relaxation (Shi 2017) is the right comparison, not linear
   |ΔP|.** A convex rainflow surrogate may already match Cao's RL — making
   the gap a gap *between MILP formulations*, not LP-vs-RL.

## Implication for project rework

The proposed pivot — battery_gym → hourly arbitrage RL benchmark beating
LP — is not supported at the simplest level. **Don't refactor.**

Three viable forward paths in order of effort:

**(a) Forecast-uncertainty pilot.** Re-run with rolling-horizon LP that
sees only k-step forecast (with noise). Compare to RL trained on
(forecast, realized) pairs. This would isolate the forecast-handling
gain. Estimated effort: 2-3 days. **Recommended next.**

**(b) Multi-year horizon pilot.** Run LP + heuristics over 25-year
lifetime with replacement iteration. Does the gap open up at long
horizon? Estimated effort: 3-5 days. Needs a more efficient LP solver
than `scipy.linprog`.

**(c) Stop pivoting; keep battery_gym standalone.** The current repo
(Srinivasa repro + heterogeneous-fleet RL on regulation signals) has a
clean honest finding. The lit review motivated a pivot; the data didn't
support it. Park the rework, keep the existing positioning.

## What this rules out

- "RL beats LP at hourly arbitrage with in-loop rainflow" — false at
  perfect foresight, single-week, this battery configuration.
- "Linear cycling penalty fails to capture rainflow at hourly" — false.
  Median 1.7% gap.

## What survives

- Lit review (`litreview.tex`) is unaffected; it correctly cites the
  Cao 2020 number, but the **mechanism** behind that number is now
  uncertain.
- battery_gym's regulation-signal mode + heterogeneous-fleet result is
  unchanged.

## Files

- `sanity_arbitrage.py` — the experiment.
- `fig_arbitrage_pareto.png` — Pareto plot.

## Next decision

Pick one:
- Run pilot (a) — forecast uncertainty — 2-3 days. Highest likelihood
  of finding a real gap given the lit review.
- Run pilot (b) — multi-year — 3-5 days, harder infra.
- Stop the rework, keep current repo as-is.
