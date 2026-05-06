# Pilot (a) consolidated memo: forecast-uncertainty angle (UPDATED)

**Date:** 2026-05-06
**Status:** **REVISED** — combined nonlinear-cost + forecast-uncertainty
gap is alive. Linear-cost-only or perfect-foresight-only versions are dead.

## TL;DR

| Test | Cost | Foresight | Gap exists? |
|---|---|---|---|
| Day 1 (in-loop rainflow) | nonlinear (rainflow) | perfect | NO |
| Day 2 (shrinkage) | linear | noisy | NO |
| Day 3-5 (Jensen) | quadratic | noisy | **YES, +16-66%** |

The two prior nulls hid the result that requires **both** ingredients:
nonlinear cycling cost AND forecast uncertainty. Either alone collapses
the gap. Together they open a 16-66% lift over single-forecast QP at
realistic-to-high forecast noise.

## Day 1 — value-of-perfect-foresight ceiling

S-curve from 0% (noise=1) to 49% (noise=25). Realistic forecasts → 2-8%
gap. Sets the upside ceiling for any forecast-robust policy.

## Day 2 — shrinkage with linear cost

Linear-cost LP is rank-invariant in price. Affine shrinkage preserves
rank. w ∈ {0.25, 0.5, 0.75, 1.0} all give identical revenue. RL given
the same forecast extracts no info LP doesn't already use. Decisive null
under linear cost.

## Day 3-5 — Jensen-gap test with quadratic cycling cost

Replaced linear cycling cost with `α · Σ (P_chg + P_dis)^2`. Convex QP
solved via `scipy.optimize.minimize(SLSQP)`. Compared:

- **Single-forecast QP** — baseline, what hydesign-style would do.
- **Ensemble (K=16) forecast QP** — average K independent forecast
  realizations into a single mean, solve QP once on that mean. Cheap
  proxy for forecast-robust policy.
- **Oracle QP** — perfect foresight upper bound.

T=72 h, B_E=2 MWh, B_P=2 MW, α=0.005.

| Noise std | Single R | Ensemble R | Single D | Ensemble D | Lift R |
|---|---|---|---|---|---|
| 3 | 640 | 651 | 0.00070 | 0.00074 | +1.8% |
| 8 | 550 | 638 | 0.00084 | 0.00071 | **+16.1%** |
| 18 | 368 | 609 | 0.00104 | 0.00077 | **+65.6%** |

Ensemble both *raises* revenue and *lowers* degradation simultaneously
at noise≥8. Strict Pareto improvement.

## Mechanism

Quadratic cycling cost depends on price *magnitudes*, not just rank
order. A noisy forecast with large positive deviation in a single hour
makes single-forecast QP plan deep cycling that's not warranted under
realized prices, paying excess cycling cost. Ensemble forecast averages
the magnitude noise (via central limit), giving a smoother price
estimate that drives more conservative (and on-average-correct) cycling
depth.

This is the empirical signature of Jensen's inequality:
`E[QP(forecast)] ≠ QP(E[forecast])` for nonlinear cost. The deterministic
substitution that linear-cost LP enjoys breaks. Stochastic / ensemble
methods are strictly better in expectation.

## Implication for the rework

**Reverse the prior kill verdict.** The combined angle is a real
research result.

- The lit-review claim that RL can beat LP at hourly arbitrage is
  *conditionally* true: only when cycling cost is nonlinear AND forecasts
  are noisy.
- Ensemble-mean QP is a strong simple baseline that captures most of the
  Jensen lift. RL trained on the joint (forecast, realized) distribution
  should match or beat ensemble-mean QP.
- Cao 2020's 58% MILP-vs-RL number is consistent: their MILP linearized
  degradation, their RL had nonlinear degradation in reward, their
  prices had forecast noise.

## What the rework should now look like

Clean reframe: `battery_gym` becomes a benchmark for **hourly arbitrage
with nonlinear cycling cost under forecast uncertainty.** Three
policies to ship:

1. `LPDeterministic(forecast)` — linear cycling cost, baseline. Captures
   one published convention (hydesign).
2. `QPDeterministic(forecast)` — quadratic cycling cost, single-forecast.
   Captures a stronger published convention.
3. `QPEnsemble(forecasts)` — quadratic cycling cost on mean of K
   forecasts. Captures the Jensen lift cheaply.
4. `RLPolicy(forecast)` — learned on (forecast, realized) pairs. Goal:
   match or beat (3).

Headline figure: lift of (3) and (4) over (2) as a function of forecast
noise. Reproduce this paper's Figure 1.

## Refactor scope

The existing `battery_gym` codebase already has the rainflow tracker,
which is the central asset. Extending to PriceEnv + arbitrage-mode
agents is real work but bounded:

| Phase | Effort | Deliverable |
|---|---|---|
| 1. PriceEnv + QP solver wrappers | 2-3 days | Reproduce Jensen lift at scale |
| 2. RL policy on joint distribution | 4-5 days | Match ensemble lift |
| 3. Real DK1 day-ahead prices, headline figure | 2-3 days | Production figure |
| 4. README rewrite, demote RegEnv to legacy | 1 day | Clean repo |

~10-12 days total. Bounded.

## Caveats

1. **Quadratic ≠ true rainflow.** The Jensen-gap test used `(P_chg+P_dis)^2`,
   a smooth surrogate. True rainflow is non-Markovian. Need to confirm
   the lift survives when cost is the actual rainflow proxy used in
   `degradation.py`. ~1 day.
2. **K=16 is a lot of forecasts.** Real operators don't have 16
   independent day-ahead forecasts. Need to test K=2-4 (realistic) and
   show meaningful lift. If K=2 lift collapses, the angle is weaker than
   it looks.
3. **The ensemble baseline beats single-forecast QP, but does it beat
   the LINEAR-COST LP?** That's the actual hydesign comparison. Need
   to verify the QP-with-quadratic-cost itself meaningfully outperforms
   LP-with-linear-cost (Pareto-frontier-wise). If linear LP can match
   QP by re-tuning μ, the nonlinear-cost premise fails.

## Next steps

Three optional gates before committing to refactor:

**(a)** Test caveat 1 — replace QP quadratic surrogate with rainflow-in-loop
proxy (sum of dev-from-switching-points^2 from `env.RainflowStack`). 1 day.
Go/no-go: Jensen lift survives at ≥10% with realistic K=2-4.

**(b)** Test caveat 3 — Pareto-compare LP-linear vs QP-quadratic on
realized data. Sweep both penalty terms. 1 day. Go/no-go: QP-quadratic
strictly dominates LP-linear at a meaningful (R, D) point.

**(c)** Both. 2 days total.

After (a) + (b) clear, commit to the 10-12 day refactor.

## Files

- `sanity_forecast.py`, `fig_forecast_ceiling.png` — Day 1.
- `sanity_robust.py`, `fig_forecast_robust.png` — Day 2.
- `sanity_jensen.py`, `fig_jensen_gap.png` — Day 3-5.
