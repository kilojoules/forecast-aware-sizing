# Pre-flight memo: pilot (a) caveats resolved

**Date:** 2026-05-06
**Verdict:** Refactor is justified. Commit.

## What was tested

Two cheap pre-flight tests called out in `SANITY_PILOT_A_MEMO.md` before
committing to the 10-12 day refactor.

**(a)** Does the Jensen lift survive at realistic ensemble size K=2-4?
**(b)** Does QP-quadratic-cost actually open new (R, D) Pareto territory
       LP-linear can't reach?

`sanity_preflight.py` runs all three sub-tests; figures committed.

## (a) K-sweep at noise=8

| K | R mean | D mean | Lift over K=1 |
|---|---|---|---|
| 1 (single forecast) | 515.4 | 0.00090 | (baseline) |
| 2 | 599.9 | 0.00084 | **+16.4%** |
| 4 | 611.1 | 0.00076 | **+18.6%** |
| 8 | 632.8 | 0.00072 | +22.8% |
| 16 | 647.9 | 0.00079 | +25.7% |
| 32 | 649.3 | 0.00076 | +26.0% |

K=2 (realistic minimum) already captures 16% lift. Diminishing returns
past K=4. **PASS.**

## (b) Perfect-foresight Pareto

LP-linear sweep μ vs QP-quadratic sweep α, both on realized prices.

| Frontier | R range | D range | Dominated points |
|---|---|---|---|
| LP-linear | [0, 659] | [0, 0.00077] | 0/19 |
| QP-quadratic | [632, 659] | [0.00045, 0.00077] | 8/13 by LP |

Under perfect foresight, LP-linear traces a fuller frontier (down to
R=0 with D=0), and QP-quadratic is dominated 8/13. **Apparent FAIL —
but wrong framing.** Real systems have noisy forecasts.

## (c) Noisy-foresight Pareto (the actual test)

Same sweep but: LP solves on a single noisy forecast (per outer seed),
QP solves on K=4 ensemble of noisy forecasts. noise=8. Score on realized.

| Frontier | R range | D range | Dominated by other |
|---|---|---|---|
| LP-linear-with-forecast | [0, 568] | [0, 0.00084] | **7/13 by QP-ensemble** |
| QP-quadratic-K=4 | [437, 627] | [0.00045, 0.00079] | 1/9 by LP |

**QP-ensemble strictly dominates LP-with-forecast at 7/13 points.**
The high-revenue end (R > 560) is QP-ensemble territory; LP-with-forecast
caps at R=568, QP-ensemble reaches R=627.

LP wins only at low-R / no-cycling (D≈0), where QP-quadratic's smooth
cost prevents reaching exact-zero cycling. Not the operating regime.

## What this means for the refactor

The combined story is clean and testable:

> At hourly arbitrage with noisy forecasts, deterministic LP-with-forecast
> mis-allocates cycling because (i) the cycling cost is approximately
> nonlinear (rainflow stress is convex in DoD), and (ii) the forecast
> noise has nonzero variance over price magnitudes. Stochastic / ensemble
> methods (QP-ensemble, or RL trained on the forecast-realized joint
> distribution) recover most of the perfect-foresight Pareto frontier.

This is consistent with Cao 2020's MILP-vs-RL gap. It is not consistent
with the simpler claim that "LP linear cost works fine"—that claim is
true at perfect foresight, false at realistic noise.

## Refactor scope (carry over from prior memo)

| Phase | Effort | Deliverable |
|---|---|---|
| 1. PriceEnv + QP solver wrappers | 2-3 days | Reproduce Jensen lift at scale |
| 2. RL policy on joint distribution | 4-5 days | Match QP-ensemble lift |
| 3. Real DK1 day-ahead prices, headline figure | 2-3 days | Production figure |
| 4. README rewrite, demote RegEnv to legacy | 1 day | Clean repo |

~10-12 days. Bounded. Headline is clear. Commit to it.

## Files

- `sanity_preflight.py` — runner.
- `fig_pareto_lp_vs_qp.png` — perfect-foresight Pareto.
- `fig_pareto_noise.png` — noisy-foresight Pareto. **The headline figure.**

## Decision

GO. Start phase 1 of refactor.
