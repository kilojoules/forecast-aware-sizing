# Sizing-under-uncertainty: research scope

> **STATUS — KILLED 2026-05-08 by pilot S1.** See `SIZING_S1_KILL_MEMO.md`.
> Sizing decision is invariant to dispatch fidelity across noise levels
> 3-18. NPV-at-optimum changes (+1.6 to +55%), but `b_E*` does not.
> Document retained as design rationale; do not implement S2-S4.

**Date:** 2026-05-07
**Thesis:** For hybrid power plants with batteries, the optimal sizing
under deterministic-perfect-foresight inner dispatch differs systematically
from sizing under stochastic-realistic-forecast inner dispatch, by an
amount that matters for investment decisions.

This is the rework that survives the kill memo. Operations is taken
(Tesla / Fluence / enspired). Sizing under uncertainty is open.

## Why this is open

1. Sizing tools (hydesign, REopt, PyPSA capacity-expansion) approximate
   inner dispatch as deterministic LP for tractability — outer sizing
   loop evaluates inner ~10⁴ times per run.
2. Stochastic capacity expansion exists (PyPSA-Eur, REopt-stochastic,
   ReEDS-2-stage) but rarely battery-focused, rarely with rainflow
   degradation, never benchmarked against deterministic at HPP scale
   with realistic forecast errors.
3. Battery cycling cost is convex in DoD: forecast errors lead to
   suboptimal cycling, systematic underestimate of lifetime degradation.
   Deterministic sizing **oversizes** battery's apparent value.
4. Replacement scheduling (hydesign's `num_batteries`) compounds the
   error: more cycling than predicted → more replacements → CAPEX
   underestimated.

## What success looks like

A 1-figure result: NPV shift (deterministic-sizing − stochastic-sizing,
both scored on realized prices) vs forecast quality, on a published
hydesign case. If the curve is meaningfully above zero, sizing tools
need stochastic inner dispatch.

Headline target: **Applied Energy** or **Energy Economics**.

## Pilots, cheapest-falsifying-first

### Pilot S1 — Single-config smoke test (1 day)

One hydesign-style case. Sweep battery energy capacity `b_E ∈ {1, 2, 4, 8, 16}` MWh,
fixed `b_P = b_E / 4`. For each `b_E`:

- Solve **deterministic**: inner dispatch = single-forecast QP using one
  synthetic-realized price trace (T=8760, hourly diurnal).
- Solve **stochastic**: inner dispatch = K=4-ensemble QP using K=4
  forecasts of the same realized.
- Score both on lifetime NPV using rainflow degradation + replacement
  cost model from hydesign (already in `degradation.py`).
- Find `b_E*` for each.

**Go threshold:** |`b_E*_det` − `b_E*_stoch`| / `b_E*_det` ≥ **10%**.
**Null:** <5% shift; sizing robust to dispatch fidelity. **Kill.**
**Deliverable:** `pilot_s1.py` + 1 figure (NPV vs `b_E` for both regimes).

### Pilot S2 — Forecast-quality sensitivity (3 days)

If S1 passes, sweep noise std ∈ {1, 3, 5, 8, 12, 18} on the same case.
For each noise level, compute the optimal `b_E` for both dispatch regimes
and the realized NPV gap.

**Go threshold:** monotone gap growth with noise; max gap ≥ **15%** at
realistic noise (std=5–8) on lifetime NPV.
**Null:** flat or non-monotone gap.
**Deliverable:** `pilot_s2.py` + 1 figure (NPV gap vs noise).

### Pilot S3 — Real-data anchoring (5 days)

Replace synthetic prices with **ENTSO-E DK1 day-ahead 2024** (publicly
downloadable), with realistic forecast errors derived from Energinet's
day-ahead forecast vs realized statistics. Replace synthetic wind/PV
with **NREL WIND Toolkit** Denmark site + Sandia PV forecast errors.

Re-run S1+S2 on real data. Confirm shift persists.

**Go threshold:** shift ≥ **10%** at realistic forecast quality on real
DK1 prices.
**Null:** shift <5% on real data → synthetic-only effect. Kill.
**Deliverable:** `pilot_s3.py` + 1 figure + data-loader for ENTSO-E.

### Pilot S4 — Full paper-grade study (10–15 days)

If S1–S3 clear:

1. Replicate the hydesign Indian peak-power case from Murcia Le\'on 2024.
2. Add stochastic dispatch inner loop to hydesign (PR-able if accepted).
3. Run sizing optimization under both regimes for 5 sites
   (varying renewable penetration, price volatility).
4. Multi-year forecast error distributions; bootstrap NPV CIs.
5. Paper draft: framing, methods, 4–5 figures, discussion.

**Deliverable:** journal submission. Estimated 6 weeks calendar.

## Required infrastructure

| Component | Status | Effort |
|---|---|---|
| `PriceEnv` | shipped (phase 1) | — |
| `RainflowStack` + post-hoc D | shipped | — |
| LP / QP / ensemble dispatch | shipped (`arbitrage_agents.py`) | — |
| Sizing outer loop | TODO | 1 day (scipy.optimize on b_E) |
| Lifetime NPV with replacement | TODO | 1 day (port from hydesign battery_degradation.py) |
| ENTSO-E DK1 data loader | TODO | 1 day (`price_signal.load_csv` already stubbed) |
| Hydesign integration (S4) | optional | 5 days |
| Multi-year scenario generation | TODO (S3+) | 2 days |

## Risks

1. **Effect size too small.** Sizing aggregates over 25 years; hourly
   forecast errors may wash out at NPV scale. Pilot S1 falsifies cheaply.
2. **Computational explosion at S4.** Outer-sizing × stochastic-inner is
   1000× current cost. Mitigate with surrogate models (Gaussian process
   regression on `(config) → NPV` from a sparse sample of stochastic
   inner solves).
3. **Real-data uglier than synthetic.** ENTSO-E forecasts are biased
   (not zero-mean). Need to handle bias separately from variance. Adds
   work but is itself publishable.
4. **Audience mismatch.** Sizing investors care about price-scenario
   sensitivity, not dispatch-fidelity sensitivity. Frame paper around
   robust sizing for battery-marginal regimes (>40% renewable
   penetration), where the effect should be biggest.
5. **Hydesign integration friction.** Replacing inner LP requires
   touching openMDAO subsystem code. Mitigate by running S1–S3 entirely
   inside `battery_gym` first; only do hydesign PR if effect survives.

## Decision tree

```
[Pilot S1] -- 1 day, synthetic, single config
   |
   |-- shift >= 10%  -> [Pilot S2] (sweep noise)
   |                       |
   |                       |-- max gap >= 15% -> [Pilot S3] (real DK1)
   |                       |                        |
   |                       |                        |-- gap persists -> [Pilot S4]
   |                       |                        |                     -> paper draft
   |                       |                        |-- gap <5%  -> KILL (synthetic-only)
   |                       |-- flat gap -> KILL
   |-- shift <5%  -> KILL (sizing robust)
```

Run S1 first. **The whole project lives or dies on its result.**

## Why this fits the BMS-application-ready story

Independent of paper outcome, this work demonstrates:
1. End-to-end systems design (hydesign integration, ENTSO-E data,
   stochastic optimization, RL/QP solvers).
2. Honest quantitative experimentation (clear go/no-go thresholds,
   negative-result publishing if it fails).
3. Real industry tools (hydesign, ENTSO-E, NREL WIND Toolkit).
4. Adjacent to BMS work: dispatch-aware sizing → battery requirement
   spec → BMS thermal / cell-balance / FMEA derived from realized
   cycling profile, not nominal.

## Sources

- [Murcia Le\'on et al. 2024, hydesign WES paper](https://wes.copernicus.org/articles/9/759/2024/)
- [PyPSA-Eur stochastic capacity expansion](https://pypsa-eur.readthedocs.io/)
- [REopt-stochastic, NREL](https://www.nrel.gov/transportation/reopt.html)
- [ENTSO-E Transparency Platform (DK1 day-ahead)](https://transparency.entsoe.eu/)
- [Energinet day-ahead forecast statistics](https://www.energidataservice.dk/)
- [NREL WIND Toolkit](https://www.nrel.gov/grid/wind-toolkit.html)
- [Cao 2020 RL+rainflow arbitrage](https://ieeexplore.ieee.org/abstract/document/9061038)
- [Birge & Louveaux 1997, Introduction to Stochastic Programming](https://link.springer.com/book/10.1007/978-1-4614-0237-4)
