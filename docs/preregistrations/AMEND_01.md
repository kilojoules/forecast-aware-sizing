# Pre-registration amendment 01: data source switch

**Date:** 2026-05-08
**Original commitment** (PREREGISTRATION_ERCOT.md, commit 2ce93ae):
- Settlement point: ERCOT North Hub
- Forecast model: day-ahead-as-forecast-of-real-time
- Years: 2021 / 2022 / 2023

**Reason for amendment:**
ERCOT public-data archive (gridstatus / ercot.com MIS) only retains a
recent rolling window (~2-3 months) of public reports. Historical
2021-2023 DA + RT settlement prices are not freely accessible without
a paid API subscription (gridstatus.io, BloombergPaper, etc). Tested
empirically: gridstatus 0.36.0 returns NoDataFoundException for
ERCOT DA queries on 2022-06-15. Same applies to CAISO OASIS for the
same date.

This is a structural blocker on the pre-registered data source.

## New commitment

**Data source:** Energinet (Danish TSO) public API
(`api.energidataservice.dk`). Free, no auth, hourly historical going
back to ~2019.

**Settlement points:** **DK1** (West Denmark, includes Jutland +
Funen). Most-traded Danish hub. DK2 (Zealand + Copenhagen) and the
DE-LU coupling area are out of scope to keep the slice 1-D as
originally pre-committed.

**Years analyzed:** 2021 (typical, pre-energy-crisis), 2022 (European
energy crisis spikes, analogous to ERCOT 2021 Storm Uri in regime
character), 2023 (post-crisis recovery, intermediate). Same three-row
structure as the original pre-registration.

**Realized price:** Nordpool day-ahead spot price for DK1, hourly,
EUR/MWh (`Elspotprices`).

**Forecast model:** Persistence forecast at $t-24h$. Forecast for hour
$t$ in day $D$ is the cleared DA spot price for hour $t$ of day $D-1$.
This is a published-baseline forecast model used in the academic
literature \citep{nogales2002forecasting}. Persistence is the simplest
non-trivial forecast and gives a real, asymmetric, regime-conditional
error distribution.

The empirical $\sigma_f$ is measured directly from
$\epsilon_t = \mathrm{DA}_t - \mathrm{DA}_{t-24h}$ on the realized
trace. No synthetic AR(1) noise on the real-data section.

## Why this is a defensible pivot

1. **Same underlying experimental design.** The 2x2 factorial, the
   $b_{\mathrm{sat}}^\epsilon$ diagnostic, the bootstrap CIs, the
   significance criteria — all unchanged. Only the price series
   substrate changes.
2. **Same regime character.** DK1 has all the multi-timescale
   structure that motivated using ERCOT: diurnal cycles, weekly
   patterns, multi-day weather-driven price clustering, and 2022
   energy-crisis spikes (multiple hours $> \euro$500/MWh, occasional
   negative prices). Storm Uri analogue: DK1's August-November 2022
   gas-driven spikes. Boundary-probe for the proposition.
3. **Persistence forecast has known fat-tailed empirical residuals.**
   Standard baseline in load/price forecasting literature. The
   forecast-error distribution will be regime-conditional (worse on
   high-volatility days), which is exactly the structure that should
   activate the Jensen lift if it activates at all.
4. **Free, public, no auth.** The pre-registration's reproducibility
   commitment is preserved.

## What remains LOCKED from the original pre-registration

- All three years analyzed; no year-selection.
- 2x2 factorial: {linear, quadratic} × {single, K=4 ensemble}.
- $b_P = 1$ MW fixed; sweep $b_E$.
- 32 bootstrap seeds.
- Significance thresholds: 95\% CI overlap $\geq 50\%$ for
  invariance-survives; disjoint $\geq 25\%$ for invariance-breaks.
- $\epsilon / c_E \in \{0.01, 0.05, 0.10\}$, headline at 0.05.

## What changes

- Settlement point: ERCOT North Hub $\to$ DK1.
- Forecast model: DA-as-forecast-of-RT $\to$ persistence-at-$t-24h$.
- Currency: \$/MWh $\to$ EUR/MWh (rescale CAPEX accordingly:
  $C_E = $ \euro100,000 / MWh, $C_P = $ \euro75,000 / MW).
- The "ensemble" forecast: $K=4$ noisy persistence forecasts, where
  each is perturbed by an iid sample from the empirical $(\mathrm{DA}_t -
  \mathrm{DA}_{t-24h})$ residual distribution observed on the trace.

The empirical residual distribution is fat-tailed; bootstrap-from-empirical
is the right ensemble construction. NOT iid Gaussian.

## Non-amendments

- The Wiener-style $q(\omega, \pi)$ derivation in THEORY_DRAFT.md is a
  Gaussian-noise heuristic. It survives qualitatively under fat-tailed
  noise (forecast resolution still degrades faster at long timescales)
  but the closed form is even more heuristic. Acknowledged as a
  caveat in the paper, not a blocker.
- The diagnostic $b_{\mathrm{sat}}^\epsilon$ test is fully
  empirical and indifferent to the data source. Survives unchanged.

## What this commitment is

A genuine amendment, written before any DK1 data is loaded, before any
results are computed. The reason is structural-data-availability,
not "the original setup gave bad results." There are no DK1 results
yet to cherry-pick from.

This document is committed to git BEFORE any ERCOT-replacement data
processing.

## Sources

- Energinet datadocs:
  \url{https://www.energidataservice.dk/tso-electricity/elspotprices}
- DK1 / DK2 boundary description:
  Nordpool market split documentation
- 2022 European energy crisis price statistics:
  ENTSO-E Market Monitoring Report 2022
- Persistence forecast baseline literature:
  Nogales et al. 2002, "Forecasting next-day electricity prices by
  time series models", IEEE TPS
