# Pre-registration: ERCOT empirical extension

**Date:** 2026-05-08
**Purpose:** lock the experimental design before touching ERCOT data.
Eliminates degrees of freedom for cherry-picking regimes / forecast
models post-hoc. Committed to git in this state; deviations require an
explicit "amendment" commit with reason.

## Commitments

### 1. Years analyzed: ALL THREE of 2021, 2022, 2023.

Reason: each year is qualitatively different in regime, and reporting
only the most favorable would be a cherry-pick.

- **2021**: Storm Uri (February 14-19, 2021) — extreme price spikes
  (\$9000/MWh cap hit for days). Tests the rare-event-driven-revenue
  case explicitly mentioned in `THEORY_DRAFT.md` §"Where the condition
  breaks." If invariance breaks anywhere, it should break here.
- **2022**: Typical year. Tests the central regime of the theory; if
  invariance breaks even on a non-anomalous year, the academic-tool
  convention is broadly suspect.
- **2023**: Summer scarcity events (June heat dome, August scarcity
  hours). Intermediate regime — multi-day clustering without
  catastrophic spikes.

Reporting structure: one row per year in the headline table; one
$b_{\mathrm{sat}}^{\epsilon}$ pair plot per year as a panel. No
year-selection downstream.

### 2. Forecast model: day-ahead-as-forecast-of-real-time.

Reason: standard ERCOT setup, $\sigma_f$ from data not synthetic, no
choice degrees of freedom in the noise model.

Specifically:
- **Real-time prices**: ERCOT real-time settlement-point prices (5-min
  or 15-min, aggregated to hourly).
- **Day-ahead prices**: ERCOT day-ahead market prices at the same
  settlement point, hourly.
- **"Forecast"** at hour $t$ = day-ahead clearing price for hour $t$.
- **"Realized"** at hour $t$ = real-time hourly average for hour $t$.

The forecast errors $\epsilon_t = \mathrm{DA}_t - \mathrm{RT}_t$ are
empirically observable and have known statistical properties (biased,
fat-tailed, regime-conditional). $\sigma_f$ for use in
$\tau_{\mathrm{res}}$ derivations comes from the empirical residual
distribution — no synthetic AR(1).

Drop the synthetic AR(1) noise model entirely for the real-price
section. Keep it only in the synthetic-validation section as the
proof-case for the proposition.

### 3. Settlement point: ERCOT North Hub.

Reason: most-traded hub, broad public-data availability, the canonical
benchmark in the ERCOT-arbitrage literature. Avoids the deniability of
"we picked an unusual location."

If North Hub data is unavailable for any of the three years, fall back
to the ERCOT load-zone weighted average. Document the substitution.

### 4. Battery configuration: $b_P = 1$ MW fixed across all years.

Reason: keep $b_P$ constant so the $b_E$ sweep is the single varying
axis (matching the theoretical proposition's 1-D slice).

Sweep grid: $b_E \in \{0.5, 1, 2, 4, 8, 16, 32, 64\}$ MWh. Add finer
linear sweep $\{4, 6, 8, 10, ..., 60\}$ MWh near the apparent argmax
for argmax-CI estimation, separately for each year.

### 5. Dispatch policies: 2x2 factorial.

Reason: separate the cost-function and forecast-handling axes per
reviewer's earlier critique of the original paper.

- $\pi_{\mathrm{LP-single}}$: linear cost, single forecast.
- $\pi_{\mathrm{LP-ens}}$: linear cost, $K{=}4$ ensemble forecast.
- $\pi_{\mathrm{QP-single}}$: quadratic cost, single forecast.
- $\pi_{\mathrm{QP-ens}}$: quadratic cost, $K{=}4$ ensemble forecast.

The "ensemble" for real ERCOT data: bootstrap from the empirical
day-ahead-vs-real-time residual distribution. Generate $K{=}4$
synthetic forecasts by perturbing the day-ahead price with iid samples
from the empirical residual distribution. **NOT** by re-sampling the
real-price-history-conditional distribution (that would leak realized
information into the forecast).

### 6. Number of bootstrap seeds: 32.

Reason: enough for tight CI on argmax via percentile bootstrap;
$\sqrt{32} = 5.7\%$ standard error baseline.

### 7. Significance criterion: $b_{\mathrm{sat}}^\epsilon$ overlap.

Pre-committed thresholds:
- **"Invariance survives"**: 95\% bootstrap CIs of
  $b_{\mathrm{sat}}^{\epsilon}(\pi_{\mathrm{det}})$ and
  $b_{\mathrm{sat}}^{\epsilon}(\pi_{\mathrm{ens}})$ overlap by $\geq 50\%$
  of the smaller CI's width.
- **"Invariance breaks"**: 95\% CIs are disjoint with a gap $\geq 25\%$
  of the smaller CI's lower bound.
- **"Inconclusive"**: anything in between. Report as inconclusive, do
  not strain interpretation.

$\epsilon$ threshold: report for
$\epsilon / c_E \in \{0.01, 0.05, 0.10\}$. Headline number uses
$\epsilon / c_E = 0.05$. Sensitivity stated explicitly.

### 8. Replacement-cost model: keep current step-function.

Reason: changing the model post-hoc to "smooth out" anomalies is a
cherry-pick. Acknowledge the discreteness as a limitation in §6
(already done).

## What this pre-registration LOCKS

- All three years analyzed; no year-selection.
- Day-ahead-as-forecast; no synthetic noise on real data.
- ERCOT North Hub; no settlement-point shopping.
- $b_P = 1$ MW; no power-sweep mid-paper.
- 2x2 factorial; no ad-hoc policy comparisons.
- 32 bootstrap seeds; CI thresholds committed.

## What it does NOT lock

- Implementation details (data parser, missing-hour handling, holiday
  treatment) — to be decided before reporting numbers but not before
  pulling data.
- Plot styling.
- Whether to extend to CAISO or Nord Pool intraday in a follow-up
  paper. Out of scope here.

## Amendment policy

If a commitment above must be changed (e.g., 2023 data missing in a
key date range), commit an explicit `PREREGISTRATION_AMEND_NN.md` with
the reason, before re-running. No silent edits to this file.

## Why this matters

The reviewer flagged:
> "Pre-register which year(s) you'll use and why, otherwise you'll be
> tempted to cherry-pick the regime that makes your proposition look
> best."

This is correct. Locking the design before data touch eliminates that
degree of freedom. The honest move is "all three, report all three"
even if 2022 turns out null.

Forecast-model upgrade: day-ahead-as-forecast-of-real-time is a strict
upgrade over synthetic AR(1) — empirical $\sigma_f$, real bias, real
fat tails. Drops a degree of freedom.

These two together substantially harden the empirical claim.

## Schedule

- Day 1-2: pre-registration committed (this document).
- Day 3: ERCOT data download (gridstatus.io or ERCOT public data,
  ENTSO-E for cross-check).
- Day 4: data parsing, sanity check spectrum + DA-RT residuals match
  literature.
- Day 5-7: run 2x2 factorial × 3 years × full sweep grid + bootstrap.
- Day 8-9: analysis + figures.
- Day 10: paper revision integrating ERCOT results + proposition.
- Day 11: submit (CCAI / RLEM / Energy & AI).

## Sources

- ERCOT public data: \url{https://www.ercot.com/mp/data-products/}
- gridstatus.io ERCOT API for parsed CSVs
- DA vs RT residual statistics for ERCOT Hub Average:
  Wood, Walter, et al. various ERCOT market reports
- Storm Uri analysis: ERCOT, "February 2021 Cold Weather Event Final Report"
