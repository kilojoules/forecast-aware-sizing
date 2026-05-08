# Pre-registration amendment 02: ensemble construction fix

**Date:** 2026-05-08
**Earlier commitment** (PREREGISTRATION_AMEND_01.md, commit ebb035c):
> "K=4 noisy persistence forecasts, where each is perturbed by an iid
> sample from the empirical (DA_t - DA_{t-24h}) residual distribution
> observed on the trace."

**Reason for amendment:**
The above ensemble construction is information-theoretically a no-op
under linear cost. Each "ensemble member" is `persistence + iid_noise_k`;
the ensemble mean is `persistence + mean(noise_k)` $\to$ persistence as
$K \to \infty$. No Jensen lift can arise — the ensemble carries strictly
less information per member than the persistence forecast itself, and
averaging just denoises the additive perturbation.

This was a planning error caught before reporting any results. Caught
empirically: smoke test on DK1 2022 quick run showed ensemble revenue
*lower* than single-forecast revenue at every $b_E$, exactly opposite
to the predicted Jensen direction.

## New ensemble construction

**Multi-lag persistence ensemble.** $K=4$ forecasts at different lags:

- $\hat{p}^{(1)}_t = p_{t - 24}$ (yesterday at this hour)
- $\hat{p}^{(2)}_t = p_{t - 48}$ (two days ago)
- $\hat{p}^{(3)}_t = p_{t - 168}$ (one week ago at this hour)
- $\hat{p}^{(4)}_t = p_{t - 336}$ (two weeks ago at this hour)

These are genuinely different forecasts of the same target. Each
captures a different hypothesis about which past pattern is the best
predictor. Averaging across lags is a standard ensemble technique in
load + price forecasting (Hong et al. 2014, Weron 2014).

The forecast errors at each lag have systematically different
properties:
- Short lag (24h): captures intra-week pattern, fails on day-of-week
  switches.
- Mid lag (48h, 168h): captures weekly pattern, fails on
  weather-driven multi-day events.
- Long lag (336h): captures fortnightly trend, fails on regime shifts.

Averaging tends to cancel the lag-specific failures.

The single-forecast policy still uses persistence at $t - 24$h.
The ensemble policy averages all four.

## Why this still satisfies the spirit of the original pre-registration

Original AMEND_01 intent: "K=4 noisy forecasts of the same target,
averaging captures the conditional mean and produces Jensen lift under
quadratic cost."

The multi-lag construction satisfies this intent properly: K=4
genuinely-different forecasts, averaged, with empirically observable
error distributions per lag.

## What remains LOCKED

- All three years analyzed.
- 2x2 factorial.
- $b_P = 1$ MW fixed.
- 95% CI thresholds.
- $\epsilon$ sweep and headline.

## What changes

- `dk_loader.py:ensemble_forecasts` rewritten to do multi-lag
  persistence, not iid noise perturbation.
- `paper_benchmark.py` re-run from scratch on the corrected ensemble.

## Diagnostic

After this fix, the reported per-year statistics should show:
- ensemble revenue $\geq$ single revenue on multi-timescale years
  (predicted by Jensen on quadratic cost + multi-lag information gain),
- ensemble revenue $\approx$ single revenue on linear-cost (rank
  invariance, but with multi-lag the ranks may also differ in subtle
  ways).

## Decision

This amendment is committed BEFORE running any analysis on the
corrected ensemble. The smoke test that exposed the bug used the
flawed construction; results are discarded.

## Sources

- Hong, Pinson, Fan 2014. Global energy forecasting competition 2012.
  Int J Forecasting.
- Weron 2014. Electricity price forecasting: a review of the
  state-of-the-art with a look into the future. Int J Forecasting.
