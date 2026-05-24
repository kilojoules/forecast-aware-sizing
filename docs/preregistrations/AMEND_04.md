# Pre-registration amendment 04: quantile-regression ensemble

**Date:** 2026-05-08
**Earlier commitment:** ensemble = K=4 multi-lag persistence at lags
{24, 48, 168, 336} h (AMEND_02).

**Reason for amendment:** the most consequential limitation called out
in paper §6 is "multi-lag persistence is a weak ensemble; modern
probabilistic forecasts could break invariance, and the negative result
is conditional on this choice." With LUMI compute granted, we can
test the proposition under a richer ensemble.

**New commitments:**

- Forecast model: `sklearn.ensemble.GradientBoostingRegressor` with
  loss="quantile", alpha=$q$ for $q \in \{0.05, 0.10, ..., 0.95\}$
  (K=20 quantile levels).
- Features: lagged prices at {24, 48, 168, 336} h + sin/cos calendar
  features (hour-of-day, day-of-week).
- Trained on year $y - 1$, applied to year $y$ (no test-set leakage).
- Single-forecast baseline = predicted MEDIAN (q=0.5).
- Ensemble = mean of all K=20 quantile predictions.
- 6 array tasks: (dk1, ercot) × (2021, 2022, 2023).
- 12 b_E values, b_P=1 MW fixed (matches main paper run).

**Same significance criteria:** $b_E^*$ overlap test for invariance.

**Output:** `results_quantile/{src}_{year}.json` per task.

Predicted outcomes:
- If quantile ensemble breaks invariance → diagnostic correctly fires
  ("disjoint"), proposition's prediction supported.
- If quantile ensemble preserves invariance → invariance is structurally
  more robust than just persistence-ensemble; broadens the empirical
  claim.

Committed before LUMI submission.
