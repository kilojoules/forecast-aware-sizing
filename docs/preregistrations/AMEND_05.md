# Pre-registration amendment 05: scenario stochastic LP

**Date:** 2026-05-08
**Earlier commitment:** ensemble = mean-of-K-forecasts (deterministic
substitution into a deterministic LP). Limitation called out in §6:
"no scenario stochastic LP comparison."

**Reason for amendment:** Krishnamurthy 2018 / Birge & Louveaux 1997
argue stochastic LP with N≥50 scenarios is the production-grade
benchmark. With LUMI compute granted, run the proper SLP comparison.

**New commitments:**

- Two-stage stochastic LP, rolling-horizon.
- Window: 24 h.
- N = 50 scenarios per window. Each scenario = persistence forecast +
  iid sample from empirical (DA - DA(t-24h)) residual distribution.
- Quadratic cycling cost only (LP linear is rank-invariant; SLP-LP
  reduces to deterministic-LP-on-mean-forecast which we already have).
- 6 array tasks: (dk1, ercot) × (2021, 2022, 2023).
- 12 b_E values, b_P=1 MW.
- Walltime budget: 24 h per task.
- Solver: cvxpy + CLARABEL, fallback SCS.

**Same significance criteria.**

**Output:** `results_slp/{src}_{year}.json` per task.

Compares against the existing 4-policy factorial; SLP becomes a 5th
policy. Argmax invariance test now over 5 policies per (market, year).

Committed before LUMI submission.
