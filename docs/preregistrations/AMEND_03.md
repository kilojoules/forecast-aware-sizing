# Pre-registration amendment 03: 2-D (b_E, b_P) sweep

**Date:** 2026-05-08
**Earlier commitment:** $b_P = 1$ MW fixed; only $b_E$ swept. Limitation
acknowledged in the paper as the 1-D-slice caveat.

**Reason for amendment:** LUMI compute granted; can run a full 2-D
sweep without time pressure. The proposition's $b_{\mathrm{sat}}(\pi)
\propto b_P$ form predicts that 2-D surfaces could behave differently
than the 1-D slice. Now testable.

**New commitments:**

- $b_E \in \{1, 2, 4, 8, 16, 32\}$ MWh (6 values)
- $b_P \in \{0.25, 0.5, 1.0, 2.0, 4.0\}$ MW (5 values)
- 30 (b_E, b_P) configurations per (market, year, cost-form)
- 2 markets × 3 years × 4 policies × 30 = 720 dispatch evaluations
- LUMI Slurm array job, 30 tasks (one per (b_P, b_E) with all years/markets/policies inside)
- Same significance criteria as before: argmax invariance test ON the 2-D surface — i.e., does the maximum over (b_E, b_P) of the NPV surface coincide across single-vs-ensemble dispatch?

**Locked from prior amendments:** all 3 years, both markets, multi-lag persistence ensemble construction.

**Output:** `results_2d/2d_bP{X}_bE{Y}.json` per task.

Committed before LUMI submission.
