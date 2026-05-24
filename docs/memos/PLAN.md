# Plan: ERCOT extension → workshop paper

**Generated:** 2026-05-08
**Pre-registration locked at:** PREREGISTRATION_ERCOT.md (commit 2ce93ae)
**Theory locked at:** THEORY_DRAFT.md v2 (commit 3b0e090)
**Target:** workshop paper submission in 10-11 days

## Sequencing

```
Day 3   ERCOT data pull + parser     [no DAY 1-2 — done]
Day 4   Forecast + spectrum tooling
Day 5   b_sat^eps diagnostic + synthetic validation
Day 6   2x2 factorial × 3 years run
Day 7   Bootstrap + aggregate results
Day 8   Figures + tables
Day 9   Paper revision (proposition + ERCOT section + intro rewrite)
Day 10  Format-convert for venue, final compile
Day 11  Submit
```

## Tasks (granular, gated)

### Block A — ERCOT data pipeline (Day 3-4)

- [ ] **A1** Install `gridstatus` package; verify ERCOT API works.
- [ ] **A2** Pull ERCOT North Hub day-ahead + real-time prices for 2021, 2022, 2023.
- [ ] **A3** Aggregate RT to hourly. Verify continuity (no gaps).
- [ ] **A4** Compute DA-RT residuals; report mean, std, kurtosis. Cross-check against published statistics.
- [ ] **A5** Save canonical CSVs to `data/ercot/{year}_north.csv` (gitignored).

### Block B — Tooling extensions (Day 4)

- [ ] **B1** `price_signal.py`: add `load_ercot(year, hub)` returning realized + DA. Empirical residual sampler for ensemble forecast generation.
- [ ] **B2** `spectrum.py`: Welch PSD on a price series; dominant timescales report; figure helpers.
- [ ] **B3** Sanity check: synthetic AR(1) spectrum recovers expected peaks at 24h + 12h.

### Block C — Diagnostic (Day 5)

- [ ] **C1** `b_sat_classifier.py`: fine sweep `b_E` near apparent argmax + percentile bootstrap on
      argmax across 32 forecast seeds → return $b_{\mathrm{sat}}^\epsilon(\pi)$ and 95% CI.
- [ ] **C2** Validate on synthetic AR(1) (Pilot S1 setup): both policies yield overlapping CIs, recovers invariance.
- [ ] **C3** Add helper to render the $b_{\mathrm{sat}}^\epsilon$-pair plot.

### Block D — 2x2 factorial benchmark (Day 6-7)

- [ ] **D1** Refactor `phase1_benchmark.py` → `paper_benchmark.py` with config:
      `(year, policy ∈ {LP-single, LP-ens, QP-single, QP-ens})`.
- [ ] **D2** Local run on a 1-week ERCOT slice for smoke test (each policy × each year).
- [ ] **D3** Full run: 3 years × 4 policies × 8 b_E values × 32 forecast seeds.
      Estimate: 3 × 4 × 8 × 32 = 3072 QP solves; ~50 min on this laptop with cvxpy CLARABEL.
      If too slow, parallelize across years via Python multiprocessing.
- [ ] **D4** Save full results JSON + per-config artifacts.

### Block E — Analysis + figures (Day 8)

- [ ] **E1** Aggregate JSON → summary table: per (year, policy) the mean/std of NPV at each b_E.
- [ ] **E2** Headline figure: $b_{\mathrm{sat}}^\epsilon$ panel-per-year with CIs, det vs ens.
- [ ] **E3** Supporting figure: NPV vs b_E for each year, det vs ens, with bootstrap bands.
- [ ] **E4** 2x2 factorial decomposition figure: cost-axis vs forecast-axis effects separated.
- [ ] **E5** Spectrum overlay: ERCOT 2021/2022/2023 PSD vs synthetic AR(1).

### Block F — Paper revision (Day 9)

- [ ] **F1** Add §3 "Theoretical condition" with proposition + sketch + diagnostic + 1-D-slice caveat.
- [ ] **F2** Replace §4 (single-noise table) with §4 "Synthetic validation" tightened.
- [ ] **F3** Add §5 "Empirical regime characterization on ERCOT" with all three years.
- [ ] **F4** Rewrite intro around Tesla-vs-NREL puzzle.
- [ ] **F5** Update abstract: lead with realistic regime (not 55%), state two findings + diagnostic.
- [ ] **F6** Fix citation hygiene (Schultheis vs "Evaluation 2025"; Hancock date stamp).
- [ ] **F7** Acknowledge limitations: 1-D slice, no SLP, post-hoc rainflow, replacement step function.
- [ ] **F8** Compile clean PDF.

### Block G — Submit (Day 10-11)

- [ ] **G1** Pick venue (CCAI workshop vs RLEM vs Energy & AI). User decision.
- [ ] **G2** Format-convert (template, page limit).
- [ ] **G3** Final proofread.
- [ ] **G4** Submit.

## Blocker map (live)

| Blocker | Affects | Workaround |
|---|---|---|
| ERCOT data unavailable for a year | Block A onward | Substitute load-zone weighted avg; document in amendment |
| `gridstatus` package missing | A1 | Direct ERCOT public-data CSV download |
| cvxpy slow at full scale | D3 | Multiprocess across years; reduce K-bootstrap to 16 |
| Storm Uri data gap (Feb 2021) | A2 | Treat gap as data; spike values are real |
| User decision on venue | G1 | Default CCAI; user can override last-minute |
| ControlMaster expired (gbar) | unrelated to this paper; ignore | n/a |

## Self-audit cadence

Monitor running `audit.sh`:
- Polls every 5 minutes.
- Lists unchecked PLAN.md tasks.
- Reports artifacts present in repo + their commit timestamps.
- Surfaces TODOs by file.

Notification per change in unchecked-count or new artifact creation.

## Decision points (require user input)

- **D7**: venue choice (CCAI / RLEM / Energy & AI). Default: CCAI workshop.
- **F8 → G1**: ready-for-submit signoff.
- Any time: reviewer-style critique on a draft section.

## What runs autonomously

Everything in Blocks A-F above except the decision points. ControlMaster failures don't apply (this work is local-only). cvxpy failures are local recoverable.

If a task fails three times in a row, escalate by writing a `BLOCKER_<task>.md` with diagnostic and pause until user input.

## Schedule diff vs SIZING_PILOTS.md

Original plan (SIZING_PILOTS.md, killed): pilot tree S1-S4, sizing-shift hypothesis. **Killed** by Pilot S1.

Current plan (this doc): operational-vs-sizing decomposition + theoretical condition + ERCOT regime characterization. **Live**.

Theory upgrade vs original paper draft:
- Was: empirical observation of sizing-invariance.
- Is now: proposition deriving $b_{\mathrm{sat}}$ from price spectrum + practitioner-runnable diagnostic + ERCOT regime probe.
