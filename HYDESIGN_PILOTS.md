# Hydesign integration: pilot plan

> **OUTCOME — Pilot 1 ran, killed the project (2026-05-05).** See
> `HYDESIGN_PILOT1_MEMO.md`. Pilots 2-4 cancelled or decoupled.

Idea-critic vetted the question list (`HYDESIGN_INTEGRATION.md`). Verdict:
**REFINE-leaning-KILL**. Critique below, then 4 ordered pilots.

## Critique of the question list

### The 0th question I missed
**Does any real hydesign user run a heterogeneous fleet AND sell sub-hourly
services?** Q1 and Q2 are not independent — they collapse. If hydesign's user
community (DTU/IEA Wind, single-aggregate-battery convention) has zero
published use cases needing both, the two-tier architecture is a solution
looking for a problem. Answer first; subsumes ~50% of the list.

### Three hidden risks I under-flagged

1. **Audience mismatch.** BMS/DC-DC EE hiring committee wants firmware, BMS
   chemistry models, CAN, IEC 61508 — not LP+RL hybrid-plant coupling.
   FAR.AI / NeurIPS won't care about hydesign at all. Even if the technical
   work succeeds, the deliverable is orphaned. "Make battery_gym look
   credible" framing is doing too much work — credible to whom?
2. **Falsified finding is load-bearing.** ELM-RL = smooth Greedy was
   confirmed in `action_match.py`. Heterogeneity is the *hypothesized*
   rescue, not a verified one. Until RL > Greedy on a het fleet *in
   battery_gym alone*, the integration argument has no foundation.
3. **hydesign rainflow is ground truth, battery_gym proxy is a learnable
   surrogate.** That's one sentence, not a question.

### Cull these — answers don't change architecture

- **Q6 (resolution K):** pick 15-min, hydesign HiFiEMS already supports it.
- **Q7 (signal source):** use real PJM RegD trace.
- **Q8 (λ sweep):** always a sweep, not a gating question.
- **Q4 (degradation truth):** see point 3 above.

### Keep

- **Q1 + Q2 fused as 0th question** (revenue model + heterogeneous fleet).
- **Q3** (reg-cap reservation in the LP).
- **Q5** (energy-balance handoff).

## Pilots (cheapest-falsifying first)

### Pilot 1 — User-need survey (2 days, no code)

Search hydesign GitHub issues, DTU/Risø theses 2022-2026, IEA Wind Task 50
reports, Energinet/EirGrid hybrid-plant studies. Count distinct published
or planned hydesign use cases with (>1 physical battery) AND (sub-hourly
market participation).

- **Go threshold:** ≥3 distinct cases → proceed.
- **No-go:** <3 → kill or pivot tool.
- **Null:** zero credible use cases exist.
- **Deliverable:** 1-page table of cases with citations.

### Pilot 2 — Heterogeneous RL vs Greedy in battery_gym alone (1 week)

2-pack heterogeneous fleet (Li-NMC + LFP, asymmetric C-rates and cycle
costs). Compare RL, smooth-Greedy, Naïve on lifetime $/MWh under PJM RegD
trace.

- **Go threshold:** RL beats best non-RL baseline by **≥5% lifetime $/MWh**
  at p<0.05 over 30 seeds.
- **No-go:** RL ≤ Greedy (consistent with the action-match finding).
- **Null:** RL ≤ Greedy.
- **Deliverable:** boxplot + table. If null holds, write negative-result
  memo and stop.

### Pilot 3 — hydesign sub-hour sensitivity (1 week)

Run hydesign on one canonical hybrid-plant case at 60-min, 15-min, 5-min
dispatch resolution. Hold everything else fixed.

- **Go threshold:** NPV or end-of-life SoH shifts by **≥2%** between 60-min
  and 5-min.
- **No-go / null:** <2% drift — hourly LP is already adequate; sub-min layer
  is decoration.
- **Deliverable:** 3-row sensitivity table.

### Pilot 4 — Reg-cap reservation matters in the LP (1 week, only if 1-3 pass)

Add reg-up/reg-down headroom constraints to hydesign LP. Solve with
realistic 2024 reg-prices (PJM or Nordic FCR-N).

- **Go threshold:** optimal reserved capacity is **≥10% of battery
  nameplate** AND objective improves by **≥3%** vs energy-only.
- **No-go / null:** reservation is negligible — timescale-coupling problem
  doesn't bite, no need for sub-min tracker.
- **Deliverable:** Pareto plot (reg-revenue vs energy-revenue) for one site.

## Decision path

Run **Pilot 1 this week**. Base rate for hydesign-with-heterogeneous-
multi-service plants in published literature is likely low. If Pilot 1
returns <3 cases:

- Kill the integration.
- Keep battery_gym as a standalone heterogeneous-fleet RL paper *if*
  Pilot 2 ever passes.
- Otherwise shelve.

Don't run Pilots 3-4 until 1 and 2 both clear.

## Honest framing for the BMS application

"battery_gym reproduces a recent paper, finds RL ≈ Greedy on the proxy
reward, identifies heterogeneous fleets as the regime where the gap might
exist."

Clean, credible story without the hydesign integration. The integration
is upside, not load-bearing.

## Sources

- arXiv:2601.22865v2 — Srinivasa et al. 2026 (paper reproduced)
- https://github.com/IEAWindTask37/hydesign
- https://iea-wind.org/task50/
