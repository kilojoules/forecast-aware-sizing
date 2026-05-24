# Pilot 1 memo: hydesign user-need survey

**Question:** Are there ≥3 published or planned hydesign use cases with
(>1 physical battery in parallel) AND (sub-hourly market participation)?

**Verdict: NO — count is 0. Kill the integration as currently framed.**

## Evidence — hydesign codebase

- **Single-aggregate battery only.** `b_P` (MW) and `b_E` (MWh) are scalars
  in `hydesign/ems/ems.py:84-91`, `hydesign/assembly/hpp_assembly.py:396-414`.
- **`num_batteries` ≠ fleet.** It means **serial replacements over the
  25-year plant lifetime** — battery 1 dies, battery 2 installed, etc.
  See `hydesign/battery_degradation.py:445` (`for i in range(num_batteries-1):`)
  and the docstring at line 409: "num_batteries: number of battery
  replacements". No parallel multi-pack semantics anywhere.
- **Sub-hourly already supported.** `hydesign/ems/ems_hifi_dems.py:18`
  defaults to `intervals_per_hour=4` (15-min). `hydesign/ems/ems_BM.py`
  + `hydesign/assembly/hpp_assembly_BM.py` model spot + balancing-market
  participation directly.

## Evidence — published work

| Source | Multi-physical-battery? | Sub-hourly? | Uses hydesign? |
|---|---|---|---|
| Murcia Leon et al., WES 9:759, 2024 (canonical hydesign paper) | No (single aggregate) | Hourly | Yes |
| Zhu, Das, Sørensen, Hansen 2024 (Int J Electr Power Energy Syst, "Enhancing profits of hybrid wind-battery plants in spot and balancing markets using data-driven two-level optimization") | No | Yes (BM) | Yes — built into `ems_BM.py` |
| DTU PhD thesis "Optimal Energy Management of Hybrid Power Plants in Electricity Markets" (orbit.dtu.dk/files/351171759) | No | Yes (DA + HA + re-dispatch) | Adjacent / hydesign-like |
| Multi-energy balancing services (Applied Energy 2024, S0306261924013497) | No (PV+BESS+H2, single BESS) | Yes | No |
| IEA Wind Task 50 Annual Report 2024 | No clean cases found | Mixed | Tool overview only |
| NREL FY25 BESS-in-HPP (osti.gov 84791) | No | n/a | n/a |

**Cases meeting BOTH criteria: 0.**

## What this means

The two-tier wedge in `HYDESIGN_INTEGRATION.md` rested on a hypothesis:
hydesign would benefit from a sub-hourly RL layer because heterogeneous
fleets break Naïve. Two collapses:

1. **hydesign already has sub-hourly + BM participation.** No gap to fill at
   the sub-hour level for single-battery dispatch. The DTU group already
   shipped two-level (spot+BM) optimization in `ems_BM.py`.
2. **Heterogeneous fleets are not a hydesign use case.** Tool, papers, and
   adjacent literature uniformly model single aggregate storage. There's
   no published demand for multi-physical-battery dispatch in this
   community.

`battery_gym`'s RL story (RL beats Greedy on heterogeneous fleets) is
**orthogonal** to hydesign's design space, not complementary.

## Cancelled pilots

- **Pilot 2** (RL vs Greedy on het fleet, battery_gym alone) — still useful
  as standalone work, decoupled from hydesign integration. Run only if
  pursuing a het-fleet RL paper.
- **Pilot 3** (sub-hour sensitivity) — not needed; `ems_BM.py` exists.
- **Pilot 4** (reg-cap reservation in LP) — not needed for the integration;
  would be a hydesign PR, not a battery_gym deliverable.

## Two viable forward paths

1. **Keep `battery_gym` standalone.** Present as a paper reproduction +
   honest finding (ELM-RL = smooth Greedy on this proxy reward). Use as
   credibility object in BMS application as-is. **Recommended.**
2. **Reframe as a hydesign PR: heterogeneous-fleet support.** Extend
   hydesign to vectorize `b_P`/`b_E`. Open question: who would use it?
   Until a hydesign user asks, this is push not pull. **Defer.**

## Decision

Kill the integration. Update `HYDESIGN_INTEGRATION.md` and
`HYDESIGN_PILOTS.md` headers to reflect this.

## Sources

- [WES 9:759, 2024 — Murcia Leon et al.](https://wes.copernicus.org/articles/9/759/2024/)
- [hydesign GitHub mirror](https://github.com/DTUWindEnergy/hydesign)
- [hydesign HPP+BM tutorial](https://topfarm.pages.windenergy.dtu.dk/hydesign/notebooks/HPP_evaluation_BM.html)
- [Zhu et al. 2024 SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4499864)
- [DTU PhD thesis on HPP EMS](https://orbit.dtu.dk/files/351171759/PhD_Thesis.pdf)
- [IEA Wind Task 50](https://iea-wind.org/task50/)
- [WES 10:559, 2025 — surrogate-based EMS for HPP sizing](https://wes.copernicus.org/articles/10/559/2025/)
