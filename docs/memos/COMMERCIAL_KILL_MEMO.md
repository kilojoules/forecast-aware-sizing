# Commercial potential: KILL (product) / PURSUE (paper)

**Date:** 2026-05-07
**Idea-critic verdict:** KILL the commercial framing. Keep as a paper +
portfolio piece. Reasoning condensed below.

## Why the commercial market is closed

| Player | Funding / Scale | Moat |
|---|---|---|
| **enspired** | €40M Series B Oct 2025; 1 GW BESS managed | Trading-as-a-service revenue share, not software sale |
| **GridBeyond** | $12M D + €52M C; 2.6 GW; Samsung/ABB backed | Vertical integration |
| **Tesla Autobidder** | >100 GWh dispatched | Closed-source, ships with Megapack |
| **Fluence Mosaic** | 16 GW deployed/awarded | Stochastic + ML, hardware-bundled |
| **Wärtsilä GEMS** | >9 GWh monitored | Hardware-bundled |
| **Modo, 3E, Sympower, Enery EPO** | various | Local market expertise |

No underserved segment large enough for a new entrant. Small IPPs
*outsource* dispatch to enspired-style trading-as-a-service rather than
buying software.

## The closest open-source analogue

**PyPSA** — MIT-licensed, 8+ years old, gold standard for power-system
optimization. Commercial layer is consultancy (OET, d-fine, Energynautics,
CLIMACT). No PyPSA acquihire. That's the realistic ceiling for a Python
optimization library in this domain. Consulting also conflicts with the
LBL position.

## Why our Jensen-gap result isn't a moat

- K=4 forecast averaging is straw-man baseline. Production BESS dispatch
  uses 50-200 scenario stochastic LP with copula-correlated forecast
  noise (Birge & Louveaux 1997).
- Cao et al. 2020's 58% RL-vs-MILP is six years old. If end-to-end RL
  were a commercial moat, it would be deployed by now. Public stack
  descriptions still say "ML forecasts feeding stochastic optimization" —
  RL is *not* the dispatch primitive in production.
- Fluence engineer's likely reaction: "we do that, with 200 scenarios and
  risk-aware tilts."

## Other blockers

1. **No moat.** Env + baselines + lit review are months for solo, zero
   days for Tesla. Reproducible from the abstract.
2. **No buyer with money.** Consultancies want free tools, academics have
   no budget, hardware vendors ship own firmware, IPPs outsource.
3. **Degradation-aware dispatch is gated by fleet telemetry, not better
   algorithms.** battery_gym has zero operational data.
4. **LBL conflict-of-interest** is real friction — outside work in a
   subject area related to Lab work requires institutional review and
   approval. LBL has active BESS programs (ESI, EETD).
5. **Strategic opportunity cost.** Commercial path is 12-18 months minimum
   to reach PyPSA-consultancy stage. Displaces the FAR.AI / MATS Autumn /
   Anthropic Fellows track. Per memory, those are Julian's stated near-
   perfect-fit applications.

## What to do instead

- **Submit to a paper venue.** Workshop fits: RLEM @ BuildSys, Climate
  Change AI @ NeurIPS. Journal: Applied Energy, Energy & AI. Headline =
  Jensen-gap on (revenue, D) Pareto under nonlinear cost + forecast noise.
- **Drop hydesign integration as a commercial vehicle.** Keep as a
  research collaboration if useful, no commercial wrapper.
- **Do not engage DTU TTO or LBL outside-employment review.** Both
  signal commitment without proportionate upside.

## Last cheap test before fully closing

Two hours, no code: read enspired's technical blog, 3E's BESS revenue-
optimization white paper, Fluence Mosaic's architecture page. Name **one
specific production feature** that K-scenario QP-ensemble beats. If you
can't name one in 2 hours, the KILL is confirmed and we move on. If you
can, surface it for a reconcile pass.

## Sources

- [enspired €40M Series B](https://www.ess-news.com/2025/10/15/enspired-raises-over-e40-million-to-expand-battery-optimization-beyond-europe/)
- [GridBeyond Series D](https://www.datacenterdynamics.com/en/news/energy-management-startup-gridbeyond-funding-round-raises-12m/)
- [LBL Outside Employment Policy](https://commons.lbl.gov/display/rpm2/Outside+Employment+and+Consulting)
- [Fluence Mosaic](https://fluenceenergy.com/mosaic-intelligent-bidding-software/)
- [Tesla Autobidder](https://www.tesla.com/support/energy/tesla-software/autobidder)
- [PyPSA + OET consultancy](https://openenergytransition.org/)
- [Cao et al. 2020 RL arbitrage](https://www.researchgate.net/publication/340535096_Deep_Reinforcement_Learning-Based_Energy_Storage_Arbitrage_With_Accurate_Lithium-Ion_Battery_Degradation_Model)
- [Energy-Storage.News on degradation operational reality](https://www.energy-storage.news/unfettered-optimism-on-us-bess-degradation-hits-wall-of-operational-reality/)
