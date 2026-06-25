# forecast-aware-sizing

**Do you need good forecasts to pick the right battery size?**

You're building a battery (maybe next to a wind farm). Two separate
decisions:

- **How to operate it** — better price forecasts always pay here:
  0.9–37% more lifetime revenue in our tests.
- **How big to build it** — surprisingly, *for any forecast at least
  as good as naive persistence* (i.e. the whole range a real operator
  occupies), forecast quality does **not** change this answer. The
  cheap deterministic dispatch model inside academic sizing tools picks
  the same capacity as a stochastic dispatcher. (Push the forecast
  *worse* than persistence and the optimal battery does shrink — but
  that range loses money anyway, so no real operator is there.) The
  invariance breaks for a different reason: a **penalty for energy you
  promised but didn't deliver**.

Once that imbalance penalty enters the divergence band (opening around
25–35 €/MWh in normal years, and as low as 10–15 €/MWh in the 2022
crisis year for wind-heavy plants), better forecasts buy you a
*smaller battery* — up to half the capacity for the same job:

![NPV at the optimal size vs the penalty for energy promised but not delivered. Grey bands: where a cheap forecast and a good forecast disagree on the best battery size.](paper/figures/fig_readme_npv.png)

*Each panel: lifetime NPV at the best battery size, as the imbalance
penalty grows, for a cheap point forecast (blue) vs a better ensemble
forecast (orange). In the grey bands the two pick different optimal
sizes — that's where forecast quality drives the capacity decision.*

**Where does reality sit?** We settled the same plant against *actual*
Danish (DK1) imbalance prices (effective penalty 11–28 €/MWh in
2021–2023). In normal years, cheap-forecast sizing got the capacity
right at every plant configuration tested. In the 2022 crisis year,
**wind-heavy plants split: the cheap forecast buys 8 MWh where the
ensemble buys 4** under real two-price settlement. The March-2025
Nordic balancing reforms sustain crisis-level spreads (92–123 €/MWh
average up-regulation) — the regime 2022 previewed is plausibly the
new normal:

(Measured per plant configuration and against both real settlement
regimes — see `paper/figures/fig_paper_real_imbalance.png` and §5.2 of
the paper.)

**How much is knowing the future worth?** Plan a year of dispatch for
the same 16 MWh battery (DK1 2022) two ways: a **deterministic** plan
that optimizes against one assumed trajectory (the ensemble mean) as
if it were certain, and a **robust** plan that hedges across the K=4
ensemble members (max-min). Then evaluate each plan on its assumed
future, across the ensemble of futures, and on what actually happened:

| plan ↓ / evaluated on → | its assumed future | the 4 ensemble members | realized 2022 |
|---|---|---|---|
| **deterministic** (treats the mean as truth) | €346.5k | mean €346.5k ± 12.1k, worst €326.4k | €293.9k |
| **robust** (max-min across members) | €340.7k | mean €340.7k ± 3.3k, worst €337.5k | €287.1k |

Read it row against row: the robust plan gives up €6k of average to
buy an €11k-better floor and a 4× tighter spread across futures —
insurance, priced. (For the deterministic row, "assumed" and
"ensemble mean" coincide because revenue is linear in price: planning
on the mean *is* the expectation-optimal plan.) And read the last
column against the first: both plans promise ≈€340–347k but collect
≈€290k, while a perfect-foresight oracle collects €418.6k — the
actual 2022 lay outside what any of the past-pattern members
predicted. Forecast uncertainty costs ~30% of the oracle's revenue,
and no amount of clever planning over a bad ensemble buys it back.

Here is what those information levels look like on the worst week of
the crisis:

![Battery state of charge over a 2022 spike week: perfect foresight, cheap forecast, ensemble.](paper/figures/fig_readme_soc.png)

Perfect foresight (grey, dashed) cycles hard *and* times every swing
right — week revenue €21.1k. The cheap forecast (blue) cycles just as
hard but chases day-old price patterns, slamming into full/empty at
the wrong moments — €14.5k. The ensemble (orange) hedges across
several past patterns, cycles shallower, and lands in between —
€17.3k. Under a delivery penalty the rail-slamming costs twice: wrong
arbitrage timing *and* no headroom left to absorb wind-forecast
misses. That's why the cheap-forecast plant needs a bigger battery
once penalties bite — and why better forecasts substitute for
capacity.

---

## Two loops: scheduling vs sizing

Battery economics is two nested optimizations, and this repo is about
whether they can be **decoupled**:

- **Inner loop — scheduling (dispatch).** Given a *fixed* battery, how
  to operate it hour to hour under uncertain prices and wind.
  Re-solved continuously in production. This is where forecast quality
  and stochastic optimization live (Tesla Autobidder, Fluence Mosaic,
  Wärtsilä GEMS). It sets your **operating revenue**.
- **Outer loop — sizing.** The one-shot capacity investment ($b_E$
  MWh, $b_P$ MW). Evaluated by running the inner loop $10^3$–$10^4$
  times over candidate sizes and picking the NPV argmax. This is what
  academic sizing tools (HyDesign, REopt, PyPSA) do — with a *cheap,
  deterministic* inner loop, because a stochastic one would cost
  100–1000× more per evaluation.

**The question:** does the *fidelity of the inner loop* change the
*outer-loop answer* (the optimal size)?

**The finding:** in the pure-merchant limit, **no**. Better forecasts
and stochastic dispatch make you **more money** (0.9–37% NPV uplift;
in stochastic-programming terms the value of the stochastic solution
and the expected value of perfect information are both positive for
*operation*) but they do **not** move the optimal capacity (VSS and
EVPI for *sizing* are zero). The two loops decouple — **in the merchant
limit (no delivery penalty), and for any forecast at least as good as
persistence, you can size with the cheap deterministic inner loop and
operate with the fancy stochastic one.** (A continuous forecast-error
dial confirms it: optimal size is flat across the whole realistic skill
range and only shrinks once forecasts are worse than persistence, which
loses money anyway.) The loops re-couple when an imbalance penalty for
undelivered energy pushes the plant into a divergence band — and even
then it is two *other* axes, not forecast quality, that move sizing most:

1. **The imbalance penalty**, above a band that opens at ≈25–35 €/MWh
   in normal years and ≈10–15 €/MWh in a crisis year for wind-heavy
   plants.
2. **Risk attitude.** Switching the sizing objective from expected NPV
   to CVaR (worst-case-averse) halves-to-quarters the optimal battery —
   a *bigger* swing than forecast quality ever produces. And a
   pessimist who designs around the worst-case forecast sizes *smaller*
   still, and loses money doing it (see "three reference strategies"
   below). Pick your risk measure before you argue about your forecast.

---

## Background

Why was this ever in doubt? Because the two communities that touch this
problem run different math. Academic sizing tools (DTU hydesign, NREL
REopt, PyPSA) embed deterministic-LP inner dispatch with point
forecasts; commercial operators (Tesla Autobidder, Fluence Mosaic,
Wärtsilä GEMS) dispatch tens of GW with ML-driven stochastic
optimization. Until now there was no empirical test of whether the
academic shortcut produces the wrong **capacity** recommendation.

**Answer.** With no imbalance settlement (the pure-merchant limit),
optimal capacity is identical across cheap and stochastic dispatch in
**17 of 18 regimes** on DK1 and ERCOT North Hub, 2021–2023 — including
the 2022 EU energy crisis and Storm Uri. The one break (DK1 2022 under
a K=20 quantile ensemble) is the diagnostic firing correctly on the
most-stressed regime. With imbalance settlement, **divergence bands**
appear — penalty ranges where a single-forecast plant sizes bigger
than an ensemble one — opening at ≈25–35 €/MWh in normal years and
≈10–15 €/MWh in the 2022 crisis year for wind-heavy plants. Against
*actual* eSett DK1 settlement: normal years stay invariant at every
configuration; **the 2022 crisis year already split wind-heavy
two-price plants (single 8 MWh vs ensemble 4 MWh)**. The March-2025
Nordic reforms sustain crisis-level spreads, so that divergence regime
is plausibly the post-reform normal.

Full writeup: `paper/paper.pdf` (22 pages, submission draft). Every
headline number is reproducible from the committed `results/` JSONs
(see `sizing/paper_regret_ci.py` for the bootstrap/regret audit).

### Three reference strategies (and a pessimist that sizes the wrong way)

For the co-located wind+battery system with imbalance settlement we run
three reference dispatch strategies (`sizing/paper_three_baselines.py`):

1. **Perfect foresight** — bid realized wind, dispatch on realized
   prices. Residual ≡ 0, no imbalance cost, NPV flat in the penalty.
   The clairvoyant upper bound (the merchant optimizer expanded with
   the imbalance market and settlement mechanics).
2. **Honest forecast-error** — bid the forecast at gate closure, let
   the battery correct the error at the grid point in real time
   (`residual = bid − realized + BESS_correction`). The "single" and
   "ensemble" policies.
3. **Adversarial forecast-error** — feed a *deterministic worst-case*
   input within a physical-plausibility (SCADA) band. `adv_design` bids
   the per-hour **lower bound** on delivered power (a pessimist
   designing around a floor, no peeking); `adv_stress` bids the band
   endpoint that maximizes imbalance volume (an adversary maximizing
   the plant's imbalance cost / TSO balancing procurement).

NPV orders cleanly — perfect > ensemble ≥ single > adv_design >
adv_stress — but the **sizing is the surprise**: the pessimist
(`adv_design`) sizes *smaller* than the honest policies, not larger
(ratio 20, λ=200, 2022: 24 MWh vs the honest 48). Bidding a delivery
floor makes the imbalance chronic and one-signed, the battery
saturates, and extra capacity can't chase a deficit the bid itself
created — so the pessimist eats the penalty instead of sizing up.
**"Design around the worst case" is not conservative here; it is
wrong.** The pessimist↔adversary sizing span exceeds the
single-vs-ensemble gap at every λ > 0 — decision attitude moves
wind-coupled sizing more than forecast quality does.

## Recommendations for the sector

- **Sizing-tool builders (HyDesign / REopt / PyPSA).** Your
  deterministic-LP inner dispatch is *empirically safe for capacity* in
  the merchant limit — keep it; do not pay 100–1000× for a stochastic
  inner loop just to size. **But** add the cheap diagnostic and the
  settlement term before trusting a size for an imbalance-exposed,
  wind-heavy plant: that is the regime where the shortcut breaks.
- **Dispatch-optimizer vendors (Tesla / Fluence / Wärtsilä).**
  Stochastic dispatch is justified by *operating revenue* (positive
  VSS/EVPI everywhere), not by sizing. Mosaic-style pre-construction
  "better sizing" claims do not hold in the merchant limit, where the
  sizing value of stochastic simulation is ≈0 — sell the ops uplift,
  which is real (up to 37%).
- **HPP developers with imbalance exposure.** Forecast quality buys a
  capacity reduction *only inside the divergence band*. Wind-heavy
  ratios plus crisis-level spreads (post-2025 Nordic) put you in it —
  size with an ensemble/settlement-aware loop, because a single-forecast
  sizing exercise can buy you **twice** the battery you need.
- **Anyone sizing under risk.** Choose your risk measure *first*:
  expected-NPV vs CVaR moves the optimal battery more than any forecast
  upgrade. And do not naively "design around the worst case" — the
  pessimist sizes smaller and loses money.
- **Run it on your own market.** The $b_{\mathrm{sat}}^{\epsilon}$
  overlap test needs one year of price data and a laptop; it returns
  {invariance survives, disjoint, inconclusive} for *your* market
  rather than ours.

## In context of prior work

This sits between two established literatures and reconciles a
contradiction running between them.

- **Why invariance is even plausible.** Sioshansi et al. (2009) showed
  a naive backcast captures ~85% of perfect-foresight arbitrage value —
  if forecasts barely change *revenue*, the NPV-vs-capacity surface
  barely shifts, so the *argmax* can hold. We make that argmax
  invariance an explicit, falsifiable test.
- **Forecast-error-driven storage sizing** (Bludszuweit &
  Domínguez-Navarro 2011; Haessig et al. 2015; Pinson et al. 2007 on
  quantile bids; Mokhtare et al. 2024 on penalty sweeps) established
  the *direction* — worse forecasts need more storage under penalties.
  Our contribution is locating the **policy-divergence bands** (where
  single and ensemble sizing actually part) and their dependence on
  plant configuration and year, on real settlement data.
- **Fidelity-sensitive sizing** in load-serving and system-scale
  settings (Dinh et al. 2025, 22% loss from simplified sizing; Cauz et
  al. 2023, MILP-vs-RL investment differences; Schmidt 2025, limited
  foresight shifts LDES mix) reaches the *opposite* answer — but all of
  it lives on the *penalized* side of our boundary. The regime map says
  when each answer applies.
- **The closest "cheap-inner-loop, same size" result**, BOOST (Chehade
  & Karaki 2025), shows LP-vs-MILP picks the same microgrid size — but
  varies *constraint* fidelity under perfect foresight, not *forecast*
  fidelity across real market years, and offers no market diagnostic.
- **Stochastic-programming framing.** We report the canonical
  decomposition (Birge & Louveaux): value of the stochastic solution
  and expected value of perfect information are positive for operation,
  zero for capacity in the merchant limit.
- **Reconciling our own group.** Assaad et al. (2025, shared author)
  argued low-fidelity EMS yields "faulty sizing evaluations." We refine
  it: true for operating NPV and above the penalty band — *not* for
  capacity in the merchant limit, where the cheap inner loop was already
  enough.

## Heilmeier Catechism

**What are we trying to do?** Tell a battery developer whether their sizing tool's perfect-foresight-LP assumption is good enough on their specific market — without making them rerun the whole sizing exercise under a stochastic dispatcher.

**How is it done today.** Two camps. Academic sizing tools (hydesign, REopt, PyPSA) call a deterministic LP inner solver $10^3$–$10^4$ times per design evaluation; switching to a stochastic LP costs $100$–$1000\times$ more compute per evaluation. Commercial operators run ML-driven stochastic optimization at dispatch time (tens of GW deployed). Two communities, two architectures, no empirical test of whether the academic shortcut affects the capacity recommendation it produces.

**What is new.** (1) A practitioner-runnable regime-classification diagnostic — the $b_{\mathrm{sat}}^{\epsilon}$ overlap test — computable on one year of price data; returns {invariance survives, disjoint, inconclusive}. (2) First empirical sweep comparing deterministic-LP vs.\ stochastic dispatch sizing on DK1 + ERCOT North Hub, 2021-2023, with three orthogonal LUMI HPC stress tests (2-D $(b_E, b_P)$ surface, K=20 quantile-regression ensemble, N=50 scenario SLP) plus post-hoc max-min-robust and CVaR risk sweeps. (3) Off-the-shelf-cost characterization of hydesign defaults applied to a merchant battery. (4) Imbalance-penalty extension that recovers forecast-quality-dependent divergence bands in sizing, mapped across wind/battery ratios, under corrected residual-at-DA settlement. (5) Real-settlement anchor: the same residuals settled against actual eSett DK1 imbalance prices — normal years invariant, the 2022 crisis year splitting wind-heavy plants. (6) The stochastic-programming decomposition (VSS/EVPI: positive for operation, zero for merchant capacity) and the three-baseline framework (perfect / honest / adversarial), showing decision attitude outweighs forecast quality for sizing.

**Who cares.** Anyone running hydesign / REopt / PyPSA for grid-tied battery sizing (40+ GW of hybrid-power-plant projects in pipeline use these tools). Knowing whether your market sits below or above $\lambda^*$ tells you whether your deterministic-LP sizing answer is robust or off by $\sim 50\%$ on capacity.

**Risks.** (1) Persistence ensemble is a weak stochastic baseline; richer forecasts can break invariance — partially confirmed by quantile-regression ensemble on DK1 2022; wind-side skilled forecasts at $\lambda > 0$ untested. (2) Only DK1 + ERCOT North Hub tested; CAISO, PJM, Nord Pool intraday have different structure; real-settlement anchor is DK1-only and hourly (post-2025 15-minute settlement adds intra-hour residuals this model omits). (3) Divergence-band onset depends on plant configuration — non-increasing in wind/battery ratio, reaching the realized-spread range in the crisis year at wind-heavy ratios. (4) Factorial dispatch is open-loop per 8-week chunk (anticipative beyond 24 h); revenue *levels* are upper bounds, though the nonanticipative rolling SLP confirms the *argmax*. (5) Real-time recourse is "absorb always," value-destroying at small spreads; a price-rational recourse rule is future work.

**Cost.** Diagnostic runs on a laptop in minutes. Stress tests $\sim 50$ node-h on LUMI HPC (research allocation, no marginal cost).

**Time.** Pre-registered design, dataset acquisition (Energinet + gridstatus + eSett), the dispatch-policy factorial, three LUMI stress tests, hydesign baseline integration, the imbalance-penalty + real-settlement extensions, the stochastic-programming decomposition and three-baseline framework, and three rounds of adversarial review (each of which materially changed the numbers — the settlement-accounting fix came out of the last). Submission draft ready; targeting a journal (Energy & AI / Applied Energy / Wind Energy Science) rather than a 4-page workshop, since the diagnostic and stress battery are the contribution.

**Mid-term exam.** Diagnostic returns "invariance survives" on synthetic AR(1) where invariance must hold by construction. ✓
**Final exam.** Diagnostic correctly fires "disjoint" on the one stress-test regime (of 18) where sizing actually shifts (DK1 2022 quantile ensemble). ✓ Divergence bands reproducible across years and wind ratios. ✓ Under *real* eSett DK1 settlement the diagnostic's prediction holds on both sides: normal years invariant at every configuration, and the 2022 crisis year splits wind-heavy two-price plants (single 8 MWh vs ensemble 4) exactly where the bands say it should. ✓

## Headline results

| Test | Result |
|---|---|
| Argmax invariance, persistence ensemble, 6 (market, year) | **6/6 survive** at $\lambda=0$ |
| Argmax invariance, 3 LUMI stress tests × 6 regimes (18 total) | **17/18 survive**; DK1 2022 quantile-K=20 breaks |
| Max-min robust dispatch (post-hoc, worst-case across K=4 members) | same $b_E^*$ in **5/6 regimes**; ERCOT-2021/Uri lands on the quadratic policies' optimum |
| Hydesign-default operational constraints vs unrestricted LP | **5.5–35.9% NPV gap** at argmax; $b_E^*$ shifts 2/6 regimes |
| Imbalance divergence bands (wind + 1 MW battery, DK1, corrected settlement) | open at **≈25–35 €/MWh** normal years, **≈10–15** in 2022 crisis at ratios ≥ 10 |
| Bands vs wind/battery ratio (W = 1/2/5/10/20 MW) | onset non-increasing in ratio; reaches realized-spread range in the crisis year |
| Real DK1 settlement (eSett two-price + one-price), 2021–23 | effective penalty **11–28 €/MWh**; normal years invariant; 2022 wind-heavy splits only under a *counterfactual* two-price regime (retired Nov 2021), not one-price |
| Continuous forecast-error dial γ (perfect → persistence → worse) | b_E\* **flat across the realistic skill range** (γ≤1), shrinks only for worse-than-persistence forecasts (which lose money) |
| Stochastic-programming decomposition (WS/RP/EEV) | VSS & EVPI **positive for operation, zero for merchant capacity** |
| CVaR sizing sweep (mean → CVaR₀.₈₅) | optimal $b_E$ **halves-to-quarters** — risk attitude moves sizing more than forecast quality |
| Three baselines (perfect / honest / adversarial) | NPV: perfect > ensemble ≥ single > pessimist > adversary; **pessimist sizes smaller, not larger** |
| Settlement-accounting fix (residual valued at DA) | removed an overbidder money-pump the quantile bidder exposed; all imbalance results regenerated |
| Settlement-aware reserve dispatch (ρ ≤ 0.3) + single-site wind errors (γ ≤ 3) | reserve never pays below λ=500, bands unchanged; single-site errors push the **crisis-year** band into the realized-spread range |
| Real settlement at wind-heavy ratios (10:1, 20:1) | normal years invariant; **2022 splits 8 vs 4 MWh under the *counterfactual* two-price regime only**, not the in-force one-price |
| Quantile (newsvendor) wind bidding from the K=4 ensemble | within ±0.5% NPV of mean bidding; bands unchanged (member spread too narrow) |
| Operational stochastic-dispatch realized-NPV uplift at $b_E^*$ | **0.9–37%** across (market, year) |

## Repo layout

(formerly `battery_gym`; prior RL-ELM degradation reproduction kept under `rl_elm/`.)

```
forecast-aware-sizing/
├── paper/              workshop paper source + figures
│   ├── paper.tex
│   ├── paper.pdf
│   └── figures/        fig_paper_*.png  (referenced via \graphicspath)
├── sizing/             workshop paper code (flat package)
│   ├── env.py, arbitrage_agents.py, b_sat_classifier.py
│   ├── dk_loader.py, ercot_loader.py, price_signal.py, spectrum.py
│   ├── hydesign_merchant_fork.py, hydesign_local_check.py
│   ├── paper_benchmark.py, paper_2d_task.py, paper_quantile.py,
│   ├── paper_slp.py, paper_timeseries.py, paper_hydesign.py,
│   ├── paper_imbalance.py, paper_real_imbalance.py, ratio_sweep_cell.py,
│   ├── paper_regret_ci.py     bootstrap/regret + replacement audit
│   ├── paper_robust.py        max-min robust dispatch sweep
│   ├── paper_ws_cvar.py       WS/RP/EEV decomposition + CVaR risk sweep
│   ├── paper_quantile_bid.py  newsvendor wind bidding
│   ├── paper_settlement_aware.py  reserve bands + wind-error inflation
│   ├── paper_three_baselines.py   perfect / honest / adversarial
│   ├── paper_stress_figures.py, paper_figures.py, real_imbalance_figures.py
│   └── sanity_*.py
├── rl_elm/             prior project: degradation-aware RL (see below)
├── results/            tracked JSON outputs (2d/, quantile/, slp/,
│                       hydesign/, imbalance/, main/, gbar/)
├── scripts/lumi/       Slurm submission scripts (2d, quantile, slp, run)
├── scripts/gbar/       LSF (DTU gbar) scripts; memrun.sh RSS watchdog
├── docs/memos/         design + kill memos
├── docs/preregistrations/  pre-registered amendments
├── tests/              pytest entry points
├── data/               raw market data (gitignored cache)
└── README.md, pixi.toml
```

## Reproduce

```bash
pixi install
# Workshop paper §4 (main invariance test)
pixi run python sizing/paper_benchmark.py
# §4.6 hydesign-default off-the-shelf baseline
pixi run python sizing/paper_hydesign.py --source dk1 --year 2022 \
    --out results/hydesign/dk1_2022.json
# §5.2 imbalance-penalty break-point
pixi run python sizing/paper_imbalance.py --year 2022 \
    --out results/imbalance/dk1_2022.json
# All paper figures (writes into paper/figures/)
pixi run python sizing/paper_stress_figures.py
# Compile paper.pdf  (graphicspath = {figures/})
cd paper && pdflatex paper.tex
```

LUMI stress tests (§4-§5): `sbatch scripts/lumi/2d.sh`, `scripts/lumi/quantile.sh`, `scripts/lumi/slp.sh`. gbar (DTU): `scripts/gbar/run.sh deploy`.

---

## Prior work in this repo: RL-ELM degradation-aware regulation

Reproduction of Srinivasa, Deulkar, Bhargava, Hajiesmaili, Shenoy 2026,
*"Degradation-Aware Frequency Regulation of a Heterogeneous Battery Fleet via
Reinforcement Learning"* (arXiv:2601.22865v2). Self-contained below.

### Files

- `env.py` — fleet MDP env. Per-battery ramp + capacity (eq 1, 3), collective
  regulation tracking (eq 6), deterministic SoC update (eq 2), online rainflow
  switching-point tracker (sec 3.5).
- `reg_signal.py` — Markov regulation signal over `S_r = {-4, -1, 1, 5}`. (Paper
  does not specify the transition matrix; we use a sticky mean-reverting DTMC.)
- `degradation.py` — rainflow cycle counting + footnote-7 stress function
  `f(δ) = (k1·δ^k2 + k3)^-1` with `k1=1.4e5, k2=-0.501, k3=-1.23e5`.
- `agents.py` — Naive (eq 19), Greedy (eq 20), Tabular Q-learning (sec 4.1),
  ELM-RL (sec 4.4-4.5) with SiLU activation, fixed random `W,b`, replay buffer,
  minibatch semi-gradient TD.
- `run.py` — driver. Trains each agent, evaluates on a fresh signal seed, computes
  accumulated reward + per-battery rainflow degradation + DoD histogram.
- `plot_results.py` — DoD histogram (Fig 2-style).

### Reproduce

```bash
pip install rainflow numpy matplotlib
python3 rl_elm/run.py --B 2 3 --c 2 3 --d 2 3 --T 100000
python3 rl_elm/plot_results.py results.json fig_dod.png
```

### Validation outcome (B=(2,3), c=d=(2,3), T=10^5, 3 signal seeds: 42, 7, 123)

| Agent     | Reward             | D₁              | D₂              | D₁+D₂           |
|-----------|-------------------:|----------------:|----------------:|----------------:|
| Naive     | -6044.6 ± 4.4      | 0.288 ± 0.001   | 0.244 ± 0.001   | 0.532 ± 0.001   |
| Greedy    | -6017.6 ± 3.9      | 0.282 ± 0.001   | 0.233 ± 0.001   | 0.515 ± 0.000   |
| Tabular Q | -6034.4 ± 21.1     | 0.256 ± 0.054   | 0.252 ± 0.033   | **0.508 ± 0.022** |
| ELM-RL    | -6027.5 ± 4.4      | 0.299 ± 0.014   | 0.235 ± 0.005   | 0.534 ± 0.009   |

What matches the paper qualitatively:
- Sanity: my feasible-action count summed over `S` is 77 for B=(2,3); paper's
  Table-1 entry `|S × A| = 420` is the **unconstrained** product
  `∏(B_i+1)·∏(c_i+d_i+1)` (ignores the eq-6 collective constraint and `|S_r|`).
- Greedy reliably beats Naive (~3% reduction in summed degradation, σ = 0.001).
  Direction matches Table 1.
- Tabular Q is the best agent on summed degradation (0.508 vs Naive 0.532).
  Order is **Tabular Q < Greedy < Naive**, matching the paper's tabular row.
- DoD histograms (`fig_dod_tuned.png`) reproduce Fig 2 qualitatively. Tabular Q
  shifts cycle mass from `DoD ≈ 1.0` toward `DoD ≈ 0.5–0.7`.
- Stress function reproduces depth-monotone shape from Wankmüller et al. 2017
  (footnote 7).

What does **not** match the paper:
- **ELM-RL does not improve over Naive at this small config** (0.534 vs 0.532).
  Paper Table 1 row (2,3) reports `D ≈ 0.02` per battery for RL-ELM — a ~85%
  improvement over Naive. With my hyperparameters (hidden=100, α=2e-3, ε₀=0.6,
  linear decay) ELM's bias and variance are both worse than tabular's at this
  scale. The framework is correct (tabular wins), but ELM's headline advantage
  in the paper (scaling to large state spaces) cannot be quantitatively replicated
  without fitting the unspecified hyperparameters.
- **Tabular Q has high cross-seed variance** in per-battery D (σ ≈ 0.05 vs
  Naive σ ≈ 0.001). It often converges to an asymmetric policy that uses one
  battery as the variation buffer; the asymmetry direction flips between seeds,
  so the mean-per-battery is balanced even though individual seeds are not.
- **Absolute D values are ~10× larger than the paper's**. The stress-function
  constants from footnote 7 are reproduced exactly, but the paper does not
  pin down the units convention (δ as fraction vs. percent); the unspecified
  Markov transition matrix for the toy signal also differs from theirs.

### Implementation note: SP-update timing

Eq (9) reads `r(t) = -(α e^{β|b(t)+a(t)-b_SP(t)|} - α e^{β|b(t)-b_SP(t)|})`.
A literal "compute reward, then update SP" ordering is **gameable**: after a
direction reversal, `b_SP` slides to the just-passed extremum, and the next
reversal step rewards the agent `+α(e^{β·B}−1)` for collapsing the deviation
back to zero — sustained full-amplitude oscillation pays positive reward every
step. Tabular Q under that ordering pegs at DoD=1.0 and degrades worse than
Naive.

The intended ordering (per §3.5: "switching points can be detected and updated
incrementally by monitoring changes in the direction of SoC evolution") is to
update the rainflow stack **first** with the new SoC, push the freshly-confirmed
extremum, then evaluate the reward against that updated SP. Under this ordering,
every step of full-amplitude oscillation costs `α(e^{β·B}−1)`; depth-1
oscillation costs `α(e^β−1)` per step; no motion costs 0. RL prefers shallow
cycles. `env.step()` and `agents.proxy_reward_for_action()` use this ordering.

### Files produced

- `results.json` — per-agent metrics + full SoC traces from the latest run.
- `fig_dod_tuned.png` — DoD histograms equivalent to paper Fig 2.
- `fig_headline.png` — repro.py output: per-battery DoD at B=(10,100).
- `fig_soc_traces.png` — Naive vs ELM SoC time series, B=(10,100), 1000 steps.
- `fig_dod_grid.png` — 5-config × 2-battery DoD grid (Naive vs ELM).
- `fig_action_match.png` — bar chart: ELM matches Greedy 95.7-99.6% on held-out states.
- `fig_proxy_vs_d.png` — scatter of cumulative proxy R vs rainflow D across 5 configs × 2 seeds.

Regenerate the last four with `python plot_all.py` (uses cached run artifacts).

### Honest framing

ELM-RL is, in practice, a **smooth function-approximator copy of Greedy** on
this proxy reward. Action-match analysis (`action_match.py`) shows ELM picks
the same action as Greedy 95.7-99.6% of the time on held-out trajectories at
B=(2,50) and B=(10,100). Per-step degradation is identical to 4 decimals.

| Config         | Seed | ELM≡Greedy match | sign-match | D_ELM | D_Greedy |
|----------------|-----:|----------------:|-----------:|------:|---------:|
| B=(2, 50)      | 42   | **99.5%**       | 99.8%      | 0.0165| 0.0165   |
| B=(2, 50)      |  7   | **99.6%**       | 99.9%      | 0.0170| 0.0170   |
| B=(10, 100)    | 42   | 95.7%           | 97.8%      | 0.0086| 0.0085   |
| B=(10, 100)    |  7   | 97.4%           | 98.8%      | 0.0085| 0.0085   |

So the headline result ("ELM beats Naive 17-74%") is real, but the mechanism
is `ELM ≈ Greedy ≫ Naive`, not multi-step temporal-difference credit
assignment. The win comes from the **proxy reward** plus the heterogeneous
fleet structure -- not from RL discovering a non-myopic policy. RL would only
provide additional value if (a) the reward were sparse (e.g., true rainflow
on cycle completion), (b) the dynamics had hidden state RL needs to model, or
(c) the action enumeration itself were intractable.

Note that ELM does **not** beat Greedy on compute either: both enumerate
feasible actions per state. ELM's per-action cost is one ELM forward pass;
Greedy's is one proxy-reward formula evaluation. They scale identically in N.

### Headline (one command, ~5 min on a laptop)

```bash
pixi run python rl_elm/repro.py
```

Trains ELM-RL on B=(10,100) heterogeneous fleet, then plots per-battery DoD
histograms (`fig_headline.png`): Naive over-cycles the small battery (mass at
high DoD); ELM-RL redistributes regulation across the fleet so the small
battery stays shallow. Total degradation drops ~55% vs Naive.

### Heterogeneous fleet result

Symmetric fleets (B=(N,N) with c=d) hit a structural ceiling: at large N,
Naive's proportional allocation is near-optimal under the proxy reward, so
Greedy ≈ Naive and no RL can do better. Heterogeneous fleets (asymmetric
capacity / ramp) break this — Naive over-cycles the small battery.

Heterogeneous results (T=200k, 2 seeds, quad_growth reward, ELM with per-batt
normalization + rich features [(b−sp), |b−sp|, sign, one-hot signal]):

| Config       | Naive D | Greedy D | ELM-RL D | ELM vs Naive | ELM vs Greedy |
|--------------|--------:|---------:|---------:|-------------:|--------------:|
| B=(5, 20)    | 0.474   | 0.353    | 0.391    | **+17.4%**   | −10%          |
| B=(2, 20)    | 0.737   | 0.364    | 0.376    | **+49.0%**   | −3.5%         |

ELM nearly matches the analytic Greedy optimum at maximum heterogeneity.

### Sanity checklist

- [x] Closed-form sanity (`tests/sanity.py` — 11 tests, all passing). Covers
  stress-fn monotonicity, rainflow on constant / single-cycle / N-fixed-depth
  trajectories, SP-tracker invariants, env SoC update, feasible-action
  enumeration, zero-policy → D=0.
- [x] Multi-seed reporting (`multi_seed.py`).
- [x] ELM hyperparam sweep (`elm_sweep.py`) at B=(2,3), T=30k screening.
- [x] Mid-scale check (`scale_check.py`) at B=(10,10) and (20,20).
- [x] Pixi env (`pixi.toml`).
- [x] ELM checkpoint + per-step training log hooks (`agent.train(log_every=...,
  checkpoint_path=...)`).
