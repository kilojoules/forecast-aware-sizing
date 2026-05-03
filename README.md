# battery_gym

Reproduction of Srinivasa, Deulkar, Bhargava, Hajiesmaili, Shenoy 2026,
*"Degradation-Aware Frequency Regulation of a Heterogeneous Battery Fleet via
Reinforcement Learning"* (arXiv:2601.22865v2).

## Files

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

## Reproduce

```bash
pip install rainflow numpy matplotlib
python3 run.py --B 2 3 --c 2 3 --d 2 3 --T 100000
python3 plot_results.py results.json fig_dod.png
```

## Validation outcome (B=(2,3), c=d=(2,3), T=10^5, 3 signal seeds: 42, 7, 123)

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

## Implementation note: SP-update timing

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

## Files produced

- `results.json` — per-agent metrics + full SoC traces from the latest run.
- `fig_dod_tuned.png` — DoD histograms equivalent to paper Fig 2.

## LUMI run (T=1M, 2 seeds, het fleets)

10-task array on `small` partition. Job 18086082. ~hour each, all in parallel.

| Config | Naive D | Greedy D | ELM-RL D | ELM vs Naive | ELM vs Greedy |
|---|---:|---:|---:|---:|---:|
| B=(2, 20)   | 3.688 ± 0.010 | 1.825 ± 0.002 | 1.858 ± 0.004 | **−49.6%** | −1.8% |
| B=(5, 20)   | 2.374 ± 0.005 | 1.770 ± 0.001 | 1.949 ± 0.077 | **−17.9%** | −10.1% |
| B=(2, 50)   | 3.134 ± 0.005 | 0.788 ± 0.009 | 0.792 ± 0.010 | **−74.7%** | −0.5% |
| B=(5, 50)   | 1.796 ± 0.002 | 0.771 ± 0.008 | 0.816 ± 0.009 | **−54.6%** | −5.9% |
| B=(10,100)  | 0.963 ± 0.003 | 0.378 ± 0.001 | 0.399 ± 0.002 | **−58.6%** | −5.6% |

ELM holds at scale: B=(10,100) still −58.6% over Naive, within 5.6% of Greedy.
Cross-seed σ tiny (~0.005-0.08). See `fig_lumi_summary.png`.

## Big het runs (T=500k, single seed, post-fix ELM)

These four configs are what `lumi_run.sh` will scale up.

| Config | Naive D | Greedy D | ELM-RL D | ELM vs Naive | ELM vs Greedy |
|--------|--------:|---------:|---------:|-------------:|--------------:|
| B=(2,20) | 1.84 | 0.91 | 0.93 | **−49.5%** | −2.2% |
| B=(2,50) | 1.57 | 0.39 | 0.40 | **−74.2%** | −2.0% |
| B=(5,20) | 1.19 | 0.88 | 0.98 | −17.6% | −11.5% |
| B=(5,50) | 0.90 | 0.39 | 0.41 | −54.4% | −5.9% |

## Heterogeneous fleet result (the LUMI use case)

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

## Pre-LUMI checklist

- [x] Closed-form sanity (`tests/sanity.py` — 11 tests, all passing). Covers
  stress-fn monotonicity, rainflow on constant / single-cycle / N-fixed-depth
  trajectories, SP-tracker invariants, env SoC update, feasible-action
  enumeration, zero-policy → D=0.
- [x] Multi-seed reporting (`multi_seed.py`).
- [x] ELM hyperparam sweep (`elm_sweep.py`) at B=(2,3), T=30k screening.
- [x] Mid-scale check (`scale_check.py`) at B=(10,10) and (20,20) before LUMI.
- [x] Pixi env (`pixi.toml`).
- [x] Slurm submission script (`lumi_run.sh`) — single-node CPU, OMP threads via
  `$SLURM_CPUS_PER_TASK`. Edit `--account`, partition, modules.
- [x] ELM checkpoint + per-step training log hooks (`agent.train(log_every=...,
  checkpoint_path=...)`).
