# Phase 2 BC: honest negative result

**Date:** 2026-05-06
**Verdict:** Behavior cloning of QP-ensemble actions from a single-forecast
state is **information-theoretically limited**; cannot beat QP-single. Pivot to PPO.

## Setup

`phase2_benchmark.py` v1: lookahead=24, 200 traces, 80 epochs, 64-unit 2-layer MLP.
v2: lookahead=72, 400 traces, 200 epochs, 128-unit 3-layer MLP.

Target: QP-ensemble (K=4) action vectors per trace.
Inputs: (SoC_t / b_E, normalized forecast lookahead window).
Trained on T=168, alpha=0.005, train_noise=8.
Eval: 30 held-out (realized, single-forecast) pairs at noise ∈ {3, 5, 8, 12, 18}.

## Results

### v1 (lookahead=24, smaller MLP)

| Noise | LP-fc | QP-s | QP-e | RL | RL vs QP-s | RL vs QP-e |
|---|---|---|---|---|---|---|
| 3 | 1522 | 1632 | 1661 | 1205 | **−26%** | **−27%** |
| 8 | 1436 | 1417 | 1598 | 1147 | **−19%** | **−28%** |
| 18 | 1089 | 958 | 1384 | 917 | −4% | −34% |

### v2 (lookahead=72, bigger MLP, 200 epochs)

| Noise | LP-fc | QP-s | QP-e | RL | RL vs QP-s | RL vs QP-e |
|---|---|---|---|---|---|---|
| 3 | 1522 | 1632 | 1661 | 1071 | **−34%** | **−36%** |
| 8 | 1436 | 1417 | 1598 | 999 | **−30%** | **−38%** |
| 18 | 1089 | 958 | 1384 | 769 | −20% | −44% |

**v2 is WORSE.** Overfitting:
- v2 train loss collapsed 0.475 → 0.102 over 200 epochs.
- v2 val loss rose 0.551 → 0.823.
- Best val loss reached at ~epoch 20 (early stopping would help marginally).

## Why behavior cloning failed

Information-theoretic argument:
1. The QP-ensemble action at step t depends on the full ensemble forecast
   (K independent samples). Different ensembles → different actions even
   for the same realized prices.
2. The policy state contains a single forecast. It cannot recover the
   ensemble action via any function of (SoC, single_forecast). The best
   it can learn is `E[ensemble_action | single_forecast]`.
3. By Jensen, `E[ensemble_action | single_forecast]` ≈ QP-single's action
   (the deterministic-LP-on-this-forecast solution).
4. Therefore even a perfect BC of QP-ensemble targets converges to
   QP-single performance — which we already know underperforms QP-ensemble.

The Jensen lift requires accessing ensemble information at inference time,
not just at training time. Behavior cloning of ensemble actions from
single-forecast state cannot achieve this.

What v2 demonstrates is the additional failure mode: with stronger
function approximation, the model overfits the noisy targets, generalizes
worse, and falls below QP-single (which is the BC information ceiling).

## What would work

**Proper RL (PPO/SAC).** The agent samples many realized trajectories
during training. The reward signal `R = revenue − α · action²` evaluates
each action against the realized prices, not against a target action. The
policy learns to be robust under the conditional distribution of realized
given forecast — naturally capturing the Jensen lift.

PPO/SAC would target: at test time, given a single forecast, output an
action that is good in expectation under the true conditional
P(realized | forecast). This is precisely what QP-ensemble approximates
via Monte Carlo, and what PPO learns from samples.

Estimated effort: 5-7 days. Need to:
1. Set up gym-compatible PriceEnv with proper reset() / step() / observation_space.
2. Wrap stable-baselines3 PPO or SAC.
3. Train 1-5M steps.
4. Evaluate on same held-out set as phase 2 BC.

## Alternative (cheaper) approaches to consider

1. **DAGGER**: roll out current policy, query QP-ensemble expert for each
   visited state, retrain. Reduces compounding rollout error. Doesn't fix
   the information-limit issue. Modest improvement expected.

2. **Stronger features**: include time-of-day one-hot, summary statistics
   of forecast (mean, std, percentile), longer lookahead. Might lift the
   information ceiling slightly but doesn't change the structural issue.

3. **Simpler policy class** (linear regression): less overfitting, cheaper
   to train, gives a clean lower bound. Would likely match QP-single.

4. **Ensemble at inference time**: at test, generate K synthetic forecasts
   from a learned forecast-conditional distribution, run QP-ensemble.
   Trivial to implement but doesn't really exercise the RL claim.

## Files

- `rl_agent.py` — featurize, MLPPolicy, build_dataset, train_policy,
  policy_dispatch. All retained for the PPO version (state featurization
  reusable).
- `phase2_benchmark.py` — runner. Numbers reproducible.
- `fig_phase2_benchmark.png`, `fig_phase2_train.png` — plots.
- `rl_policy.pt` — trained MLP weights (for reference; not useful).

## Decision

Commit phase 2 BC as a negative-result milestone. Pivot phase 2 v2 to
PPO. Rerun benchmark. If PPO achieves ≥ QP-ensemble, the rework's RL
claim is supported empirically.
