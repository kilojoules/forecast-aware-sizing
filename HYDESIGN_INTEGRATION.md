# Integrating battery_gym into hydesign

> **STATUS — KILLED 2026-05-05.** Pilot 1 (`HYDESIGN_PILOT1_MEMO.md`) found
> zero published hydesign use cases needing both heterogeneous fleets AND
> sub-hourly markets. hydesign already supports sub-hourly + balancing
> markets on a single aggregate battery; battery_gym's heterogeneous-fleet
> RL story is orthogonal, not complementary. Document retained as the
> design rationale; do not implement.

Goal: make the `battery_gym` RL controller (env + ELM-RL agent) drop cleanly into the hydesign hybrid-plant pipeline, replacing or augmenting the existing `ems` openMDAO component.

## What hydesign exposes

Canonical seam: `hydesign/ems/ems.py:18` — `class ems(om.ExplicitComponent)`.
Wired at `hydesign/assembly/hpp_assembly.py:396` as subsystem `'ems'`.

**Compute contract**
- Inputs (per call): `wind_t`, `solar_t`, `price_t` (length-N hourly arrays, MW / MW / EUR), scalars `b_P` (MW), `b_E` (MWh), `G_MW`, `battery_depth_of_discharge`, `battery_charge_efficiency`, `peak_hr_quantile`, …
- Outputs: `b_t` (battery dispatch [MW], +discharge / −charge), `b_E_SOC_t` (SoC [MWh]), `hpp_t`, `hpp_curt_t`, `penalty_t`.
- Time resolution: 1 h default (`intervals_per_hour=1`); HiFiEMS variant runs 15-min internally.
- Solver: full-horizon LP via CPLEX, week-by-week batches (`batch_size≈110`), no warm-start across batches.

**Degradation feedback**
- Post-hoc rainflow on `b_E_SOC_t` → `battery_degradation.py` → `SoH` → fed into `ems_long_term_operation` for SoH-stepped re-dispatch (loop in `assembly/hpp_assembly.py:614`).
- Degradation **not** in the dispatch policy itself.

**Fleet support:** none. `b_P`, `b_E` are scalars; storage is one aggregate device.

## What battery_gym exposes

- `FleetEnv` (multi-battery), integer SoC units, per-battery ramp + capacity, collective signal-following constraint, regulation signal `S_r = {-4, -1, 1, 5}`, reward = degradation proxy on rainflow switching points.
- Online step-by-step Markov decision process — no lookahead, no price awareness.
- `ELMQAgent`, `GreedyAgent`, `NaiveAgent`. ELM ≈ smooth Greedy.

## Mismatches to bridge

| dimension | hydesign | battery_gym | bridge |
|---|---|---|---|
| Storage shape | scalar (`b_P`, `b_E`) MW/MWh | fleet (`B`, `c`, `d`) integer units | Aggregate fleet → scalar; or extend hydesign to vectorize |
| Time step | 1 h | regulation cadence (sub-min) | Two-tier (hourly EMS sets setpoint, RL fills sub-hour) — preferred |
| Reward | revenue (price·dispatch) − peak penalty | degradation proxy | Joint reward `R = revenue − λ·D_proxy` |
| Lookahead | full-week LP | Markov, no lookahead | Add price/wind/PV forecast features to RL state |
| Degradation | post-hoc rainflow + SoH iter | in-the-loop proxy | Keep hydesign post-hoc; use proxy only as RL signal |
| Driving signal | price + grid cap + peak target | discrete `S_r` regulation signal | RL state = (SoC, price, residual = wind+PV−G_target) |

## Recommended integration: two-tier wedge

Don't replace `ems`. Sit **below** it.

```
[hydesign EMS / LP, hourly]            sets hour-h energy setpoint  E_h*
        │
        ▼
[battery_gym RL / sub-hourly]          tracks E_h* with intra-hour profile
        │                              chosen to minimize rainflow D
        ▼
[battery_degradation / rainflow]       post-hoc on the high-res SoC
```

Rationale: hydesign's strength = revenue-optimal dispatch with full lookahead; battery_gym's strength = degradation-aware shaping at fast timescales. Two-tier matches how grid-tied BESS actually run (market layer + frequency-response layer). Keeps both code-bases monolithic; the wedge is a single new component.

### Minimum work to land it

1. **Wrap battery_gym as openMDAO component.** New file `hydesign/ems/ems_rl_subhourly.py`. Input contract: take `b_t` (hourly setpoints from `ems`) + `b_E_SOC_t[0]`. Output: `b_E_SOC_t_high_res` at `intervals_per_hour=K` (e.g. K=60 for one-min). Internally: at each minute, RL agent picks ramp action that respects per-step ramp limit + integrates to `E_h*` over the hour.

2. **Aggregate-to-fleet mapping.** Decide once: scalar `b_P`/`b_E` → single FleetSpec(B=(B,)) with B inferred from energy capacity at the chosen integer-unit resolution. Or split aggregate into fictitious sub-batteries with the same total capacity (heterogeneous fleet) — **only useful if hydesign actually has heterogeneous physical units**, otherwise it's a synthetic split.

3. **Re-train RL with a setpoint-tracking term.** New reward:
   `R_t = -α·(D_t_proxy) - μ·(E_running − E_target_running)²`
   where `E_target_running` linearly interpolates the hourly setpoint. ELM/ELMQAgent retrain runs already exist locally.

4. **Wire into assembly.** In `hpp_assembly.py:396` block, insert the new component after `ems` and route `ems.b_t` → `ems_rl_subhourly.b_t_setpoints`, then route `ems_rl_subhourly.b_E_SOC_t` (high-res) into `battery_degradation` instead of EMS's hourly trace. Keep degradation iteration loop intact.

### Risks

- **Synthetic fleet split** undermines battery_gym's main result (heterogeneous fleets break Naive). If hydesign's storage is one device, the RL controller has nothing structurally to do beyond minimizing within-hour cycling — which Greedy already does. Need to confirm there's a real heterogeneous fleet in scope (e.g., multi-pack, multi-chemistry, or BESS + flywheel hybrids).
- **Hourly LP already smooths SoC.** If hydesign's hourly setpoint is already low-cycling, the RL controller's improvement headroom shrinks. Need to measure ΔD baseline before committing.
- **CPLEX dependency.** hydesign's `ems` requires CPLEX. battery_gym is dependency-light. Keep new component CPLEX-free.

## Alternative integration: full replacement

Replace `ems` outright with a battery_gym-style RL agent that takes `(SoC, price_forecast, wind, PV, G_MW)` → dispatch.

Cost:
- ELM-Q won't work for this — state is continuous, lookahead is essential. Need PPO or DQN with forecast features.
- Need price-aware reward, peak-hour penalty term, grid-cap projection.
- Loses LP optimality guarantee that hydesign provides; would need to benchmark RL revenue vs CPLEX.
- Re-implements what hydesign already does well.

Recommend **against** this unless the goal is specifically "RL replaces LP for dispatch" research, which is a different paper.

## Alternative integration: benchmark only

Don't integrate code. Use hydesign's CPLEX dispatch as ground truth for revenue; run battery_gym's controller on hydesign's traces (price + wind + PV); compute (revenue, D) Pareto frontier.

Lowest risk, no code change to hydesign. Useful for a paper figure but doesn't make battery_gym available to hydesign users.

## Open questions for the user

1. **Two-tier or replacement?** Two-tier (recommended) keeps hydesign's LP and adds battery_gym below; replacement is a different (and bigger) research project.
2. **Real heterogeneous fleet?** Does any hydesign use case have >1 physical battery with different capacity / ramp / chemistry? If yes, fleet support in hydesign is the unlock and is worth a hydesign PR. If no, battery_gym's multi-battery story doesn't translate cleanly.
3. **Sub-hourly resolution.** What's the right K? hydesign HiFiEMS already supports 15-min; battery_gym was implicitly per-second. Pick a target (1-min? 15-min?) — drives env retraining and signal regeneration.
4. **Driving signal at sub-hourly resolution.** Where does the within-hour residual come from? Synthetic Markov signal (current battery_gym), AGC trace, or from hydesign's own wind/PV high-res inputs? Affects realism of the result.
5. **Reward weights.** `λ` in `R = revenue − λ·D` is the headline knob. Sweep, or fix from a battery-replacement-cost dollar value?
6. **Degradation source of truth.** Keep hydesign's post-hoc rainflow + SoH iteration as authoritative, and use battery_gym's proxy only as a training signal? (Recommended — avoids divergent definitions.)
7. **CPLEX-free path.** Should the new sub-hourly component be CPLEX-free (so battery_gym's lightweight install carries over), or is CPLEX assumed in any integrated install?
8. **Where does it live?** New directory `hydesign/ems/` module, new top-level repo, or git submodule of `battery_gym` into hydesign?

## Suggested first deliverable

Notebook + one-page write-up:
- Pull a 1-week hydesign run (synthetic price + wind + PV).
- Run hydesign EMS to get `b_t` hourly setpoints.
- Run battery_gym RL at 1-min resolution to track those setpoints with degradation proxy.
- Plot: hourly SoC (LP) vs 1-min SoC (RL) overlaid.
- Report: revenue (unchanged by construction) + D_rainflow on each trace.
- If ΔD is meaningful (>10%) → green-light the openMDAO wrapper. If not → kill the integration, do the benchmark-only path.

~3 days of work. De-risks the bigger integration before any hydesign code change.
