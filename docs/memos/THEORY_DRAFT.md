# When does forecast-handling translate uniformly across capacity?

**Date:** 2026-05-08 (v2 — non-circular)
**Goal:** derive a sufficient condition under which the argmax of the
sizing curve is invariant to the choice of dispatch policy. Earn this
condition from price-process primitives, not from "the policies happen
to plateau at the same place."

## Critique of v1: the circular version

Original proposition: argmax invariance holds when $\Delta(b)$ is
locally flat at $b^*$ and replacement counts agree. **Circular.** "Lift
near $b^*$ doesn't shift the argmax" is just argmax invariance restated.

Real content has to derive $b_{\mathrm{sat}}(\pi)$ from primitives. v2:

## Setup (unchanged)

- $b$: battery energy capacity (MWh). Power capacity $b_P$ fixed.
- $\pi$: dispatch policy mapping (state, forecast) $\to$ action.
- $R(b, \pi)$: expected lifetime revenue.
- $C(b) = c_E b + c_P b_P$.
- $\mathrm{NPV}(b, \pi) = R(b, \pi) - C(b)$ (replacements absorbed
  into $C$ for cleanness; revisit below).

## Derivation from price-process primitives

Let $p(t)$ be the realized price process with spectral density
$S_p(\omega)$. A pure single-tone process at frequency $\omega_0$
(period $\tau_0 = 2\pi/\omega_0$) with amplitude $A$ has a peak-to-trough
spread $2A$ over $\tau_0/2$. A battery with power $b_P$ extracting this
arbitrage in one cycle uses energy throughput

$$b_E^{\mathrm{cycle}}(\tau_0) = b_P \cdot \tau_0 / 2.$$

Capacity past this threshold contributes nothing to one-cycle revenue:
$R'(b, \pi) = 0$ for $b > b_E^{\mathrm{cycle}}(\tau_0)$. The marginal
revenue plateau $b_{\mathrm{sat}}$ is exactly $b_P \tau_0 / 2$.

For a multi-timescale process with weight at frequencies
$\{\omega_1, \omega_2, ...\}$, capacity unlocks longer-period arbitrage.
The *resolvable* arbitrage at frequency $\omega_k$ requires the policy
to identify the cycle's peak and trough timing within its forecast
horizon.

**Key primitive: forecast resolution.** Let $q(\omega, \pi) \in [0, 1]$
be the fraction of price variance at frequency $\omega$ that policy
$\pi$ can resolve from its forecast input. For the deterministic
single-forecast policy $\pi_{\mathrm{det}}$ on a forecast with stationary
AR(1) noise of marginal variance $\sigma_f^2$:

$$q(\omega, \pi_{\mathrm{det}}) \approx \frac{S_p(\omega)}{S_p(\omega) + \sigma_f^2 / (1 - \rho_f^2)}$$

(Wiener-filter-style SNR; signal vs noise at each frequency.) Long
timescales — where signal power is concentrated in a few low-frequency
modes — are most affected by the noise floor.

For a $K$-forecast ensemble policy with iid forecast noise samples,
ensemble-mean reduces noise variance by $1/K$:

$$q(\omega, \pi_{\mathrm{ens},K}) \approx \frac{S_p(\omega)}{S_p(\omega) + \sigma_f^2 / (K(1 - \rho_f^2))}.$$

So $q(\omega, \pi_{\mathrm{ens},K}) > q(\omega, \pi_{\mathrm{det}})$ for
all $\omega$, and the gap is widest where signal is comparable to or
weaker than noise.

**Resolvable timescale.** Define $\tau_{\mathrm{res}}(\pi) =
2\pi / \omega_{\mathrm{cut}}(\pi)$, where $\omega_{\mathrm{cut}}$ is the
lowest frequency at which $q$ drops below some threshold (e.g., 0.5).
Past this timescale, the policy cannot reliably identify arbitrage.

## Proposition (revised)

**Proposition (sketch).** Suppose:

(P1) $R(b, \pi)$ is non-decreasing concave in $b$ with plateau at
$b_{\mathrm{sat}}(\pi) = b_P \cdot \tau_{\mathrm{res}}(\pi) / 2$,
where $\tau_{\mathrm{res}}(\pi)$ is the resolvable-timescale primitive
defined above.

(P2) CAPEX is linear in $b$ with constant $c_E > 0$ such that
$c_E < R'(b, \pi)$ for all $b < b_{\mathrm{sat}}(\pi)$ (the battery is
worth building at all).

Then $b^*(\pi) = b_{\mathrm{sat}}(\pi)$, and the sizing argmax is
invariant across policies iff
$\tau_{\mathrm{res}}(\pi_1) = \tau_{\mathrm{res}}(\pi_2)$, which is iff
the price spectrum $S_p$ has all of its dispatch-relevant power at
frequencies above both policies' cutoffs.

**Corollary (regime characterization).**

- *Single-timescale regime* (all $S_p$ power at $\omega \gg
  \omega_{\mathrm{cut}}$): both policies resolve everything;
  $\tau_{\mathrm{res}}$ identical; argmax invariant.
- *Multi-timescale regime* (significant $S_p$ power at frequencies near
  or below $\omega_{\mathrm{cut}}(\pi_{\mathrm{det}})$): the
  deterministic policy fails to resolve longer-period arbitrage;
  $\tau_{\mathrm{res}}(\pi_{\mathrm{ens}}) > \tau_{\mathrm{res}}(\pi_{\mathrm{det}})$;
  argmax shift $\Delta b^* \approx b_P \cdot
  (\tau_{\mathrm{res}}(\pi_{\mathrm{ens}}) - \tau_{\mathrm{res}}(\pi_{\mathrm{det}})) / 2$.

Now $b_{\mathrm{sat}}$ is derived from $(S_p, \sigma_f, \rho_f, K, b_P)$,
not defined as "where the curve happens to flatten." This earns the
condition.

## Consistency with Pilot S1 numerics

Pilot S1 used diurnal AR(1):
$p(t) = \mu + A_1 \sin(2\pi t / 24) + A_2 \sin(4\pi t / 24)
+ \nu_t$,
$\nu_t$ AR(1) at hourly scale. The spectrum $S_p$ has dominant peaks at
$\tau_1 = 24$ h and $\tau_2 = 12$ h plus broadband AR(1) noise. No power
at $\tau > 24$ h beyond what AR(1) carries.

The dominant arbitrage timescale is 24 h. Both single-forecast and
$K{=}4$ ensemble resolve a 24-h cycle robustly at the noise levels
tested (noise std 3-18 vs price amplitude $A_1 = 35$, so SNR at 24 h is
high). $\tau_{\mathrm{res}}$ is identical $\to$ $b_{\mathrm{sat}}$
identical $\to$ argmax invariant. Confirmed empirically.

## What ERCOT 2021-2023 should look like

ERCOT prices have spectral content at:
- Diurnal (24 h), weekly (168 h), seasonal (~2000 h)
- Aperiodic large-amplitude spikes (rare events with broadband
  contribution)

The diagnostic prediction: at moderate forecast noise, $\tau_{\mathrm{res}}(\pi_{\mathrm{det}})$
clips below the weekly mode, while $\pi_{\mathrm{ens}}$ extends past it.
$b_{\mathrm{sat}}$ shifts; argmax shifts; sizing tools using
deterministic inner-LP under-recommend battery capacity by an amount
proportional to $b_P \cdot (168 - 24) / 2 = 72 b_P$ MWh in the
weekly-arbitrage-bound regime.

Concretely for $b_P = 1$ MW: predicted shift $\sim 72$ MWh past
$b_{\mathrm{sat}}(\pi_{\mathrm{det}}) \approx 12$ MWh, i.e., argmax
moves from 12 MWh to ~80 MWh. A factor-of-7 shift.

This is testable on real data with the existing pipeline.

## The diagnostic test (highlighted, standalone)

Independent of whether the proposition holds rigorously, the diagnostic
test stands alone:

**$b_{\mathrm{sat}}^{\epsilon}$ regime classifier.**
For a given price process and dispatch policy $\pi$:

1. Sweep $b$ on a fine linear grid.
2. Estimate $R'(b, \pi)$ via finite differences with bootstrap CIs
   across forecast realizations.
3. Define $b_{\mathrm{sat}}^{\epsilon}(\pi)$ = smallest $b$ where
   $R'(b, \pi) < \epsilon$.

The pair $(b_{\mathrm{sat}}^{\epsilon}(\pi_{\mathrm{det}}),
b_{\mathrm{sat}}^{\epsilon}(\pi_{\mathrm{ens}}))$ classifies the regime:

- Overlap within CIs $\to$ deterministic-LP sizing tools are robust on
  this market.
- Significant gap $\to$ deterministic-LP sizing tools systematically
  under-size battery capacity by approximately
  $b_{\mathrm{sat}}^{\epsilon}(\pi_{\mathrm{ens}}) -
  b_{\mathrm{sat}}^{\epsilon}(\pi_{\mathrm{det}})$ on this market.

**Cheap to compute** (no SLP, no theorem). **Practitioner-runnable**
on their own market data. **Falsifiable.** Lead figure for the paper:
the $b_{\mathrm{sat}}^{\epsilon}$ pair plotted across markets
(synthetic-AR(1), ERCOT, CAISO, Nord Pool intraday).

## Caveats (now including the b_P slice)

1. **Replacement-count step-function** discreteness can mask continuous
   shifts. The proposition treats replacement cost as absorbed into
   linear CAPEX; in practice replacement schedule is discrete and
   policy-dependent. A continuous SoH-decay model would clean this up.

2. **Concavity of $R(b, \pi)$** is assumed. Holds for arbitrage on
   stationary processes; can fail with discrete strategy switching.

3. **One-dimensional sizing slice** (only $b_E$ varies, $b_P$ fixed).
   The proposition's $b_{\mathrm{sat}}(\pi) = b_P \tau_{\mathrm{res}}(\pi)/2$
   linear-in-$b_P$ form predicts that on a 2-D $(b_E, b_P)$ surface,
   invariance might hold along the $b_E$ axis while breaking along the
   $b_P$ axis (a higher-power battery can resolve faster cycles or
   exploit price spikes that the lower-power one cannot, and the
   forecast-quality-dependence of that resolution is a separate effect
   from the capacity-dependence). The full 2-D analysis is left for
   future work; the slice result is what we test.

4. **The Wiener-style $q(\omega, \pi)$ approximation** assumes Gaussian
   linearity. Real forecast errors are fat-tailed and
   regime-conditional; the resolvable-timescale concept holds
   intuitively but the closed form is heuristic.

5. **The threshold parameter $\epsilon$** in $b_{\mathrm{sat}}^{\epsilon}$
   has to be chosen. Sensitivity to $\epsilon$ is a real concern but
   tractable: report results for $\epsilon \in \{0.01, 0.05, 0.1\} \cdot
   c_E$.

## Self-test against reviewer's circularity charge

v2 does derive $b_{\mathrm{sat}}(\pi)$ from primitives:
$(S_p, \sigma_f, \rho_f, K, b_P) \to \tau_{\mathrm{res}}(\pi)
\to b_{\mathrm{sat}}(\pi)$. The proposition then states a relationship
between argmax invariance and $\tau_{\mathrm{res}}$ equality. Whether
$\tau_{\mathrm{res}}$ values agree on a given price process is an
empirical question answerable from the spectrum, not a definitional one.

The Wiener-style derivation of $q(\omega, \pi)$ is the load-bearing
part. If that derivation is too loose for a paper, the proposition
collapses. But the diagnostic $b_{\mathrm{sat}}^{\epsilon}$ test
survives independently as an empirical regime classifier.

## Decision criterion for the paper

- Theory holds up: integrate proposition + diagnostic into paper.tex
  as a new \S{}3 between Methods and Results. Run ERCOT to test the
  factor-of-7 prediction. ~10 days to submission.
- Theory falls apart on closer reading: keep the diagnostic
  $b_{\mathrm{sat}}^{\epsilon}$ as the standalone contribution. Reframe
  paper as "regime classifier for sizing-tool fidelity," empirical only.
  ~7 days to submission.
- Both fall apart: withdraw.

Re-read tomorrow. Most-likely failure mode: the Wiener-filter $q(\omega,
\pi)$ form is too cute for a battery-dispatch context where the
"signal" isn't easily disentangled from the "decision" (the policy is
nonlinear, not a linear filter). Accept that and downgrade to the
empirical diagnostic — it's the gem regardless.
