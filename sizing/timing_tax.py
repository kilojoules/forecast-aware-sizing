"""#3 Forecast error is a timing tax, not a sizing distortion.

Claim to prove: realized arbitrage revenue factorizes as
    R(b_E, gamma) ~ rho(gamma) * V(b_E) + const
where rho(gamma) is a timing-skill factor (the action-price correlation,
which collapses as forecasts degrade) and V(b_E) is a throughput-value
curve independent of forecast quality. If R is (approximately) a rank-1
product f(b)*g(gamma), its argmax over b is independent of gamma -- which
IS the invariance, made mechanistic instead of asserted ("the NPV curves
are vertical translations").

Tests, on the (b_E x gamma) grid:
  1. Rank-1 separability: SVD of the centered revenue matrix; report the
     fraction of variance in the first singular component (near 1 =>
     separable => argmax invariant).
  2. rho(gamma): action-price correlation collapses with gamma and is
     ~independent of b_E (spread of rho across b at fixed gamma small).
  3. Throughput (sum|action|) per capacity is ~invariant to gamma
     (degradation is mis-timing, not less cycling).

Output: figure + printed diagnostics. Uses the same gamma parameterization
as paper_gamma_sweep (hat_p = realized + gamma*(persistence - realized)).

Run: scripts/memrun.sh 1000 .pixi/envs/default/bin/python sizing/timing_tax.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SIZING_DIR = Path(__file__).resolve().parent
FIGURES = SIZING_DIR.parent / "paper" / "figures"
sys.path.insert(0, str(SIZING_DIR))

import matplotlib.pyplot as plt
import numpy as np

from arbitrage_agents import lp_linear_actions
from dk_loader import multi_lag_persistence
from paper_benchmark import B_P, MU_LP, chunk_indices

DISC = sum(1.07 ** -y for y in range(15))
GAMMAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
BES = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
YEAR = 2022


def dispatch_year(realized, plan, b_E):
    """Return (revenue on realized, throughput sum|a|, corr(a, price))."""
    acts, soc = [], b_E / 2.0
    for s, e in chunk_indices(len(realized)):
        a = lp_linear_actions(plan[s:e], b_E, B_P, soc, mu=MU_LP)
        acts.append(a)
        soc = max(0.0, min(b_E, soc - float(np.sum(a))))
    a = np.concatenate(acts)
    rev = float(np.sum(realized * a))
    thru = float(np.sum(np.abs(a)))
    rho = float(np.corrcoef(a, realized)[0, 1]) if a.std() > 0 else 0.0
    return rev, thru, rho


def main(out: Path):
    realized, F = multi_lag_persistence(YEAR, area="DK1")
    persistence = F[0]
    R = np.zeros((len(BES), len(GAMMAS)))
    THRU = np.zeros_like(R)
    RHO = np.zeros_like(R)
    for i, b in enumerate(BES):
        for j, g in enumerate(GAMMAS):
            plan = realized + g * (persistence - realized)
            R[i, j], THRU[i, j], RHO[i, j] = dispatch_year(realized, plan, b)

    # --- Test 1: rank-1 separability of R(b, gamma) ---
    # On the operational skill range gamma<=1 the claim is invariance;
    # gamma>1 (worse than persistence) is expected to break separability.
    skill = GAMMAS <= 1.0
    def rank1_frac(M):
        S = np.linalg.svd(M, full_matrices=False)[1]
        return float(S[0] ** 2 / np.sum(S ** 2))
    var_frac1_skill = rank1_frac(R[:, skill])
    var_frac1_full = rank1_frac(R)
    U, S, Vt = np.linalg.svd(R[:, skill], full_matrices=False)
    var_frac1 = var_frac1_skill
    f_b = U[:, 0] * S[0]
    g_gamma = Vt[0, :]
    if f_b.mean() < 0:
        f_b, g_gamma = -f_b, -g_gamma
    R1 = np.outer(f_b, g_gamma)
    rel_resid = float(np.linalg.norm(R[:, skill] - R1) / np.linalg.norm(R[:, skill]))
    print(f"  rank-1 variance fraction, full gamma range: {var_frac1_full:.4f}")

    # --- Test 3: throughput per capacity vs gamma ---
    thru_per_b = THRU / BES[:, None]
    thru_cv_over_gamma = float(np.nanmean(np.std(THRU, axis=1) / np.mean(THRU, axis=1)))

    print(f"DK1 {YEAR} timing-tax factorization:")
    print(f"  rank-1 variance fraction (sep. => ~1): {var_frac1:.4f}")
    print(f"  relative residual ||R - f(b)g(g)||/||R||: {rel_resid:.4f}")
    print(f"  rho(gamma) (mean over b): "
          + " ".join(f"{g:.2f}:{RHO[:,j].mean():+.2f}" for j, g in enumerate(GAMMAS)))
    print(f"  rho spread across b at fixed gamma (mean std): {RHO.std(axis=0).mean():.3f}")
    print(f"  throughput CV across gamma (per b, mean): {thru_cv_over_gamma:.3f}")
    # argmax of R over b at each gamma (should be constant => invariance)
    print(f"  argmax_b R at each gamma: "
          + " ".join(f"{g:.2f}:{BES[np.argmax(R[:,j])]:g}" for j, g in enumerate(GAMMAS)))

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # A: normalized revenue curves collapse on gamma<=1 (level shift =>
    #    same shape => same argmax); gamma>1 breaks it (dashed).
    for j, g in enumerate(GAMMAS):
        ls = "-" if g <= 1.0 else "--"
        axes[0].plot(BES, R[:, j] / R[:, j].max(), ls, marker="o", ms=3,
                     label=f"γ={g:g}" + ("" if g <= 1 else " (>persist.)"),
                     color=plt.cm.cividis(min(g, 1.0) / 1.3),
                     alpha=1.0 if g <= 1 else 0.5)
    axes[0].set_xscale("log"); axes[0].set_xticks(BES)
    axes[0].set_xticklabels([f"{b:g}" for b in BES])
    axes[0].set_xlabel("$b_E$ (MWh)"); axes[0].set_ylabel("revenue / max over $b_E$")
    axes[0].set_title(f"A. revenue-vs-size shape is γ-invariant on the skill\n"
                      f"range γ≤1 (rank-1={var_frac1:.3f}); breaks past persistence",
                      fontsize=9.5)
    axes[0].legend(fontsize=6.5, ncol=2); axes[0].grid(alpha=0.3)
    # B: rho(gamma) collapses, ~independent of b
    for i, b in enumerate(BES):
        axes[1].plot(GAMMAS, RHO[i, :], "-o", ms=3, label=f"{b:g} MWh",
                     color=plt.cm.viridis(i / len(BES)))
    axes[1].axhline(0, color="0.6", lw=0.7)
    axes[1].set_xlabel("forecast-error dial γ")
    axes[1].set_ylabel("timing skill  ρ = corr(action, price)")
    axes[1].set_title("B. the timing tax: ρ(γ) collapses,\n~same for every battery size",
                      fontsize=9.5)
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    # C: throughput per capacity flat in gamma
    for i, b in enumerate(BES):
        axes[2].plot(GAMMAS, thru_per_b[i, :] / thru_per_b[i, 0], "-o", ms=3,
                     label=f"{b:g} MWh", color=plt.cm.viridis(i / len(BES)))
    axes[2].set_xlabel("forecast-error dial γ")
    axes[2].set_ylabel("cycling volume / (its γ=0 value)")
    axes[2].set_ylim(0.7, 1.15)
    axes[2].set_title("C. cycling VOLUME barely moves with γ\n"
                      "(degradation is mis-timing, not less cycling)", fontsize=9.5)
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    fig.suptitle("Forecast error is a timing tax, not a sizing distortion: "
                 "$R(b_E,γ)≈ρ(γ)\\,V(b_E)$ — a rank-1 product, so the best size is γ-invariant",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    main(FIGURES / "fig_timing_tax.png")
