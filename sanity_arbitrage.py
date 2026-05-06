"""Day-1 sanity check: does in-loop rainflow open a Pareto gap vs LP-with-linear-penalty?

Single battery, hourly trace, three dispatch policies on the same (price_t):
  1. LP-perfect-foresight + linear cycling penalty  (hydesign-style baseline)
     Sweep mu to trace the (Revenue, D) Pareto frontier.
  2. Hand-crafted "sparse-deep-cycle" heuristic  (k charge hours / k discharge hours)
     A simple alternative with deep cycles, sweep k.
  3. Hand-crafted "broad-shallow-cycle" heuristic  (charge below pct, discharge above)
     Threshold-driven, sweep threshold.

For each, compute lifetime rainflow D on the resulting SoC trace.
Plot Pareto frontier. Pareto-dominance check: do (2) or (3) reach (R,D) points that no LP-mu can reach?

This is the Day-1 falsifier from HYDESIGN_PILOTS.md / litreview.tex pilot list.

Run:
  python sanity_arbitrage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from degradation import cycle_degradation


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
T = 168                # 1 week, hourly
B_E = 2.0              # MWh capacity  (1-h battery)
B_P = 2.0              # MW charge/discharge limit  (full discharge in 1 h)
ETA = 1.0              # round-trip efficiency (sanity: 1.0 to keep LP clean)
SOC0 = 0.5 * B_E       # initial SoC
SEED = 42


# -----------------------------------------------------------------------------
# Synthetic price trace -- mean-reverting + diurnal
# -----------------------------------------------------------------------------
def synth_price(T: int, seed: int = 42) -> np.ndarray:
    """Diurnal AR(1) price: morning + evening peaks, AR-noise around the cycle."""
    rng = np.random.default_rng(seed)
    hours = np.arange(T)
    diurnal = (
        50.0
        + 35.0 * np.sin(2 * np.pi * (hours - 8) / 24)        # main daily wave
        + 15.0 * np.sin(2 * np.pi * (hours - 19) / 24 * 2)   # evening sub-peak
    )
    noise = np.zeros(T)
    rho = 0.7
    for t in range(1, T):
        noise[t] = rho * noise[t-1] + rng.normal(0, 8.0)
    return diurnal + noise


# -----------------------------------------------------------------------------
# LP with linear cycling penalty (perfect-foresight)
# -----------------------------------------------------------------------------
def lp_dispatch(price: np.ndarray, mu: float) -> np.ndarray:
    """
    Variables (length 2T):  [P_charge[0..T-1], P_discharge[0..T-1]]
    All >= 0.  Power into battery = P_charge - P_discharge.
    Net dispatch (sold to grid) = P_discharge - P_charge.
    Cycling proxy = mu * sum(P_charge + P_discharge).

    Maximize  sum(price * (P_dis - P_chg)) - mu * sum(P_chg + P_dis)
       <==>   minimize  -price . P_dis + price . P_chg + mu * 1 . (P_chg + P_dis)

    s.t.  P_chg, P_dis in [0, B_P]
          0 <= SoC0 + sum_{s<=t}(P_chg[s] - P_dis[s]) <= B_E   for each t
    Returns SoC trajectory length T+1.
    """
    n = 2 * T
    # Objective: minimize  c . x
    c = np.concatenate([price + mu, -price + mu])  # P_chg coeff, P_dis coeff
    # Bounds
    bounds = [(0, B_P)] * n
    # SoC constraints: 0 <= SoC0 + cumsum(P_chg - P_dis) <= B_E
    A_ub = np.zeros((2 * T, n))
    b_ub = np.zeros(2 * T)
    for t in range(T):
        # upper: SoC0 + sum(P_chg[0..t]) - sum(P_dis[0..t]) <= B_E
        A_ub[t, 0:t+1] = 1.0          # P_chg
        A_ub[t, T:T+t+1] = -1.0       # -P_dis
        b_ub[t] = B_E - SOC0
        # lower: -[SoC0 + sum(P_chg) - sum(P_dis)] <= 0  ==>  -SoC0 - cumsum <= 0
        A_ub[T+t, 0:t+1] = -1.0
        A_ub[T+t, T:T+t+1] = 1.0
        b_ub[T+t] = SOC0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    P_chg = res.x[:T]
    P_dis = res.x[T:]
    soc = SOC0 + np.cumsum(P_chg - P_dis)
    return np.concatenate([[SOC0], soc]), P_chg, P_dis


def revenue(price: np.ndarray, P_chg: np.ndarray, P_dis: np.ndarray) -> float:
    return float(np.sum(price * (P_dis - P_chg)))


def D_of(soc: np.ndarray, B: float) -> float:
    try:
        D, _, _ = cycle_degradation(soc, B)
    except (IndexError, ValueError):
        return 0.0
    return D


# -----------------------------------------------------------------------------
# Heuristic 1: sparse-deep-cycle (k cheapest hours charge full, k priciest discharge)
# -----------------------------------------------------------------------------
def heuristic_sparse(price: np.ndarray, k: int):
    P_chg = np.zeros(T)
    P_dis = np.zeros(T)
    sorted_idx = np.argsort(price)
    cheap = sorted_idx[:k]
    expensive = sorted_idx[-k:]
    P_chg[cheap] = B_P
    P_dis[expensive] = B_P
    soc = SOC0 + np.cumsum(P_chg - P_dis)
    soc = np.clip(soc, 0, B_E)
    # if SoC clipped, recompute clipped power
    P_eff = np.diff(np.concatenate([[SOC0], soc]))
    P_chg_eff = np.maximum(P_eff, 0)
    P_dis_eff = np.maximum(-P_eff, 0)
    return np.concatenate([[SOC0], soc]), P_chg_eff, P_dis_eff


# -----------------------------------------------------------------------------
# Heuristic 2: threshold (charge below low pct, discharge above high pct)
# -----------------------------------------------------------------------------
def heuristic_threshold(price: np.ndarray, low_pct: float, high_pct: float):
    p_low = np.percentile(price, low_pct)
    p_high = np.percentile(price, high_pct)
    soc = SOC0
    socs = [soc]
    P_chg = np.zeros(T)
    P_dis = np.zeros(T)
    for t in range(T):
        if price[t] < p_low and soc < B_E:
            chg = min(B_P, B_E - soc)
            P_chg[t] = chg
            soc += chg
        elif price[t] > p_high and soc > 0:
            dis = min(B_P, soc)
            P_dis[t] = dis
            soc -= dis
        socs.append(soc)
    return np.array(socs), P_chg, P_dis


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def main():
    print("Day-1 sanity check: in-loop rainflow vs LP-linear-penalty Pareto gap?")
    print(f"T={T}h  B_E={B_E}MWh  B_P={B_P}MW  seed={SEED}\n")
    price = synth_price(T, seed=SEED)
    print(f"Price trace: mean={price.mean():.1f} std={price.std():.1f} "
          f"min={price.min():.1f} max={price.max():.1f}")

    # 1) LP sweep mu wide -- 0 then log-spaced to past max price
    mus = np.concatenate([[0.0], np.logspace(-1, np.log10(price.max()*1.5), 30)])
    lp_pts = []
    for mu in mus:
        soc, P_chg, P_dis = lp_dispatch(price, mu)
        R = revenue(price, P_chg, P_dis)
        D = D_of(soc, B_E)
        lp_pts.append((R, D, mu))
    lp_pts = np.array(lp_pts)
    print(f"\nLP sweep mu in [0, 30], {len(mus)} points")
    print(f"  R range: {lp_pts[:,0].min():.0f} .. {lp_pts[:,0].max():.0f}")
    print(f"  D range: {lp_pts[:,1].min():.4f} .. {lp_pts[:,1].max():.4f}")

    # 2) Sparse heuristic sweep k in [1..30]
    sparse_pts = []
    for k in range(1, 31):
        soc, P_chg, P_dis = heuristic_sparse(price, k)
        R = revenue(price, P_chg, P_dis)
        D = D_of(soc, B_E)
        sparse_pts.append((R, D, k))
    sparse_pts = np.array(sparse_pts)
    print(f"\nSparse heuristic sweep k in [1,30]")
    print(f"  R range: {sparse_pts[:,0].min():.0f} .. {sparse_pts[:,0].max():.0f}")
    print(f"  D range: {sparse_pts[:,1].min():.4f} .. {sparse_pts[:,1].max():.4f}")

    # 3) Threshold heuristic sweep (low,high pct grid)
    thr_pts = []
    for low in [10, 20, 30, 40]:
        for high in [60, 70, 80, 90]:
            soc, P_chg, P_dis = heuristic_threshold(price, low, high)
            R = revenue(price, P_chg, P_dis)
            D = D_of(soc, B_E)
            thr_pts.append((R, D, low, high))
    thr_pts = np.array(thr_pts)
    print(f"\nThreshold heuristic sweep ({len(thr_pts)} pts)")
    print(f"  R range: {thr_pts[:,0].min():.0f} .. {thr_pts[:,0].max():.0f}")
    print(f"  D range: {thr_pts[:,1].min():.4f} .. {thr_pts[:,1].max():.4f}")

    # ---- Same-R comparison: at LP's revenue, what's heuristic D? --------
    print("\nSame-revenue comparison (interpolate heuristic D at LP's R values):")
    # Pool all heuristic points
    h_pts = np.vstack([sparse_pts[:, :2], thr_pts[:, :2]])
    # For each LP point, find any heuristic with R within 5% and compare D
    same_r_diff = []
    for R_lp, D_lp, mu in lp_pts:
        if R_lp < 100:
            continue
        mask = np.abs(h_pts[:, 0] - R_lp) / max(abs(R_lp), 1.0) < 0.05
        if mask.sum() == 0:
            continue
        D_h_min = h_pts[mask, 1].min()
        rel = (D_h_min - D_lp) / max(D_lp, 1e-9) * 100
        same_r_diff.append(rel)
        if abs(rel) > 5:
            print(f"  R={R_lp:7.0f} mu={mu:5.1f}: LP D={D_lp:.5f}  best heur D={D_h_min:.5f}  ({rel:+.1f}%)")
    if same_r_diff:
        print(f"  Median (heur - LP)/LP * 100 at matched R: {np.median(same_r_diff):+.1f}%")

    # ---- Pareto-dominance check ------------------------------------------------
    # For each non-LP point, check if any LP point Pareto-dominates it (R_lp >= R AND D_lp <= D)
    def lp_dominates(R, D):
        return np.any((lp_pts[:, 0] >= R - 1e-6) & (lp_pts[:, 1] <= D + 1e-9)
                       & ((lp_pts[:, 0] > R + 1e-6) | (lp_pts[:, 1] < D - 1e-9)))
    sparse_dominated = sum(lp_dominates(R, D) for R, D, _ in sparse_pts)
    thr_dominated = sum(lp_dominates(R, D) for R, D, *_ in thr_pts)
    print(f"\nPareto-dominance check (LP dominates non-LP point?):")
    print(f"  Sparse: {sparse_dominated}/{len(sparse_pts)} dominated by LP frontier")
    print(f"  Threshold: {thr_dominated}/{len(thr_pts)} dominated by LP frontier")

    # And vice versa: any non-LP that beats every LP at its (R_or_better, D_or_better) cell?
    def beats_lp(R, D):
        # strictly dominates SOME LP point: R > R_lp AND D < D_lp
        return np.any((lp_pts[:, 0] < R - 1e-6) & (lp_pts[:, 1] > D + 1e-9))
    sparse_beats = sum(beats_lp(R, D) for R, D, _ in sparse_pts)
    thr_beats = sum(beats_lp(R, D) for R, D, *_ in thr_pts)
    print(f"  Sparse: {sparse_beats}/{len(sparse_pts)} strictly dominate some LP point")
    print(f"  Threshold: {thr_beats}/{len(thr_pts)} strictly dominate some LP point")

    # Headline: any non-LP point that is NOT dominated by any LP point?
    def lp_strictly_better_or_equal_anywhere(R, D):
        return np.any((lp_pts[:, 0] >= R - 1e-6) & (lp_pts[:, 1] <= D + 1e-9))
    sparse_off_frontier = sum(not lp_strictly_better_or_equal_anywhere(R, D)
                              for R, D, _ in sparse_pts)
    thr_off_frontier = sum(not lp_strictly_better_or_equal_anywhere(R, D)
                           for R, D, *_ in thr_pts)
    print(f"\n  Off-LP-frontier points (not weakly dominated by any LP point):")
    print(f"    Sparse: {sparse_off_frontier}/{len(sparse_pts)}")
    print(f"    Threshold: {thr_off_frontier}/{len(thr_pts)}")

    # ---- Plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lp_pts[:, 0], lp_pts[:, 1], c="#4477aa", s=60, marker="o",
               label=f"LP linear-penalty (sweep mu, n={len(lp_pts)})", zorder=3)
    ax.plot(lp_pts[:, 0], lp_pts[:, 1], color="#4477aa", lw=1, alpha=0.5, zorder=2)
    ax.scatter(sparse_pts[:, 0], sparse_pts[:, 1], c="#cc6677", s=40, marker="s",
               label=f"Sparse-deep heuristic (sweep k, n={len(sparse_pts)})", zorder=4)
    ax.scatter(thr_pts[:, 0], thr_pts[:, 1], c="#117733", s=40, marker="^",
               label=f"Threshold heuristic (low/high grid, n={len(thr_pts)})", zorder=4)
    ax.set_xlabel("Cumulative revenue (price units)")
    ax.set_ylabel("Rainflow D (battery-1 capacity-normalized)")
    ax.set_title(
        f"Pareto: revenue vs rainflow D\n"
        f"T={T}h, B_E={B_E}MWh, B_P={B_P}MW.  "
        f"If heuristics fall ON the LP curve -> linear penalty captures rainflow.  "
        f"If heuristics give Pareto-better points -> RL has room.")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = "fig_arbitrage_pareto.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nWrote {out}")

    # Verdict
    print("\n" + "="*70)
    if sparse_off_frontier + thr_off_frontier == 0:
        print("VERDICT: LP-linear-penalty traces out the entire frontier.")
        print("  No room for in-loop rainflow at hourly resolution on this trace.")
        print("  -> Day-1 says: KILL the rework. Linear penalty is enough.")
    elif sparse_off_frontier + thr_off_frontier > 5:
        print(f"VERDICT: {sparse_off_frontier + thr_off_frontier} non-LP points lie off the LP frontier.")
        print("  In-loop rainflow opens a Pareto gap. RL has room.")
        print("  -> Day-1 says: GO. Refactor warranted.")
    else:
        print(f"VERDICT: {sparse_off_frontier + thr_off_frontier} non-LP points off frontier (small).")
        print("  Marginal gap. Rerun with more diverse heuristics + longer horizon before committing.")


if __name__ == "__main__":
    main()
