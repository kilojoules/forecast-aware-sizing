"""Rainflow-counted degradation evaluator (fatpack-based).

Footnote 7 of the paper: stress function f(delta) = (k1 * delta^k2 + k3)^-1
with k1 = 1.4e5, k2 = -0.501, k3 = -1.23e5. delta is DoD as fraction of capacity.

`fatpack.find_rainflow_cycles` returns closed cycles only; `find_rainflow_ranges`
adds residue half-cycles weighted at 0.5.
"""
from __future__ import annotations

import numpy as np
import fatpack

K1 = 1.4e5
K2 = -0.501
K3 = -1.23e5

DELTA_FLOOR = 1e-3   # avoid singularity at delta -> 0
DELTA_CEIL = 1.29    # denominator changes sign beyond this


def stress(delta: float) -> float:
    delta = max(min(delta, DELTA_CEIL - 1e-3), DELTA_FLOOR)
    denom = K1 * (delta ** K2) + K3
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def cycle_degradation(soc_trajectory: list[int], capacity: int) -> tuple[float, list[float], list[float]]:
    """Run rainflow on integer SoC trajectory, return (D_i, dod_fractions, counts)."""
    if capacity <= 0 or len(soc_trajectory) < 3:
        return 0.0, [], []
    y = np.asarray(soc_trajectory, dtype=float)
    if y.max() == y.min():
        return 0.0, [], []
    reversals, _ = fatpack.find_reversals(y)
    if len(reversals) < 2:
        return 0.0, [], []
    cycles, residue = fatpack.find_rainflow_cycles(reversals)
    ranges = np.abs(cycles[:, 1] - cycles[:, 0]) if cycles.size else np.array([])
    closed_counts = np.ones_like(ranges)
    # residue: open half-cycles, weight 0.5 each
    res_ranges = np.abs(np.diff(residue)) if len(residue) >= 2 else np.array([])
    res_counts = 0.5 * np.ones_like(res_ranges)
    all_ranges = np.concatenate([ranges, res_ranges])
    all_counts = np.concatenate([closed_counts, res_counts])
    D = 0.0
    dods, counts = [], []
    for rng_, count in zip(all_ranges, all_counts):
        delta = rng_ / capacity
        D += stress(delta) * count
        dods.append(float(delta))
        counts.append(float(count))
    return D, dods, counts


def fleet_degradation(soc_logs: list[list[int]], capacities: list[int]):
    Ds = []
    histos = []
    for traj, B in zip(soc_logs, capacities):
        D, dods, counts = cycle_degradation(traj, B)
        Ds.append(D)
        histos.append((dods, counts))
    return Ds, histos
