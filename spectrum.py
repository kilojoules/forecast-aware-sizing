"""Welch power spectral density + dominant-timescale extraction.

Used in the paper's regime characterization to relate price-process
structure to the resolvable-timescale primitive in THEORY_DRAFT.md.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def welch_psd(x: np.ndarray, fs_per_day: float = 24.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD on hourly data, returning (period_hours, PSD).

    Inverts the frequency axis to periods (hours) for natural interpretation.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    nperseg = min(len(x) // 4, 24 * 28)  # ~28 days segment if available
    f, P = welch(x, fs=fs_per_day, nperseg=nperseg, scaling="density")
    # f is cycles/day; period = 24/f_cyc/day = 1/f cycles  (in days)
    # convert to hours: T_hours = 24 / f_cycles_per_day
    with np.errstate(divide="ignore"):
        period_hours = 24.0 / np.where(f > 0, f, np.nan)
    # drop the f=0 / DC bin
    mask = np.isfinite(period_hours)
    return period_hours[mask], P[mask]


def dominant_timescales(period_hours: np.ndarray, psd: np.ndarray,
                         top_k: int = 5) -> list[tuple[float, float]]:
    """Return the top-k peaks in the spectrum sorted by power. Each item is
    (period_hours, psd_value)."""
    # local-max heuristic: keep entries whose PSD > both neighbors
    isort = np.argsort(period_hours)
    p_sorted = period_hours[isort]
    psd_sorted = psd[isort]
    is_peak = np.zeros_like(psd_sorted, dtype=bool)
    is_peak[1:-1] = (psd_sorted[1:-1] > psd_sorted[:-2]) & (psd_sorted[1:-1] > psd_sorted[2:])
    peaks = list(zip(p_sorted[is_peak], psd_sorted[is_peak]))
    peaks.sort(key=lambda x: -x[1])
    return peaks[:top_k]


def report(label: str, x: np.ndarray) -> None:
    period, psd = welch_psd(x)
    peaks = dominant_timescales(period, psd, top_k=5)
    print(f"\n[{label}]  N={len(x)}, mean={x.mean():.1f}, std={x.std():.1f}")
    print(f"  Top spectral peaks (period_h, PSD):")
    for p, val in peaks:
        print(f"    {p:7.1f}h  {val:.2e}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from price_signal import synth_diurnal
    from dk_loader import load_dk_year

    # Synthetic AR(1) diurnal: should peak at 24h, 12h
    x = synth_diurnal(24 * 90, seed=0)  # 90 days
    report("synthetic-diurnal-AR1", x)

    # DK1 each year
    for year in [2021, 2022, 2023]:
        df = load_dk_year(year)
        x = df["da_eur_per_mwh"].to_numpy()
        report(f"DK1-{year}", x)
