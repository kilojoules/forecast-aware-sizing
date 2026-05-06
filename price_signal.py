"""Price signal generators for PriceEnv.

Two synthetic processes:
  - synth_diurnal: 24-h diurnal cycle + AR(1) noise (default for sanity tests)
  - make_forecast: realized + AR(1) noise (forecast = realized + corr noise)

CSV loader (load_csv) is left as a thin stub for real-prices integration
in phase 3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def synth_diurnal(T: int, seed: int = 42, base: float = 50.0,
                  diurnal_amp: float = 35.0, evening_amp: float = 15.0,
                  ar1_std: float = 8.0, ar1_rho: float = 0.7) -> np.ndarray:
    """Synthetic hourly price: morning + evening peaks + AR(1) noise.

    Roughly mimics northern-European DA spot price shape.
    """
    rng = np.random.default_rng(seed)
    h = np.arange(T)
    diurnal = (
        base
        + diurnal_amp * np.sin(2 * np.pi * (h - 8) / 24)
        + evening_amp * np.sin(2 * np.pi * (h - 19) / 24 * 2)
    )
    noise = np.zeros(T)
    for t in range(1, T):
        noise[t] = ar1_rho * noise[t-1] + rng.normal(0, ar1_std)
    return diurnal + noise


def make_forecast(realized: np.ndarray, noise_std: float, seed: int,
                  rho: float = 0.5) -> np.ndarray:
    """Forecast = realized + AR(1) noise with stationary marginal std `noise_std`."""
    rng = np.random.default_rng(seed)
    fc_noise = np.zeros_like(realized)
    sigma_innov = noise_std * np.sqrt(max(1 - rho ** 2, 1e-9))
    for t in range(1, len(realized)):
        fc_noise[t] = rho * fc_noise[t-1] + rng.normal(0, sigma_innov)
    return realized + fc_noise


def load_csv(path: str | Path, column: str = "price") -> np.ndarray:
    """Load a price CSV (DK1, EPEX, PJM). Phase-3 stub."""
    import pandas as pd
    df = pd.read_csv(path)
    return df[column].to_numpy(dtype=float)
