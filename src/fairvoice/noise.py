"""
noise.py — Deterministic noise augmentation for FairVoice-Lite.

Implements additive Gaussian noise at controlled SNR levels.
All noise generation uses a fixed random seed for reproducibility.

IMPORTANT: This module generates SYNTHETIC Gaussian noise only.
Do NOT label results as real-world environmental noise robustness.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# SNR levels used in all noise experiments
DEFAULT_SNR_LEVELS_DB = [20, 10, 5, 0]


def add_gaussian_noise(
    waveform: np.ndarray,
    snr_db: float,
    random_seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Add additive white Gaussian noise at a target SNR.

    The noise amplitude is scaled to achieve the desired signal-to-noise ratio.
    Peak clipping is avoided by scaling down before adding noise.

    Parameters
    ----------
    waveform   : np.ndarray  float32, shape (samples,)
    snr_db     : float       Target SNR in decibels (higher = less noise)
    random_seed: int or None Fixed seed for reproducibility

    Returns
    -------
    noisy_waveform : np.ndarray  float32, same shape as input
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    signal_power = np.mean(waveform ** 2)

    if signal_power == 0.0:
        logger.warning("Silent waveform passed to add_gaussian_noise; returning as-is.")
        return waveform.copy()

    # Noise power for target SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)

    noise = rng.normal(0.0, noise_std, size=waveform.shape).astype(np.float32)
    noisy = waveform + noise

    # Clip to [-1, 1] to avoid overflow (with a warning if significant clipping)
    max_abs = np.max(np.abs(noisy))
    if max_abs > 1.0:
        noisy = noisy / max_abs
        logger.debug("Clipping avoided by scaling (max_abs=%.4f) at SNR=%ddB", max_abs, snr_db)

    return noisy.astype(np.float32)


def augment_with_noise(
    waveform: np.ndarray,
    snr_db: float,
    random_seed: int = 42,
) -> np.ndarray:
    """Convenience wrapper around add_gaussian_noise."""
    return add_gaussian_noise(waveform, snr_db=snr_db, random_seed=random_seed)


def snr_label(snr_db: float) -> str:
    """Return a human-readable label for a given SNR level."""
    return f"{int(snr_db)}dB_SNR"


def validate_snr_levels(snr_levels: List[float]) -> None:
    """Raise ValueError if SNR levels are not in expected range."""
    for snr in snr_levels:
        if not (-10 <= snr <= 100):
            raise ValueError(f"Unusual SNR value: {snr} dB. Check your config.")
