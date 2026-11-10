"""
audio.py — Audio loading, validation, and preprocessing for FairVoice-Lite.

All audio is converted to mono 16 kHz float32 numpy arrays.
No silent failures: every problematic file is logged and excluded with a reason.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

TARGET_SR = 16_000          # Hz
MIN_DURATION = 0.1          # seconds
MAX_DURATION = 30.0         # seconds


def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SR,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, resample to target_sr, and convert to mono.

    Returns
    -------
    waveform : np.ndarray  shape (samples,)  float32
    sample_rate : int

    Raises
    ------
    RuntimeError if the file cannot be read or produces invalid audio.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"soundfile failed on {path}: {exc}") from exc

    # Convert to mono
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform[:, 0]

    # Resample if necessary
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return waveform.astype(np.float32), sr


def validate_audio(
    waveform: np.ndarray,
    sr: int,
    path: str | Path = "",
    min_duration: float = MIN_DURATION,
    max_duration: float = MAX_DURATION,
) -> Tuple[bool, str]:
    """
    Validate a loaded waveform.

    Returns
    -------
    (is_valid, reason)
        reason is empty string when valid.
    """
    duration = len(waveform) / sr

    if len(waveform) == 0:
        return False, "empty waveform"
    if not np.all(np.isfinite(waveform)):
        return False, "contains NaN or Inf"
    if np.max(np.abs(waveform)) == 0.0:
        return False, "silent recording (all zeros)"
    if duration < min_duration:
        return False, f"too short ({duration:.3f}s < {min_duration}s)"
    if duration > max_duration:
        return False, f"too long ({duration:.1f}s > {max_duration}s)"

    return True, ""


def get_duration(path: str | Path, sr: int = TARGET_SR) -> Optional[float]:
    """Return the duration of an audio file in seconds without loading fully."""
    try:
        info = sf.info(str(path))
        return info.duration
    except Exception:
        try:
            waveform, actual_sr = load_audio(path, target_sr=sr)
            return len(waveform) / actual_sr
        except Exception:
            return None
