"""
tests/test_audio.py — Unit tests for audio loading and validation.
Run: pytest tests/
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.audio import validate_audio, TARGET_SR
from fairvoice.noise import add_gaussian_noise


class TestValidateAudio:
    def test_valid_waveform(self):
        waveform = np.random.randn(TARGET_SR * 2).astype(np.float32) * 0.1
        ok, reason = validate_audio(waveform, TARGET_SR)
        assert ok, reason

    def test_empty_waveform(self):
        ok, reason = validate_audio(np.array([], dtype=np.float32), TARGET_SR)
        assert not ok
        assert "empty" in reason

    def test_silent_waveform(self):
        waveform = np.zeros(TARGET_SR * 2, dtype=np.float32)
        ok, reason = validate_audio(waveform, TARGET_SR)
        assert not ok
        assert "silent" in reason

    def test_nan_waveform(self):
        waveform = np.full(TARGET_SR, float("nan"), dtype=np.float32)
        ok, reason = validate_audio(waveform, TARGET_SR)
        assert not ok

    def test_too_short(self):
        waveform = np.random.randn(100).astype(np.float32) * 0.1
        ok, reason = validate_audio(waveform, TARGET_SR)
        assert not ok
        assert "short" in reason

    def test_too_long(self):
        waveform = np.random.randn(TARGET_SR * 40).astype(np.float32) * 0.1
        ok, reason = validate_audio(waveform, TARGET_SR, max_duration=30.0)
        assert not ok
        assert "long" in reason


class TestGaussianNoise:
    def test_output_shape(self):
        waveform = np.random.randn(TARGET_SR).astype(np.float32) * 0.1
        noisy = add_gaussian_noise(waveform, snr_db=10, random_seed=42)
        assert noisy.shape == waveform.shape

    def test_noise_added(self):
        waveform = np.random.randn(TARGET_SR).astype(np.float32) * 0.1
        noisy = add_gaussian_noise(waveform, snr_db=10, random_seed=42)
        assert not np.allclose(waveform, noisy)

    def test_reproducibility(self):
        waveform = np.random.randn(TARGET_SR).astype(np.float32) * 0.1
        n1 = add_gaussian_noise(waveform, snr_db=10, random_seed=42)
        n2 = add_gaussian_noise(waveform, snr_db=10, random_seed=42)
        np.testing.assert_array_equal(n1, n2)

    def test_clipping_avoided(self):
        waveform = np.random.randn(TARGET_SR).astype(np.float32) * 0.9
        noisy = add_gaussian_noise(waveform, snr_db=0, random_seed=42)
        assert np.max(np.abs(noisy)) <= 1.0 + 1e-6

    def test_different_seeds_differ(self):
        waveform = np.random.randn(TARGET_SR).astype(np.float32) * 0.1
        n1 = add_gaussian_noise(waveform, snr_db=10, random_seed=1)
        n2 = add_gaussian_noise(waveform, snr_db=10, random_seed=2)
        assert not np.allclose(n1, n2)
