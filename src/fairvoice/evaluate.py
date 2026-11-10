"""
src/fairvoice/evaluate.py — Batch evaluation utilities for FairVoice-Lite.

Used by evaluate_baseline.py, evaluate_noise.py, evaluate_model.py.
Provides a single run_evaluation() function that handles the full loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from fairvoice.audio import load_audio
from fairvoice.noise import add_gaussian_noise
from fairvoice.metrics import (
    UtteranceResult,
    EvaluationSummary,
    evaluate_utterance,
    aggregate_results,
    save_results,
    print_summary,
)

logger = logging.getLogger(__name__)


def run_evaluation(
    manifest_csv: str | Path,
    transcribe_fn: Callable[[any], str],
    model_id: str,
    split: str = "validation",
    condition: str = "clean",
    snr_db: Optional[float] = None,
    max_samples: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    prefix: str = "eval",
    target_sr: int = 16_000,
) -> EvaluationSummary:
    """
    Run a full evaluation loop over a manifest CSV.

    Parameters
    ----------
    manifest_csv   : Path to CSV with columns [file_id, transcript, processed_audio_path]
    transcribe_fn  : Callable that takes a np.ndarray waveform and returns a str
    model_id       : Model identifier string for logging/saving
    split          : 'train', 'validation', 'test'
    condition      : e.g. 'clean', '20dB_SNR'
    snr_db         : If not None, add Gaussian noise at this SNR
    max_samples    : Limit number of samples (for quick testing)
    output_dir     : Directory to save results. If None, results are not saved.
    prefix         : Filename prefix for saved results
    target_sr      : Audio sample rate

    Returns
    -------
    EvaluationSummary
    """
    df = pd.read_csv(manifest_csv, encoding="utf-8")
    if max_samples:
        df = df.head(max_samples)
        logger.info("Evaluation limited to %d samples.", max_samples)

    logger.info(
        "Starting evaluation | model=%s | split=%s | condition=%s | n=%d",
        model_id, split, condition, len(df),
    )

    utterance_results: List[UtteranceResult] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}/{condition}"):
        file_id = str(row.get("file_id", row.name))
        audio_path = row["processed_audio_path"]
        reference = str(row["transcript"])

        try:
            waveform, sr = load_audio(audio_path, target_sr=target_sr)

            if snr_db is not None:
                noise_seed = abs(hash(file_id)) % (2 ** 31)
                waveform = add_gaussian_noise(waveform, snr_db=snr_db, random_seed=noise_seed)

            hypothesis = transcribe_fn(waveform)

        except Exception as exc:
            logger.warning("Failed on %s: %s — using empty hypothesis.", file_id, exc)
            hypothesis = ""

        result = evaluate_utterance(file_id, reference, hypothesis)
        utterance_results.append(result)

    summary = aggregate_results(
        utterance_results,
        model_id=model_id,
        split=split,
        condition=condition,
    )

    print_summary(summary)

    if output_dir is not None:
        save_results(utterance_results, summary, output_dir, prefix=prefix)

    return summary
