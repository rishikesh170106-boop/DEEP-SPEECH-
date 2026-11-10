"""
dataset.py — Telugu ASR PyTorch Dataset for FairVoice-Lite.

Loads audio files and transcripts from a CSV manifest.
Applies preprocessing and feature extraction on-the-fly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from fairvoice.audio import load_audio, validate_audio
from fairvoice.text import normalize_text
from fairvoice.noise import add_gaussian_noise

logger = logging.getLogger(__name__)


class TeluguASRDataset(Dataset):
    """
    Dataset for Telugu ASR fine-tuning.

    Each item returns:
        input_values : Tensor (samples,)  — processor-extracted features
        labels       : Tensor (seq_len,)  — CTC label token IDs
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        processor: Wav2Vec2Processor,
        audio_col: str = "processed_audio_path",
        transcript_col: str = "transcript",
        target_sr: int = 16_000,
        max_duration: float = 20.0,
        augment_snr_db: Optional[Union[float, List[float]]] = None,
        augment_prob: float = 0.5,
        random_seed: int = 42,
    ) -> None:
        self.processor = processor
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.augment_snr_db = augment_snr_db
        self.augment_prob = augment_prob
        self.rng = np.random.default_rng(random_seed)

        df = pd.read_csv(manifest_csv, encoding="utf-8")
        df = df.dropna(subset=[audio_col, transcript_col])
        df[transcript_col] = df[transcript_col].apply(normalize_text)
        df = df[df[transcript_col].str.len() > 0]
        self.records = df[[audio_col, transcript_col]].reset_index(drop=True)

        logger.info("Dataset loaded: %d samples from %s (augment_snr=%s)", len(self.records), manifest_csv, self.augment_snr_db)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.records.iloc[idx]
        audio_path = row.iloc[0]
        transcript = row.iloc[1]

        # Load audio
        try:
            waveform, sr = load_audio(audio_path, target_sr=self.target_sr)
        except Exception as exc:
            logger.error("Failed to load %s: %s", audio_path, exc)
            # Return empty item; collate_fn should handle this
            raise

        # Optional noise augmentation (variable SNR support)
        if self.augment_snr_db is not None:
            if self.rng.random() < self.augment_prob:
                seed = int(self.rng.integers(0, 2**31))
                if isinstance(self.augment_snr_db, (list, tuple)):
                    snr = float(self.rng.choice(self.augment_snr_db))
                else:
                    snr = float(self.augment_snr_db)
                waveform = add_gaussian_noise(waveform, snr_db=snr, random_seed=seed)

        # Extract features using processor
        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)  # (samples,)

        # Encode labels
        with self.processor.as_target_processor():
            labels = self.processor(transcript, return_tensors="pt").input_ids.squeeze(0)

        return {"input_values": input_values, "labels": labels}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length audio and labels."""
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad input_values
    max_len = max(x.shape[0] for x in input_values)
    padded_inputs = torch.zeros(len(batch), max_len)
    for i, x in enumerate(input_values):
        padded_inputs[i, : x.shape[0]] = x

    # Pad labels with -100 (ignored by CTC loss)
    max_label_len = max(y.shape[0] for y in labels)
    padded_labels = torch.full((len(batch), max_label_len), fill_value=-100, dtype=torch.long)
    for i, y in enumerate(labels):
        padded_labels[i, : y.shape[0]] = y

    return {
        "input_values": padded_inputs,
        "labels": padded_labels,
    }
