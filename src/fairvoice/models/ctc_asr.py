"""
models/ctc_asr.py — Telugu Wav2Vec2 CTC model loader for FairVoice-Lite.

Handles:
- Loading the pretrained CTC model and processor.
- Verifying model architecture, vocab size, and sample rate.
- Freezing the feature extractor (recommended for small-data fine-tuning).
- Single-file inference.
- Forward/backward pass smoke test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from fairvoice.audio import load_audio
from fairvoice.text import normalize_text, word_delimiter_decode

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "biswa921/wav2vec2-clsril-23-10k-telugu-init"


class TeluguCTCModel:
    """
    Wrapper around Wav2Vec2ForCTC for Telugu ASR.

    Usage
    -----
    model = TeluguCTCModel.load()
    transcript = model.transcribe("path/to/audio.wav")
    """

    def __init__(
        self,
        model: Wav2Vec2ForCTC,
        processor: Wav2Vec2Processor,
        model_id: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.processor = processor
        self.model_id = model_id
        self.device = device

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        freeze_feature_extractor: bool = True,
        cache_dir: Optional[str] = None,
    ) -> "TeluguCTCModel":
        """
        Load processor and model from HuggingFace Hub.

        Parameters
        ----------
        model_id                : HuggingFace model ID
        device                  : 'cuda', 'cpu', or None (auto-detect)
        freeze_feature_extractor: Freeze CNN layers (recommended)
        cache_dir               : Optional custom HF cache directory
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        logger.info("Loading processor: %s", model_id)
        processor = Wav2Vec2Processor.from_pretrained(model_id, cache_dir=cache_dir)

        logger.info("Loading model: %s", model_id)
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        model = model.to(dev)

        if freeze_feature_extractor:
            if hasattr(model, "freeze_feature_encoder"):
                model.freeze_feature_encoder()
            else:
                model.freeze_feature_extractor()
            logger.info("Feature extractor frozen.")

        # Log key properties
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model loaded | Total params: %.1fM | Trainable: %.1fM | Vocab: %d",
            n_params / 1e6,
            n_trainable / 1e6,
            model.config.vocab_size,
        )

        return cls(model=model, processor=processor, model_id=model_id, device=dev)

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: str | Path,
        target_sr: int = 16_000,
    ) -> str:
        """
        Transcribe a single audio file to Telugu text.

        Parameters
        ----------
        audio_path : path to WAV file (mono, 16kHz recommended)
        target_sr  : expected sample rate

        Returns
        -------
        str — decoded Telugu transcript
        """
        waveform, sr = load_audio(audio_path, target_sr=target_sr)

        inputs = self.processor(
            waveform,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.processor.batch_decode(predicted_ids)[0]
        transcript = word_delimiter_decode(transcript)
        transcript = normalize_text(transcript)
        return transcript

    def smoke_test(self, waveform: np.ndarray, sr: int = 16_000) -> dict:
        """
        Run a single forward + backward pass on a waveform.

        Returns a dict with: loss_value, gradients_exist, gradients_finite, success.
        Used before full training to confirm the pipeline works.
        """
        self.model.train()
        self.model.zero_grad()

        inputs = self.processor(
            waveform, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        # Dummy label: encode a simple Telugu word
        dummy_text = "నమస్కారం"
        with self.processor.as_target_processor():
            labels = self.processor(dummy_text, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model(input_values=input_values, labels=labels)
        loss = outputs.loss

        loss_val = loss.item()
        loss_finite = np.isfinite(loss_val)

        if loss_finite:
            loss.backward()

        grad_params = [p for p in self.model.parameters() if p.grad is not None]
        grads_exist = len(grad_params) > 0
        grads_finite = all(torch.isfinite(p.grad).all() for p in grad_params)

        self.model.eval()

        return {
            "loss_value": loss_val,
            "loss_finite": loss_finite,
            "gradients_exist": grads_exist,
            "gradients_finite": grads_finite,
            "success": loss_finite and grads_exist and grads_finite,
        }
