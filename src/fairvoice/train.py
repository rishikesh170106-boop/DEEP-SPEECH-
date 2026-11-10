"""
src/fairvoice/train.py — Full CTC fine-tuning trainer for FairVoice-Lite.

Wraps HuggingFace Trainer with:
- CTC data collator (handles variable-length padding)
- WER-based best model selection
- Gradient checkpointing DISABLED (avoids the known backward-pass bug)
- Conservative config for RTX 4050 (6GB VRAM)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction
import time
import json

import jiwer
from fairvoice.text import normalize_text, word_delimiter_decode

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Pads input_values and labels for CTC training.
    Labels are padded with -100 (ignored by CTC loss).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad audio inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding token id with -100 (ignored by loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def build_compute_metrics(processor: Wav2Vec2Processor):
    """
    Returns a compute_metrics function for WER-based evaluation.
    Called by Trainer after each eval step.
    """

    def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
        pred_logits = pred.predictions
        if isinstance(pred_logits, tuple):
            pred_logits = pred_logits[0]
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Replace -100 in labels (padding)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        # Normalize
        pred_str_norm  = [normalize_text(word_delimiter_decode(p)) for p in pred_str]
        label_str_norm = [normalize_text(word_delimiter_decode(l)) for l in label_str]

        try:
            wer = jiwer.wer(label_str_norm, pred_str_norm)
        except Exception:
            wer = 1.0

        return {"wer": round(wer, 4)}

    return compute_metrics


def build_training_args(cfg: dict, output_dir: str) -> TrainingArguments:
    """Build TrainingArguments from config dict."""
    t = cfg.get("training", {})

    return TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=t.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        evaluation_strategy=t.get("evaluation_strategy", "steps"),
        num_train_epochs=t.get("num_train_epochs", 5),
        fp16=t.get("fp16", True) and torch.cuda.is_available(),
        save_steps=t.get("save_steps", 100),
        eval_steps=t.get("eval_steps", 100),
        logging_steps=t.get("logging_steps", 10),
        learning_rate=t.get("learning_rate", 3e-5),
        weight_decay=t.get("weight_decay", 0.005),
        warmup_steps=t.get("warmup_steps", 100),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "wer"),
        greater_is_better=t.get("greater_is_better", False),
        # IMPORTANT: gradient_checkpointing must be False
        # The Wav2Vec2 model has a known incompatibility with PyTorch
        # gradient checkpointing that causes backward-pass errors.
        gradient_checkpointing=False,
        report_to="none",
        seed=t.get("seed", 42),
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
    )


class LiveProgressBarCallback(TrainerCallback):
    """
    Console progress bar and live status writer for FairVoice-Lite.
    Writes real-time metrics to reports/training_status.json for web dashboard & UI.
    """
    def __init__(self, status_path=None, bar_len: int = 24):
        self.status_path = Path(status_path) if status_path else None
        self.bar_len = bar_len
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "=" * 65)
        print("  >> FairVoice-Lite: Training Pipeline Engaged")
        print("=" * 65)

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        max_steps = state.max_steps
        if max_steps <= 0:
            return
        pct = current_step / max_steps
        filled = int(self.bar_len * pct)
        bar = "#" * filled + "-" * (self.bar_len - filled)
        elapsed = time.time() - (self.start_time or time.time())
        speed = current_step / elapsed if elapsed > 0 else 0
        rem_sec = (max_steps - current_step) / speed if speed > 0 else 0
        m, s = divmod(int(rem_sec), 60)
        em, es = divmod(int(elapsed), 60)

        last_loss = "N/A"
        last_eval_wer = "N/A"
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry and last_loss == "N/A":
                    last_loss = f"{entry['loss']:.4f}"
                if "eval_wer" in entry and last_eval_wer == "N/A":
                    last_eval_wer = f"{entry['eval_wer']*100:.1f}%"

        print(
            f"\r[{bar}] {pct*100:5.1f}% | Step {current_step:>3}/{max_steps} | "
            f"Ep {state.epoch:.2f} | Loss: {last_loss} | {speed:4.2f} it/s | "
            f"Elapsed: {em:02d}:{es:02d} | ETA: {m:02d}:{s:02d}   ",
            end="",
            flush=True,
        )

        if self.status_path:
            try:
                status = {
                    "step": current_step,
                    "max_steps": max_steps,
                    "percent": round(pct * 100, 1),
                    "epoch": round(state.epoch or 0.0, 2),
                    "loss": last_loss,
                    "eval_wer": last_eval_wer,
                    "speed_steps_sec": round(speed, 2),
                    "elapsed_sec": round(elapsed, 1),
                    "eta_str": f"{m:02d}m {s:02d}s",
                }
                self.status_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.status_path, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2)
            except Exception:
                pass

    def on_train_end(self, args, state, control, **kwargs):
        print("\n\n" + "=" * 65)
        print("  [SUCCESS] FairVoice-Lite: Training Pipeline Finished")
        print("=" * 65 + "\n")
