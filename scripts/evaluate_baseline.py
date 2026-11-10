"""
scripts/evaluate_baseline.py — Baseline ASR evaluation for FairVoice-Lite (E1).

Evaluates openai/whisper-small on the Telugu validation split.
Uses Whisper with forced Telugu language decoding.

Usage:
    python scripts/evaluate_baseline.py --split validation
    python scripts/evaluate_baseline.py --split test
    python scripts/evaluate_baseline.py --split validation --max_samples 50

Note: whisper-small (~244MB) will be downloaded from HuggingFace on first run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.config import PROJECT_ROOT, ensure_dir
from fairvoice.audio import load_audio
from fairvoice.text import normalize_text
from fairvoice.metrics import (
    evaluate_utterance,
    aggregate_results,
    save_results,
    print_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "evaluate_baseline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

MODEL_ID = "openai/whisper-small"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_whisper(device: str):
    """Load Whisper small model."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    logger.info("Loading Whisper processor: %s", MODEL_ID)
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    logger.info("Loading Whisper model: %s", MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Whisper loaded | %.1fM parameters | device=%s", n_params, device)
    return model, processor


@torch.no_grad()
def transcribe_whisper(
    waveform,
    model,
    processor,
    device: str,
    language: str = "te",
    task: str = "transcribe",
) -> str:
    """Transcribe a waveform using Whisper with Telugu forced decoding."""
    inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Forced Telugu decoding
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)

    generated_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=200,
    )
    transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcript.strip()


def main():
    parser = argparse.ArgumentParser(description="Baseline ASR evaluation (Whisper-small)")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples (for quick testing)")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto-detect if omitted)")
    args = parser.parse_args()

    ensure_dir(PROJECT_ROOT / "logs")
    ensure_dir(REPORTS_DIR / "predictions")
    ensure_dir(REPORTS_DIR / "tables")

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load manifest
    csv_path = PROJECT_ROOT / "data" / "metadata" / f"{args.split}.csv"
    if not csv_path.exists():
        logger.error("Manifest not found: %s", csv_path)
        logger.error("Run: python scripts/prepare_data.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8")
    if args.max_samples:
        df = df.head(args.max_samples)
        logger.info("Limited to %d samples for quick evaluation.", args.max_samples)

    logger.info("Evaluating %d samples from '%s' split.", len(df), args.split)

    # Load model
    model, processor = load_whisper(device)

    # Evaluate
    utterance_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
        file_id = str(row.get("file_id", row.name))
        audio_path = row["processed_audio_path"]
        reference = str(row["transcript"])

        try:
            waveform, sr = load_audio(audio_path, target_sr=16_000)
            hypothesis = transcribe_whisper(waveform, model, processor, device)
        except Exception as exc:
            logger.warning("Failed on %s: %s", file_id, exc)
            hypothesis = ""

        result = evaluate_utterance(file_id, reference, hypothesis)
        utterance_results.append(result)

    # Aggregate
    summary = aggregate_results(
        utterance_results,
        model_id=MODEL_ID,
        split=args.split,
        condition="clean",
    )

    print_summary(summary)

    # Save
    prefix = f"baseline_whisper_{args.split}"
    if args.max_samples:
        prefix += f"_n{args.max_samples}"
    save_results(utterance_results, summary, REPORTS_DIR / "predictions", prefix=prefix)

    logger.info("Evaluation complete. Results saved to %s", REPORTS_DIR / "predictions")


if __name__ == "__main__":
    main()
