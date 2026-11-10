"""
scripts/evaluate_noise.py — Noise robustness evaluation for FairVoice-Lite (E2).

Evaluates the baseline model under synthetic Gaussian noise at multiple SNR levels.
Uses the same model and normalization as evaluate_baseline.py.

Usage:
    python scripts/evaluate_noise.py --split validation
    python scripts/evaluate_noise.py --split validation --snr_levels 20 10 5 0
    python scripts/evaluate_noise.py --split validation --max_samples 50
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
from fairvoice.noise import add_gaussian_noise, snr_label, DEFAULT_SNR_LEVELS_DB
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "evaluate_noise.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

MODEL_ID = "openai/whisper-small"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_whisper(device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def transcribe_whisper(waveform, model, processor, device: str) -> str:
    from transformers import WhisperProcessor
    inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="te", task="transcribe")
    generated_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=200)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def evaluate_at_snr(
    df: pd.DataFrame,
    model,
    processor,
    device: str,
    snr_db: float,
    split: str,
    max_samples: int | None = None,
) -> None:
    """Evaluate under a single noise condition and save results."""
    condition = snr_label(snr_db)
    logger.info("\n--- Evaluating condition: %s ---", condition)

    if max_samples:
        df = df.head(max_samples)

    utterance_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=condition):
        file_id = str(row.get("file_id", row.name))
        audio_path = row["processed_audio_path"]
        reference = str(row["transcript"])

        try:
            waveform, sr = load_audio(audio_path, target_sr=16_000)
            # Add noise deterministically (seed based on file_id hash for reproducibility)
            noise_seed = abs(hash(file_id)) % (2**31)
            noisy_waveform = add_gaussian_noise(waveform, snr_db=snr_db, random_seed=noise_seed)
            hypothesis = transcribe_whisper(noisy_waveform, model, processor, device)
        except Exception as exc:
            logger.warning("Failed on %s: %s", file_id, exc)
            hypothesis = ""

        result = evaluate_utterance(file_id, reference, hypothesis)
        utterance_results.append(result)

    summary = aggregate_results(
        utterance_results,
        model_id=MODEL_ID,
        split=split,
        condition=condition,
    )
    print_summary(summary)

    prefix = f"noise_{condition}_{split}"
    if max_samples:
        prefix += f"_n{max_samples}"
    save_results(utterance_results, summary, REPORTS_DIR / "predictions", prefix=prefix)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Noise robustness evaluation")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--snr_levels", nargs="+", type=float, default=DEFAULT_SNR_LEVELS_DB)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ensure_dir(PROJECT_ROOT / "logs")
    ensure_dir(REPORTS_DIR / "predictions")
    ensure_dir(REPORTS_DIR / "tables")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | SNR levels: %s dB", device, args.snr_levels)

    csv_path = PROJECT_ROOT / "data" / "metadata" / f"{args.split}.csv"
    if not csv_path.exists():
        logger.error("Manifest not found: %s. Run prepare_data.py first.", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8")
    logger.info("Loaded %d samples from %s split.", len(df), args.split)

    model, processor = load_whisper(device)

    all_summaries = []
    for snr_db in args.snr_levels:
        summary = evaluate_at_snr(df, model, processor, device, snr_db, args.split, args.max_samples)
        if summary:
            all_summaries.append(summary)

    # Print comparison table
    print("\n" + "=" * 65)
    print("  NOISE ROBUSTNESS SUMMARY TABLE")
    print(f"  Model: {MODEL_ID} | Split: {args.split}")
    print("=" * 65)
    print(f"  {'Condition':<15} {'WER %':>8} {'CER %':>8} {'ExactMatch %':>13}")
    print("-" * 65)
    for s in all_summaries:
        print(f"  {s.condition:<15} {s.wer:>8.2f} {s.cer:>8.2f} {s.exact_match_accuracy:>13.2f}")
    print("=" * 65)

    # Save comparison table
    table = [
        {"condition": s.condition, "wer": s.wer, "cer": s.cer,
         "exact_match_accuracy": s.exact_match_accuracy, "model_id": s.model_id}
        for s in all_summaries
    ]
    with open(REPORTS_DIR / "tables" / f"noise_comparison_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    logger.info("Comparison table saved.")


if __name__ == "__main__":
    main()
