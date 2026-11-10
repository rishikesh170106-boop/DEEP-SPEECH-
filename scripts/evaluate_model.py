"""
scripts/evaluate_model.py — General model evaluation script for FairVoice-Lite.

Evaluates any saved fine-tuned CTC checkpoint on the validation or test split.
Used for E4 (clean fine-tuning) and E6 (robustness comparison).

Usage:
    python scripts/evaluate_model.py --model_path artifacts/models/wav2vec2_telugu_ctc_best
    python scripts/evaluate_model.py --model_path artifacts/models/wav2vec2_telugu_ctc_best --split test
    python scripts/evaluate_model.py --model_path artifacts/models/wav2vec2_telugu_ctc_best --snr_db 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.config import PROJECT_ROOT, ensure_dir
from fairvoice.audio import load_audio
from fairvoice.noise import add_gaussian_noise, snr_label
from fairvoice.models.ctc_asr import TeluguCTCModel
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "evaluate_model.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned CTC model")
    parser.add_argument("--model_path", required=True, help="Path to saved model directory")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--snr_db", type=float, default=None,
                        help="If set, add noise at this SNR (dB). Omit for clean evaluation.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ensure_dir(PROJECT_ROOT / "logs")
    ensure_dir(REPORTS_DIR / "predictions")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)

    if not model_path.exists():
        logger.error("Model path not found: %s", model_path)
        logger.error("Run: python scripts/train_ctc.py first.")
        sys.exit(1)

    logger.info("Loading fine-tuned model from: %s", model_path)
    ctc_model = TeluguCTCModel.load(
        model_id=str(model_path),
        device=device,
        freeze_feature_extractor=False,  # Evaluation only
    )

    # Load manifest
    csv_path = PROJECT_ROOT / "data" / "metadata" / f"{args.split}.csv"
    df = pd.read_csv(csv_path, encoding="utf-8")
    if args.max_samples:
        df = df.head(args.max_samples)

    condition = snr_label(args.snr_db) if args.snr_db is not None else "clean"
    logger.info("Evaluating %d samples | condition=%s", len(df), condition)

    utterance_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating ({condition})"):
        file_id = str(row.get("file_id", row.name))
        audio_path = row["processed_audio_path"]
        reference = str(row["transcript"])

        try:
            waveform, sr = load_audio(audio_path, target_sr=16_000)
            if args.snr_db is not None:
                noise_seed = abs(hash(file_id)) % (2**31)
                waveform = add_gaussian_noise(waveform, snr_db=args.snr_db, random_seed=noise_seed)
            hypothesis = ctc_model.transcribe(audio_path) if args.snr_db is None else \
                         _transcribe_waveform(ctc_model, waveform)
        except Exception as exc:
            logger.warning("Failed on %s: %s", file_id, exc)
            hypothesis = ""

        result = evaluate_utterance(file_id, reference, hypothesis)
        utterance_results.append(result)

    summary = aggregate_results(
        utterance_results,
        model_id=str(model_path),
        split=args.split,
        condition=condition,
    )
    print_summary(summary)

    prefix = f"finetuned_{condition}_{args.split}"
    if args.max_samples:
        prefix += f"_n{args.max_samples}"
    save_results(utterance_results, summary, REPORTS_DIR / "predictions", prefix=prefix)


@torch.no_grad()
def _transcribe_waveform(ctc_model: TeluguCTCModel, waveform) -> str:
    """Transcribe a pre-loaded waveform (for noisy evaluation)."""
    import numpy as np
    from fairvoice.text import normalize_text, word_delimiter_decode

    inputs = ctc_model.processor(
        waveform, sampling_rate=16_000, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(ctc_model.device)
    logits = ctc_model.model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = ctc_model.processor.batch_decode(predicted_ids)[0]
    return normalize_text(word_delimiter_decode(transcript))


if __name__ == "__main__":
    main()
