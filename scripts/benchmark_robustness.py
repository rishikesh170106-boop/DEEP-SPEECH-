"""
scripts/benchmark_robustness.py — Phase 3 Deliverable: Multi-condition SNR Degradation Matrix.

Compares:
1. Whisper-Small Baseline
2. Wav2Vec2 CTC Clean Model
3. Wav2Vec2 CTC Noise-Robust Model

Across:
- Clean Speech
- 20 dB SNR (Mild Noise)
- 10 dB SNR (Moderate Noise)
- 5 dB SNR (High Noise)
- 0 dB SNR (Severe Noise)

Outputs:
- reports/tables/snr_degradation_matrix.json
- reports/tables/snr_degradation_matrix.csv
"""

from __future__ import annotations

import json
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
from fairvoice.metrics import evaluate_utterance, aggregate_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"


@torch.no_grad()
def transcribe_waveform_ctc(ctc_model: TeluguCTCModel, waveform, target_sr=16_000) -> str:
    from fairvoice.text import normalize_text, word_delimiter_decode
    inputs = ctc_model.processor(waveform, sampling_rate=target_sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(ctc_model.device)
    logits = ctc_model.model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcript = ctc_model.processor.batch_decode(pred_ids)[0]
    return normalize_text(word_delimiter_decode(transcript))


def evaluate_model_on_snr(model, df: pd.DataFrame, snr_db: float | None, condition: str) -> dict:
    utterance_results = []
    for _, row in df.iterrows():
        file_id = str(row.get("file_id", row.name))
        audio_path = row["processed_audio_path"]
        reference = str(row["transcript"])
        try:
            waveform, sr = load_audio(audio_path, target_sr=16_000)
            if snr_db is not None:
                noise_seed = abs(hash(file_id)) % (2**31)
                waveform = add_gaussian_noise(waveform, snr_db=snr_db, random_seed=noise_seed)
            hypothesis = transcribe_waveform_ctc(model, waveform)
        except Exception:
            hypothesis = ""
        utterance_results.append(evaluate_utterance(file_id, reference, hypothesis))

    return aggregate_results(utterance_results, model_id=model.model_id, split="validation", condition=condition)


def main():
    ensure_dir(TABLES_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Engaging benchmark evaluation on device: %s", device)

    clean_path = PROJECT_ROOT / "artifacts" / "models" / "wav2vec2_telugu_ctc_best"
    robust_path = PROJECT_ROOT / "artifacts" / "models" / "wav2vec2_telugu_ctc_robust"

    logger.info("Loading Clean Model...")
    clean_model = TeluguCTCModel.load(str(clean_path), device=device, freeze_feature_extractor=False)
    logger.info("Loading Robust Model...")
    robust_model = TeluguCTCModel.load(str(robust_path), device=device, freeze_feature_extractor=False)

    df_val = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "validation.csv", encoding="utf-8")
    logger.info("Loaded %d validation utterances.", len(df_val))

    # Existing Whisper validation numbers from E1 & E2
    whisper_results = {
        "Clean": {"wer": 153.33, "insertions": 641, "cer": 151.87},
        "20dB_SNR": {"wer": 138.61, "insertions": 482, "cer": 160.61},
        "10dB_SNR": {"wer": 125.48, "insertions": 412, "cer": 166.08},
        "5dB_SNR": {"wer": 135.52, "insertions": 447, "cer": 158.05},
        "0dB_SNR": {"wer": 123.55, "insertions": 395, "cer": 163.23},
    }

    conditions = [
        ("Clean", None),
        ("20dB_SNR", 20.0),
        ("10dB_SNR", 10.0),
        ("5dB_SNR", 5.0),
        ("0dB_SNR", 0.0),
    ]

    matrix_rows = []

    print("\n" + "=" * 80)
    print("  🏆 PHASE 3: MULTI-CONDITION SNR ROBUSTNESS BENCHMARK MATRIX")
    print("=" * 80)

    for cond_name, snr_val in conditions:
        logger.info("Benchmarking Condition: %s ...", cond_name)
        clean_res = evaluate_model_on_snr(clean_model, df_val, snr_val, cond_name)
        robust_res = evaluate_model_on_snr(robust_model, df_val, snr_val, cond_name)

        w_info = whisper_results.get(cond_name, {})

        row = {
            "condition": cond_name,
            "snr_db": "Clean" if snr_val is None else f"{int(snr_val)} dB",
            "whisper_wer": w_info.get("wer", 0.0),
            "whisper_insertions": w_info.get("insertions", 0),
            "clean_ctc_wer": clean_res.wer,
            "clean_ctc_insertions": clean_res.insertions,
            "robust_ctc_wer": robust_res.wer,
            "robust_ctc_insertions": robust_res.insertions,
            "robust_ctc_exact_match": robust_res.exact_match_accuracy,
        }
        matrix_rows.append(row)

    df_matrix = pd.DataFrame(matrix_rows)
    json_path = TABLES_DIR / "snr_degradation_matrix.json"
    csv_path = TABLES_DIR / "snr_degradation_matrix.csv"

    df_matrix.to_json(json_path, orient="records", indent=2)
    df_matrix.to_csv(csv_path, index=False)

    print("\n" + df_matrix.to_string(index=False))
    print("\n" + "=" * 80)
    print(f"✅ Matrix exported to: {json_path}")
    print(f"✅ CSV exported to: {csv_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
