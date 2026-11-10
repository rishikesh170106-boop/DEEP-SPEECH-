"""
scripts/validate_data.py — Post-preparation dataset validation for FairVoice-Lite.

Run after prepare_data.py to confirm the manifests are correct.

Usage:
    python scripts/validate_data.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.config import PROJECT_ROOT, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

META_DIR = PROJECT_ROOT / "data" / "metadata"


def validate_csv(csv_path: Path, split_name: str) -> dict:
    logger.info("\n--- Validating: %s ---", split_name)

    if not csv_path.exists():
        logger.error("  NOT FOUND: %s", csv_path)
        return {"split": split_name, "status": "missing"}

    df = pd.read_csv(csv_path, encoding="utf-8")
    n = len(df)
    logger.info("  Rows: %d", n)

    # Check required columns
    required_cols = ["file_id", "transcript", "processed_audio_path", "duration"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.warning("  Missing columns: %s", missing_cols)

    # Check audio files exist
    audio_col = "processed_audio_path"
    if audio_col in df.columns:
        missing_audio = df[~df[audio_col].apply(lambda p: Path(str(p)).exists())]
        logger.info("  Audio files found     : %d / %d", n - len(missing_audio), n)
        if len(missing_audio) > 0:
            logger.warning("  Missing audio files   : %d", len(missing_audio))

    # Check transcripts
    if "transcript" in df.columns:
        empty_trans = df[df["transcript"].isna() | (df["transcript"].astype(str).str.strip() == "")]
        logger.info("  Non-empty transcripts : %d / %d", n - len(empty_trans), n)

    # Duration stats
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
        valid_dur = df["duration"].dropna()
        logger.info(
            "  Duration (sec)  min=%.2f  max=%.2f  mean=%.2f  total=%.1f min",
            valid_dur.min(), valid_dur.max(), valid_dur.mean(), valid_dur.sum() / 60
        )

    return {"split": split_name, "n": n, "status": "ok"}


def main():
    logger.info("=== FairVoice-Lite: Data Validation ===")

    results = []
    for split in ["train", "validation", "test"]:
        csv_path = META_DIR / f"{split}.csv"
        r = validate_csv(csv_path, split)
        results.append(r)

    # Check for overlap between splits
    dfs = {}
    for split in ["train", "validation", "test"]:
        csv_path = META_DIR / f"{split}.csv"
        if csv_path.exists():
            dfs[split] = set(pd.read_csv(csv_path)["file_id"].astype(str))

    if len(dfs) == 3:
        tr_val = dfs["train"] & dfs["validation"]
        tr_te  = dfs["train"] & dfs["test"]
        val_te = dfs["validation"] & dfs["test"]
        logger.info("\n--- Split Overlap Check ---")
        logger.info("  Train ∩ Validation : %d (should be 0)", len(tr_val))
        logger.info("  Train ∩ Test       : %d (should be 0)", len(tr_te))
        logger.info("  Validation ∩ Test  : %d (should be 0)", len(val_te))

        if tr_val or tr_te or val_te:
            logger.error("DATA LEAKAGE DETECTED! Overlap between splits.")
        else:
            logger.info("  No overlap — splits are clean.")

    logger.info("\n=== Validation complete. ===")
    print("\nNext step:")
    print("  python scripts/evaluate_baseline.py --split validation --max_samples 10")


if __name__ == "__main__":
    main()
