"""
scripts/prepare_data.py — Dataset preparation pipeline for FairVoice-Lite.

Steps:
1. Looks for te_in_female.zip in data/raw/
2. Extracts the archive
3. Discovers WAV files and transcript lines
4. Validates each audio file
5. Converts to mono 16kHz if needed
6. Creates train/validation/test CSV manifests

Usage:
    python scripts/prepare_data.py

Prerequisites:
    Place te_in_female.zip in data/raw/ before running.
    Download from: https://www.openslr.org/resources/66/te_in_female.zip
"""

from __future__ import annotations

import csv
import logging
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# ── Add src/ to path so fairvoice package is importable ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.config import PROJECT_ROOT, ensure_dir
from fairvoice.audio import load_audio, validate_audio, TARGET_SR
from fairvoice.text import normalize_text, is_valid_transcript

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "prepare_data.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
META_DIR = PROJECT_ROOT / "data" / "metadata"
ARCHIVE = RAW_DIR / "te_in_female.zip"


def find_transcript_file(base_dir: Path) -> Path | None:
    """Find the transcript text file inside the extracted archive."""
    for pattern in ["*.tsv", "**/*.tsv", "*.txt", "**/*.txt"]:
        candidates = list(base_dir.glob(pattern))
        for f in candidates:
            if "line" in f.name.lower() or "transcript" in f.name.lower() or "index" in f.name.lower():
                return f
        if candidates:
            return candidates[0]
    return None


def parse_transcripts(transcript_file: Path) -> dict[str, str]:
    """
    Parse transcript file into {file_id: transcript} dict.
    Handles tab-separated (OpenSLR line_index.tsv) or whitespace-separated formats.
    """
    mapping = {}
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) == 2:
                file_id = parts[0].strip()
                transcript = parts[1].replace("\\n", " ").strip()
                transcript = unicodedata.normalize("NFC", transcript)
                mapping[file_id] = transcript
    logger.info("Parsed %d transcripts from %s", len(mapping), transcript_file.name)
    return mapping


def download_archive_if_needed() -> None:
    """Download te_in_female.zip from OpenSLR if not already present."""
    if ARCHIVE.exists():
        logger.info("Found existing archive: %s (%.1f MB)", ARCHIVE.name, ARCHIVE.stat().st_size / (1024 * 1024))
        return

    import urllib.request
    from tqdm import tqdm

    ensure_dir(RAW_DIR)
    url = "https://www.openslr.org/resources/66/te_in_female.zip"
    logger.info("Downloading %s ...", url)
    print(f"Downloading OpenSLR-66 (te_in_female.zip, ~482 MB)...")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=ARCHIVE.name) as t:
        urllib.request.urlretrieve(url, filename=ARCHIVE, reporthook=t.update_to)
    logger.info("Downloaded %s successfully.", ARCHIVE)


def extract_archive() -> Path:
    """Extract te_in_female.zip to data/interim/."""
    ensure_dir(INTERIM_DIR)

    # Check if already extracted (either flat in INTERIM_DIR or in a subfolder)
    if list(INTERIM_DIR.glob("*.wav")):
        logger.info("Interim directory already contains extracted WAV files: %s — skipping extraction.", INTERIM_DIR)
        return INTERIM_DIR

    extract_to = INTERIM_DIR / "te_in_female"
    if extract_to.exists() and any(extract_to.iterdir()):
        logger.info("Interim directory already exists: %s — skipping extraction.", extract_to)
        return extract_to

    download_archive_if_needed()
    if not ARCHIVE.exists():
        logger.error("Archive not found: %s", ARCHIVE)
        sys.exit(1)

    logger.info("Extracting %s ...", ARCHIVE.name)
    with zipfile.ZipFile(ARCHIVE, "r") as z:
        z.extractall(INTERIM_DIR)

    if list(INTERIM_DIR.glob("*.wav")):
        logger.info("Extracted directly into: %s", INTERIM_DIR)
        return INTERIM_DIR

    extracted = [d for d in INTERIM_DIR.iterdir() if d.is_dir()]
    if not extracted:
        logger.error("Extraction produced no directories or WAV files in %s", INTERIM_DIR)
        sys.exit(1)

    actual = extracted[0]
    if actual != extract_to:
        actual.rename(extract_to)

    logger.info("Extracted to: %s", extract_to)
    return extract_to


def process_audio(wav_path: Path, out_dir: Path) -> tuple[bool, str, float | None]:
    """
    Load, validate, and save audio as mono 16kHz WAV.

    Returns (success, reason, duration)
    """
    out_path = out_dir / wav_path.name
    if out_path.exists():
        # Already processed — verify it
        try:
            info = sf.info(str(out_path))
            return True, "", info.duration
        except Exception:
            pass  # Reprocess

    try:
        waveform, sr = load_audio(wav_path, target_sr=TARGET_SR)
    except Exception as exc:
        return False, f"load_error: {exc}", None

    ok, reason = validate_audio(waveform, sr, path=wav_path)
    if not ok:
        return False, reason, None

    duration = len(waveform) / sr

    try:
        sf.write(str(out_path), waveform, TARGET_SR, subtype="PCM_16")
    except Exception as exc:
        return False, f"write_error: {exc}", None

    return True, "", duration


def build_manifests(records: list[dict]) -> None:
    """
    Split records into train/validation/test and save CSVs.
    Uses the original SLR66 split counts: 1835 / 229 / 230.
    """
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    n_train = min(1835, int(0.80 * n))
    n_val   = min(229,  int(0.10 * n))
    n_test  = n - n_train - n_val

    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train: n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val:].copy()

    train_df["split"] = "train"
    val_df["split"]   = "validation"
    test_df["split"]  = "test"

    ensure_dir(META_DIR)

    train_df.to_csv(META_DIR / "train.csv",      index=False, encoding="utf-8")
    val_df.to_csv(  META_DIR / "validation.csv", index=False, encoding="utf-8")
    test_df.to_csv( META_DIR / "test.csv",       index=False, encoding="utf-8")
    pd.concat([train_df, val_df, test_df]).to_csv(META_DIR / "all.csv", index=False, encoding="utf-8")

    logger.info("Manifests saved to %s", META_DIR)
    logger.info("  Train      : %d", len(train_df))
    logger.info("  Validation : %d", len(val_df))
    logger.info("  Test       : %d", len(test_df))


def main():
    ensure_dir(PROJECT_ROOT / "logs")
    logger.info("=== FairVoice-Lite: Data Preparation ===")

    # Step 1: Extract archive
    interim_dir = extract_archive()

    # Step 2: Find transcript file
    transcript_file = find_transcript_file(interim_dir)
    if transcript_file is None:
        logger.error("No transcript file found in %s", interim_dir)
        logger.info("Contents: %s", list(interim_dir.rglob("*"))[:20])
        sys.exit(1)
    logger.info("Transcript file: %s", transcript_file)

    # Step 3: Parse transcripts
    transcripts = parse_transcripts(transcript_file)

    # Step 4: Find WAV files
    wav_files = list(interim_dir.rglob("*.wav"))
    logger.info("Found %d WAV files.", len(wav_files))
    if not wav_files:
        logger.error("No WAV files found in %s", interim_dir)
        sys.exit(1)

    # Step 5: Process each file
    ensure_dir(PROCESSED_DIR)
    records = []
    excluded = []

    for wav_path in sorted(wav_files):
        file_id = wav_path.stem
        transcript = transcripts.get(file_id)

        if transcript is None:
            excluded.append({"file_id": file_id, "reason": "no_transcript"})
            continue

        if not is_valid_transcript(transcript):
            excluded.append({"file_id": file_id, "reason": "empty_transcript_after_normalization"})
            continue

        ok, reason, duration = process_audio(wav_path, PROCESSED_DIR)
        if not ok:
            excluded.append({"file_id": file_id, "reason": reason})
            logger.warning("Excluded %s: %s", file_id, reason)
            continue

        records.append({
            "file_id": file_id,
            "transcript": transcript,
            "audio_path": str(wav_path),
            "sample_rate": TARGET_SR,
            "duration": round(duration, 4),
            "processed_audio_path": str(PROCESSED_DIR / wav_path.name),
        })

    logger.info("Processed: %d valid | %d excluded", len(records), len(excluded))

    # Save exclusions log
    if excluded:
        excl_df = pd.DataFrame(excluded)
        excl_df.to_csv(META_DIR / "excluded.csv", index=False, encoding="utf-8")
        logger.info("Exclusion log saved.")

    if not records:
        logger.error("No valid records! Check audio and transcript files.")
        sys.exit(1)

    # Step 6: Build manifests
    build_manifests(records)

    logger.info("=== Data preparation complete. ===")
    print("\nDone! Next step:")
    print("  python scripts/validate_data.py")


if __name__ == "__main__":
    main()
