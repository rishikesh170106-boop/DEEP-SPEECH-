"""
scripts/download_data.py — Dataset download helper for FairVoice-Lite.

Downloads the SLR66 Telugu dataset if not already present.
The file is ~482 MB.

Usage:
    python scripts/download_data.py

Manual alternative:
    Download from: https://www.openslr.org/resources/66/te_in_female.zip
    Save to: data/raw/te_in_female.zip
"""

from __future__ import annotations

import sys
import logging
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.config import PROJECT_ROOT, ensure_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_URL = "https://www.openslr.org/resources/66/te_in_female.zip"
EXPECTED_SIZE_MB = 482
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "te_in_female.zip"


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            print(f"\r  Downloading: {pct:5.1f}% ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)

    print(f"Downloading: {url}")
    print(f"Saving to  : {dest}")
    print(f"Expected   : ~{EXPECTED_SIZE_MB} MB\n")

    urllib.request.urlretrieve(url, str(dest), reporthook)
    print()  # newline after progress


def main():
    ensure_dir(RAW_DIR)

    if ARCHIVE_PATH.exists():
        size_mb = ARCHIVE_PATH.stat().st_size / 1e6
        logger.info("Archive already exists: %s (%.1f MB)", ARCHIVE_PATH.name, size_mb)
        if size_mb < EXPECTED_SIZE_MB * 0.9:
            logger.warning("File seems incomplete (%.1f MB < %.1f MB expected). Re-downloading.", size_mb, EXPECTED_SIZE_MB)
            ARCHIVE_PATH.unlink()
        else:
            logger.info("Skipping download — file looks complete.")
            print("\nNext step:")
            print("  python scripts/prepare_data.py")
            return

    logger.info("Starting download...")
    logger.info("NOTE: This is ~%d MB. Ensure you have a stable connection.", EXPECTED_SIZE_MB)
    logger.info("If the download fails, you can:")
    logger.info("  1. Download manually from: %s", DATASET_URL)
    logger.info("  2. Save to: %s", ARCHIVE_PATH)

    try:
        download_with_progress(DATASET_URL, ARCHIVE_PATH)
        size_mb = ARCHIVE_PATH.stat().st_size / 1e6
        logger.info("Downloaded successfully: %.1f MB", size_mb)
        print("\nNext step:")
        print("  python scripts/prepare_data.py")
    except KeyboardInterrupt:
        logger.warning("Download cancelled.")
        if ARCHIVE_PATH.exists():
            ARCHIVE_PATH.unlink()
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        logger.error("Try downloading manually from: %s", DATASET_URL)
        sys.exit(1)


if __name__ == "__main__":
    main()
