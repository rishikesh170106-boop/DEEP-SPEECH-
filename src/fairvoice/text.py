"""
text.py — Telugu text normalization for FairVoice-Lite.

Provides a consistent normalize_text() function used identically
across ALL model comparisons. Never change this function mid-experiment.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize a Telugu transcript for ASR evaluation.

    Steps:
    1. Convert to string and strip whitespace.
    2. Apply Unicode NFC normalization (handles Telugu compound characters).
    3. Remove punctuation characters (keep word characters and spaces).
    4. Collapse multiple spaces into one.

    This function MUST be applied identically to references and hypotheses.
    Do not alter it between model comparisons without documenting the change.

    Parameters
    ----------
    text : str

    Returns
    -------
    str — normalized text
    """
    text = unicodedata.normalize("NFC", str(text)).strip()
    # Remove punctuation and symbol characters while preserving Telugu combining marks (Mn, Mc)
    chars = [" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text]
    text = "".join(chars)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def is_valid_transcript(text: str) -> bool:
    """Return True if the transcript is non-empty after normalization."""
    return len(normalize_text(text)) > 0


def char_set(texts: list[str]) -> set[str]:
    """Return the set of unique characters across a list of normalized texts."""
    normalized = [normalize_text(t) for t in texts]
    return set(c for t in normalized for c in t)


def word_delimiter_encode(text: str, delimiter: str = "|") -> str:
    """
    Encode spaces as word delimiters for CTC tokenizers.

    Example: 'యాభై మీటర్లలో' -> 'యాభై|మీటర్లలో'
    """
    return text.replace(" ", delimiter)


def word_delimiter_decode(text: str, delimiter: str = "|") -> str:
    """Decode CTC word delimiters back to spaces."""
    return text.replace(delimiter, " ").strip()
