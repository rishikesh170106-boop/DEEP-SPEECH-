"""
tests/test_text.py — Unit tests for Telugu text normalization.
Run: pytest tests/
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.text import normalize_text, is_valid_transcript, word_delimiter_encode, word_delimiter_decode


class TestNormalizeText:
    def test_basic_telugu(self):
        result = normalize_text("యాభై మీటర్లలో")
        assert result == "యాభై మీటర్లలో"

    def test_nfc_normalization(self):
        # NFC should be idempotent on already-normalized text
        text = "నమస్కారం"
        assert normalize_text(text) == normalize_text(normalize_text(text))

    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert normalize_text("a   b   c") == "a b c"

    def test_removes_punctuation(self):
        result = normalize_text("నమస్కారం!")
        assert "!" not in result

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_like(self):
        # Passing a non-string should coerce via str()
        result = normalize_text(123)
        assert isinstance(result, str)


class TestIsValidTranscript:
    def test_valid(self):
        assert is_valid_transcript("నమస్కారం") is True

    def test_empty(self):
        assert is_valid_transcript("") is False

    def test_only_punctuation(self):
        # Punctuation is stripped — result empty
        assert is_valid_transcript("!!!") is False


class TestWordDelimiter:
    def test_encode(self):
        assert word_delimiter_encode("యాభై మీటర్లలో") == "యాభై|మీటర్లలో"

    def test_decode(self):
        assert word_delimiter_decode("యాభై|మీటర్లలో") == "యాభై మీటర్లలో"

    def test_roundtrip(self):
        text = "నమస్కారం మీకు"
        assert word_delimiter_decode(word_delimiter_encode(text)) == text
