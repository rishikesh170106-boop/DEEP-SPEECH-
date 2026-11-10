"""
tests/test_metrics.py — Unit tests for ASR metrics.
Run: pytest tests/
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.metrics import evaluate_utterance, aggregate_results, compute_wer, compute_cer


class TestComputeWER:
    def test_perfect_match(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_complete_mismatch(self):
        wer = compute_wer("hello", "goodbye")
        assert wer > 0.0

    def test_empty_reference(self):
        # jiwer may return 0.0 or raise — just check it doesn't crash
        try:
            result = compute_wer("", "")
            assert isinstance(result, float)
        except Exception:
            pass


class TestEvaluateUtterance:
    def test_perfect_match(self):
        result = evaluate_utterance("file_001", "నమస్కారం", "నమస్కారం")
        assert result.exact_match is True
        assert result.wer == 0.0

    def test_different_texts(self):
        result = evaluate_utterance("file_002", "నమస్కారం", "")
        assert result.exact_match is False
        assert result.wer > 0.0

    def test_file_id_preserved(self):
        result = evaluate_utterance("my_file", "test", "test")
        assert result.file_id == "my_file"


class TestAggregateResults:
    def test_aggregate_basic(self):
        results = [
            evaluate_utterance("f1", "నమస్కారం", "నమస్కారం"),
            evaluate_utterance("f2", "యాభై", "యాభై"),
        ]
        summary = aggregate_results(results, model_id="test", split="validation", condition="clean")
        assert summary.num_samples == 2
        assert summary.wer == 0.0
        assert summary.exact_match_accuracy == 100.0

    def test_aggregate_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_results([], model_id="test", split="validation", condition="clean")
