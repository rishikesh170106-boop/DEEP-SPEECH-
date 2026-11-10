"""
metrics.py — ASR evaluation metrics for FairVoice-Lite.

Implements WER, CER, exact-match accuracy, and per-utterance error counts.
Uses jiwer for reliable WER/CER computation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import jiwer

from fairvoice.text import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class UtteranceResult:
    """Per-utterance evaluation result."""
    file_id: str
    reference: str
    hypothesis: str
    ref_normalized: str
    hyp_normalized: str
    wer: float
    cer: float
    exact_match: bool
    ref_words: int
    word_errors: int


@dataclass
class EvaluationSummary:
    """Aggregate evaluation summary."""
    model_id: str
    split: str
    condition: str              # e.g. 'clean', '20dB_SNR', '10dB_SNR'
    num_samples: int
    total_ref_words: int
    total_word_errors: int
    wer: float
    cer: float
    exact_match_accuracy: float
    insertions: int
    deletions: int
    substitutions: int


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute WER between two normalized strings."""
    try:
        return jiwer.wer(reference, hypothesis)
    except Exception:
        return 1.0


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute CER between two normalized strings."""
    try:
        return jiwer.cer(reference, hypothesis)
    except Exception:
        return 1.0


def evaluate_utterance(
    file_id: str,
    reference: str,
    hypothesis: str,
) -> UtteranceResult:
    """Compute per-utterance metrics. Normalizes both ref and hyp."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    wer_val = compute_wer(ref_norm, hyp_norm)
    cer_val = compute_cer(ref_norm, hyp_norm)
    exact = ref_norm == hyp_norm

    ref_words = len(ref_norm.split()) if ref_norm else 0

    # jiwer word-level counts
    try:
        output = jiwer.process_words(ref_norm, hyp_norm)
        word_errors = output.substitutions + output.deletions + output.insertions
    except Exception:
        word_errors = int(round(wer_val * max(ref_words, 1)))

    return UtteranceResult(
        file_id=file_id,
        reference=reference,
        hypothesis=hypothesis,
        ref_normalized=ref_norm,
        hyp_normalized=hyp_norm,
        wer=round(wer_val, 6),
        cer=round(cer_val, 6),
        exact_match=exact,
        ref_words=ref_words,
        word_errors=word_errors,
    )


def aggregate_results(
    utterance_results: List[UtteranceResult],
    model_id: str = "unknown",
    split: str = "validation",
    condition: str = "clean",
) -> EvaluationSummary:
    """Aggregate per-utterance results into a summary."""
    if not utterance_results:
        raise ValueError("No utterance results to aggregate.")

    refs = [r.ref_normalized for r in utterance_results]
    hyps = [r.hyp_normalized for r in utterance_results]

    # Corpus-level WER / CER (more accurate than mean of per-utterance WER)
    corpus_wer = jiwer.wer(refs, hyps)
    corpus_cer = jiwer.cer(refs, hyps)

    # Word-level counts
    output = jiwer.process_words(refs, hyps)
    total_ref_words = sum(r.ref_words for r in utterance_results)
    total_word_errors = output.substitutions + output.deletions + output.insertions

    exact_match_acc = sum(1 for r in utterance_results if r.exact_match) / len(utterance_results)

    return EvaluationSummary(
        model_id=model_id,
        split=split,
        condition=condition,
        num_samples=len(utterance_results),
        total_ref_words=total_ref_words,
        total_word_errors=total_word_errors,
        wer=round(corpus_wer * 100, 4),
        cer=round(corpus_cer * 100, 4),
        exact_match_accuracy=round(exact_match_acc * 100, 4),
        insertions=output.insertions,
        deletions=output.deletions,
        substitutions=output.substitutions,
    )


def save_results(
    utterance_results: List[UtteranceResult],
    summary: EvaluationSummary,
    output_dir: str | Path,
    prefix: str = "eval",
) -> None:
    """Save utterance predictions and summary to JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictions_path = out / f"{prefix}_predictions.json"
    summary_path = out / f"{prefix}_summary.json"

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in utterance_results], f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    logger.info("Saved predictions to %s", predictions_path)
    logger.info("Saved summary to %s", summary_path)


def print_summary(summary: EvaluationSummary) -> None:
    """Print a formatted evaluation summary to stdout."""
    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY")
    print(f"  Model   : {summary.model_id}")
    print(f"  Split   : {summary.split}")
    print(f"  Condition: {summary.condition}")
    print("=" * 60)
    print(f"  Samples          : {summary.num_samples}")
    print(f"  Reference words  : {summary.total_ref_words}")
    print(f"  Word errors      : {summary.total_word_errors}")
    print(f"  WER              : {summary.wer:.2f}%")
    print(f"  CER              : {summary.cer:.2f}%")
    print(f"  Exact-match acc  : {summary.exact_match_accuracy:.2f}%")
    print(f"  Insertions       : {summary.insertions}")
    print(f"  Deletions        : {summary.deletions}")
    print(f"  Substitutions    : {summary.substitutions}")
    print("=" * 60 + "\n")
