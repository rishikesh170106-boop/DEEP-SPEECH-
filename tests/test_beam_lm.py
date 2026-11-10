"""
tests/test_beam_lm.py — Unit tests for Phase 6: CTC Beam Search and Telugu KenLM Rescoring.
"""

import pytest
from pathlib import Path

from fairvoice.models.lm_rescorer import TeluguKenLMRescorer, levenshtein_distance
from fairvoice.models.beam_decoder import BeamHypothesis, log_add


def test_levenshtein_distance():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("తెలుగు", "తెలుగు") == 0
    assert levenshtein_distance("గొంగలి", "గొంగళి") == 1
    assert levenshtein_distance("ఏలుగులు", "ఎలుగులు") == 1


def test_log_add():
    import math
    a, b = -2.0, -3.0
    expected = math.log(math.exp(a) + math.exp(b))
    assert abs(log_add(a, b) - expected) < 1e-5
    assert log_add(-float("inf"), a) == a
    assert log_add(b, -float("inf")) == b


def test_kenlm_rescorer_initialization():
    rescorer = TeluguKenLMRescorer()
    assert len(rescorer.vocabulary) > 1000
    assert len(rescorer.unigram_log_probs) > 1000


def test_kenlm_sentence_scoring():
    rescorer = TeluguKenLMRescorer()
    valid_sentence = "చాలా గొంగళి పురుగులు షాకాహారులే"
    typo_sentence = "చాలా గొంగలి పురుగులు శాఖహారులే"

    valid_score = rescorer.score_sentence(valid_sentence)
    typo_score = rescorer.score_sentence(typo_sentence)

    # The grammatically correct Telugu sentence must have a strictly higher score than the typo sentence
    assert valid_score > typo_score


def test_kenlm_phonetic_word_correction():
    rescorer = TeluguKenLMRescorer()
    # Typo word 'గొంగలి' should correct to 'గొంగళి'
    assert rescorer.correct_phonetic_word("గొంగలి") == "గొంగళి"
    # Typo word 'శాఖహారులే' should correct to 'షాకాహారులే'
    assert rescorer.correct_phonetic_word("శాఖహారులే") == "షాకాహారులే"
    # Typo word 'ఏలుగులు' should correct to 'ఎలుగులు'
    assert rescorer.correct_phonetic_word("ఏలుగులు") == "ఎలుగులు"


def test_kenlm_hypothesis_reranking():
    rescorer = TeluguKenLMRescorer(alpha=0.8, beta=1.0)
    hyps = [
        BeamHypothesis(text="చాలా గొంగలి పురుగులు శాఖహారులే", token_ids=(1, 2), acoustic_score=-2.34, rank=1),
        BeamHypothesis(text="చాలా గొంగళి పురుగులు షాకాహారులే", token_ids=(1, 3), acoustic_score=-2.80, rank=2),
    ]
    rescored = rescorer.rescore_hypotheses(hyps)
    assert len(rescored) == 2
    # The corrected hypothesis with valid Telugu words should rank #1
    assert rescored[0].text == "చాలా గొంగళి పురుగులు షాకాహారులే"

