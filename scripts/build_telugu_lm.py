"""
scripts/build_telugu_lm.py — Phase 6A: Telugu KenLM & Statistical Language Model Builder.

Builds:
1. Standard ARPA n-gram format (artifacts/lm/telugu_kenlm.arpa) compatible with KenLM.
2. Fast JSON statistical dictionary (artifacts/lm/telugu_lm.json) for pure-Python zero-overhead beam decoding.
3. Unigram vocabulary list (artifacts/lm/telugu_unigrams.txt) for lexicon-constrained CTC decoding.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from fairvoice.text import normalize_text


def build_language_model():
    print("=" * 65)
    print("📚 PHASE 6A: TELUGU STATISTICAL LANGUAGE MODEL & ARPA BUILDER")
    print("=" * 65)

    metadata_path = PROJECT_ROOT / "data" / "metadata" / "all.csv"
    if not metadata_path.exists():
        metadata_path = PROJECT_ROOT / "data" / "metadata" / "train.csv"

    print(f"📖 Ingesting Telugu speech corpus from: {metadata_path.relative_to(PROJECT_ROOT)}")
    df = pd.read_csv(metadata_path)
    raw_transcripts = df["transcript"].dropna().tolist()
    print(f"✓ Total raw utterances: {len(raw_transcripts):,}")

    # Normalize sentences under NFC
    sentences: list[list[str]] = []
    all_words: list[str] = []

    for text in raw_transcripts:
        clean = normalize_text(text)
        tokens = [w for w in clean.split() if len(w) > 0]
        if tokens:
            sentences.append(tokens)
            all_words.extend(tokens)

    total_words = len(all_words)
    unigram_counts = Counter(all_words)
    vocab = sorted(unigram_counts.keys())
    vocab_size = len(vocab)

    print(f"✓ Total word tokens: {total_words:,}")
    print(f"✓ Unique Telugu vocabulary words: {vocab_size:,}")

    # Build Bigrams
    bigram_counts: Counter[tuple[str, str]] = Counter()
    for tokens in sentences:
        for i in range(len(tokens) - 1):
            bigram_counts[(tokens[i], tokens[i + 1])] += 1

    total_bigrams = sum(bigram_counts.values())
    unique_bigrams = len(bigram_counts)
    print(f"✓ Total bigrams: {total_bigrams:,} (Unique: {unique_bigrams:,})")

    # Compute Unigram log10 probabilities
    unigram_log_probs: dict[str, float] = {}
    for word, count in unigram_counts.items():
        # MLE log10 probability
        unigram_log_probs[word] = round(math.log10(count / total_words), 5)

    # Compute Bigram log10 conditional probabilities with Laplace smoothing
    # P(w2 | w1) = (Count(w1, w2) + 0.1) / (Count(w1) + 0.1 * vocab_size)
    bigram_log_probs: dict[str, float] = {}
    for (w1, w2), count in bigram_counts.items():
        w1_count = unigram_counts[w1]
        p = (count + 0.05) / (w1_count + 0.05 * vocab_size)
        bigram_log_probs[f"{w1} {w2}"] = round(math.log10(p), 5)

    # Ensure output directory exists
    output_dir = PROJECT_ROOT / "artifacts" / "lm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Unigrams Text File
    unigrams_file = output_dir / "telugu_unigrams.txt"
    with open(unigrams_file, "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(f"{w}\n")
    print(f"💾 Saved Unigram Lexicon -> {unigrams_file.relative_to(PROJECT_ROOT)}")

    # 2. Save Fast JSON Language Model
    lm_json_file = output_dir / "telugu_lm.json"
    lm_data = {
        "metadata": {
            "name": "FairVoice-Lite Telugu N-Gram LM",
            "vocab_size": vocab_size,
            "total_tokens": total_words,
            "unique_bigrams": unique_bigrams,
            "oov_penalty": -8.0,
        },
        "unigram_log_probs": unigram_log_probs,
        "bigram_log_probs": bigram_log_probs,
        "vocabulary": vocab,
    }
    with open(lm_json_file, "w", encoding="utf-8") as f:
        json.dump(lm_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved Fast JSON Telugu LM -> {lm_json_file.relative_to(PROJECT_ROOT)}")

    # 3. Save Standard ARPA Format (KenLM Compatible)
    arpa_file = output_dir / "telugu_kenlm.arpa"
    with open(arpa_file, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        f.write(f"ngram 1={vocab_size}\n")
        f.write(f"ngram 2={unique_bigrams}\n")
        f.write("\n")

        # 1-grams: log10_prob \t word \t backoff_weight
        f.write("\\1-grams:\n")
        for word, log_p in sorted(unigram_log_probs.items()):
            # Backoff weight placeholder -0.4
            f.write(f"{log_p:.4f}\t{word}\t-0.4000\n")
        f.write("\n")

        # 2-grams: log10_prob \t word1 word2
        f.write("\\2-grams:\n")
        for bg_str, log_p in sorted(bigram_log_probs.items()):
            f.write(f"{log_p:.4f}\t{bg_str}\n")
        f.write("\n")

        f.write("\\end\\\n")
    print(f"💾 Saved Universal ARPA Format -> {arpa_file.relative_to(PROJECT_ROOT)}")

    # Verification on sample benchmark words
    test_words = ["చాలా", "గొంగళి", "పురుగులు", "షాకాహారులే", "శాఖహారులే", "వ్యవసాయానికి", "ప్రసిద్ధి"]
    print("\n🔍 Verification on Sample Vocabulary Tokens:")
    print("-" * 65)
    print(f"{'Word':<20} | {'In Lexicon?':<12} | {'Log10 Prob':<12} | {'Frequency':<10}")
    print("-" * 65)
    for tw in test_words:
        in_lex = tw in unigram_counts
        prob = unigram_log_probs.get(tw, -8.0)
        cnt = unigram_counts.get(tw, 0)
        status = "✅ YES" if in_lex else "❌ NO (Typo/OOV)"
        print(f"{tw:<20} | {status:<12} | {prob:<12.4f} | {cnt:<10}")
    print("-" * 65)

    print("\n✨ PHASE 6A COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    build_language_model()

