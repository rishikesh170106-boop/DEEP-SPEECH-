"""
scripts/test_phase6b_beam.py — Standalone verification script for Phase 6B: CTC Beam Search Decoder.

Tests:
1. Greedy decoding vs. CTC Beam Search (beam_width=5).
2. Inspection of top candidate phonetic beams.
3. Execution latency on the GPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from fairvoice.audio import load_audio
from fairvoice.models.beam_decoder import CTCBeamSearchDecoder


def test_phase_6b():
    print("=" * 65)
    print("🎙️ PHASE 6B VERIFICATION: CTC MULTI-HYPOTHESIS BEAM SEARCH DECODER")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100"

    print(f"📦 Loading Telugu Wav2Vec2 CTC Model ({model_id}) on {device.upper()}...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()

    # Audio sample to test
    audio_path = PROJECT_ROOT / "data" / "processed" / "tef_08318_01333265498.wav"
    ground_truth = "చాలా గొంగళి పురుగులు షాకాహారులే"

    print(f"🎵 Testing Benchmark Audio: {audio_path.name}")
    print(f"🎯 Ground Truth Reference:  {ground_truth}")

    waveform, sr = load_audio(audio_path, target_sr=16_000)

    # Prepare inputs
    inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    # 1. Forward Pass
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(input_values).logits
        forward_time_ms = (time.perf_counter() - t0) * 1000

    # 2. Greedy Decoding (Current Phase 4 Method)
    t0 = time.perf_counter()
    greedy_ids = torch.argmax(logits, dim=-1)
    greedy_text = processor.batch_decode(greedy_ids)[0].replace("<s>", "").replace("  ", " ").strip()
    greedy_time_ms = (time.perf_counter() - t0) * 1000

    print("\n--- 1. GREEDY DECODING (Ears Only / Argmax) ---")
    print(f"Text Output:   \"{greedy_text}\"")
    print(f"Decode Time:   {greedy_time_ms:.2f} ms")

    # 3. CTC Beam Search Decoding (Phase 6B Method)
    beam_width = 5
    decoder = CTCBeamSearchDecoder(processor, beam_width=beam_width, top_k_per_frame=8)

    t0 = time.perf_counter()
    hypotheses = decoder.decode_logits(logits[0])
    beam_time_ms = (time.perf_counter() - t0) * 1000

    print(f"\n--- 2. CTC BEAM SEARCH (Top {beam_width} Candidate Hypotheses) ---")
    print(f"Decode Time:   {beam_time_ms:.2f} ms")
    print("-" * 65)
    print(f"{'Rank':<6} | {'Acoustic Score':<16} | {'Candidate Text'}")
    print("-" * 65)
    for hyp in hypotheses:
        print(f"#{hyp.rank:<5} | {hyp.acoustic_score:<16.2f} | \"{hyp.text}\"")
    print("-" * 65)

    print(f"\n⚡ Total Pipeline Latency: {forward_time_ms + beam_time_ms:.2f} ms on {device.upper()}")
    print("✨ PHASE 6B VERIFICATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    test_phase_6b()

