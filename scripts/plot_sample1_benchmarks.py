"""
scripts/plot_sample1_benchmarks.py — Generate two publication-quality graphs:
Graph 1: Model A (Clean CTC) vs. Model B (Robust + KenLM) across SNR dB levels
Graph 2: Ablation of Enable Telugu KenLM Grammar Rescoring across SNR dB levels
Using the audio example: "చాలా గొంగళి పురుగులు షాకాహారులే"
With pure English labels and explicit transliterations to ensure 100% clean rendering!
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# SNR Levels from clean down to 0 dB
snr_labels = ["Clean", "30 dB", "25 dB", "20 dB", "15 dB", "10 dB", "5 dB", "0 dB"]
x = np.arange(len(snr_labels))

# Empirical measured data from our RTX 4050 run on Sample 1
# Model A: Baseline CTC (No Denoising, No LM)
cer_model_a = [12.9, 9.7, 9.7, 12.9, 12.9, 12.9, 22.6, 35.5]
wer_model_a = [50.0, 50.0, 50.0, 50.0, 75.0, 50.0, 75.0, 75.0]

# Model B: Noise-Robust Pipeline (Spectral Denoising + KenLM Rescoring)
cer_model_b = [0.0, 0.0, 0.0, 0.0, 0.0, 19.4, 16.1, 19.4]
wer_model_b = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 25.0]

# KenLM Ablation on Denoised Audio:
# Without KenLM Rescoring (Greedy Argmax on Denoised Audio)
cer_no_lm = [12.9, 12.9, 9.7, 12.9, 12.9, 12.9, 12.9, 29.0]
wer_no_lm = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

# With KenLM Rescoring (Beam Search + KenLM Prior on Denoised Audio)
cer_with_lm = [0.0, 0.0, 0.0, 0.0, 0.0, 19.4, 16.1, 19.4]
wer_with_lm = [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 25.0]

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1: Model A (Clean CTC) vs. Model B (Robust FairVoice-Lite + KenLM)
# ══════════════════════════════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

# Left Subplot: Character Error Rate (CER)
ax1.plot(x, cer_model_a, marker="o", linewidth=2.5, markersize=7, color="#e11d48", label="Model A: Clean CTC Baseline", linestyle="--")
ax1.plot(x, cer_model_b, marker="s", linewidth=2.5, markersize=7, color="#059669", label="Model B: FairVoice-Lite (DSP + KenLM)", linestyle="-")
ax1.set_title("Character Error Rate (CER %) vs. Noise Level", fontsize=12.5, fontweight="bold", pad=12, color="#0f172a")
ax1.set_xlabel("Acoustic Noise Condition (SNR dB)", fontsize=11, labelpad=8)
ax1.set_ylabel("CER (%) — Lower is Better", fontsize=11, labelpad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(snr_labels, fontsize=10)
ax1.set_ylim(-2, 45)
ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10.5)

# Annotate the perfect 0% region
ax1.annotate("0.0% CER (100% Exact Match)\nAcross Clean to 15 dB SNR", xy=(2, 0), xytext=(1.0, 8),
             arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight="bold", color="#059669",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#059669", alpha=0.9))

# Right Subplot: Word Error Rate (WER)
ax2.plot(x, wer_model_a, marker="o", linewidth=2.5, markersize=7, color="#e11d48", label="Model A: Clean CTC Baseline", linestyle="--")
ax2.plot(x, wer_model_b, marker="s", linewidth=2.5, markersize=7, color="#059669", label="Model B: FairVoice-Lite (DSP + KenLM)", linestyle="-")
ax2.set_title("Word Error Rate (WER %) vs. Noise Level", fontsize=12.5, fontweight="bold", pad=12, color="#0f172a")
ax2.set_xlabel("Acoustic Noise Condition (SNR dB)", fontsize=11, labelpad=8)
ax2.set_ylabel("WER (%) — Lower is Better", fontsize=11, labelpad=8)
ax2.set_xticks(x)
ax2.set_xticklabels(snr_labels, fontsize=10)
ax2.set_ylim(-5, 90)
ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10.5)

fig.suptitle("Graph 1: Model A (Clean CTC) vs. Model B (Robust FairVoice-Lite + KenLM)\nBenchmark: Sample 1 ('chala gongali purugulu shaakaharule')", 
             fontsize=13.5, fontweight="bold", color="#1e293b", y=1.03)
fig.tight_layout()
graph1_path = out_dir / "graph1_model_a_vs_model_b_snr.png"
fig.savefig(graph1_path, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved Graph 1: {graph1_path}")

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2: Ablation of 'Enable Telugu KenLM Grammar Rescoring'
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

# Left Subplot: CER Ablation
ax1.plot(x, cer_no_lm, marker="^", linewidth=2.5, markersize=7, color="#d97706", label="Without KenLM Rescoring (Ears Only)", linestyle=":")
ax1.plot(x, cer_with_lm, marker="D", linewidth=2.5, markersize=7, color="#2563eb", label="With Telugu KenLM Rescoring (Ears + Brain)", linestyle="-")
ax1.fill_between(x, cer_no_lm, cer_with_lm, where=[c1 >= c2 for c1, c2 in zip(cer_no_lm, cer_with_lm)], color="#3b82f6", alpha=0.15, label="KenLM Accuracy Advantage")
ax1.set_title("Ablation: Character Error Rate (CER %)", fontsize=12.5, fontweight="bold", pad=12, color="#0f172a")
ax1.set_xlabel("Acoustic Noise Condition (SNR dB)", fontsize=11, labelpad=8)
ax1.set_ylabel("CER (%) — Lower is Better", fontsize=11, labelpad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(snr_labels, fontsize=10)
ax1.set_ylim(-2, 40)
ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10.5)

# Annotate phonetic auto-correction
ax1.annotate("Phonetic Auto-Correction:\n'gongali' -> 'gongali (retroflex)'\n'shakhaharule' -> 'shaakaharule'", xy=(0, 0), xytext=(0.3, 6),
             arrowprops=dict(facecolor="#2563eb", shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight="bold", color="#1d4ed8",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#2563eb", alpha=0.9))

# Right Subplot: WER Ablation
ax2.plot(x, wer_no_lm, marker="^", linewidth=2.5, markersize=7, color="#d97706", label="Without KenLM Rescoring (Ears Only)", linestyle=":")
ax2.plot(x, wer_with_lm, marker="D", linewidth=2.5, markersize=7, color="#2563eb", label="With Telugu KenLM Rescoring (Ears + Brain)", linestyle="-")
ax2.fill_between(x, wer_no_lm, wer_with_lm, where=[w1 >= w2 for w1, w2 in zip(wer_no_lm, wer_with_lm)], color="#3b82f6", alpha=0.15, label="WER Slashed to 0% up to 15 dB")
ax2.set_title("Ablation: Word Error Rate (WER %)", fontsize=12.5, fontweight="bold", pad=12, color="#0f172a")
ax2.set_xlabel("Acoustic Noise Condition (SNR dB)", fontsize=11, labelpad=8)
ax2.set_ylabel("WER (%) — Lower is Better", fontsize=11, labelpad=8)
ax2.set_xticks(x)
ax2.set_xticklabels(snr_labels, fontsize=10)
ax2.set_ylim(-5, 90)
ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10.5)

fig.suptitle("Graph 2: Impact of 'Enable Telugu KenLM Grammar Rescoring'\nAblation across Graded Noise Levels on Sample 1", 
             fontsize=13.5, fontweight="bold", color="#1e293b", y=1.03)
fig.tight_layout()
graph2_path = out_dir / "graph2_kenlm_ablation_snr.png"
fig.savefig(graph2_path, bbox_inches="tight")
plt.close(fig)
print(f"✅ Saved Graph 2: {graph2_path}")
