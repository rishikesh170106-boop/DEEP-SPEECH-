# FairVoice-Lite 🎙️ | తెలుగు వాక్ గుర్తింపు
## Noise-Robust Dravidian Automatic Speech Recognition with Joint CTC + Statistical Language Model

---

## 📌 Project Overview

**FairVoice-Lite** is a production-grade, noise-robust speech recognition system tailored for the **Telugu language (Dravidian family)**. It investigates and mitigates the severe performance degradation exhibited by standard transformer-based ASR models under ambient acoustic noise and phonetic transcription slips.

### Key Contributions:
1. **Dual-Model Architecture:**
   - **Model A (Baseline CTC):** Clean Fine-Tuned Wav2Vec2 CTC model (94.4M parameters) trained on studio speech.
   - **Model B (FairVoice-Lite):** Real-time Spectral Subtraction DSP ($\alpha=1.2, \beta=0.05$) combined with CTC Prefix Beam Search ($W=5$) and a 4,216-word **Telugu KenLM Statistical Language Model** ($98.61\%$ character accuracy).
2. **Zero-Hallucination Monotonic Decoding:** CTC alignment eliminates runaway hallucination loops (0 token insertions even under severe 0 dB SNR), outperforming autoregressive models like OpenAI Whisper-Small.
3. **Lexicon-Guided Phonetic Correction:** Resolves frequent Dravidian consonant-vowel acoustic slips (e.g., retroflex `ళ` vs `ల`, sibilants `శ` vs `ష`, aspirated `ఖ` vs `క`) and compound word splitting.
4. **Interactive Dual-Model Web Dashboard:** Built with Gradio 4, providing real-time side-by-side transcription, dual Mel-spectrogram spectral analysis, audio playback, and latency diagnostics.

---

## 👥 Authors & Academic Context
* **Institution:** B.Tech Data Science — Term Evaluation
* **Course:** Deep Speech Processing
* **Authors:**
  - Rishikesh (24BDS030)
  - Arjun (24BDS039)
  - Vineel (24BDS010)

---

## 📊 Benchmark Results

### 1. Telugu KenLM Decoding Ablation (Sample 1: *"చాలా గొంగళి పురుగులు షాకాహారులే"*)
| Decoding Paradigm | Character Error Rate (CER) | Character Accuracy | Word Error Rate (WER) | Exact Sentence Match |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy Argmax (Ears Only)** | 5.06% | 94.94% | 29.17% | 25.0% |
| **CTC Beam + Telugu KenLM (Ears + Brain)** | **1.39%** | **98.61%** | **4.17%** | **75.0%** |

### 2. Multi-Condition Gaussian Noise Sweep (OpenSLR-66 Telugu Female)
| Noise Condition (SNR) | Model A (Clean CTC) CER | Model B (DSP + KenLM) CER | Whisper-Small Insertions | CTC Insertions |
| :---: | :---: | :---: | :---: | :---: |
| **Clean Signal** | 12.9% | **0.0%** | 0 | **0** |
| **20 dB SNR (Mild)** | 12.9% | **0.0%** | 22 | **0** |
| **10 dB SNR (Moderate)** | 12.9% | **0.0%** | 89 | **0** |
| **5 dB SNR (Heavy)** | 22.6% | **16.1%** | 314 | **0** |
| **0 dB SNR (Severe)** | 35.5% | **19.4%** | 641 (Hallucination loops) | **0 (Strict Monotonicity)** |

---

## 🛠️ Project Structure

```
DEEP SPEECH PROCESSING/
├── app/
│   └── app.py                     # Gradio Dual-Model Interactive Web Application
├── src/
│   └── fairvoice/
│       ├── audio.py               # 16kHz mono audio loading & validation
│       ├── noise.py               # Deterministic Gaussian noise injection (SNR 0-30dB)
│       ├── text.py                # Telugu Unicode NFC normalization & cleaning
│       ├── metrics.py             # Word/Character Error Rate computation (jiwer)
│       └── models/
│           ├── ctc_asr.py         # Wav2Vec2 CTC model wrapper
│           ├── beam_decoder.py    # CTC Prefix Beam Search decoder
│           └── lm_rescorer.py     # Dual-backend Telugu KenLM rescorer & morphology
├── artifacts/
│   └── lm/
│       ├── telugu_kenlm.arpa      # ARPA statistical language model
│       ├── telugu_lm.json         # Fast Python N-Gram statistical dictionary
│       └── telugu_unigrams.txt    # 4,216-word Telugu unigram vocabulary
├── configs/                       # YAML configurations for training and evaluation
├── data/
│   ├── metadata/                  # Train / Val / Test metadata splits
│   └── processed/                 # Offline benchmark audio samples
├── reports/
│   ├── figures/                   # Benchmark plots (CER vs SNR, KenLM ablation)
│   ├── tables/                    # SNR degradation matrix CSVs
│   ├── FairVoice_Lite_Project_Report.html   # Publication-grade HTML report
│   └── FairVoice_Lite_Presentation.pptx     # Academic presentation deck
├── scripts/                       # Automated evaluation, plotting & build scripts
├── tests/                         # Full Pytest test suite (38/38 passing)
├── run_app.bat                    # One-click Windows launch script
└── pyproject.toml                 # Package dependencies and project config
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10.x
- NVIDIA GPU with CUDA support (tested on RTX 4050 Laptop GPU, 6GB VRAM)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/rishikesh170106-boop/DEEP-SPEECH-.git
cd DEEP-SPEECH-

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run Unit Tests
Verify all 38 unit tests pass:
```powershell
python -m pytest tests/ -v
```

### 4. Launch the Web Studio
```powershell
# Using the Windows batch script:
run_app.bat

# Or directly via Python:
$env:PYTHONPATH = "src;."
python app/app.py
```
Open **`http://127.0.0.1:8080`** in your browser.

---

## 🔬 Scientific Highlights
* **STFT Spectral Subtraction:** Computes stationary noise profiles across the quietest 15% of frames and oversubtracts noise energy ($\alpha=1.2, \beta=0.05$) to reconstruct voice formants.
* **Telugu Nominal Morphology:** Generates anusvara nominative forms (`+ ం`) at startup across stem vocabularies to handle Dravidian noun declensions seamlessly.
* **Low Latency:** Average GPU inference time $< 180\text{ ms}$ on NVIDIA RTX 4050, conforming to real-time telecommunication standards.
