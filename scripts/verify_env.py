"""
scripts/verify_env.py — Environment verification for FairVoice-Lite.

Run this FIRST before installing any models or training anything.
Checks: Python version, PyTorch, CUDA, GPU memory, key imports.

Usage:
    python scripts/verify_env.py
"""

import sys
import platform
import subprocess


def section(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


def check_python():
    section("Python Environment")
    print(f"  Python version : {sys.version}")
    print(f"  Executable     : {sys.executable}")
    v = sys.version_info
    ok = v.major == 3 and v.minor in (10, 11)
    print(f"  Version OK     : {'YES' if ok else 'WARNING — expected 3.10 or 3.11'}")


def check_torch():
    section("PyTorch + CUDA")
    try:
        import torch
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version    : {torch.version.cuda}")
            print(f"  GPU name        : {torch.cuda.get_device_name(0)}")
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  VRAM total      : {total:.2f} GB")
            print(f"  VRAM reserved   : {reserved:.2f} GB")
            print(f"  VRAM free       : {total - reserved:.2f} GB")
            # Quick tensor test on GPU
            x = torch.randn(100, 100, device="cuda")
            y = x @ x.T
            print(f"  GPU tensor test : OK (shape={y.shape})")
        else:
            print("  WARNING: No CUDA GPU detected — training will be very slow on CPU")
    except ImportError:
        print("  ERROR: PyTorch not installed. Run: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")


def check_torchaudio():
    section("Torchaudio")
    try:
        import torchaudio
        print(f"  torchaudio version : {torchaudio.__version__}")
        backends = torchaudio.list_audio_backends()
        print(f"  Audio backends     : {backends}")
    except Exception as exc:
        print(f"  WARNING: torchaudio load issue ({exc}). Using soundfile/librosa instead.")


def check_transformers():
    section("HuggingFace Transformers")
    try:
        import transformers
        print(f"  transformers version : {transformers.__version__}")
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        print(f"  Wav2Vec2ForCTC       : OK")
        print(f"  Wav2Vec2Processor    : OK")
    except ImportError as e:
        print(f"  ERROR: {e}")


def check_audio_libs():
    section("Audio Libraries")
    for lib in ["librosa", "soundfile", "numpy", "scipy"]:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {lib:<12} : {ver}")
        except ImportError:
            print(f"  {lib:<12} : NOT INSTALLED")


def check_other_libs():
    section("Other Dependencies")
    for lib in ["pandas", "jiwer", "yaml", "tqdm", "matplotlib", "gradio"]:
        import_name = "yaml" if lib == "yaml" else lib
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {lib:<12} : {ver}")
        except ImportError:
            print(f"  {lib:<12} : NOT INSTALLED")


def check_disk():
    section("Disk Space")
    import shutil
    total, used, free = shutil.disk_usage("C:\\")
    print(f"  C: drive total : {total / 1e9:.1f} GB")
    print(f"  C: drive used  : {used / 1e9:.1f} GB")
    print(f"  C: drive free  : {free / 1e9:.1f} GB")
    if free / 1e9 < 10:
        print("  WARNING: Less than 10 GB free — dataset and models need space!")


def main():
    print("\n" + "=" * 55)
    print("  FAIRVOICE-LITE — ENVIRONMENT VERIFICATION")
    print("=" * 55)
    print(f"  Platform : {platform.system()} {platform.release()} {platform.machine()}")

    check_python()
    check_torch()
    check_torchaudio()
    check_transformers()
    check_audio_libs()
    check_other_libs()
    check_disk()

    print("\n" + "=" * 55)
    print("  Verification complete.")
    print("  If all checks pass, proceed to: python scripts/prepare_data.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
