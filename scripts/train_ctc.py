"""
scripts/train_ctc.py — Telugu Wav2Vec2 CTC fine-tuning for FairVoice-Lite.

Implements staged training:
  Stage 1: Smoke test (forward + backward on tiny batch)
  Stage 2: Small training run (verify predictions are non-blank)
  Stage 3: Full fine-tuning

Usage:
    python scripts/train_ctc.py --config configs/training.yaml
    python scripts/train_ctc.py --smoke_test_only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fairvoice.config import PROJECT_ROOT, ensure_dir, load_training_config
from fairvoice.audio import load_audio
from fairvoice.models.ctc_asr import TeluguCTCModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "train_ctc.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def run_smoke_test(model: TeluguCTCModel) -> bool:
    """Stage 1: Verify forward + backward pass works."""
    logger.info("=" * 50)
    logger.info("STAGE 1: SMOKE TEST")
    logger.info("=" * 50)

    # Use a short synthetic waveform (2 seconds of near-silence with tiny signal)
    sr = 16_000
    duration = 2.0
    waveform = np.random.randn(int(sr * duration)).astype(np.float32) * 0.01

    result = model.smoke_test(waveform, sr=sr)

    logger.info("Smoke test results:")
    for k, v in result.items():
        logger.info("  %-22s : %s", k, v)

    if result["success"]:
        logger.info("SMOKE TEST PASSED — Forward and backward pass OK.")
        return True
    else:
        logger.error("SMOKE TEST FAILED:")
        if not result["loss_finite"]:
            logger.error("  Loss is not finite: %s", result["loss_value"])
        if not result["gradients_exist"]:
            logger.error("  No gradients computed.")
        if not result["gradients_finite"]:
            logger.error("  Gradients contain NaN/Inf.")
        return False


def run_small_training(cfg: dict, model: TeluguCTCModel) -> None:
    """Stage 2: Small training experiment to verify non-blank predictions."""
    import pandas as pd
    from transformers import TrainingArguments, Trainer
    from fairvoice.dataset import TeluguASRDataset, collate_fn
    from fairvoice.metrics import compute_wer
    from fairvoice.text import normalize_text

    logger.info("=" * 50)
    logger.info("STAGE 2: SMALL TRAINING EXPERIMENT")
    logger.info("=" * 50)

    smoke_cfg = cfg.get("smoke_test", {})
    n_samples = smoke_cfg.get("num_samples", 16)
    n_steps = smoke_cfg.get("num_steps", 5)
    batch_size = smoke_cfg.get("batch_size", 2)

    train_csv = PROJECT_ROOT / cfg["data"]["train_csv"]
    df = pd.read_csv(train_csv, encoding="utf-8").head(n_samples)
    df.to_csv(PROJECT_ROOT / "data" / "metadata" / "_smoke_train.csv", index=False, encoding="utf-8")

    dataset = TeluguASRDataset(
        manifest_csv=PROJECT_ROOT / "data" / "metadata" / "_smoke_train.csv",
        processor=model.processor,
    )

    train_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "artifacts" / "checkpoints" / "smoke_test"),
        max_steps=n_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=n_steps + 1,
        fp16=torch.cuda.is_available(),
        no_cuda=not torch.cuda.is_available(),
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    from fairvoice.train import DataCollatorCTCWithPadding, LiveProgressBarCallback
    data_collator = DataCollatorCTCWithPadding(processor=model.processor, padding=True)

    progress_cb = LiveProgressBarCallback(PROJECT_ROOT / "reports" / "training_status.json")
    trainer = Trainer(
        model=model.model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[progress_cb],
    )

    logger.info("Running %d training steps on %d samples...", n_steps, n_samples)
    trainer.train()
    logger.info("Stage 2 training complete.")

    # Check predictions are non-blank
    logger.info("Testing predictions on 3 samples...")
    model.model.eval()
    for i in range(min(3, len(dataset))):
        row = dataset.records.iloc[i]
        audio_path = row.iloc[0]
        reference = row.iloc[1]
        hypothesis = model.transcribe(audio_path)
        logger.info("  Ref : %s", reference[:60])
        logger.info("  Hyp : %s", hypothesis[:60] or "(empty)")
        logger.info("  WER : %.4f", compute_wer(normalize_text(reference), normalize_text(hypothesis)))


def run_full_training(cfg: dict, model: TeluguCTCModel) -> None:
    """Stage 3: Full CTC fine-tuning."""
    from transformers import Trainer
    from fairvoice.dataset import TeluguASRDataset
    from fairvoice.train import DataCollatorCTCWithPadding, build_compute_metrics, build_training_args, LiveProgressBarCallback

    logger.info("=" * 50)
    logger.info("STAGE 3: FULL CTC FINE-TUNING")
    logger.info("=" * 50)

    train_csv = PROJECT_ROOT / cfg["data"]["train_csv"]
    val_csv = PROJECT_ROOT / cfg["data"]["val_csv"]

    noise_cfg = cfg.get("noise_augmentation", {})
    noise_enabled = noise_cfg.get("enabled", False)
    augment_snr = noise_cfg.get("snr_levels_db") if noise_enabled else None
    augment_prob = noise_cfg.get("augment_prob", 0.5)

    logger.info("Loading training dataset: %s (noise_augment=%s, snrs=%s)", train_csv, noise_enabled, augment_snr)
    train_dataset = TeluguASRDataset(
        manifest_csv=train_csv,
        processor=model.processor,
        augment_snr_db=augment_snr,
        augment_prob=augment_prob,
    )

    logger.info("Loading validation dataset: %s", val_csv)
    val_dataset = TeluguASRDataset(
        manifest_csv=val_csv,
        processor=model.processor,
    )

    if noise_enabled:
        checkpoint_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "wav2vec2_telugu_ctc_robust"
        best_model_dir = PROJECT_ROOT / cfg["output"].get("robust_model_dir", "artifacts/models/wav2vec2_telugu_ctc_robust")
    else:
        checkpoint_dir = PROJECT_ROOT / cfg["output"]["checkpoint_dir"]
        best_model_dir = PROJECT_ROOT / cfg["output"]["best_model_dir"]
    ensure_dir(checkpoint_dir)
    ensure_dir(best_model_dir)

    training_args = build_training_args(cfg, output_dir=str(checkpoint_dir))
    data_collator = DataCollatorCTCWithPadding(processor=model.processor, padding=True)
    compute_metrics = build_compute_metrics(model.processor)

    progress_cb = LiveProgressBarCallback(PROJECT_ROOT / "reports" / "training_status.json")
    trainer = Trainer(
        model=model.model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model.processor.feature_extractor,
        callbacks=[progress_cb],
    )

    logger.info("Starting training loop...")
    train_result = trainer.train()
    logger.info("Training complete. Train loss: %.4f", train_result.training_loss)

    logger.info("Saving best fine-tuned model to: %s", best_model_dir)
    trainer.save_model(str(best_model_dir))
    model.processor.save_pretrained(str(best_model_dir))
    logger.info("Model saved successfully! Ready for evaluation.")


def main():
    parser = argparse.ArgumentParser(description="Telugu CTC Fine-tuning")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--smoke_test_only", action="store_true")
    parser.add_argument("--full_train", action="store_true", help="Execute full Stage 3 fine-tuning")
    parser.add_argument("--init_from_clean", action="store_true", help="Initialize from Phase 2 clean checkpoint")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ensure_dir(PROJECT_ROOT / "logs")
    ensure_dir(PROJECT_ROOT / "artifacts" / "checkpoints")
    ensure_dir(PROJECT_ROOT / "artifacts" / "models")

    cfg = load_training_config()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_id = cfg["model"]["hf_id"]
    if args.init_from_clean:
        clean_path = PROJECT_ROOT / "artifacts" / "models" / "wav2vec2_telugu_ctc_best"
        if clean_path.exists():
            model_id = str(clean_path)
            logger.info("Initializing from Phase 2 clean checkpoint: %s", model_id)

    logger.info("Loading Telugu CTC model...")
    model = TeluguCTCModel.load(
        model_id=model_id,
        device=device,
        freeze_feature_extractor=cfg["model"].get("freeze_feature_extractor", True),
    )

    # Stage 1: Smoke test
    smoke_passed = run_smoke_test(model)
    if not smoke_passed:
        logger.error("Smoke test failed. Fix the error before proceeding.")
        logger.error("Do NOT run full training until smoke test passes.")
        sys.exit(1)

    if args.smoke_test_only:
        logger.info("--smoke_test_only flag set. Stopping after Stage 1.")
        sys.exit(0)

    # Check manifest exists before Stages 2 & 3
    train_csv = PROJECT_ROOT / cfg["data"]["train_csv"]
    if not train_csv.exists():
        logger.error("Training manifest not found: %s", train_csv)
        logger.error("Run: python scripts/prepare_data.py first.")
        sys.exit(1)

    # Stage 2: Small training
    run_small_training(cfg, model)

    # Stage 3: Full training if requested
    if args.full_train:
        run_full_training(cfg, model)
    else:
        logger.info("\n" + "=" * 50)
        logger.info("Stages 1 and 2 complete.")
        logger.info("To run full fine-tuning, run with --full_train flag.")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
