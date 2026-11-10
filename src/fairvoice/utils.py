"""
src/fairvoice/utils.py — General utilities for FairVoice-Lite.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Save an object to JSON with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gpu_memory_info() -> Dict[str, float]:
    """Return GPU memory info in GB. Returns empty dict if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        return {
            "total_gb": round(total, 2),
            "reserved_gb": round(reserved, 2),
            "allocated_gb": round(allocated, 2),
            "free_gb": round(total - reserved, 2),
        }
    except Exception:
        return {}


def setup_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> None:
    """Configure root logger with optional file handler."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
