"""
config.py — Configuration loader for FairVoice-Lite.

Loads and validates YAML configs. Provides a single point of truth for
all configuration values used across scripts and modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ── Project root (two levels up from src/fairvoice/) ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load configs/data.yaml relative to project root."""
    cfg_dir = config_dir or PROJECT_ROOT / "configs"
    return load_yaml(cfg_dir / "data.yaml")


def load_model_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load configs/model.yaml relative to project root."""
    cfg_dir = config_dir or PROJECT_ROOT / "configs"
    return load_yaml(cfg_dir / "model.yaml")


def load_training_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load configs/training.yaml relative to project root."""
    cfg_dir = config_dir or PROJECT_ROOT / "configs"
    return load_yaml(cfg_dir / "training.yaml")


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative_path


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
