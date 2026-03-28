"""Runtime utilities for efficient YOLO inference.

This module centralizes device and precision tuning for edge and desktop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RuntimeConfig:
    device: str
    use_half: bool
    imgsz: int


def resolve_device(preferred: str = "auto") -> str:
    """Resolve the best available inference device."""
    pref = preferred.lower()
    if pref in {"cpu", "mps", "cuda"}:
        if pref == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if pref == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return pref

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(precision: str, device: str) -> bool:
    """Return whether FP16 should be enabled for this device."""
    mode = precision.lower()
    if mode == "fp16" and device == "cuda":
        return True
    return False


def build_predict_kwargs(config: RuntimeConfig) -> dict:
    """Build kwargs for Ultralytics model inference."""
    kwargs = {
        "verbose": False,
        "device": config.device,
        "imgsz": config.imgsz,
    }
    if config.use_half:
        kwargs["half"] = True
    return kwargs


def format_runtime_summary(config: RuntimeConfig) -> str:
    precision = "FP16" if config.use_half else "FP32"
    return (
        f"Device={config.device.upper()} | Precision={precision} | "
        f"imgsz={config.imgsz}"
    )
