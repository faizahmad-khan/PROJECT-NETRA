"""Runtime utilities for efficient YOLO inference.

This module centralizes device and precision tuning for edge and desktop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class RuntimeConfig:
    """Configuration for model runtime optimization."""
    device: str
    use_half: bool
    imgsz: int


def resolve_device(preferred: str = "auto") -> str:
    """Resolve the best available inference device.
    
    Args:
        preferred: Preferred device ('auto', 'cpu', 'mps', 'cuda')
    
    Returns:
        str: Available device name
    """
    pref: str = preferred.lower()
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
    """Return whether FP16 should be enabled for this device.
    
    Args:
        precision: Requested precision ('fp32' or 'fp16')
        device: Target device ('cpu', 'mps', 'cuda')
    
    Returns:
        bool: True if FP16 should be used, False otherwise
    """
    mode: str = precision.lower()
    if mode == "fp16" and device == "cuda":
        return True
    return False


def build_predict_kwargs(config: RuntimeConfig) -> Dict[str, Any]:
    """Build kwargs for Ultralytics model inference.
    
    Args:
        config: RuntimeConfig with device and precision settings
    
    Returns:
        Dict containing prediction arguments
    """
    kwargs: Dict[str, Any] = {
        "verbose": False,
        "device": config.device,
        "imgsz": config.imgsz,
    }
    if config.use_half:
        kwargs["half"] = True
    return kwargs


def format_runtime_summary(config: RuntimeConfig) -> str:
    """Format runtime configuration as human-readable string.
    
    Args:
        config: RuntimeConfig to format
    
    Returns:
        str: Formatted runtime summary
    """
    precision: str = "FP16" if config.use_half else "FP32"
    return (
        f"Device={config.device.upper()} | Precision={precision} | "
        f"imgsz={config.imgsz}"
    )
