"""
INT8 Quantization Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Generates:
1. Full-precision FP32 model
2. Quantized INT8 model via PyTorch Dynamic Post-Training Quantization (PTQ)

Measures:
- Physical model storage reduction (%)
- Latency improvement (%)
- Accuracy retention / trade-off
"""

import copy
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn

from src import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Applies post-training dynamic quantization to INT8 for Linear layers.
    Quantizes 32-bit floating-point weights to 8-bit integers (qint8).
    """
    logger.info("Applying dynamic INT8 post-training quantization to linear layers...")
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def save_quantized_model(model: nn.Module, filepath: Path) -> float:
    """Saves quantized model state and returns file size in KB."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    size_kb = round(filepath.stat().st_size / 1024.0, 2)
    logger.info(f"Saved INT8 quantized model to {filepath} ({size_kb:.2f} KB)")
    return size_kb

