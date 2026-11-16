"""
Model Pruning Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Implements L1 unstructured magnitude weight pruning across candidate sparsity levels (25%, 50%, 75%).
Evaluates:
- Theoretical weight sparsity (%)
- Non-zero parameter count
- Fine-tuning with gradient masking for accuracy recovery
- Actual compressed storage and latency impact
"""

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

from src import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def apply_unstructured_pruning(
    model: nn.Module,
    sparsity: float = 0.50,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    finetune_epochs: int = 2,
    lr: float = 0.001,
) -> nn.Module:
    """
    Applies L1-norm unstructured pruning to all Linear layers in the model.
    Sparsity represents the fraction of smallest magnitude weights set to zero.
    Optionally performs mask-preserving fine-tuning to recover anomaly detection accuracy.
    Permanently consolidates the mask into the weight tensors.
    """
    pruned_model = copy.deepcopy(model)
    pruned_model.eval()

    linear_modules = [m for m in pruned_model.modules() if isinstance(m, nn.Linear)]

    for module in linear_modules:
        prune.l1_unstructured(module, name="weight", amount=sparsity)

    # If training data provided, perform fine-tuning with frozen pruned weights
    if X_train is not None and y_train is not None and finetune_epochs > 0:
        logger.info(f"Fine-tuning pruned model ({int(sparsity*100)}% sparsity) for {finetune_epochs} epochs...")
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=512, shuffle=True)
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Extract current weight masks
        masks = {}
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
                masks[module] = module.weight_mask

        pruned_model.train()
        for ep in range(finetune_epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = pruned_model(bx)
                loss = criterion(out, by)
                loss.backward()

                # Re-apply zero gradients on pruned connections
                for module, mask in masks.items():
                    if module.weight.grad is not None:
                        module.weight.grad.data.mul_(mask)

                optimizer.step()

    # Consolidate and permanently bake the zeroed weights into the parameter
    for module in linear_modules:
        prune.remove(module, "weight")

    pruned_model.eval()
    return pruned_model


def calculate_model_sparsity(model: nn.Module) -> Tuple[int, int, float]:
    """
    Calculates exact zero vs non-zero weights across all weight parameters.
    Returns: (total_elements, zero_elements, actual_sparsity_pct)
    """
    total_elements = 0
    zero_elements = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            num = param.numel()
            zeros = (param == 0).sum().item()
            total_elements += num
            zero_elements += zeros

    actual_sparsity = (zero_elements / max(1, total_elements)) * 100.0
    return total_elements, zero_elements, round(actual_sparsity, 2)
