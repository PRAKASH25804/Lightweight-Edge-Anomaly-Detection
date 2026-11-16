"""
Model Architectures and Baselines Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Contains:
1. Classical ML Baselines: Logistic Regression, Random Forest
2. Standard Deep Neural Network Reference (Standard MLP: 128 -> 64 -> 32 -> 1)
3. Compact Neural Architectures for Architecture Search:
   - Arch A: 32 -> 16
   - Arch B: 64 -> 32 -> 16 (PPT Proposed Baseline)
   - Arch C: 32 -> 16 -> 8
   - Arch D: 16 -> 8
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config


class StandardMLP(nn.Module):
    """
    Standard MLP Baseline (High-capacity reference).
    Architecture: Input -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Output(1)
    """

    def __init__(self, in_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LightweightMLP(nn.Module):
    """
    Compact Feed-Forward Neural Architecture designed for resource-constrained edge devices.
    Configurable layer topology supporting architecture search.
    Default follows the project PPT: 64 -> 32 -> 16 -> 1 with optional BatchNorm and Dropout during training.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [64, 32, 16],
        use_batch_norm: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm

        layers: List[nn.Module] = []
        prev_dim = in_features

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm and i == 0:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0 and i == 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # Final binary classification head
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_file_size_kb(filepath: Path) -> float:
    """Returns file size in Kilobytes (KB)."""
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / 1024.0, 2)
    return 0.0


def create_baseline_models(
    random_seed: int = config.RANDOM_SEED,
) -> Dict[str, Any]:
    """Instantiates baseline models with reproducible random seed and balanced weighting."""
    return {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_seed,
            solver="lbfgs",
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
    }


def save_sklearn_model(model: Any, filepath: Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_sklearn_model(filepath: Path) -> Any:
    return joblib.load(filepath)

