"""
Edge Device Adapter and Real-Time Detection Interface
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Implements:
Raw Input Record -> Preprocessing -> Feature Selection -> Optimized Neural Inference -> Thresholding -> Local Alert
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch

from src import config
from src.models import LightweightMLP
from src.preprocess import EdgeDataPreprocessor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EdgeAnomalyDetector:
    """
    On-device embedded runtime engine for real-time anomaly detection.
    Guarantees strict parity between training preprocessing and edge deployment.
    """

    def __init__(
        self,
        config_path: Path = config.RESULTS_DIR / "final_model_config.json",
        preprocessor_path: Path = config.ARTIFACTS_DIR / "preprocessor.joblib",
        model_path: Optional[Path] = None,
    ):
        # 1. Load deployment config
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.threshold = float(self.cfg.get("decision_threshold", 0.50))
        self.selected_features = self.cfg["selected_features"]
        self.hidden_layers = self.cfg["hidden_layers"]
        self.input_dim = len(self.selected_features)

        # 2. Load preprocessor
        self.preprocessor = EdgeDataPreprocessor.load(preprocessor_path)

        # Align selected feature indices in preprocessor active feature list
        self.feature_indices = [
            self.preprocessor.active_features.index(f_name)
            for f_name in self.selected_features
        ]

        # 3. Load lightweight model
        resolved_model_path = model_path or (config.BASE_DIR / self.cfg["final_model_artifact"])
        dropout_rate = float(self.cfg.get("dropout_rate", 0.05))
        self.model = LightweightMLP(
            in_features=self.input_dim,
            hidden_dims=self.hidden_layers,
            use_batch_norm=False,
            dropout_rate=dropout_rate,
        )
        self.model.load_state_dict(torch.load(resolved_model_path, weights_only=True, map_location="cpu"))
        self.model.eval()
        self.model.to("cpu")

        logger.info(
            f"EdgeAnomalyDetector initialized: Arch={self.hidden_layers}, "
            f"Inputs={self.input_dim}, Threshold={self.threshold:.2f}, Artifact={resolved_model_path.name}"
        )

    def preprocess_record(self, raw_record: Dict[str, Any]) -> np.ndarray:
        """Converts raw sensor/network packet dictionary to standardized feature vector."""
        df = pd.DataFrame([raw_record])
        scaled_full = self.preprocessor.transform(df)
        selected_vector = scaled_full[:, self.feature_indices]
        return selected_vector

    def predict_vector(self, feature_vector: np.ndarray) -> Tuple[float, bool]:
        """Runs feed-forward pass on feature vector and applies locked threshold."""
        x_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        with torch.no_grad():
            logit = self.model(x_tensor).squeeze()
            prob = float(torch.sigmoid(logit).item())
        is_anomaly = prob >= self.threshold
        return prob, is_anomaly

    def detect(self, raw_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        End-to-end edge screening pipeline:
        Raw Sensor Observation -> Cleaned Features -> Inference -> Alert Decision
        """
        t0 = time.perf_counter()
        vector = self.preprocess_record(raw_record)
        t_prep = time.perf_counter()

        prob, is_anomaly = self.predict_vector(vector)
        t_inf = time.perf_counter()

        prep_time_ms = round((t_prep - t0) * 1000.0, 3)
        inf_time_ms = round((t_inf - t_prep) * 1000.0, 3)
        total_time_ms = round((t_inf - t0) * 1000.0, 3)

        alert_level = "CRITICAL" if prob >= 0.85 else "HIGH" if is_anomaly else "NORMAL"

        return {
            "verdict": "ANOMALY" if is_anomaly else "NORMAL",
            "is_anomaly": bool(is_anomaly),
            "anomaly_probability": round(prob, 4),
            "confidence_pct": round((prob if is_anomaly else 1.0 - prob) * 100.0, 2),
            "threshold": self.threshold,
            "alert_level": alert_level,
            "preprocessing_ms": prep_time_ms,
            "inference_ms": inf_time_ms,
            "total_latency_ms": total_time_ms,
        }

    def detect_batch(self, raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a batch of raw records with vectorized preprocessing."""
        if not raw_records:
            return []

        df = pd.DataFrame(raw_records)
        scaled_full = self.preprocessor.transform(df)
        sel_vectors = scaled_full[:, self.feature_indices]

        x_tensor = torch.tensor(sel_vectors, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

        if isinstance(probs, float):
            probs = [probs]

        results = []
        for prob in probs:
            is_anomaly = prob >= self.threshold
            alert = "CRITICAL" if prob >= 0.85 else "HIGH" if is_anomaly else "NORMAL"
            results.append(
                {
                    "verdict": "ANOMALY" if is_anomaly else "NORMAL",
                    "is_anomaly": bool(is_anomaly),
                    "anomaly_probability": round(prob, 4),
                    "confidence_pct": round((prob if is_anomaly else 1.0 - prob) * 100.0, 2),
                    "threshold": self.threshold,
                    "alert_level": alert,
                }
            )
        return results
