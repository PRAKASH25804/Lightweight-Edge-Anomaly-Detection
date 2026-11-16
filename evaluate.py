"""
Standalone Test Evaluation Script
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Evaluates all trained models strictly on the unseen held-out test split (15%, 23,548 samples)
using the locked validation thresholds.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src import config
from src.evaluate import compute_metrics
from src.models import StandardMLP, LightweightMLP
from src.quantization import quantize_model_int8

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation():
    logger.info("=" * 65)
    logger.info("STANDALONE TEST EVALUATION ON UNSEEN HELD-OUT TEST DATA")
    logger.info("=" * 65)

    data = np.load(config.ARTIFACTS_DIR / "processed_data.npz", allow_pickle=True)
    X_test_full = data["X_test"]
    y_test = data["y_test"]

    with open(config.ARTIFACTS_DIR / "selected_features.json", "r") as f:
        feat_meta = json.load(f)
    feature_names = data["feature_names"].tolist()
    selected_features = feat_meta["selected_features"]
    selected_indices = [feature_names.index(f) for f in selected_features]
    X_test_sel = X_test_full[:, selected_indices]

    with open(config.RESULTS_DIR / "final_model_config.json", "r") as f:
        final_cfg = json.load(f)
    locked_threshold = final_cfg.get("decision_threshold", 0.50)

    results = []

    # 1. Logistic Regression
    lr_path = config.MODELS_DIR / "baseline_logistic_regression.joblib"
    if lr_path.exists():
        lr_m = joblib.load(lr_path)
        probs = lr_m.predict_proba(X_test_sel)[:, 1]
        m = compute_metrics(y_test, probs, threshold=locked_threshold)
        results.append({"Model": "Baseline: Logistic Regression", **m})

    # 2. Random Forest
    rf_path = config.MODELS_DIR / "baseline_random_forest.joblib"
    if rf_path.exists():
        rf_m = joblib.load(rf_path)
        probs = rf_m.predict_proba(X_test_sel)[:, 1]
        m = compute_metrics(y_test, probs, threshold=0.50)
        results.append({"Model": "Baseline: Random Forest", **m})

    # 3. Standard MLP
    std_path = config.MODELS_DIR / "baseline_standard_mlp.pt"
    if std_path.exists():
        std_m = StandardMLP(in_features=X_test_full.shape[1])
        std_m.load_state_dict(torch.load(std_path, weights_only=True, map_location="cpu"))
        std_m.eval()
        with torch.no_grad():
            probs = torch.sigmoid(std_m(torch.tensor(X_test_full, dtype=torch.float32)).squeeze()).numpy()
        m = compute_metrics(y_test, probs, threshold=locked_threshold)
        results.append({"Model": "Baseline: Standard MLP (Full 40)", **m})

    # 4. Proposed Lightweight (FP32)
    prop_path = config.MODELS_DIR / "proposed_lightweight_model.pt"
    if prop_path.exists():
        prop_m = LightweightMLP(in_features=len(selected_indices), hidden_dims=[32, 16], dropout_rate=0.05)
        prop_m.load_state_dict(torch.load(prop_path, weights_only=True, map_location="cpu"))
        prop_m.eval()
        with torch.no_grad():
            probs = torch.sigmoid(prop_m(torch.tensor(X_test_sel, dtype=torch.float32)).squeeze()).numpy()
        m = compute_metrics(y_test, probs, threshold=locked_threshold)
        results.append({"Model": "Proposed Lightweight (FP32, Top 30)", **m})

    # 5. Proposed Lightweight (Pruned 75%)
    pruned_path = config.MODELS_DIR / "proposed_lightweight_model_pruned.pt"
    if pruned_path.exists():
        p_m = LightweightMLP(in_features=len(selected_indices), hidden_dims=[32, 16], dropout_rate=0.05)
        p_m.load_state_dict(torch.load(pruned_path, weights_only=True, map_location="cpu"))
        p_m.eval()
        with torch.no_grad():
            probs = torch.sigmoid(p_m(torch.tensor(X_test_sel, dtype=torch.float32)).squeeze()).numpy()
        m = compute_metrics(y_test, probs, threshold=locked_threshold)
        results.append({"Model": "Proposed Lightweight (Pruned 75%)", **m})

    # 6. Proposed Lightweight (INT8 Quantized)
    q_path = config.MODELS_DIR / "proposed_lightweight_model_quantized.pt"
    if q_path.exists() and prop_path.exists():
        base_m = LightweightMLP(in_features=len(selected_indices), hidden_dims=[32, 16], dropout_rate=0.05)
        base_m.load_state_dict(torch.load(prop_path, weights_only=True, map_location="cpu"))
        q_m = quantize_model_int8(base_m)
        q_m.eval()
        with torch.no_grad():
            probs = torch.sigmoid(q_m(torch.tensor(X_test_sel, dtype=torch.float32)).squeeze()).numpy()
        m = compute_metrics(y_test, probs, threshold=locked_threshold)
        results.append({"Model": "Proposed Lightweight (INT8 Quantized)", **m})

    df = pd.DataFrame(results)
    out_csv = config.RESULTS_DIR / "test_evaluation_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Test evaluation results saved to {out_csv}")

    print("\n" + "=" * 90)
    print("UNSEEN TEST SET PERFORMANCE SUMMARY (N = 23,548 samples)")
    print("=" * 90)
    cols_to_print = ["Model", "accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "tp", "fn", "tn", "fp"]
    print(df[cols_to_print].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    run_evaluation()
