"""
Evaluation and Metrics Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Computes:
- Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, FPR, Confusion Matrix
- Validation-set Threshold Optimization (prioritizing recall and FPR control)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes all essential detection metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, Confusion Matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate: FP / (FP + TN)
    fpr = float(fp / max(1, (fp + tn)))

    return {
        "accuracy": round(acc, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "fpr": round(fpr, 5),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": round(float(threshold), 3),
    }


def optimize_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    candidate_thresholds: List[float] = config.CANDIDATE_THRESHOLDS,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates candidate thresholds strictly on the validation set.
    Prioritizes high anomaly recall while strictly penalizing excessive false positive rates.
    Selects the optimal threshold on validation data ONLY (never tuning on the final test set).
    """
    threshold_results = []
    best_threshold = 0.50
    best_score = -1.0
    best_metrics = {}

    for thresh in candidate_thresholds:
        metrics = compute_metrics(y_val, val_probs, threshold=thresh)
        # Selection criterion: Harmonic balance of F1 with penalty if FPR > 0.05
        # Score = F1 - 2.0 * max(0, FPR - 0.02)
        score = metrics["f1"] - 2.0 * max(0.0, metrics["fpr"] - 0.02)

        metrics["selection_score"] = round(float(score), 5)
        threshold_results.append(metrics)

        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = metrics

    logger.info(
        f"Validation Threshold Optimization: Selected threshold = {best_threshold:.2f} "
        f"(Val F1={best_metrics['f1']:.4f}, Recall={best_metrics['recall']:.4f}, FPR={best_metrics['fpr']:.4f})"
    )

    return best_threshold, best_metrics, threshold_results

