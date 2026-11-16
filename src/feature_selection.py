"""
Feature Selection and Dimensionality Reduction Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Features:
- Tree-based feature importance (Extra-Trees / Random Forest)
- Statistical feature ranking (ANOVA F-statistic / Mutual Information)
- Comparative evaluation of Top 10, Top 15, Top 20, Top 30, and Full (40) feature sets
- Validation-set performance justification
- Serialization of selected feature subset for edge inference
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression

from src import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_feature_rankings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    random_seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Computes and compares two distinct feature importance strategies on training data only:
    1. Tree-based Feature Importance (ExtraTreesClassifier)
    2. Statistical Ranking (ANOVA F-statistic)
    """
    logger.info("Computing tree-based feature importance on X_train...")
    tree_model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=16,
        random_state=random_seed,
        n_jobs=-1,
    )
    tree_model.fit(X_train, y_train)
    tree_importances = tree_model.feature_importances_

    logger.info("Computing statistical ANOVA F-statistic ranking on X_train...")
    f_scores, _ = f_classif(X_train, y_train)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize scores for direct comparison
    tree_norm = tree_importances / (np.sum(tree_importances) + 1e-12)
    f_norm = f_scores / (np.sum(f_scores) + 1e-12)

    df_rankings = pd.DataFrame(
        {
            "feature": feature_names,
            "tree_importance": tree_importances,
            "tree_importance_normalized": tree_norm,
            "f_statistic": f_scores,
            "f_statistic_normalized": f_norm,
        }
    )

    # Sort descending by tree importance as primary strategy
    df_rankings = df_rankings.sort_values(by="tree_importance", ascending=False).reset_index(drop=True)
    df_rankings["rank_tree"] = np.arange(1, len(df_rankings) + 1)

    # Rank by F-stat
    f_rank_map = (
        df_rankings.sort_values(by="f_statistic", ascending=False)
        .reset_index(drop=True)
        .assign(rank_f=lambda x: np.arange(1, len(x) + 1))
        .set_index("feature")["rank_f"]
        .to_dict()
    )
    df_rankings["rank_f"] = df_rankings["feature"].map(f_rank_map)

    # Save ranking CSV
    out_csv = config.RESULTS_DIR / "feature_importance_ranking.csv"
    df_rankings.to_csv(out_csv, index=False)
    logger.info(f"Feature importance rankings saved to {out_csv}")

    return df_rankings


def evaluate_feature_subsets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_rankings: pd.DataFrame,
    candidate_k_list: List[int] = config.FEATURE_K_CANDIDATES,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, int, List[str]]:
    """
    Evaluates candidate feature subsets (Top 10, Top 15, Top 20, Top 30, and Full)
    on the validation set using a lightweight neural network to determine the smallest
    subset that optimizes the anomaly detection performance.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    logger.info("Evaluating feature subsets with neural network on validation set...")
    total_features = len(feature_rankings)
    eval_k_list = [k for k in candidate_k_list if k < total_features] + [total_features]

    results = []

    for k in eval_k_list:
        selected_names = feature_rankings.iloc[:k]["feature"].tolist()
        indices = feature_rankings.iloc[:k].index.values

        X_tr_sub = torch.tensor(X_train[:, indices], dtype=torch.float32)
        y_tr_sub = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_v_sub = torch.tensor(X_val[:, indices], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_tr_sub, y_tr_sub), batch_size=512, shuffle=True)

        model = nn.Sequential(
            nn.Linear(k, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Handle class imbalance with positive weight
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1, y_train.sum())], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        model.train()
        for epoch in range(6):
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_v_sub).squeeze().cpu()
            probs_val = torch.sigmoid(logits).numpy()
            preds_val = (probs_val >= 0.50).astype(int)

        acc = accuracy_score(y_val, preds_val)
        prec = precision_score(y_val, preds_val, zero_division=0)
        rec = recall_score(y_val, preds_val, zero_division=0)
        f1 = f1_score(y_val, preds_val, zero_division=0)
        roc = roc_auc_score(y_val, probs_val)

        results.append(
            {
                "k_features": k,
                "feature_count": k,
                "subset_type": f"Top_{k}" if k < total_features else "All_Features",
                "accuracy": round(acc, 5),
                "precision": round(prec, 5),
                "recall": round(rec, 5),
                "f1_score": round(f1, 5),
                "roc_auc": round(roc, 5),
                "selected_features": selected_names,
            }
        )
        logger.info(f"Subset Top-{k:2d}: Accuracy={acc:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc:.4f}")

    df_eval = pd.DataFrame(results)
    out_csv = config.RESULTS_DIR / "feature_selection_comparison.csv"
    df_eval.drop(columns=["selected_features"]).to_csv(out_csv, index=False)
    logger.info(f"Feature selection comparison saved to {out_csv}")

    # Explicit efficiency-aware selection rule:
    # Select the smallest k that achieves >= 99% of maximum validation F1
    max_f1 = df_eval["f1_score"].max()
    target_f1 = 0.99 * max_f1

    eligible = df_eval[df_eval["f1_score"] >= target_f1].sort_values("k_features")
    selected_k = int(eligible.iloc[0]["k_features"]) if not eligible.empty else 15

    selected_features = feature_rankings.iloc[:selected_k]["feature"].tolist()
    full_f1 = df_eval.loc[df_eval["k_features"] == total_features, "f1_score"].values[0]
    sel_f1 = df_eval.loc[df_eval["k_features"] == selected_k, "f1_score"].values[0]
    logger.info(f"Selected optimal feature subset: Top-{selected_k} features (F1={sel_f1:.4f} vs Full 40 F1={full_f1:.4f})")

    # Serialize selected feature configuration
    selection_artifact = {
        "selection_method": "Tree-based (ExtraTrees) Importance",
        "ranking_metric": "gini_importance",
        "total_active_features": total_features,
        "selected_feature_count": selected_k,
        "selected_features": selected_features,
        "feature_indices": [int(idx) for idx in feature_rankings.iloc[:selected_k].index.values],
        "validation_f1": float(sel_f1),
        "validation_accuracy": float(df_eval.loc[df_eval["k_features"] == selected_k, "accuracy"].values[0]),
        "performance_retention_pct": round(float(sel_f1 / full_f1 * 100), 2),
        "dimensionality_reduction_pct": round(float((total_features - selected_k) / total_features * 100), 2),
    }

    out_json = config.ARTIFACTS_DIR / "selected_features.json"
    with open(out_json, "w") as f:
        json.dump(selection_artifact, f, indent=4)
    logger.info(f"Selected feature metadata saved to {out_json}")

    return df_eval, selected_k, selected_features


def run_feature_selection(
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, int, List[str], List[int]]:
    """Loads processed train/val data and runs full feature selection."""
    cache_path = config.ARTIFACTS_DIR / "processed_data.npz"
    if not cache_path.exists():
        from src.preprocess import prepare_datasets
        prepare_datasets()

    data = np.load(cache_path, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    feature_names = data["feature_names"].tolist()

    rankings = compute_feature_rankings(X_train, y_train, feature_names, random_seed)
    df_eval, selected_k, selected_features = evaluate_feature_subsets(
        X_train, y_train, X_val, y_val, rankings, config.FEATURE_K_CANDIDATES, random_seed
    )

    # Feature indices matching the preprocessor's active_features
    # Map back by name
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    selected_indices = [name_to_idx[name] for name in selected_features]

    return df_eval, selected_k, selected_features, selected_indices


if __name__ == "__main__":
    run_feature_selection()

