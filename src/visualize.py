"""
Research Publication Visualization Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Generates 14 publication-grade figures from real experimental results:
1. Class distribution (Normal vs 14 Attack subcategories)
2. Feature importance ranking (Tree importance vs ANOVA F-stat)
3. Confusion matrix for the final selected model
4. ROC curves (Baselines vs Proposed Lightweight models)
5. Precision-Recall curves
6. Model F1 score comparison
7. Model storage size comparison (KB)
8. Inference latency comparison (ms)
9. RAM memory consumption comparison (MB)
10. Trainable parameter count comparison
11. Accuracy vs Model Storage Size trade-off
12. F1 Score vs Inference Latency Pareto frontier
13. Normalized Accuracy-Efficiency composite trade-off
14. Ablation study progression (Base MLP -> Compression -> Quantization)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from src import config
from src.models import StandardMLP, LightweightMLP, load_sklearn_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Research styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)


def generate_all_plots():
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating 14 publication-grade plots from real experimental results...")

    # Load experimental data
    final_df = pd.read_csv(config.RESULTS_DIR / "final_results.csv")
    ablation_df = pd.read_csv(config.RESULTS_DIR / "ablation_results.csv")
    rankings_df = pd.read_csv(config.RESULTS_DIR / "feature_importance_ranking.csv")
    bench_df = pd.read_csv(config.RESULTS_DIR / "edge_benchmark.csv")
    with open(config.RESULTS_DIR / "final_model_config.json") as f:
        final_cfg = json.load(f)
    with open(config.RESULTS_DIR / "dataset_profile.json") as f:
        profile_cfg = json.load(f)

    # Load test set arrays
    data = np.load(config.ARTIFACTS_DIR / "processed_data.npz", allow_pickle=True)
    X_test_full = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    with open(config.ARTIFACTS_DIR / "selected_features.json") as f:
        feat_meta = json.load(f)
    selected_indices = feat_meta["feature_indices"]
    X_test_sel = X_test_full[:, selected_indices]

    # Helper: resolve clean display names
    def get_display_name(row):
        name = str(row.get("Model_Name", row.get("Model", row.get("Experiment"))))
        if "Baseline 1: Logistic Regression" in name:
            return "Logistic Reg"
        if "Baseline 2: Random Forest" in name:
            return "Random Forest"
        if "Baseline 3: Standard MLP" in name:
            return "Standard MLP"
        if "Exp A:" in name:
            return "Std MLP (Full)"
        if "Exp B:" in name:
            return "Light MLP (Full)"
        if "Exp C:" in name:
            return "Light MLP + FS (Proposed)"
        if "Exp D:" in name:
            return "Pruned (75%)"
        if "Exp E:" in name:
            return "INT8 Quantized"
        if "Exp F:" in name:
            return "Pruned + INT8"
        return name[:18]

    final_df["Display_Name"] = final_df.apply(get_display_name, axis=1)

    # -------------------------------------------------------------
    # PLOT 1: Class Distribution
    # -------------------------------------------------------------
    logger.info("Plot 1: Class Distribution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Binary distribution
    bin_counts = profile_cfg["target_binary_distribution"]
    colors_bin = ["#2b5c8f", "#d95f02"]
    ax1.bar(["Normal (0)", "Anomaly (1)"], [bin_counts["Normal (0)"], bin_counts["Anomaly (1)"]], color=colors_bin, width=0.5, edgecolor="black")
    for i, v in enumerate([bin_counts["Normal (0)"], bin_counts["Anomaly (1)"]]):
        pct = v / profile_cfg["total_rows"] * 100
        ax1.text(i, v + 2500, f"{v:,}\n({pct:.1f}%)", ha="center", fontweight="bold")
    ax1.set_title("Edge-IIoTset Binary Class Distribution", fontweight="bold")
    ax1.set_ylabel("Record Count")
    ax1.set_ylim(0, max(bin_counts.values()) * 1.18)

    # Attack Subcategories distribution
    atk_dist = profile_cfg["attack_types_distribution"]
    sorted_atk = sorted(atk_dist.items(), key=lambda x: x[1], reverse=True)
    atk_names = [k for k, v in sorted_atk]
    atk_vals = [v for k, v in sorted_atk]
    palette = ["#2b5c8f" if k == "Normal" else "#e7298a" if "DDoS" in k else "#7570b3" for k in atk_names]

    y_pos = np.arange(len(atk_names))
    ax2.barh(y_pos, atk_vals, color=palette, edgecolor="black", height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(atk_names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_title("Traffic Distribution across 14 Attack Subclasses", fontweight="bold")
    ax2.set_xlabel("Records")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "01_class_distribution.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 2: Feature Importance Ranking
    # -------------------------------------------------------------
    logger.info("Plot 2: Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    top20_feats = rankings_df.head(20)
    sns.barplot(data=top20_feats, x="tree_importance", y="feature", palette="viridis", edgecolor="black", ax=ax)
    ax.set_title("Top-20 Features Ranked by Tree-Based Gini Importance", fontweight="bold")
    ax.set_xlabel("Relative Feature Importance Score")
    ax.set_ylabel("Feature Name")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "02_feature_importance.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 3: Confusion Matrix for Final Model
    # -------------------------------------------------------------
    logger.info("Plot 3: Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cm_dict = final_cfg["test_metrics"]["confusion_matrix"]
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal (0)", "Anomaly (1)"],
        yticklabels=["Normal (0)", "Anomaly (1)"],
        annot_kws={"size": 13, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: Proposed Lightweight Model\n(F1: {final_cfg['test_metrics']['f1']:.4f}, Recall: {final_cfg['test_metrics']['recall']:.4f})", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "03_confusion_matrix_final_model.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 4 & 5: ROC Curves & PR Curves
    # -------------------------------------------------------------
    logger.info("Plot 4 & 5: ROC and PR Curves")
    # Generate probabilities for key models
    lr_m = load_sklearn_model(config.MODELS_DIR / "baseline_logistic_regression.joblib")
    rf_m = load_sklearn_model(config.MODELS_DIR / "baseline_random_forest.joblib")
    std_m = StandardMLP(in_features=X_test_full.shape[1])
    std_m.load_state_dict(torch.load(config.MODELS_DIR / "baseline_standard_mlp.pt", weights_only=True))
    std_m.eval()

    prop_m = LightweightMLP(in_features=len(selected_indices), hidden_dims=[32, 16], dropout_rate=0.05)
    prop_m.load_state_dict(torch.load(config.MODELS_DIR / "proposed_lightweight_model.pt", weights_only=True))
    prop_m.eval()

    with torch.no_grad():
        p_lr = lr_m.predict_proba(X_test_sel)[:, 1]
        p_rf = rf_m.predict_proba(X_test_sel)[:, 1]
        p_std = torch.sigmoid(std_m(torch.tensor(X_test_full, dtype=torch.float32)).squeeze()).numpy()
        p_prop = torch.sigmoid(prop_m(torch.tensor(X_test_sel, dtype=torch.float32)).squeeze()).numpy()

    # ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for p, label, col, ls in [
        (p_lr, "Logistic Regression", "#1f78b4", "--"),
        (p_rf, "Random Forest", "#33a02c", ":"),
        (p_std, "Standard MLP (Full 40)", "#ff7f00", "-."),
        (p_prop, "Proposed Lightweight (Top 30)", "#e31a1c", "-"),
    ]:
        fpr_v, tpr_v, _ = roc_curve(y_test, p)
        roc_val = auc(fpr_v, tpr_v)
        ax.plot(fpr_v, tpr_v, label=f"{label} (AUC = {roc_val:.4f})", color=col, linestyle=ls, linewidth=2.2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "04_roc_curves.png", dpi=300)
    plt.close(fig)

    # PR Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for p, label, col, ls in [
        (p_lr, "Logistic Regression", "#1f78b4", "--"),
        (p_rf, "Random Forest", "#33a02c", ":"),
        (p_std, "Standard MLP (Full 40)", "#ff7f00", "-."),
        (p_prop, "Proposed Lightweight (Top 30)", "#e31a1c", "-"),
    ]:
        prec_v, rec_v, _ = precision_recall_curve(y_test, p)
        pr_val = auc(rec_v, prec_v)
        ax.plot(rec_v, prec_v, label=f"{label} (PR-AUC = {pr_val:.4f})", color=col, linestyle=ls, linewidth=2.2)

    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (PR) Curves", fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "05_precision_recall_curves.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 6: Model F1 Score Comparison
    # -------------------------------------------------------------
    logger.info("Plot 6: Model F1 Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_sorted = final_df.drop_duplicates(subset=["Display_Name"]).sort_values("f1", ascending=True)
    colors = ["#2b5c8f" if "Proposed" in n or "Light" in n else "#999999" for n in df_sorted["Display_Name"]]
    bars = ax.barh(df_sorted["Display_Name"], df_sorted["f1"], color=colors, edgecolor="black", height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w - 0.05, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="right", color="white", fontweight="bold")
    ax.set_title("Detection Performance: F1-Score Comparison", fontweight="bold")
    ax.set_xlabel("F1-Score")
    ax.set_xlim(0.7, 1.02)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "06_model_f1_comparison.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 7: Model Size Comparison (KB)
    # -------------------------------------------------------------
    logger.info("Plot 7: Model Size Comparison")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    df_size = final_df.drop_duplicates(subset=["Display_Name"]).sort_values("Model_Size_KB", ascending=False)
    colors_size = ["#d95f02" if "Random Forest" in n else "#2b5c8f" if "Proposed" in n or "INT8" in n else "#7570b3" for n in df_size["Display_Name"]]
    bars = ax.bar(df_size["Display_Name"], df_size["Model_Size_KB"], color=colors_size, edgecolor="black", width=0.6)
    ax.set_yscale("log")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15, f"{h:.1f} KB", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Physical Storage Footprint (Log Scale KB)", fontweight="bold")
    ax.set_ylabel("Model Size (KB, Log Scale)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "07_model_size_comparison.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 8: Latency Comparison (ms)
    # -------------------------------------------------------------
    logger.info("Plot 8: Latency Comparison")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    df_lat = final_df.drop_duplicates(subset=["Display_Name"]).sort_values("avg_latency_ms", ascending=False)
    bars = ax.bar(df_lat["Display_Name"], df_lat["avg_latency_ms"], color="#1b9e77", edgecolor="black", width=0.6)
    ax.set_yscale("log")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15, f"{h:.3f} ms", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Single-Sample Inference Latency on CPU (Log Scale ms)", fontweight="bold")
    ax.set_ylabel("Inference Latency (ms, Log Scale)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "08_latency_comparison.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 9: RAM Comparison (MB)
    # -------------------------------------------------------------
    logger.info("Plot 9: RAM Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    b1_ram = bench_df[bench_df["Batch_Size"] == 1]
    b1_ram = b1_ram.drop_duplicates(subset=["Model"])
    bars = ax.bar(b1_ram["Model"], b1_ram["Peak_RAM_MB"], color="#386cb0", edgecolor="black", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.1f} MB", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Peak Resident RAM Usage during Edge Inference", fontweight="bold")
    ax.set_ylabel("Peak RAM (MB)")
    ax.set_ylim(0, max(b1_ram["Peak_RAM_MB"]) * 1.25)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "09_ram_comparison.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 10: Parameter Count Comparison
    # -------------------------------------------------------------
    logger.info("Plot 10: Parameter Count Comparison")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    df_p = final_df.drop_duplicates(subset=["Display_Name"]).sort_values("Parameters", ascending=False)
    bars = ax.bar(df_p["Display_Name"], df_p["Parameters"], color="#7570b3", edgecolor="black", width=0.6)
    ax.set_yscale("log")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.15, f"{int(h):,}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Model Complexity: Parameter Count (Log Scale)", fontweight="bold")
    ax.set_ylabel("Parameters (Log Scale)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "10_parameter_count_comparison.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 11: Accuracy vs Model Size
    # -------------------------------------------------------------
    logger.info("Plot 11: Accuracy vs Model Size")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in final_df.drop_duplicates(subset=["Display_Name"]).iterrows():
        ax.scatter(row["Model_Size_KB"], row["accuracy"], s=140, edgecolor="black", alpha=0.85)
        ax.annotate(row["Display_Name"], (row["Model_Size_KB"] * 1.08, row["accuracy"] - 0.003), fontsize=9)
    ax.set_xscale("log")
    ax.set_title("Trade-off: Accuracy vs Model Storage Size", fontweight="bold")
    ax.set_xlabel("Model Size (KB, Log Scale)")
    ax.set_ylabel("Detection Accuracy")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "11_accuracy_vs_model_size.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 12: F1 vs Latency (Pareto Frontier)
    # -------------------------------------------------------------
    logger.info("Plot 12: F1 vs Latency")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in final_df.drop_duplicates(subset=["Display_Name"]).iterrows():
        color = "#e31a1c" if "Proposed" in row["Display_Name"] else "#1f78b4"
        ax.scatter(row["avg_latency_ms"], row["f1"], s=150, color=color, edgecolor="black", alpha=0.9)
        ax.annotate(row["Display_Name"], (row["avg_latency_ms"] * 1.08, row["f1"] - 0.004), fontsize=9)
    ax.set_xscale("log")
    ax.set_title("Pareto Analysis: F1-Score vs Inference Latency", fontweight="bold")
    ax.set_xlabel("Average Single-Sample Latency (ms, Log Scale)")
    ax.set_ylabel("Detection F1-Score")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "12_f1_vs_latency.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 13: Accuracy-Efficiency Trade-off Score
    # -------------------------------------------------------------
    logger.info("Plot 13: Accuracy-Efficiency Trade-off Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_eff = final_df.drop_duplicates(subset=["Display_Name"]).sort_values("Efficiency_Score", ascending=True)
    colors = ["#2ca02c" if "Proposed" in n or "Light" in n else "#1f77b4" for n in df_eff["Display_Name"]]
    bars = ax.barh(df_eff["Display_Name"], df_eff["Efficiency_Score"], color=colors, edgecolor="black", height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontweight="bold")
    ax.set_title("Normalized Accuracy–Efficiency Score (Higher is Better)", fontweight="bold")
    ax.set_xlabel("Efficiency Score = Detection Quality / (1.0 + Normalized Resource Cost)")
    ax.set_xlim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "13_accuracy_efficiency_tradeoff.png", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # PLOT 14: Ablation Study Comparison
    # -------------------------------------------------------------
    logger.info("Plot 14: Ablation Study Progression")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    exp_labels = ["Exp A\n(Base MLP)", "Exp B\n(Light MLP)", "Exp C\n(+Feat Sel)", "Exp D\n(+Pruning)", "Exp E\n(+INT8)", "Exp F\n(+Prune+INT8)"]

    f1_vals = ablation_df["f1"].tolist()
    size_vals = ablation_df["Model_Size_KB"].tolist()
    params_vals = ablation_df["Parameters"].tolist()

    x = np.arange(len(exp_labels))
    ax1.plot(x, f1_vals, marker="o", color="#e41a1c", linewidth=2.5, markersize=8, label="F1-Score")
    for i, txt in enumerate(f1_vals):
        ax1.annotate(f"{txt:.4f}", (x[i], f1_vals[i] + 0.005), ha="center", fontweight="bold", fontsize=9)
    ax1.set_ylabel("F1-Score")
    ax1.set_ylim(0.90, 1.01)
    ax1.set_title("Ablation Study: Detection Retention vs Resource Compression", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left")

    bar_width = 0.35
    ax2.bar(x - bar_width / 2, size_vals, width=bar_width, color="#377eb8", edgecolor="black", label="Model Size (KB)")
    ax2.set_ylabel("Model Storage (KB)")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_labels, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "14_ablation_study_comparison.png", dpi=300)
    plt.close(fig)

    logger.info("All 14 publication plots successfully generated and saved to plots/.")


if __name__ == "__main__":
    generate_all_plots()

