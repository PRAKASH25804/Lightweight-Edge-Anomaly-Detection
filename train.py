"""
Master Experimentation and Training Pipeline
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Executes:
- Phase 3: Classical & Deep Baselines (Logistic Regression, Random Forest, Standard MLP)
- Phase 4: Architecture Search (Arch A: 32->16, Arch B: 64->32->16, Arch C: 32->16->8, Arch D: 16->8)
- Phase 5: Systematic Ablation Study (Exp A through Exp F)
- Phase 6: Post-Training INT8 Quantization & Trade-off Analysis
- Phase 7: Weight Pruning across Sparsity Levels (25%, 50%, 75%)
- Phase 10: Validation-Only Decision Threshold Optimization
- Phase 11: Class Imbalance Handling
- Phase 12: Comprehensive CSV Experiment Tracking
- Phase 14: Efficiency-Aware Model Selection
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.benchmark import benchmark_inference
from src.evaluate import compute_metrics, optimize_threshold
from src.models import (
    StandardMLP,
    LightweightMLP,
    count_parameters,
    get_model_file_size_kb,
    create_baseline_models,
    save_sklearn_model,
)
from src.pruning import apply_unstructured_pruning, calculate_model_sparsity
from src.quantization import quantize_model_int8, save_quantized_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = config.RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_neural_network(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    use_class_weight: bool = True,
    model_name: str = "NeuralNetwork",
) -> Tuple[nn.Module, float]:
    """
    Trains a neural network on CPU using PyTorch.
    Implements numerical stability, Adam optimizer, class weighting,
    and validation checkpointing.
    """
    set_seed(config.RANDOM_SEED)
    model.train()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Phase 11: Class Imbalance Weighting
    if use_class_weight:
        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = -1.0
    best_weights = copy_weights = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val, dtype=torch.float32)).squeeze().cpu()
            val_probs = torch.sigmoid(val_logits).numpy()
            val_metrics = compute_metrics(y_val, val_probs, threshold=0.50)

        scheduler.step(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 3 == 0 or epoch == epochs:
            logger.info(
                f"[{model_name}] Epoch {epoch:2d}/{epochs:2d} - Loss: {epoch_loss/len(loader):.4f} - Val F1: {val_metrics['f1']:.4f} (Rec: {val_metrics['recall']:.4f})"
            )

    train_time = round(time.time() - t0, 2)
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model, train_time


def predict_probs_nn(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Computes calibrated probability outputs from neural network logits."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze().cpu()
        probs = torch.sigmoid(logits).numpy()
    return probs


def run_full_pipeline():
    set_seed(config.RANDOM_SEED)
    logger.info("=" * 70)
    logger.info("STARTING NNDP RESEARCH EXPERIMENT PIPELINE")
    logger.info("=" * 70)

    # 1. Load preprocessed dataset
    data_path = config.ARTIFACTS_DIR / "processed_data.npz"
    if not data_path.exists():
        from src.preprocess import prepare_datasets
        prepare_datasets()

    data = np.load(data_path, allow_pickle=True)
    X_train_full = data["X_train"]
    y_train = data["y_train"]
    X_val_full = data["X_val"]
    y_val = data["y_val"]
    X_test_full = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"].tolist()

    # Load selected features
    feat_path = config.ARTIFACTS_DIR / "selected_features.json"
    if not feat_path.exists():
        from src.feature_selection import run_feature_selection
        run_feature_selection()

    with open(feat_path, "r") as f:
        feat_meta = json.load(f)

    selected_feature_names = feat_meta["selected_features"]
    selected_indices = [feature_names.index(name) for name in selected_feature_names]
    logger.info(f"Loaded {len(selected_indices)} selected features: {selected_feature_names}")

    # Create sub-matrices with selected features
    X_train_sel = X_train_full[:, selected_indices]
    X_val_sel = X_val_full[:, selected_indices]
    X_test_sel = X_test_full[:, selected_indices]

    # Sample for latency benchmarking
    bench_sample_full = X_test_full[:100]
    bench_sample_sel = X_test_sel[:100]

    # =========================================================================
    # PHASE 3 — BASELINE MODELS
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: TRAINING AND EVALUATING BASELINE MODELS")
    logger.info("=" * 50)

    baseline_records = []
    baselines = create_baseline_models(config.RANDOM_SEED)

    # Baseline 1: Logistic Regression
    logger.info("Training Baseline 1: Logistic Regression...")
    lr_model = baselines["Logistic_Regression"]
    t0 = time.time()
    lr_model.fit(X_train_sel, y_train)
    lr_train_time = round(time.time() - t0, 2)
    lr_path = config.MODELS_DIR / "baseline_logistic_regression.joblib"
    save_sklearn_model(lr_model, lr_path)
    lr_size_kb = get_model_file_size_kb(lr_path)

    # Val threshold optimization
    lr_val_probs = lr_model.predict_proba(X_val_sel)[:, 1]
    lr_thresh, lr_val_met, _ = optimize_threshold(y_val, lr_val_probs)

    # Test evaluation
    lr_test_probs = lr_model.predict_proba(X_test_sel)[:, 1]
    lr_test_met = compute_metrics(y_test, lr_test_probs, threshold=lr_thresh)
    lr_bench = benchmark_inference(lr_model, bench_sample_sel, is_torch=False)

    lr_record = {
        "Model": "Baseline 1: Logistic Regression",
        "Type": "Classical Linear",
        "Feature_Count": len(selected_indices),
        "Architecture": f"{len(selected_indices)}->1",
        "Parameters": int(len(selected_indices) + 1),
        "Model_Size_KB": lr_size_kb,
        "Threshold": lr_thresh,
        **lr_test_met,
        **lr_bench,
        "Train_Time_s": lr_train_time,
    }
    baseline_records.append(lr_record)
    logger.info(f"Logistic Regression: Test F1={lr_test_met['f1']:.4f}, Latency={lr_bench['avg_latency_ms']:.3f} ms, Size={lr_size_kb:.1f} KB")

    # Baseline 2: Random Forest
    logger.info("Training Baseline 2: Random Forest...")
    rf_model = baselines["Random_Forest"]
    t0 = time.time()
    rf_model.fit(X_train_sel, y_train)
    rf_train_time = round(time.time() - t0, 2)
    rf_path = config.MODELS_DIR / "baseline_random_forest.joblib"
    save_sklearn_model(rf_model, rf_path)
    rf_size_kb = get_model_file_size_kb(rf_path)

    rf_val_probs = rf_model.predict_proba(X_val_sel)[:, 1]
    rf_thresh, rf_val_met, _ = optimize_threshold(y_val, rf_val_probs)

    rf_test_probs = rf_model.predict_proba(X_test_sel)[:, 1]
    rf_test_met = compute_metrics(y_test, rf_test_probs, threshold=rf_thresh)
    rf_bench = benchmark_inference(rf_model, bench_sample_sel, is_torch=False)

    # Estimate node parameters in RF
    rf_total_nodes = sum(tree.tree_.node_count for tree in rf_model.estimators_)
    rf_record = {
        "Model": "Baseline 2: Random Forest",
        "Type": "Ensemble Trees",
        "Feature_Count": len(selected_indices),
        "Architecture": f"100_Trees_MaxDepth16",
        "Parameters": int(rf_total_nodes),
        "Model_Size_KB": rf_size_kb,
        "Threshold": rf_thresh,
        **rf_test_met,
        **rf_bench,
        "Train_Time_s": rf_train_time,
    }
    baseline_records.append(rf_record)
    logger.info(f"Random Forest: Test F1={rf_test_met['f1']:.4f}, Latency={rf_bench['avg_latency_ms']:.3f} ms, Size={rf_size_kb:.1f} KB")

    # Baseline 3: Standard MLP (Full features)
    logger.info("Training Baseline 3: Standard MLP (Full Features)...")
    std_mlp = StandardMLP(in_features=X_train_full.shape[1])
    std_mlp, std_train_time = train_neural_network(
        std_mlp, X_train_full, y_train, X_val_full, y_val, epochs=12, model_name="StandardMLP"
    )
    std_mlp_path = config.MODELS_DIR / "baseline_standard_mlp.pt"
    torch.save(std_mlp.state_dict(), std_mlp_path)
    std_mlp_size_kb = get_model_file_size_kb(std_mlp_path)
    std_params, _ = count_parameters(std_mlp)

    std_val_probs = predict_probs_nn(std_mlp, X_val_full)
    std_thresh, std_val_met, _ = optimize_threshold(y_val, std_val_probs)

    std_test_probs = predict_probs_nn(std_mlp, X_test_full)
    std_test_met = compute_metrics(y_test, std_test_probs, threshold=std_thresh)
    std_bench = benchmark_inference(std_mlp, bench_sample_full, is_torch=True)

    std_record = {
        "Model": "Baseline 3: Standard MLP",
        "Type": "Deep Dense",
        "Feature_Count": X_train_full.shape[1],
        "Architecture": f"{X_train_full.shape[1]}->128->64->32->1",
        "Parameters": int(std_params),
        "Model_Size_KB": std_mlp_size_kb,
        "Threshold": std_thresh,
        **std_test_met,
        **std_bench,
        "Train_Time_s": std_train_time,
    }
    baseline_records.append(std_record)
    logger.info(f"Standard MLP: Test F1={std_test_met['f1']:.4f}, Latency={std_bench['avg_latency_ms']:.3f} ms, Size={std_mlp_size_kb:.1f} KB")

    df_baselines = pd.DataFrame(baseline_records)
    df_baselines.to_csv(config.RESULTS_DIR / "baseline_results.csv", index=False)
    logger.info("Baseline results saved to results/baseline_results.csv")

    # =========================================================================
    # PHASE 4 — PROPOSED LIGHTWEIGHT ARCHITECTURE SEARCH
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: ARCHITECTURE SEARCH FOR LIGHTWEIGHT NEURAL NETWORK")
    logger.info("=" * 50)

    candidate_archs = {
        "Arch_A_32_16": [32, 16],
        "Arch_B_64_32_16": [64, 32, 16],  # PPT proposed reference
        "Arch_C_32_16_8": [32, 16, 8],
        "Arch_D_16_8": [16, 8],
    }

    arch_search_records = []
    trained_arch_models = {}

    in_dim_sel = len(selected_indices)

    for arch_name, hidden_dims in candidate_archs.items():
        logger.info(f"Evaluating candidate architecture {arch_name} (Layers: {hidden_dims})...")
        arch_model = LightweightMLP(
            in_features=in_dim_sel,
            hidden_dims=hidden_dims,
            use_batch_norm=False,
            dropout_rate=0.05,
        )
        arch_model, arch_tr_time = train_neural_network(
            arch_model, X_train_sel, y_train, X_val_sel, y_val, epochs=12, model_name=arch_name
        )
        trained_arch_models[arch_name] = arch_model

        # Model complexity
        n_params, _ = count_parameters(arch_model)
        temp_path = config.MODELS_DIR / f"temp_{arch_name}.pt"
        torch.save(arch_model.state_dict(), temp_path)
        size_kb = get_model_file_size_kb(temp_path)
        if temp_path.exists():
            temp_path.unlink()

        # Validation metrics
        v_probs = predict_probs_nn(arch_model, X_val_sel)
        opt_thresh, val_met, _ = optimize_threshold(y_val, v_probs)

        # Benchmarking
        bench = benchmark_inference(arch_model, bench_sample_sel, is_torch=True)

        rec = {
            "Architecture_ID": arch_name,
            "Hidden_Layers": str(hidden_dims),
            "Parameter_Count": n_params,
            "Model_Size_KB": size_kb,
            "Threshold": opt_thresh,
            "Val_Accuracy": val_met["accuracy"],
            "Val_Precision": val_met["precision"],
            "Val_Recall": val_met["recall"],
            "Val_F1": val_met["f1"],
            "Val_ROC_AUC": val_met["roc_auc"],
            "Val_FPR": val_met["fpr"],
            "Avg_Latency_ms": bench["avg_latency_ms"],
            "P95_Latency_ms": bench["p95_latency_ms"],
            "RAM_MB": bench["ram_mb"],
            "Throughput": bench["throughput_samples_per_sec"],
        }
        arch_search_records.append(rec)
        logger.info(f"{arch_name}: Val F1={val_met['f1']:.4f}, Params={n_params}, Latency={bench['avg_latency_ms']:.3f} ms, Size={size_kb:.1f} KB")

    df_arch = pd.DataFrame(arch_search_records)
    df_arch.to_csv(config.RESULTS_DIR / "architecture_search.csv", index=False)
    logger.info("Architecture search results saved to results/architecture_search.csv")

    # Architecture selection rule:
    # Must achieve >= 99% of max F1 while minimizing parameter count
    max_arch_f1 = df_arch["Val_F1"].max()
    target_arch_f1 = 0.99 * max_arch_f1
    eligible_archs = df_arch[df_arch["Val_F1"] >= target_arch_f1].sort_values("Parameter_Count")
    best_arch_row = eligible_archs.iloc[0]
    selected_arch_id = best_arch_row["Architecture_ID"]
    selected_hidden_dims = candidate_archs[selected_arch_id]

    logger.info(
        f"\n>>> SELECTED PROPOSED LIGHTWEIGHT ARCHITECTURE: {selected_arch_id} "
        f"with hidden layers {selected_hidden_dims} (Params: {best_arch_row['Parameter_Count']}, Val F1: {best_arch_row['Val_F1']:.4f}) <<<"
    )

    # =========================================================================
    # PHASE 5 — ABLATION STUDY
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 5: SYSTEMATIC ABLATION STUDY")
    logger.info("=" * 50)

    ablation_records = []

    # Experiment A: Standard MLP (Full features)
    ablation_records.append(
        {
            "Experiment": "Exp A: Standard MLP (Full Features)",
            "Features": X_train_full.shape[1],
            "Architecture": "128->64->32->1",
            "Pruning": "None (0%)",
            "Quantization": "FP32",
            "Sparsity_%": 0.0,
            **std_test_met,
            **std_bench,
            "Parameters": std_params,
            "Model_Size_KB": std_mlp_size_kb,
        }
    )

    # Experiment B: Lightweight MLP (Full features)
    logger.info("Evaluating Exp B: Lightweight MLP on Full Features...")
    lw_full = LightweightMLP(
        in_features=X_train_full.shape[1],
        hidden_dims=selected_hidden_dims,
        use_batch_norm=False,
        dropout_rate=0.05,
    )
    lw_full, _ = train_neural_network(
        lw_full, X_train_full, y_train, X_val_full, y_val, epochs=12, model_name="ExpB_LW_Full"
    )
    b_path = config.MODELS_DIR / "exp_b_lightweight_full.pt"
    torch.save(lw_full.state_dict(), b_path)
    b_size_kb = get_model_file_size_kb(b_path)
    b_params, _ = count_parameters(lw_full)

    b_val_probs = predict_probs_nn(lw_full, X_val_full)
    b_thresh, _, _ = optimize_threshold(y_val, b_val_probs)
    b_test_probs = predict_probs_nn(lw_full, X_test_full)
    b_test_met = compute_metrics(y_test, b_test_probs, threshold=b_thresh)
    b_bench = benchmark_inference(lw_full, bench_sample_full, is_torch=True)

    ablation_records.append(
        {
            "Experiment": "Exp B: Lightweight MLP (Full Features)",
            "Features": X_train_full.shape[1],
            "Architecture": f"Lightweight ({selected_hidden_dims})",
            "Pruning": "None (0%)",
            "Quantization": "FP32",
            "Sparsity_%": 0.0,
            **b_test_met,
            **b_bench,
            "Parameters": b_params,
            "Model_Size_KB": b_size_kb,
        }
    )

    # Experiment C: Lightweight MLP + Feature Selection
    logger.info("Evaluating Exp C: Lightweight MLP + Feature Selection...")
    lw_sel = trained_arch_models[selected_arch_id]
    c_path = config.MODELS_DIR / "proposed_lightweight_model.pt"
    torch.save(lw_sel.state_dict(), c_path)
    c_size_kb = get_model_file_size_kb(c_path)
    c_params, _ = count_parameters(lw_sel)

    c_val_probs = predict_probs_nn(lw_sel, X_val_sel)
    c_thresh, _, _ = optimize_threshold(y_val, c_val_probs)
    c_test_probs = predict_probs_nn(lw_sel, X_test_sel)
    c_test_met = compute_metrics(y_test, c_test_probs, threshold=c_thresh)
    c_bench = benchmark_inference(lw_sel, bench_sample_sel, is_torch=True)

    ablation_records.append(
        {
            "Experiment": "Exp C: Lightweight MLP + Feature Selection",
            "Features": in_dim_sel,
            "Architecture": f"Lightweight ({selected_hidden_dims})",
            "Pruning": "None (0%)",
            "Quantization": "FP32",
            "Sparsity_%": 0.0,
            **c_test_met,
            **c_bench,
            "Parameters": c_params,
            "Model_Size_KB": c_size_kb,
        }
    )

    # =========================================================================
    # PHASE 7 — PRUNING EXPERIMENTS (Exp D & Sparsity study)
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 7: MODEL PRUNING & SPARSITY STUDY")
    logger.info("=" * 50)

    pruning_records = []
    pruned_models = {}

    for sparsity in config.PRUNING_SPARSITY_LEVELS:
        logger.info(f"Applying L1-unstructured pruning at {sparsity*100:.0f}% sparsity...")
        pruned_m = apply_unstructured_pruning(
            lw_sel,
            sparsity=sparsity,
            X_train=X_train_sel,
            y_train=y_train,
            finetune_epochs=2,
        )
        total_w, zero_w, actual_sp = calculate_model_sparsity(pruned_m)
        pruned_models[sparsity] = pruned_m

        p_path = config.MODELS_DIR / f"proposed_model_pruned_{int(sparsity*100)}.pt"
        torch.save(pruned_m.state_dict(), p_path)
        p_size_kb = get_model_file_size_kb(p_path)

        p_test_probs = predict_probs_nn(pruned_m, X_test_sel)
        p_test_met = compute_metrics(y_test, p_test_probs, threshold=c_thresh)
        p_bench = benchmark_inference(pruned_m, bench_sample_sel, is_torch=True)

        rec = {
            "Target_Sparsity": f"{int(sparsity*100)}%",
            "Actual_Sparsity_%": actual_sp,
            "Total_Weights": total_w,
            "Zero_Weights": zero_w,
            "Active_Weights": total_w - zero_w,
            "Model_Size_KB": p_size_kb,
            "Accuracy": p_test_met["accuracy"],
            "Precision": p_test_met["precision"],
            "Recall": p_test_met["recall"],
            "F1_Score": p_test_met["f1"],
            "ROC_AUC": p_test_met["roc_auc"],
            "Avg_Latency_ms": p_bench["avg_latency_ms"],
            "P95_Latency_ms": p_bench["p95_latency_ms"],
            "Throughput": p_bench["throughput_samples_per_sec"],
        }
        pruning_records.append(rec)
        logger.info(f"Pruned {int(sparsity*100)}%: Test F1={p_test_met['f1']:.4f}, Active weights={total_w - zero_w}, Latency={p_bench['avg_latency_ms']:.3f} ms")

    df_pruning = pd.DataFrame(pruning_records)
    df_pruning.to_csv(config.RESULTS_DIR / "pruning_results.csv", index=False)
    logger.info("Pruning results saved to results/pruning_results.csv")

    # Select optimal pruning level (highest sparsity meeting acceptable retention)
    eligible_pruning = df_pruning[df_pruning["F1_Score"] >= 0.90 * c_test_met["f1"]]
    if not eligible_pruning.empty:
        best_pruning_row = eligible_pruning.iloc[-1]
    else:
        best_pruning_row = df_pruning.sort_values("F1_Score", ascending=False).iloc[0]

    optimal_sparsity = float(best_pruning_row["Target_Sparsity"].replace("%", "")) / 100.0
    optimal_pruned_model = pruned_models[optimal_sparsity]
    logger.info(f"Selected optimal pruning level: {optimal_sparsity*100:.0f}% sparsity (Test F1={best_pruning_row['F1_Score']:.4f})")

    # Experiment D: Lightweight MLP + Feature Selection + Pruning
    d_m = optimal_pruned_model
    d_path = config.MODELS_DIR / "proposed_lightweight_model_pruned.pt"
    torch.save(d_m.state_dict(), d_path)
    d_size_kb = get_model_file_size_kb(d_path)
    d_total_w, d_zero_w, d_actual_sp = calculate_model_sparsity(d_m)
    d_test_probs = predict_probs_nn(d_m, X_test_sel)
    d_test_met = compute_metrics(y_test, d_test_probs, threshold=c_thresh)
    d_bench = benchmark_inference(d_m, bench_sample_sel, is_torch=True)

    ablation_records.append(
        {
            "Experiment": f"Exp D: Lightweight + FS + Pruning ({int(optimal_sparsity*100)}%)",
            "Features": in_dim_sel,
            "Architecture": f"Lightweight ({selected_hidden_dims})",
            "Pruning": f"L1 Unstructured ({int(optimal_sparsity*100)}%)",
            "Quantization": "FP32",
            "Sparsity_%": d_actual_sp,
            **d_test_met,
            **d_bench,
            "Parameters": c_params - d_zero_w,
            "Model_Size_KB": d_size_kb,
        }
    )

    # =========================================================================
    # PHASE 6 — QUANTIZATION (Exp E)
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 6: POST-TRAINING INT8 QUANTIZATION")
    logger.info("=" * 50)

    # Quantize unpruned lightweight model (Exp E)
    q_model = quantize_model_int8(lw_sel)
    e_path = config.MODELS_DIR / "proposed_lightweight_model_quantized.pt"
    e_size_kb = save_quantized_model(q_model, e_path)

    e_test_probs = predict_probs_nn(q_model, X_test_sel)
    e_test_met = compute_metrics(y_test, e_test_probs, threshold=c_thresh)
    e_bench = benchmark_inference(q_model, bench_sample_sel, is_torch=True)

    ablation_records.append(
        {
            "Experiment": "Exp E: Lightweight + FS + INT8 Quantization",
            "Features": in_dim_sel,
            "Architecture": f"Lightweight ({selected_hidden_dims})",
            "Pruning": "None (0%)",
            "Quantization": "INT8 (qint8)",
            "Sparsity_%": 0.0,
            **e_test_met,
            **e_bench,
            "Parameters": c_params,
            "Model_Size_KB": e_size_kb,
        }
    )

    # Experiment F: Lightweight + Feature Selection + Pruning + INT8 Quantization
    logger.info("Evaluating Exp F: Lightweight + FS + Pruning + INT8 Quantization...")
    q_pruned_model = quantize_model_int8(d_m)
    f_path = config.MODELS_DIR / "proposed_lightweight_model_pruned_quantized.pt"
    f_size_kb = save_quantized_model(q_pruned_model, f_path)

    f_test_probs = predict_probs_nn(q_pruned_model, X_test_sel)
    f_test_met = compute_metrics(y_test, f_test_probs, threshold=c_thresh)
    f_bench = benchmark_inference(q_pruned_model, bench_sample_sel, is_torch=True)

    ablation_records.append(
        {
            "Experiment": "Exp F: Lightweight + FS + Pruning + INT8",
            "Features": in_dim_sel,
            "Architecture": f"Lightweight ({selected_hidden_dims})",
            "Pruning": f"L1 Unstructured ({int(optimal_sparsity*100)}%)",
            "Quantization": "INT8 (qint8)",
            "Sparsity_%": d_actual_sp,
            **f_test_met,
            **f_bench,
            "Parameters": c_params - d_zero_w,
            "Model_Size_KB": f_size_kb,
        }
    )

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(config.RESULTS_DIR / "ablation_results.csv", index=False)
    logger.info("Ablation study results saved to results/ablation_results.csv")

    # Quantization Trade-Off Comparison Table (Phase 6)
    size_red_pct = round(((c_size_kb - e_size_kb) / c_size_kb) * 100.0, 2)
    lat_imp_pct = round(((c_bench["avg_latency_ms"] - e_bench["avg_latency_ms"]) / c_bench["avg_latency_ms"]) * 100.0, 2)
    f1_diff = round(e_test_met["f1"] - c_test_met["f1"], 5)

    quant_comparison = pd.DataFrame(
        [
            {
                "Precision": "FP32 (Exp C)",
                "Model_Size_KB": c_size_kb,
                "Avg_Latency_ms": c_bench["avg_latency_ms"],
                "P95_Latency_ms": c_bench["p95_latency_ms"],
                "RAM_MB": c_bench["ram_mb"],
                "Test_Accuracy": c_test_met["accuracy"],
                "Test_Recall": c_test_met["recall"],
                "Test_F1": c_test_met["f1"],
                "Size_Reduction_%": 0.0,
                "Latency_Improvement_%": 0.0,
            },
            {
                "Precision": "INT8 (Exp E)",
                "Model_Size_KB": e_size_kb,
                "Avg_Latency_ms": e_bench["avg_latency_ms"],
                "P95_Latency_ms": e_bench["p95_latency_ms"],
                "RAM_MB": e_bench["ram_mb"],
                "Test_Accuracy": e_test_met["accuracy"],
                "Test_Recall": e_test_met["recall"],
                "Test_F1": e_test_met["f1"],
                "Size_Reduction_%": size_red_pct,
                "Latency_Improvement_%": lat_imp_pct,
            },
        ]
    )
    quant_comparison.to_csv(config.RESULTS_DIR / "quantization_comparison.csv", index=False)
    logger.info("Quantization comparison saved to results/quantization_comparison.csv")
    logger.info(f"Quantization Analysis: Size reduced by {size_red_pct:.1f}%, Latency change={lat_imp_pct:.1f}%, F1 delta={f1_diff:+.4f}")

    # =========================================================================
    # PHASE 14 — FINAL MODEL SELECTION & PARETO EFFICIENCY ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 14: EFFICIENCY-AWARE FINAL MODEL SELECTION")
    logger.info("=" * 50)

    # Master table combining all models evaluated on test set
    all_models_df = pd.concat([df_baselines, df_ablation], ignore_index=True)

    # Compute Normalized Efficiency Score:
    # Detection Quality = (F1 * 0.5) + (Recall * 0.3) + ((1 - FPR) * 0.2)
    # Resource Cost = normalized(Model_Size) * 0.4 + normalized(Latency) * 0.4 + normalized(RAM) * 0.2
    # Score = Detection Quality / (1.0 + Resource Cost)
    min_size, max_size = all_models_df["Model_Size_KB"].min(), all_models_df["Model_Size_KB"].max()
    min_lat, max_lat = all_models_df["avg_latency_ms"].min(), all_models_df["avg_latency_ms"].max()
    min_ram, max_ram = all_models_df["ram_mb"].min(), all_models_df["ram_mb"].max()

    norm_size = (all_models_df["Model_Size_KB"] - min_size) / max(1e-6, max_size - min_size)
    norm_lat = (all_models_df["avg_latency_ms"] - min_lat) / max(1e-6, max_lat - min_lat)
    norm_ram = (all_models_df["ram_mb"] - min_ram) / max(1e-6, max_ram - min_ram)

    detection_quality = (
        all_models_df["f1"] * 0.50
        + all_models_df["recall"] * 0.30
        + (1.0 - all_models_df["fpr"]) * 0.20
    )
    resource_cost = norm_size * 0.40 + norm_lat * 0.40 + norm_ram * 0.20
    all_models_df["Efficiency_Score"] = (detection_quality / (1.0 + resource_cost)).round(4)

    all_models_df = all_models_df.sort_values(by="Efficiency_Score", ascending=False).reset_index(drop=True)
    all_models_df.to_csv(config.RESULTS_DIR / "final_results.csv", index=False)
    logger.info("Final master results table saved to results/final_results.csv")

    # Designate the winning proposed model
    # Candidate among Exp C, Exp D, Exp E, Exp F
    name_col = all_models_df["Model"].fillna(all_models_df["Experiment"]).astype(str)
    all_models_df["Model_Name"] = name_col
    proposed_candidates = all_models_df[name_col.str.contains("Exp C|Exp D|Exp E|Exp F")]
    best_proposed_row = proposed_candidates.sort_values("Efficiency_Score", ascending=False).iloc[0]
    final_model_name = best_proposed_row["Model_Name"]

    logger.info(f"\n=======================================================")
    logger.info(f"FINAL SELECTED MODEL: {final_model_name}")
    logger.info(f"Efficiency Score: {best_proposed_row['Efficiency_Score']}")
    logger.info(f"F1: {best_proposed_row['f1']:.4f} | Recall: {best_proposed_row['recall']:.4f} | Accuracy: {best_proposed_row['accuracy']:.4f}")
    logger.info(f"Latency: {best_proposed_row['avg_latency_ms']:.3f} ms | Size: {best_proposed_row['Model_Size_KB']:.2f} KB | RAM: {best_proposed_row['ram_mb']:.1f} MB")
    logger.info(f"=======================================================\n")

    # Serialize final model deployment configuration
    final_config = {
        "project_title": "Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices",
        "dataset": "Edge-IIoTset (ML-EdgeIIoT-dataset)",
        "task": "Binary Anomaly Detection (Normal=0, Anomaly=1)",
        "selected_architecture": selected_arch_id,
        "hidden_layers": selected_hidden_dims,
        "input_features_count": in_dim_sel,
        "selected_features": selected_feature_names,
        "decision_threshold": float(c_thresh),
        "quantization": "INT8" if "INT8" in final_model_name else "FP32",
        "pruning_sparsity": float(optimal_sparsity) if "Pruning" in final_model_name else 0.0,
        "final_model_artifact": (
            "models/proposed_lightweight_model_quantized.pt"
            if "INT8" in final_model_name and "Pruning" not in final_model_name
            else "models/proposed_lightweight_model_pruned_quantized.pt"
            if "Pruning" in final_model_name and "INT8" in final_model_name
            else "models/proposed_lightweight_model.pt"
        ),
        "test_metrics": {
            "accuracy": float(best_proposed_row["accuracy"]),
            "precision": float(best_proposed_row["precision"]),
            "recall": float(best_proposed_row["recall"]),
            "f1": float(best_proposed_row["f1"]),
            "roc_auc": float(best_proposed_row["roc_auc"]),
            "pr_auc": float(best_proposed_row["pr_auc"]),
            "fpr": float(best_proposed_row["fpr"]),
            "confusion_matrix": {
                "tn": int(best_proposed_row["tn"]),
                "fp": int(best_proposed_row["fp"]),
                "fn": int(best_proposed_row["fn"]),
                "tp": int(best_proposed_row["tp"]),
            },
        },
        "hardware_metrics": {
            "parameters": int(best_proposed_row["Parameters"]),
            "model_size_kb": float(best_proposed_row["Model_Size_KB"]),
            "avg_latency_ms": float(best_proposed_row["avg_latency_ms"]),
            "median_latency_ms": float(best_proposed_row["median_latency_ms"]),
            "p95_latency_ms": float(best_proposed_row["p95_latency_ms"]),
            "throughput_samples_per_sec": float(best_proposed_row["throughput_samples_per_sec"]),
            "ram_mb": float(best_proposed_row["ram_mb"]),
            "cpu_percent": float(best_proposed_row["cpu_percent"]),
        },
        "efficiency_score": float(best_proposed_row["Efficiency_Score"]),
    }

    final_cfg_path = config.RESULTS_DIR / "final_model_config.json"
    with open(final_cfg_path, "w") as f:
        json.dump(final_config, f, indent=4)
    logger.info(f"Final model configuration saved to {final_cfg_path}")

    return all_models_df


if __name__ == "__main__":
    run_full_pipeline()

