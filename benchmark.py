"""
Dedicated Edge Inference Benchmark Script
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Profiles:
- Single-sample latency (Avg, Median, P95, P99 in ms)
- Batch latency (Batch 1, Batch 32, Batch 64 in ms)
- Throughput (samples/second)
- Peak RAM usage (MB)
- CPU utilization (%)
- Physical model storage (KB)
- Parameter count
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
import psutil
import torch
import torch.nn as nn

from src import config
from src.models import (
    StandardMLP,
    LightweightMLP,
    count_parameters,
    get_model_file_size_kb,
    load_sklearn_model,
)
from src.pruning import apply_unstructured_pruning
from src.quantization import quantize_model_int8

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def profile_model_comprehensive(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    is_torch: bool = True,
    warmup: int = config.BENCHMARK_WARMUP,
    repetitions: int = config.BENCHMARK_REPETITIONS,
    file_path: Optional[Path] = None,
    total_params: int = 0,
) -> List[Dict[str, Any]]:
    """Runs rigorous multi-batch hardware profiling on CPU runtime."""
    process = psutil.Process(os.getpid())
    results = []

    if is_torch:
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to("cpu")

    model_size_kb = get_model_file_size_kb(file_path) if file_path and file_path.exists() else 0.0

    batch_sizes = [1, 32, 64]

    for b_size in batch_sizes:
        # Prepare batch input
        if len(X_test) < b_size:
            repeats = (b_size // len(X_test)) + 1
            batch_data = np.tile(X_test, (repeats, 1))[:b_size]
        else:
            batch_data = X_test[:b_size]

        if is_torch:
            torch_batch = torch.tensor(batch_data, dtype=torch.float32)

        # Warm-up iterations
        with torch.no_grad() if is_torch else np.errstate(all="ignore"):
            for _ in range(warmup):
                if is_torch:
                    _ = model(torch_batch)
                else:
                    _ = model.predict_proba(batch_data)

        # Repetition timing
        iter_latencies_ms = []
        n_iters = repetitions if b_size == 1 else max(20, repetitions // 5)

        mem_before = process.memory_info().rss / (1024.0 * 1024.0)
        t_start_total = time.perf_counter()

        with torch.no_grad() if is_torch else np.errstate(all="ignore"):
            for i in range(n_iters):
                if b_size == 1:
                    idx = i % min(100, len(X_test))
                    if is_torch:
                        single_inp = torch.tensor(X_test[idx : idx + 1], dtype=torch.float32)
                        t0 = time.perf_counter()
                        _ = model(single_inp)
                        t1 = time.perf_counter()
                    else:
                        single_inp = X_test[idx : idx + 1]
                        t0 = time.perf_counter()
                        _ = model.predict_proba(single_inp)
                        t1 = time.perf_counter()
                    iter_latencies_ms.append((t1 - t0) * 1000.0)
                else:
                    t0 = time.perf_counter()
                    if is_torch:
                        _ = model(torch_batch)
                    else:
                        _ = model.predict_proba(batch_data)
                    t1 = time.perf_counter()
                    iter_latencies_ms.append((t1 - t0) * 1000.0)

        t_end_total = time.perf_counter()
        mem_after = process.memory_info().rss / (1024.0 * 1024.0)

        total_elapsed = max(1e-6, t_end_total - t_start_total)
        total_samples = b_size * n_iters
        throughput = round(total_samples / total_elapsed, 1)

        avg_lat = float(np.mean(iter_latencies_ms))
        med_lat = float(np.median(iter_latencies_ms))
        p95_lat = float(np.percentile(iter_latencies_ms, 95))
        p99_lat = float(np.percentile(iter_latencies_ms, 99))
        min_lat = float(np.min(iter_latencies_ms))

        # Latency per sample in ms
        per_sample_lat = avg_lat / b_size

        cpu_util = round(process.cpu_percent(interval=None), 1)

        results.append(
            {
                "Model": model_name,
                "Batch_Size": b_size,
                "Parameters": total_params,
                "Model_Size_KB": model_size_kb,
                "Avg_Batch_Latency_ms": round(avg_lat, 4),
                "Avg_Per_Sample_Latency_ms": round(per_sample_lat, 4),
                "Median_Latency_ms": round(med_lat, 4),
                "P95_Latency_ms": round(p95_lat, 4),
                "P99_Latency_ms": round(p99_lat, 4),
                "Min_Latency_ms": round(min_lat, 4),
                "Throughput_samples_per_sec": throughput,
                "Peak_RAM_MB": round(max(mem_before, mem_after), 2),
                "CPU_Utilization_%": cpu_util,
            }
        )
    return results


def run_edge_benchmark():
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE EDGE BENCHMARK SUITE")
    logger.info("=" * 60)

    data = np.load(config.ARTIFACTS_DIR / "processed_data.npz", allow_pickle=True)
    X_test_full = data["X_test"]
    feature_names = data["feature_names"].tolist()

    with open(config.ARTIFACTS_DIR / "selected_features.json", "r") as f:
        feat_meta = json.load(f)
    selected_indices = feat_meta["feature_indices"]
    X_test_sel = X_test_full[:, selected_indices]

    all_benchmark_rows = []

    # 1. Baseline: Logistic Regression
    lr_path = config.MODELS_DIR / "baseline_logistic_regression.joblib"
    if lr_path.exists():
        logger.info("Benchmarking Baseline 1: Logistic Regression...")
        lr_model = joblib.load(lr_path)
        rows = profile_model_comprehensive(
            lr_model,
            "Baseline: Logistic Regression",
            X_test_sel,
            is_torch=False,
            file_path=lr_path,
            total_params=len(selected_indices) + 1,
        )
        all_benchmark_rows.extend(rows)

    # 2. Baseline: Random Forest
    rf_path = config.MODELS_DIR / "baseline_random_forest.joblib"
    if rf_path.exists():
        logger.info("Benchmarking Baseline 2: Random Forest...")
        rf_model = joblib.load(rf_path)
        total_rf_nodes = sum(tree.tree_.node_count for tree in rf_model.estimators_)
        rows = profile_model_comprehensive(
            rf_model,
            "Baseline: Random Forest",
            X_test_sel,
            is_torch=False,
            file_path=rf_path,
            total_params=total_rf_nodes,
        )
        all_benchmark_rows.extend(rows)

    # 3. Baseline: Standard MLP
    std_path = config.MODELS_DIR / "baseline_standard_mlp.pt"
    if std_path.exists():
        logger.info("Benchmarking Baseline 3: Standard MLP...")
        std_model = StandardMLP(in_features=X_test_full.shape[1])
        std_model.load_state_dict(torch.load(std_path, weights_only=True))
        n_params, _ = count_parameters(std_model)
        rows = profile_model_comprehensive(
            std_model,
            "Baseline: Standard MLP",
            X_test_full,
            is_torch=True,
            file_path=std_path,
            total_params=n_params,
        )
        all_benchmark_rows.extend(rows)

    # 4. Proposed Lightweight Model (FP32)
    prop_path = config.MODELS_DIR / "proposed_lightweight_model.pt"
    if prop_path.exists():
        logger.info("Benchmarking Proposed Lightweight Model (FP32)...")
        prop_model = LightweightMLP(in_features=len(selected_indices), hidden_dims=[32, 16])
        prop_model.load_state_dict(torch.load(prop_path, weights_only=True))
        n_params, _ = count_parameters(prop_model)
        rows = profile_model_comprehensive(
            prop_model,
            "Proposed Lightweight Neural Network (FP32)",
            X_test_sel,
            is_torch=True,
            file_path=prop_path,
            total_params=n_params,
        )
        all_benchmark_rows.extend(rows)

    # 5. Proposed Lightweight Model (Pruned)
    pruned_path = config.MODELS_DIR / "proposed_lightweight_model_pruned.pt"
    if pruned_path.exists():
        logger.info("Benchmarking Proposed Model Pruned...")
        pruned_model = LightweightMLP(in_features=len(selected_indices), hidden_dims=[32, 16])
        pruned_model.load_state_dict(torch.load(pruned_path, weights_only=True))
        rows = profile_model_comprehensive(
            pruned_model,
            "Proposed Lightweight (Pruned 75%)",
            X_test_sel,
            is_torch=True,
            file_path=pruned_path,
            total_params=421,
        )
        all_benchmark_rows.extend(rows)

    # 6. Proposed Lightweight Model (INT8 Quantized)
    q_path = config.MODELS_DIR / "proposed_lightweight_model_quantized.pt"
    if q_path.exists():
        logger.info("Benchmarking Proposed Model (INT8 Quantized)...")
        base_m = LightweightMLP(in_features=len(selected_indices), hidden_dims=[32, 16])
        base_m.load_state_dict(torch.load(prop_path, weights_only=True))
        q_model = quantize_model_int8(base_m)
        rows = profile_model_comprehensive(
            q_model,
            "Proposed Lightweight (INT8 Quantized)",
            X_test_sel,
            is_torch=True,
            file_path=q_path,
            total_params=n_params,
        )
        all_benchmark_rows.extend(rows)

    df_bench = pd.DataFrame(all_benchmark_rows)
    out_csv = config.RESULTS_DIR / "edge_benchmark.csv"
    df_bench.to_csv(out_csv, index=False)
    logger.info(f"Edge benchmark results saved to {out_csv}")
    print("\n=== EDGE BENCHMARK SUMMARY (Single-Sample, Batch Size = 1) ===")
    b1_df = df_bench[df_bench["Batch_Size"] == 1][
        ["Model", "Model_Size_KB", "Avg_Batch_Latency_ms", "P95_Latency_ms", "Throughput_samples_per_sec", "Peak_RAM_MB"]
    ]
    print(b1_df.to_string(index=False))
    return df_bench


if __name__ == "__main__":
    run_edge_benchmark()
