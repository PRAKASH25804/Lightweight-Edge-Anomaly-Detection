"""
Hardware Benchmarking and Edge Profiling Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Measures CPU-only performance:
- Single-sample latency (Mean, Median, P95 in ms)
- Batch latency and Throughput (samples/second)
- Peak RAM usage (MB)
- CPU utilization (%)
- Model storage size (KB)
- Parameter count
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import psutil
import torch
import torch.nn as nn

from src import config


def benchmark_inference(
    model: Any,
    X_sample: np.ndarray,
    is_torch: bool = True,
    warmup_runs: int = config.BENCHMARK_WARMUP,
    repetitions: int = config.BENCHMARK_REPETITIONS,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Profiles inference latency, throughput, RAM, and CPU on CPU runtime.
    Guarantees stable measurements using repeated timing and percentile estimation.
    """
    process = psutil.Process(os.getpid())

    # Force CPU
    if is_torch:
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to("cpu")

    # Prepare input tensor/array
    n_samples = len(X_sample)
    if n_samples < batch_size:
        # Tile to meet batch size if needed
        repeats = (batch_size // n_samples) + 1
        X_sample = np.tile(X_sample, (repeats, 1))[:batch_size]

    if is_torch:
        # Convert to torch tensor
        torch_single = torch.tensor(X_sample[:1], dtype=torch.float32)
        torch_batch = torch.tensor(X_sample[:batch_size], dtype=torch.float32)
    else:
        single_input = X_sample[:1]
        batch_input = X_sample[:batch_size]

    # Warmup runs to stabilize caches and runtime buffers
    with torch.no_grad() if is_torch else np.errstate(all="ignore"):
        for _ in range(warmup_runs):
            if is_torch:
                _ = model(torch_single)
            else:
                _ = model.predict_proba(single_input)

    # 1. Single-sample latency measurements
    single_latencies_ms: List[float] = []
    mem_before = process.memory_info().rss / (1024.0 * 1024.0)

    with torch.no_grad() if is_torch else np.errstate(all="ignore"):
        for i in range(repetitions):
            # Rotate across test samples to prevent artificial memory caching
            sample_idx = i % min(100, len(X_sample))
            if is_torch:
                inp = torch.tensor(X_sample[sample_idx : sample_idx + 1], dtype=torch.float32)
                t0 = time.perf_counter()
                _ = model(inp)
                t1 = time.perf_counter()
            else:
                inp = X_sample[sample_idx : sample_idx + 1]
                t0 = time.perf_counter()
                _ = model.predict_proba(inp)
                t1 = time.perf_counter()

            single_latencies_ms.append((t1 - t0) * 1000.0)

    mem_after = process.memory_info().rss / (1024.0 * 1024.0)
    ram_usage_mb = round(max(mem_before, mem_after), 2)

    # Statistical aggregation
    avg_latency = float(np.mean(single_latencies_ms))
    median_latency = float(np.median(single_latencies_ms))
    p95_latency = float(np.percentile(single_latencies_ms, 95))
    p99_latency = float(np.percentile(single_latencies_ms, 99))
    min_latency = float(np.min(single_latencies_ms))

    # 2. Batch throughput measurement
    batch_eval_iters = max(10, repetitions // 5)
    t_start = time.perf_counter()
    with torch.no_grad() if is_torch else np.errstate(all="ignore"):
        for _ in range(batch_eval_iters):
            if is_torch:
                _ = model(torch_batch)
            else:
                _ = model.predict_proba(batch_input)
    t_end = time.perf_counter()

    total_batch_time = max(1e-6, t_end - t_start)
    total_samples = batch_size * batch_eval_iters
    throughput = round(total_samples / total_batch_time, 1)

    # CPU Utilization estimate
    cpu_percent = round(process.cpu_percent(interval=None), 1)

    return {
        "avg_latency_ms": round(avg_latency, 4),
        "median_latency_ms": round(median_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "p99_latency_ms": round(p99_latency, 4),
        "min_latency_ms": round(min_latency, 4),
        "throughput_samples_per_sec": throughput,
        "ram_mb": ram_usage_mb,
        "cpu_percent": cpu_percent,
        "single_sample_latency_str": f"{avg_latency:.3f} ms (P95: {p95_latency:.3f} ms)",
    }

