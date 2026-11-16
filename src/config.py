"""
Project Configuration Module
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

RAW_DATA_PATH = DATA_DIR / "ML-EdgeIIoT-dataset.csv"

# Ensure directories exist
for d in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for complete reproducibility
RANDOM_SEED = 42

# Train / Validation / Test Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Target columns
TARGET_BINARY = "Attack_label"
TARGET_MULTICLASS = "Attack_type"

# Raw columns to exclude from candidate features:
# Identifiers, free-form text/payloads, and testbed-specific IP addresses that cause data leakage
DROP_COLUMNS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "http.referer",
    "http.request.uri.query",
    "tcp.payload",
    "tcp.options",
    "mqtt.msg",
]

# Candidate feature subset sizes to evaluate in Phase 2
FEATURE_K_CANDIDATES = [10, 15, 20, 30]

# Candidate thresholds to evaluate on validation set in Phase 10
CANDIDATE_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

# Pruning sparsity candidates in Phase 7
PRUNING_SPARSITY_LEVELS = [0.25, 0.50, 0.75]

# Benchmarking settings
BENCHMARK_WARMUP = 50
BENCHMARK_REPETITIONS = 500
BENCHMARK_BATCH_SIZES = [1, 32, 64]

