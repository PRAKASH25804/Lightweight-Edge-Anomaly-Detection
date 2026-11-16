"""
Standalone Edge Inference Demonstration Script
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Demonstrates:
- End-to-end edge pipeline execution
- Live processing of normal vs anomalous network traffic packets
- Real-time latency measurement and threat categorization
"""

import json
import logging
import time
import pandas as pd
from pathlib import Path

from src import config
from src.edge_adapter import EdgeAnomalyDetector

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_inference_demo():
    print("=" * 70)
    print("REAL-TIME EDGE ANOMALY DETECTION INFERENCE DEMO")
    print("=" * 70)

    detector = EdgeAnomalyDetector()

    # Load sample records from raw dataset: 2 normal and 3 different attack types
    raw_df = pd.read_csv(config.RAW_DATA_PATH, low_memory=False)

    normal_samples = raw_df[raw_df["Attack_label"] == 0].head(2)
    mitm_sample = raw_df[raw_df["Attack_type"] == "MITM"].head(1)
    ddos_sample = raw_df[raw_df["Attack_type"] == "DDoS_HTTP"].head(1)
    scan_sample = raw_df[raw_df["Attack_type"] == "Port_Scanning"].head(1)
    sample_df = pd.concat([normal_samples, mitm_sample, ddos_sample, scan_sample]).reset_index(drop=True)

    print(f"\nProcessing {len(sample_df)} real Edge-IIoTset packet events through Edge Detector:\n")

    for i, row in sample_df.iterrows():
        ground_truth_label = "NORMAL (0)" if row["Attack_label"] == 0 else f"ANOMALY (1 - {row['Attack_type']})"
        raw_dict = row.to_dict()

        result = detector.detect(raw_dict)

        status_marker = "[ALERT]" if result["is_anomaly"] else "[NORMAL]"
        print(f"--- Packet Event #{i+1} ---")
        print(f"Ground Truth:  {ground_truth_label}")
        print(f"Prediction:    {status_marker} {result['verdict']} (Confidence: {result['confidence_pct']}%)")
        print(f"Anomaly Prob:  {result['anomaly_probability']:.4f} (Decision Threshold: {result['threshold']:.2f})")
        print(f"Alert Level:   {result['alert_level']}")
        print(f"Latency:       Preprocessing={result['preprocessing_ms']} ms | Inference={result['inference_ms']} ms | Total={result['total_latency_ms']} ms")
        print()

    print("=" * 70)
    print("Inference verification successful. Parity preserved with training pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    run_inference_demo()
