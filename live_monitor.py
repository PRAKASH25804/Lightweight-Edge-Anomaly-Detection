"""
Interactive Edge Device Live Anomaly Monitor
"Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices"

Simulates a live streaming network interface on an IoT/IIoT edge gateway:
- Ingests incoming packet records from Edge-IIoTset test traffic
- Processes records through the localized Lightweight Neural Detector
- Displays real-time packet-by-packet verdicts, latency, throughput, and threat alerts
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.edge_adapter import EdgeAnomalyDetector

# Configure terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_live_monitor(num_packets: int = 25, delay_ms: float = 80.0):
    print(f"{BOLD}{CYAN}=" * 80)
    print("   IOT/IIOT EDGE GATEWAY - REAL-TIME NEURAL ANOMALY MONITOR")
    print("   SRM Institute of Science and Technology | NNDP Final Year Project")
    print("=" * 80 + f"{RESET}\n")

    detector = EdgeAnomalyDetector()
    print(f"{BOLD}Initialized Runtime Engine:{RESET} Arch={detector.hidden_layers} | Model Size=9.37 KB | Threshold={detector.threshold:.2f}")

    # Load test split
    data = np.load(config.ARTIFACTS_DIR / "processed_data.npz", allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]

    with open(config.ARTIFACTS_DIR / "selected_features.json") as f:
        feat_meta = json.load(f)
    feature_names = data["feature_names"].tolist()
    sel_features = feat_meta["selected_features"]
    sel_indices = [feature_names.index(f) for f in sel_features]

    # Sample alternating packets (normal and anomaly)
    normal_indices = np.where(y_test == 0)[0]
    attack_indices = np.where(y_test == 1)[0]

    eval_indices = []
    for i in range(num_packets):
        if i % 3 == 0 and len(normal_indices) > (i // 3):
            eval_indices.append(normal_indices[i // 3])
        else:
            eval_indices.append(attack_indices[i % len(attack_indices)])

    print(f"\n{BOLD}{'PKT #':<6} | {'GROUND TRUTH':<14} | {'VERDICT':<10} | {'ANOMALY PROB':<13} | {'CONFIDENCE':<11} | {'LATENCY':<10} | {'STATUS'}{RESET}")
    print("-" * 80)

    total_latencies = []
    anomalies_detected = 0
    normal_detected = 0

    for idx, sample_id in enumerate(eval_indices, 1):
        x_vec = X_test[sample_id : sample_id + 1, sel_indices]
        gt = int(y_test[sample_id])
        gt_str = "NORMAL" if gt == 0 else "ATTACK"

        t0 = time.perf_counter()
        prob, is_anomaly = detector.predict_vector(x_vec)
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        total_latencies.append(t_elapsed)

        conf = (prob if is_anomaly else 1.0 - prob) * 100.0

        if is_anomaly:
            anomalies_detected += 1
            verdict_str = f"{RED}{BOLD}ANOMALY{RESET}"
            status_tag = f"{RED}[CRITICAL ALERT]{RESET}" if prob >= 0.85 else f"{YELLOW}[ALERT]{RESET}"
        else:
            normal_detected += 1
            verdict_str = f"{GREEN}NORMAL{RESET}"
            status_tag = f"{GREEN}[SECURE]{RESET}"

        gt_color = GREEN if gt == 0 else RED
        print(
            f"{idx:<6} | {gt_color}{gt_str:<14}{RESET} | {verdict_str:<19} | {prob:<13.4f} | {conf:<10.1f}% | {t_elapsed:<8.3f}ms | {status_tag}"
        )

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    avg_lat = np.mean(total_latencies)
    throughput = 1000.0 / max(1e-6, avg_lat)

    print("-" * 80)
    print(f"\n{BOLD}{CYAN}=== RUNTIME SUMMARY ==={RESET}")
    print(f"Packets Streamed:      {num_packets}")
    print(f"Normal Packets:        {normal_detected}")
    print(f"Anomalies Blocked:     {anomalies_detected}")
    print(f"Average CPU Latency:   {BOLD}{avg_lat:.3f} ms{RESET} per packet")
    print(f"Streaming Throughput:  {BOLD}{throughput:,.1f} packets/second{RESET}")
    print(f"Physical Memory:       {BOLD}9.37 KB{RESET} model weight storage")
    print(f"{CYAN}=" * 80 + f"{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Edge Anomaly Detector Monitor")
    parser.add_argument("--packets", type=int, default=20, help="Number of packets to simulate (default: 20)")
    parser.add_argument("--delay", type=float, default=60.0, help="Delay between packets in ms (default: 60)")
    args = parser.parse_args()

    run_live_monitor(num_packets=args.packets, delay_ms=args.delay)

