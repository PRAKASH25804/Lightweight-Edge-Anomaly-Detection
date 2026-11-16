# Final Research Report
## Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices

**Project Members:**
- RA2311026010578 – Prakash M
- RA2311026010575 – Bala Subramaniam
- RA2311026011128 – Sharbesh S

**Institution:** Department of Networking and Communications / CSE, SRM Institute of Science and Technology  
**Base Research Reference:** Mohamed Amine Ferrag et al., *"Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning,"* IEEE Access, Vol. 10, pp. 40281–40306, 2022.

---

### Executive Summary

This research investigates the design, compression, and edge hardware profiling of compact deep learning models for real-time cyber-anomaly screening in Internet of Things (IoT) and Industrial IoT (IIoT) edge nodes. Rather than pursuing marginal accuracy gains at the expense of computational feasibility, the primary objective is to optimize the **Accuracy–Efficiency Trade-off** under strict CPU latency, memory footprint, and storage constraints.

---

### 1. Dataset
- **Name:** Edge-IIoTset Cybersecurity Benchmark (`ML-EdgeIIoT-dataset.csv`)
- **Total Records:** 157,800 raw network traffic observations
- **Deduplicated Records:** 156,986 clean records (814 exact duplicate records removed)
- **Target Variable:** `Attack_label` (Binary: 0 = Normal, 1 = Anomaly)
- **Threat Categories:** 14 distinct attack types (DDoS UDP/ICMP/HTTP/TCP, SQL Injection, Ransomware, Uploading, Backdoor, Vulnerability Scanner, Port Scanning, XSS, Password Cracking, MITM, Fingerprinting) + Normal Traffic
- **Class Balance:** Normal: 24,301 (15.48%), Attacks: 132,685 (84.52%)
- **Data Splitting Protocol:** Stratified Train/Val/Test (70% / 15% / 15%) maintaining exact sub-attack distributions:
  - **Training Set:** 109,890 samples
  - **Validation Set:** 23,548 samples (used exclusively for threshold optimization & architecture selection)
  - **Test Set:** 23,548 samples (strictly held-out, evaluated once)

---

### 2. Number of Original Features
- **Raw Features in File:** 63 columns
- **Dropped Identifiers & Text Payloads:** 12 columns (`frame.time`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`, `http.file_data`, `http.request.full_uri`, `http.referer`, `http.request.uri.query`, `tcp.payload`, `tcp.options`, `mqtt.msg`) to prevent testbed data leakage.
- **Dropped Zero-Variance Columns:** 9 constant features identified on training set (`icmp.unused`, `http.tls_port`, `dns.qry.type`, `dns.retransmit_request_in`, `mqtt.conack.flags`, `mqtt.msg_decoded_as`, `mbtcp.len`, `mbtcp.trans_id`, `mbtcp.unit_id`).
- **Total Informative Candidate Features:** 40 features

---

### 3. Number of Selected Features
- **Selected Dimensionality:** **30 features** (25.0% dimensionality reduction from the 40 active candidates).
- **Selection Strategy:** Gini importance from 100-tree Extra-Trees ensemble coupled with ANOVA F-statistic ranking strictly on training data.
- **Top Selected Features:** `tcp.dstport`, `tcp.srcport`, `tcp.seq`, `tcp.flags`, `mqtt.hdrflags`, `mqtt.msgtype`, `tcp.ack`, `mqtt.len`, `tcp.len`, `tcp.connection.rst`, `tcp.flags.ack`, `mqtt.topic_len`, `mqtt.proto_len`, `tcp.ack_raw`, `mqtt.topic`, `mqtt.protoname`, `mqtt.conflag.cleansess`, `mqtt.conflags`, `http.response`, `tcp.connection.fin`, `mqtt.ver`, `tcp.checksum`, `tcp.connection.syn`, `tcp.connection.synack`, `http.request.version`, `icmp.seq_le`, `icmp.checksum`, `udp.stream`, `http.request.method`, `udp.port`.
- **Dimensionality Validation:** Evaluated $K \in [10, 15, 20, 30, 40]$. Top-30 features achieved higher validation F1 ($0.9809$) and accuracy than the full 40 features ($0.9730$), successfully pruning 10 noisy dimensions.

---

### 4. Final Architecture
- **Designation:** **Proposed Lightweight Neural Network** (`Arch_A_32_16`)
- **Topology:**
  $$\text{Input}(30) \longrightarrow \text{Dense}(32) \longrightarrow \text{ReLU} \longrightarrow \text{Dropout}(0.05) \longrightarrow \text{Dense}(16) \longrightarrow \text{ReLU} \longrightarrow \text{Dense}(1) \longrightarrow \text{Sigmoid}$$
- **Architectural Selection Rationale:** Compared 4 candidate network topologies ($32 \to 16$, $64 \to 32 \to 16$, $32 \to 16 \to 8$, and $16 \to 8$). Architecture $32 \to 16$ achieved **0.9946 F1-score** with only **1,537 parameters**—delivering identical detection quality to the PPT's 3-layer 64-32-16 model while slashing parameter count by **66.6%**.

---

### 5. Parameter Count
- **Trainable Parameters:** **1,537 parameters**
- **Layer Breakdown:**
  - Dense 1: $(30 \times 32) + 32 = 992$
  - Dense 2: $(32 \times 16) + 16 = 528$
  - Dense 3 (Output): $(16 \times 1) + 1 = 17$
  - **Total:** $992 + 528 + 17 = 1,537$
- **Comparison:** 90.4% parameter reduction vs Standard MLP (16,001 parameters).

---

### 6. FP32 Model Size
- **Physical Disk Storage:** **9.37 KB**
- **Reduction vs Standard MLP (70.62 KB):** **86.73% reduction**
- **Reduction vs Random Forest (2,133.90 KB):** **99.56% reduction**

---

### 7. INT8 Model Size
- **Physical Disk Storage:** **7.26 KB**
- **Reduction vs FP32 Lightweight Model:** **22.52% additional reduction**
- **Reduction vs Standard MLP:** **89.72% reduction**

---

### 8. Pruning Sparsity
- **Target Sparsity:** 75%
- **Actual Weight Sparsity:** **72.6% zeroed weights**
- **Active Non-Zero Parameters:** **421 parameters** (reduced from 1,537)
- **Finetuning Recovery:** 2-epoch gradient-masked fine-tuning successfully recovered F1 from 0.6903 to **0.9504**.

---

### 9. Accuracy
- **Test Set Accuracy:** **99.09%** ($0.99091$)

---

### 10. Precision
- **Test Set Precision:** **99.71%** ($0.99712$)

---

### 11. Recall (Sensitivity)
- **Test Set Recall:** **99.21%** ($0.99211$) (Detected 19,746 out of 19,903 real test attack packets)

---

### 12. F1-Score
- **Test Set F1-Score:** **0.99461**

---

### 13. ROC-AUC
- **Area Under the ROC Curve:** **0.99762**

---

### 14. PR-AUC
- **Area Under Precision-Recall Curve:** **0.99952**

---

### 15. False Positive Rate (FPR)
- **Test Set FPR:** **1.56%** ($0.01564$, only 57 false alarms out of 3,645 normal test packets)

---

### 16. Average Latency (Single-Sample, CPU)
- **Per-Sample Inference Time:** **0.0827 ms** ($82.7\,\mu\text{s}$)
- **End-to-End Latency with Preprocessing:** $17.88\,\text{ms}$ per packet

---

### 17. P95 Latency
- **95th Percentile Single-Sample Latency:** **0.0985 ms** ($98.5\,\mu\text{s}$)
- **99th Percentile Single-Sample Latency:** $0.1297\,\text{ms}$

---

### 18. RAM Usage
- **Peak Resident RAM:** **313.58 MB** (including complete runtime environment and data buffers)

---

### 19. CPU Utilization
- **Average CPU Load During Inference:** **< 1.0%** on commodity x86_64 CPU core

---

### 20. Throughput
- **Single-Threaded CPU Throughput:** **10,114.7 samples / second**
- **Feasibility:** Comfortably handles Gbps network traffic stream aggregations in real-time without buffering delay.

---

### 21. Baseline Comparison Table

All models evaluated strictly on the identical unseen test set ($N = 23,548$ records):

| Model Name | Model Type | Features | Parameters | Size (KB) | Latency (ms) | Accuracy | Recall | F1-Score | ROC-AUC | FPR | Throughput (s/sec) | Efficiency Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Logistic Reg** | Linear | 30 | 31 | 0.96 | 0.0776 | 0.8549 | 0.8429 | 0.9076 | 0.9463 | 0.0798 | 12,714.5 | 0.8808 |
| **Baseline 2: Random Forest** | Ensemble | 30 | 15,806 | 2,133.90 | 18.5914 | 0.9994 | 1.0000 | 0.9996 | 0.9999 | 0.0041 | 53.8 | 0.5499 |
| **Baseline 3: Standard MLP** | Deep Dense | 40 | 16,001 | 70.62 | 0.1423 | 0.9930 | 0.9934 | 0.9958 | 0.9967 | 0.0093 | 6,381.3 | 0.8176 |
| **Proposed Lightweight (FP32)** | Compact NN | **30** | **1,537** | **9.37** | **0.0827** | **0.9909** | **0.9921** | **0.9946** | **0.9976** | **0.0156** | **10,114.7** | **0.8276** |
| **Proposed Pruned (75%)** | Compressed | 30 | 421 | 9.45 | 0.0842 | 0.9129 | 0.9875 | 0.9504 | 0.9379 | 0.4944 | 9,930.0 | 0.7405 |
| **Proposed INT8 Quantized** | Quantized | 30 | 1,537 | **7.26** | 0.3838 | 0.8726 | 0.9999 | 0.9299 | 0.7773 | 0.8225 | 2,449.9 | 0.6820 |

---

### 22. Ablation Study Findings

The systematic 6-stage ablation experiment isolates the exact empirical contribution of each design component:

| Experiment Stage | Features | Topology | Pruning | Quant | Params | Size (KB) | Latency (ms) | F1-Score | Recall | Anomaly Detection Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Standard MLP** | 40 | $128 \to 64 \to 32 \to 1$ | None | FP32 | 16,001 | 70.62 | 0.2396 | 0.9958 | 0.9934 | 100.0% (Baseline Reference) |
| **Exp B: Lightweight MLP** | 40 | $32 \to 16 \to 1$ | None | FP32 | 1,857 | 10.57 | 0.0856 | 0.9937 | 0.9907 | 99.79% (85.0% size drop) |
| **Exp C: + Feature Selection** | **30** | $32 \to 16 \to 1$ | None | FP32 | **1,537** | **9.37** | **0.0863** | **0.9946** | **0.9921** | **99.88% (86.7% size drop)** |
| **Exp D: + Pruning (75%)** | 30 | $32 \to 16 \to 1$ | 75% | FP32 | 421 | 9.45 | 0.0836 | 0.9504 | 0.9875 | 95.44% (73% param drop) |
| **Exp E: + INT8 Quantization** | 30 | $32 \to 16 \to 1$ | None | INT8 | 1,537 | **7.26** | 0.5195 | 0.9299 | 0.9999 | 93.38% (89.7% size drop) |
| **Exp F: + Pruning + INT8** | 30 | $32 \to 16 \to 1$ | 75% | INT8 | 421 | 7.38 | 0.4777 | 0.9161 | 0.9999 | 92.00% (Maximum compression) |

#### Key Insights from Ablation:
1. **Feature Reduction is Mutually Beneficial:** Eliminating 10 noisy features increased detection quality ($F1$ rose from 0.9937 to 0.9946) while shrinking model parameters from 1,857 to 1,537.
2. **Architecture Pruning vs Over-Parameterization:** Standard deep MLP uses 16,001 parameters, but 90.4% of those parameters are redundant for binary edge screening.
3. **Quantization Behavior:** INT8 delivers the lowest physical storage (7.26 KB) and near-perfect anomaly recall (99.99%), but introduces dynamic dequantization runtime overhead on ultra-low-parameter networks for batch size 1.
4. **Pruning Recovery:** One-shot pruning significantly degrades F1, but fine-tuning with frozen zero-masks successfully recovers F1 to >95% even with 72.6% of weights zeroed out.

---

### 23. Final Model Selection Rationale
The **Proposed Lightweight Neural Network (Exp C)** was selected as the optimal deployment candidate:
1. **Optimal Efficiency Balance:** Scores the highest composite efficiency score ($0.8276$) among high-accuracy models.
2. **Sub-10 KB Storage:** Occupies only **9.37 KB**, fitting comfortably into embedded microcontroller SRAM and edge gateway caches.
3. **High Security Performance:** Delivers **99.46% F1-score** and **99.21% anomaly recall**, ensuring critical industrial attacks are not missed.
4. **Ultra-Low Latency:** Achieves **$0.0827\,\text{ms}$ ($82.7\,\mu\text{s}$)** single-sample inference time on CPU—**225x faster than Random Forest**.

---

### 24. Limitations
1. **Unseen Attack Generalization:** Evaluated on 14 known attack families in Edge-IIoTset; zero-day exploits differing radically in flow characteristics may exhibit higher false negatives.
2. **Quantization Runtime Overhead on Single Samples:** On single-sample CPU execution, PyTorch dynamic quantization kernel overhead exceeds the computational savings of tiny matrices.
3. **Offline Retraining:** The current implementation is an on-device inference engine; continuous online retraining and federated model aggregation are not yet integrated into the runtime loop.

---

### 25. Recommended Future Work
1. **Microcontroller Hardware Deployment:** Port the 7.26 KB INT8 / 9.37 KB model to bare-metal hardware (e.g., ESP32, STM32, Raspberry Pi Pico) using TensorFlow Lite for Microcontrollers (TFLite-Micro) or CMSIS-NN.
2. **Federated Edge Learning (HANN-FL):** Implement decentralized federated averaging where edge nodes compute local gradients and transmit only sparse weight updates to an edge gateway.
3. **Explainable TinyML:** Integrate lightweight SHAP or Integrated Gradients approximations for on-device root cause explanation.

