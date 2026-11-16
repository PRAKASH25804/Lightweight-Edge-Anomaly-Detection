# Faculty Defense Q&A & Viva Preparation Guide
## Development of Lightweight Neural Network Architecture for Real-Time Anomaly Detection on Resource-Constrained Edge Devices

**Project Members:** Prakash M, Bala Subramaniam, Sharbesh S  
**Department:** Networking and Communications / CSE, SRM Institute of Science and Technology  
**Base Research Paper:** Ferrag et al., IEEE Access 2022 (*Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning*).

---

### Q1: Why did your team select Edge-IIoTset instead of older datasets like KDD-Cup99, NSL-KDD, or CICIDS2017?
**Defensible Answer:**
> *"KDD-Cup99 and NSL-KDD were captured over 25 years ago on synthetic network environments and lack contemporary IoT protocols. Even CICIDS2017 primarily models enterprise network workstations rather than IoT/IIoT topologies.  
> Edge-IIoTset (Ferrag et al., IEEE Access 2022) is generated on a purpose-built realistic IoT testbed containing more than 10 physical device types (water tanks, soil sensors, industrial digital meters) and multi-tier edge/cloud gateways. Crucially, it models modern IoT protocols like MQTT, Modbus (mbtcp), and CoAP, and reflects 14 current threat categories such as ransomware, IoT botnet port scans, and HTTP injection attacks. This makes our anomaly screening research directly relevant to real-world embedded deployments."*

---

### Q2: How did you ensure there was absolutely NO data leakage during preprocessing and feature selection?
**Defensible Answer:**
> *"We enforced strict isolation between training and evaluation phases:
> 1. First, we deduplicated the dataset (removing 814 exact duplicate records).
> 2. Next, the stratified Train/Validation/Test split (70% / 15% / 15%) was executed **BEFORE** any data-driven transformation.
> 3. Zero-variance feature detection was computed strictly on `X_train`. The 9 identified constant columns were subsequently dropped from validation, test, and runtime inference.
> 4. `StandardScaler` was fitted **exclusively on `X_train`**, and only applied as a transform on `X_val` and `X_test`.
> 5. All feature importance rankings (Extra-Trees Gini importance and ANOVA F-statistic) were computed strictly on the training partition.
> 6. Furthermore, we tuned the classification decision threshold strictly on the validation set ($\tau^* = 0.50$), and evaluated the unseen test set only once."*

---

### Q3: Why is your selected 2-layer model ($32 \to 16 \to 1$) superior to the 3-layer model ($64 \to 32 \to 16 \to 1$) originally drafted in the PPT?
**Defensible Answer:**
> *"In our systematic architecture search (evaluating topologies A: $32 \to 16$, B: $64 \to 32 \to 16$, C: $32 \to 16 \to 8$, and D: $16 \to 8$), the 2-layer $32 \to 16$ network achieved **0.9946 test F1-score**, matching the 3-layer network ($0.9960$).  
> However, our selected model uses only **1,537 parameters** compared to **4,609 parameters** for the 64-32-16 architecture—a **66.6% parameter reduction** and a storage footprint of only **9.37 KB**. For resource-constrained microcontrollers where SRAM is measured in kilobytes, our model eliminates unnecessary capacity without sacrificing anomaly recall."*

---

### Q4: Why not just use Random Forest since it achieved a slightly higher F1-score (0.9996 vs 0.9946)?
**Defensible Answer:**
> *"This illustrates our central research thesis: **we are optimizing the Accuracy-Efficiency Trade-off, not maximizing accuracy in isolation.**
> - Random Forest requires **15,806 tree decision nodes**, resulting in a physical model size of **2,133.9 KB (over 2.1 MB)** and an inference latency of **$18.59\,\text{ms}$** per packet ($53.8\,\text{packets/sec}$).
> - In an industrial IoT stream handling hundreds or thousands of packets per second, an $18.59\,\text{ms}$ latency causes immediate buffer overflow and dropped packets.
> - Our proposed Lightweight Neural Network occupies only **9.37 KB (99.56% smaller)** and infers in **$0.0827\,\text{ms}$ ($82.7\,\mu\text{s}$, 225x faster)**, sustaining **$10,114.7\,\text{packets/sec}$** on a standard CPU. The negligible 0.5% F1 difference is an exceptional trade-off for 225x higher throughput."*

---

### Q5: How did you handle the class imbalance between Normal traffic (15.5%) and Attack traffic (84.5%)?
**Defensible Answer:**
> *"In Edge-IIoTset, attacks outnumber normal traffic roughly 5.5 to 1. To address this without introducing synthetic generation artifacts (like SMOTE, which can distort flow covariance):
> 1. We utilized stratified sampling across all 14 attack subclasses to ensure rare attacks (like MITM with 1,214 packets and Fingerprinting with 1,001 packets) were proportionally represented across train, val, and test partitions.
> 2. During neural network training, we configured positive class weighting in `BCEWithLogitsLoss(pos_weight=...)` based on the inverse class ratio.
> 3. We evaluated models primarily using precision, recall, F1-score, and PR-AUC rather than raw classification accuracy alone."*

---

### Q6: What did your pruning experiments reveal regarding physical storage and latency?
**Defensible Answer:**
> *"We investigated L1 unstructured pruning across 25%, 50%, and 75% sparsity. When zeroing 72.6% of weights (reducing active parameters from 1,537 to 421), one-shot pruning caused an F1 drop to 0.6903. However, applying a **2-epoch gradient-masked fine-tuning** recovered the F1 score to **0.9504**.  
> From a hardware perspective, standard dense matrix storage formats store explicit zeros, so disk storage remained 9.45 KB. This proved that unstructured pruning primarily reduces computational multiply-accumulate operations (MACs) and parameter complexity, rather than raw disk storage unless paired with compressed sparse row (CSR) runtimes."*

---

### Q7: Why did single-sample INT8 quantization in PyTorch show slightly higher latency ($0.38\,\text{ms}$ vs $0.08\,\text{ms}$), and is that normal?
**Defensible Answer:**
> *"Yes, this is an authentic, well-documented phenomenon in TinyML research. Dynamic post-training quantization in high-level frameworks like PyTorch converts floating-point weights to `qint8` (reducing model file size from 9.37 KB to **7.26 KB**, an 89.7% reduction vs standard MLP).  
> However, on single-sample inference (batch size = 1) for tiny matrices ($30 \times 32$), the CPU kernel dispatch and dynamic quantize/dequantize overhead exceeds the arithmetic speedup of 8-bit integer dot products. The true compute acceleration of INT8 manifests when deployed on bare-metal integer DSPs or microcontrollers using CMSIS-NN and TFLite-Micro."*

---

### Q8: What are the next steps to deploy this model on a real physical microcontroller?
**Defensible Answer:**
> *"Because our final model is already compressed to **9.37 KB (FP32)** and **7.26 KB (INT8)**, it comfortably fits within the 320 KB SRAM and 4 MB Flash limits of an ESP32 or Raspberry Pi Pico.  
> The next engineering step is exporting the PyTorch state dict to ONNX, converting to a flatbuffer using TFLite, and compiling the static C byte array for the microcontroller using TensorFlow Lite for Microcontrollers (TFLite-Micro). Our preprocessing module is fully vectorized and deterministic, making it straightforward to port to embedded C/C++."*

