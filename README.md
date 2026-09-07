# Conformance Testing of MQTT Brokers Using a Specification-Guided Learned Oracle

Software artifacts released alongside the paper *Conformance Testing of MQTT Brokers Using a Specification-Guided Learned Oracle* are distributed through this repository.

Oracle construction (normative adapters, Mosquitto traces, training, and saved weights) is released with the companion paper:

https://github.com/MaryamAsgari1993/spec-guided-mqtt-conformance-validation

This repository **reuses** that oracle, without retraining, on two MQTT v5.0 brokers not seen during training: **EMQX 4.4.19** and **NanoMQ 0.22.4**.

---

## Scope

The repository is organized in three parts: **replay → traces → oracle inference**. Each part has its own README with **installation** and **run** instructions.

| Phase | Directory | Role |
|-------|-----------|------|
| **1 — Brokers and replay** | [`brokers/`](brokers/README.md) | Docker: EMQX 4.4.19 and NanoMQ 0.22.4. Test cases come from the QRS `mqtt-conformance-dataset/` generators. |
| **2 — Traces** | [`traces/`](traces/README.md) | Reconstructed EMQX and NanoMQ sessions used in the paper. |
| **3 — Evaluation** | [`evaluate/`](evaluate/README.md) | Same preprocessing as QRS; CNN (binary) and GRU (multi-class) inference. |

---

## High-level results

Binary conformance on the client-side rule subset (no retraining):

| Broker | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Mosquitto (reference) | 99.87 | 99.90 | 99.73 | 99.82 |
| EMQX 4.4.19 | 97.58 | 99.88 | 93.22 | 96.44 |
| NanoMQ 0.22.4 | 99.06 | 100.0 | 97.33 | 98.65 |

Multi-class rule identification remains strong on Mosquitto and degrades on the unseen brokers (EMQX Macro-F1 67.67, NanoMQ 75.42), which we use to prioritize trace-level comparison. Summary JSON files are under [`evaluate/Result/`](evaluate/Result/).
