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

