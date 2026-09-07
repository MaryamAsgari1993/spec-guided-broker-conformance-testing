# Conformance Testing of MQTT Brokers Using a Specification-Guided Learned Oracle

Software artifacts released alongside the paper *Conformance Testing of MQTT Brokers Using a Specification-Guided Learned Oracle* are distributed through this repository.

This repository **reuses**, without retraining, the specification-guided MQTT conformance oracle from our QRS 2026 paper, and applies it to two brokers that were not seen during training: **EMQX 4.4.19** and **NanoMQ 0.22.4**.

The oracle is constructed in:

Maryam Asgari Araghi and Ferhat Khendek, *Specification-Guided Sequence Learning for MQTT Conformance Validation*, in Proceedings of the 26th International Conference on Software Quality, Reliability and Security (QRS 2026), Lecture Notes in Computer Science, Springer, 2026 (in press).

Companion artifact (generators, Mosquitto traces, training, CNN/GRU weights):

https://github.com/MaryamAsgari1993/spec-guided-mqtt-conformance-validation

---

## Scope

The repository is organized in three parts: **replay → traces → oracle inference**. Each part has its own README with **installation** and **run** instructions.

| Phase | Directory | Role |
|-------|-----------|------|
| **1 — Brokers and replay** | [`brokers/`](brokers/README.md) | Docker: EMQX 4.4.19 and NanoMQ 0.22.4. Test cases are the generators from the QRS 2026 paper (`mqtt-conformance-dataset/`). |
| **2 — Traces** | [`traces/`](traces/README.md) | Reconstructed EMQX and NanoMQ sessions used in this paper. |
| **3 — Evaluation** | [`evaluate/`](evaluate/README.md) | Same preprocessing as the QRS 2026 paper; CNN (binary) and GRU (multi-class) inference. |

---

## Citation

Please cite the **QRS 2026 paper** for oracle construction, and this ICTSS paper/artifact for cross-broker testing.

```bibtex
@inproceedings{AsgariAraghiKhendek2026QRS,
  author    = {Asgari Araghi, Maryam and Khendek, Ferhat},
  title     = {Specification-Guided Sequence Learning for {MQTT} Conformance Validation},
  booktitle = {Proceedings of the 26th International Conference on Software Quality, Reliability and Security (QRS 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {In press}
}
```
