# Phase 2 — Traces

[Broad overview → `../README.md`](../README.md)

Reconstructed MQTT v5.0 sessions captured against the unseen brokers. Labels (`Target`, `Tag`) come from the adapters in the QRS 2026 paper.

---

## Layout

| Path | Broker | Contents |
|------|--------|----------|
| `emqx/EMQX-Dataset.csv` | EMQX 4.4.19 | Reconstructed packets. |
| `emqx/EMQX_preprocessed.csv` | EMQX 4.4.19 | Session table after `evaluate/preprocessing_v3.py`. |
| `nanomq/Final-NanSET.csv` | NanoMQ 0.22.4 | Reconstructed packets. |
| `nanomq/Final-NanSET_preprocessed.csv` | NanoMQ 0.22.4 | Session table after the same preprocessor. |

Mosquitto training/reference traces stay in the QRS 2026 artifact (`Sequence Learning/mqtt_preprocessed_v2.csv`).

---

## Note

`evaluate/` consumes the **preprocessed** CSVs. Raw reconstruction uses `PCAP/pcapToCsv.py` from the QRS 2026 artifact.
