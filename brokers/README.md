# Phase 1 — Brokers and replay

[Broad overview → `../README.md`](../README.md)

Deploy the two brokers under test and replay the **same** specification-derived clients from the QRS artifact. Do not retrain the oracle here.

---

## Layout

| Path | Role |
|------|------|
| `docker-compose.yml` | EMQX **4.4.19** (host port **1883**) and NanoMQ **0.22.4** (host port **1884**). |

Generators, `statements.json`, and `pcapToCsv.py` live in the QRS repository:

https://github.com/MaryamAsgari1993/spec-guided-mqtt-conformance-validation

Use `mqtt-conformance-dataset/` (compliant + **client-side** single-rule mutants) and `PCAP/pcapToCsv.py`.

---

## Requirements

- **Docker**
- From QRS: **JDK 21**, **Gradle**, an MQTT v5 client host, **tcpdump**, **tshark**

---

## Installation (brokers)

```bash
docker compose -f brokers/docker-compose.yml up -d
docker compose -f brokers/docker-compose.yml ps
```

---

## Run — replay

1. Build the QRS generators (`mqtt-conformance-dataset/`).
2. Point each generator at `127.0.0.1:1883` (EMQX) or `127.0.0.1:1884` (NanoMQ).
3. Capture with `tcpdump`, convert with QRS `PCAP/pcapToCsv.py`.

Paper traces are already in [`../traces/`](../traces/README.md).
