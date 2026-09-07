# Replaying specification-derived test cases

This paper does **not** retrain the oracle and does not re-implement the MQTT adapters. The same compliant and client-side mutant generators from the QRS artifact are replayed against the brokers in [`brokers/`](../brokers/).

## Generators (QRS repository)

Clone and build:

```bash
git clone https://github.com/MaryamAsgari1993/spec-guided-mqtt-conformance-validation.git
cd spec-guided-mqtt-conformance-validation/mqtt-conformance-dataset
./gradlew build
```

Rule identifiers and tags live in `statements.json` in that project. For ICTSS we replay:

- compliant sessions, and
- **client-side** single-rule violations only (the subset used in Tables 2–4).

Server-side / dataset-manipulation mutants from QRS are out of scope here, because they do not exercise broker-side handling of an invalid client.

Point each generator at the broker under test (`host` / `port`). Typical mapping:

| Target | Host | Port |
|--------|------|------|
| EMQX 4.4.19 | `127.0.0.1` | `1883` |
| NanoMQ 0.22.4 | `127.0.0.1` | `1884` |

## Capture

Capture the client–broker TCP session with `tcpdump` while the generator runs, then reconstruct packets with `tshark` (`pcapToCsv.py` in the QRS `PCAP/` folder). Session labels (`Target`, `Tag`) come from the adapter (which rule was violated), not from broker crashes or error codes.

The reconstructed CSVs used in the paper are already in:

- [`traces/emqx/EMQX-Dataset.csv`](../traces/emqx/EMQX-Dataset.csv)
- [`traces/nanomq/Final-NanSET.csv`](../traces/nanomq/Final-NanSET.csv)

## Preprocess

Use the same padding / truncation / sentinel pipeline as oracle training:

```bash
cd evaluate
python preprocessing_v3.py
```

Edit `CONFIG_V2['input_file']` in `preprocessing_v3.py` to point at `../traces/emqx/EMQX-Dataset.csv` or `../traces/nanomq/Final-NanSET.csv`. Preprocessed outputs used in the paper:

- [`traces/emqx/EMQX_preprocessed.csv`](../traces/emqx/EMQX_preprocessed.csv)
- [`traces/nanomq/Final-NanSET_preprocessed.csv`](../traces/nanomq/Final-NanSET_preprocessed.csv)
