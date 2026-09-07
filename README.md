# Conformance Testing of MQTT Brokers Using a Specification-Guided Learned Oracle

Software artifacts released alongside the paper *Conformance Testing of MQTT Brokers Using a Specification-Guided Learned Oracle* (ICTSS 2026) are distributed through this repository.

The oracle was **constructed and trained** in a companion paper. That artifact is:

https://github.com/MaryamAsgari1993/spec-guided-mqtt-conformance-validation

This repository is the **testing** artifact: the published CNN (binary) and GRU (multi-class) weights are applied, without retraining, to two MQTT v5.0 brokers that were not seen during training (EMQX 4.4.19 and NanoMQ 0.22.4).

---

## Scope

| Phase | Directory | Role |
|-------|-----------|------|
| Brokers | [`brokers/`](brokers/README.md) | Docker Compose for EMQX 4.4.19 and NanoMQ 0.22.4. |
| Replay | [`replay/`](replay/README.md) | How to replay the QRS adapters and capture traces. |
| Traces | [`traces/`](traces/README.md) | Reconstructed CSVs used in the paper (EMQX, NanoMQ, Mosquitto reference). |
| Oracle | [`oracle/`](oracle/) | `best_cnn_binary_model.pth`, `best_gru_multiclass_model.pth`. |
| Evaluate | [`evaluate/`](evaluate/) | Preprocessing and inference (no nested CV / no retraining). |
| Results | [`results/`](results/README.md) | Tables 2–4, Fig. 3 heatmaps, per-session predictions. |
| Findings | [`findings/README.md`](findings/README.md) | Trace-level inspection of high-ΔF1 classes. |

---

## Requirements

- Python 3.9+ (3.10 or 3.11 recommended)
- Docker (only if you want to replay new traces)
- JDK 21 + Gradle (only if you replay generators from the QRS repo)

```bash
pip install -r requirements.txt
```

PyTorch: https://pytorch.org/get-started/locally/

---

## Reproduce the paper tables (from published traces)

The preprocessed traces and weights needed for Tables 2 and 3 are already in this repository. From `evaluate/`:

```bash
cd evaluate
bash run_paper_tables.sh
```

Or run one broker at a time:

```bash
cd evaluate

python binary_tag_metrics_pipeline.py \
  --test-data ../traces/emqx/EMQX_preprocessed.csv \
  --output-prefix ../results/tables/binary_tag_metrics_EMQX_preprocessed

python multiclass_tag_metrics_pipeline.py \
  --test-data ../traces/emqx/EMQX_preprocessed.csv \
  --output-prefix ../results/tables/multiclass_tag_metrics_EMQX_preprocessed
```

The default `--subset-tags` list is the client-side rule subset used in the paper (compliant tag `-1` plus the selected violation tags). Table 2 reports that subset: 1,660 compliant sessions and 900 client-side non-compliant sessions per unseen broker.

Mosquitto numbers in Table 2 come from applying the same subset filter to `traces/mosquitto/mqtt_preprocessed_v2.csv` (the QRS training/reference set). That file is also used to fit the feature scaler so inference matches training.

---

## Mapping results to the paper

| Paper item | File |
|------------|------|
| Table 2 EMQX | `results/tables/binary_tag_metrics_EMQX_preprocessed_binary_tag_metrics_summary.json` (`subset_metrics`) |
| Table 2 NanoMQ | `results/tables/binary_tag_metrics_Final-NanSET_preprocessed_binary_tag_metrics_summary.json` (`subset_metrics`) |
| Table 2 Mosquitto | `results/tables/binary_tag_metrics_mqtt_preprocessed_v2_binary_tag_metrics_summary.json` (`subset_metrics`) |
| Table 3 / per-class F1 | `results/tables/multiclass_tag_metrics_*_summary.json` and `multiclass_selected_tags_comparison.csv` |
| Table 4 ΔF1 | `results/tables/multiclass_selected_tags_comparison.csv` (`drop_f1_EMQX`, `drop_f1_Nano`) |
| Fig. 3 | `results/heatmaps/` |
| Section 4.3 cases | [`findings/README.md`](findings/README.md) |

---

## Citation

Please cite the ICTSS paper when using these testing artifacts, and the QRS paper when using the oracle construction pipeline or training data.

Companion oracle repository: https://github.com/MaryamAsgari1993/spec-guided-mqtt-conformance-validation
