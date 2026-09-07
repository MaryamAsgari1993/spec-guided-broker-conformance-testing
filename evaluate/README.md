# Phase 3 — Oracle inference (Python)

[Broad overview → `../README.md`](../README.md)

Apply the binary CNN and multi-class GRU **trained in the QRS 2026 paper** to EMQX and NanoMQ traces. There is **no nested CV and no retraining** here.

Weights in this folder are the published models from that paper (`best_cnn_binary_model.pth`, `best_gru_multiclass_model.pth`). Feature scaling must be fitted on the Mosquitto table `mqtt_preprocessed_v2.csv` from the QRS artifact.

---

## Requirements

- **Python 3.9+** (recommended: **3.10** or **3.11**)
- From the repository root: `pip install -r requirements.txt`
- QRS 2026 file `Sequence Learning/mqtt_preprocessed_v2.csv` (scaler / reference)

PyTorch: https://pytorch.org/get-started/locally/

---

## Layout

| Path | Role |
|------|------|
| `preprocessing_v3.py` | Same session preprocessing as the QRS 2026 paper. |
| `evaluate_cnn_on_new_data.py` | Binary CNN inference. |
| `evaluate_gru_multiclass_on_new_data.py` | Multi-class GRU inference. |
| `binary_tag_metrics_pipeline.py` | Per-session binary metrics. |
| `multiclass_tag_metrics_pipeline.py` | Per-session multi-class metrics. |
| `models.py` | CNN and GRU definitions (inference). |
| `Result/` | High-level summary JSON from the paper runs. |

---

## Run

From **`evaluate/`**. Set `--train-data` to the Mosquitto CSV from the QRS 2026 artifact.

```bash
python binary_tag_metrics_pipeline.py \
  --train-data /path/to/spec-guided-mqtt-conformance-validation/Sequence\ Learning/mqtt_preprocessed_v2.csv \
  --test-data ../traces/emqx/EMQX_preprocessed.csv \
  --pretrained-model best_cnn_binary_model.pth \
  --output-prefix Result/binary_emqx

python multiclass_tag_metrics_pipeline.py \
  --train-data /path/to/spec-guided-mqtt-conformance-validation/Sequence\ Learning/mqtt_preprocessed_v2.csv \
  --test-data ../traces/emqx/EMQX_preprocessed.csv \
  --pretrained-model best_gru_multiclass_model.pth \
  --output-prefix Result/multiclass_emqx
```

Repeat with `../traces/nanomq/Final-NanSET_preprocessed.csv` for NanoMQ.
