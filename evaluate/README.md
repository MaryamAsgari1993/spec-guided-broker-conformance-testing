# Evaluation (inference only)

Run all commands from this directory.

```bash
pip install -r ../requirements.txt
bash run_paper_tables.sh
```

| Script | Role |
|--------|------|
| `preprocessing_v3.py` | Session-level features, padding metadata, sentinel values. |
| `evaluate_cnn_on_new_data.py` | Load CNN weights and score a preprocessed CSV. |
| `evaluate_gru_multiclass_on_new_data.py` | Load GRU weights and score a preprocessed CSV. |
| `binary_tag_metrics_pipeline.py` | Table 2: per-session binary predictions + subset metrics. |
| `multiclass_tag_metrics_pipeline.py` | Table 3: per-session multi-class predictions + subset metrics. |
| `models.py` | CNN and GRU class definitions (inference). |

The Mosquitto CSV is required even for EMQX/NanoMQ evaluation: the scaler is fitted on that reference distribution so normalization matches training.
