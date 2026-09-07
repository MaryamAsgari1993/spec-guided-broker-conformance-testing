# Traces

All files are reconstructed MQTT v5.0 sessions (CSV), labeled by the specification-derived adapter (`Target`, `Tag`). They are **not** crash-labeled.

| Path | Broker | Role |
|------|--------|------|
| `emqx/EMQX-Dataset.csv` | EMQX 4.4.19 | Raw reconstructed packets. |
| `emqx/EMQX_preprocessed.csv` | EMQX 4.4.19 | Same pipeline as oracle training (`preprocessing_v3.py`). |
| `nanomq/Final-NanSET.csv` | NanoMQ 0.22.4 | Raw reconstructed packets. |
| `nanomq/Final-NanSET_preprocessed.csv` | NanoMQ 0.22.4 | Preprocessed sessions used in Tables 2–4. |
| `mosquitto/mqtt_preprocessed_v2.csv` | Mosquitto 2.0.22 | QRS reference set: fit the scaler and report Table 2 Mosquitto. |

Session counts in the paper (client-side subset): 1,660 compliant + 900 non-compliant for each of EMQX and NanoMQ (2,560 sessions). Mosquitto subset: 5,700 compliant + 3,000 non-compliant (8,700 sessions).
