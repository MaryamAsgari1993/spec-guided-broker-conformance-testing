# Trace-level findings (Section 4.3)

High-ΔF1 classes from Table 4 are prioritization signals, not automatic verdicts. Each case below was inspected against MQTT v5.0. Session-level predictions are in [`results/predictions/`](../results/predictions/).

Tag IDs follow `statements.json` in the [QRS repository](https://github.com/MaryamAsgari1993/spec-guided-mqtt-conformance-validation).

| MQTT rule | Tag | Broker | ΔF1 | Classification | Observed behavior |
|-----------|-----|--------|-----|----------------|-------------------|
| MQTT-3.1.0-2 | 12 | EMQX | 100.00 | Confirmed violation | Second CONNECT is answered with DISCONNECT (msgtype 14) then close. The spec permits CONNACK before close, not DISCONNECT. See [`results/tables/second_connect_reaction_results.csv`](../results/tables/second_connect_reaction_results.csv). |
| MQTT-3.3.1-2 | 25 | EMQX | 95.24 | Confirmed violation | QoS 0 PUBLISH with DUP=1 is silently accepted. MQTT-4.13.1-1 requires the server to close the connection when a Malformed Packet / Protocol Error is detected. |
| MQTT-3.8.3-1 | 29 | EMQX | 100.00 | Permissible variability | Invalid UTF-8 Topic Filter: EMQX disconnects with Reason Code 0x8F; Mosquitto uses 0x81. Both are allowed. |
| MQTT-3.10.3-1 | 30 | EMQX and NanoMQ | 100.00 | Permissible variability | Invalid UTF-8 in UNSUBSCRIBE Topic Filter: EMQX 0x8F, NanoMQ 0x80, Mosquitto 0x81. No unique Reason Code is mandated. |
| MQTT-3.8.3-2 | 26 | NanoMQ | 100.00 | Confirmed violation | Empty SUBSCRIBE payload is processed with no error response or connection close. |
| MQTT-3.10.3-2 | 27 | NanoMQ | 100.00 | Confirmed violation | Empty UNSUBSCRIBE payload is processed with no error response or connection close. |
| MQTT-3.1.2-1 | 3 | NanoMQ | 66.67 | Permissible variability | Invalid protocol name: NanoMQ sends CONNACK 0x81 then closes; Mosquitto closes immediately. Both are allowed. |

Per-class F1 values used to compute ΔF1 are in [`results/tables/multiclass_selected_tags_comparison.csv`](../results/tables/multiclass_selected_tags_comparison.csv).
