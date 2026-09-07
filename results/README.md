# Results corresponding to the ICTSS paper

| Path | Contents |
|------|----------|
| `tables/binary_tag_metrics_*_summary.json` | Table 2 (use `subset_metrics`). |
| `tables/multiclass_tag_metrics_*_summary.json` | Table 3 and per-tag metrics. |
| `tables/multiclass_selected_tags_comparison.csv` | Cross-broker F1 / ΔF1 (Table 4, Fig. 3b). |
| `tables/binary_per_tag_correct_wrong_all3.csv` | Binary per-class correct/wrong counts (Fig. 3a). |
| `tables/second_connect_reaction_results.csv` | MQTT-3.1.0-2 / Tag 12 reaction on EMQX. |
| `heatmaps/` | Fig. 3 figures. |
| `predictions/` | Per-session oracle outputs for trace-level inspection. |

`EMQX_preprocessed` = EMQX, `Final-NanSET_preprocessed` = NanoMQ, `mqtt_preprocessed_v2` = Mosquitto reference.
