#!/usr/bin/env bash
# Reproduce Tables 2 and 3 from the published preprocessed traces and oracle weights.
# Run from this directory (evaluate/).
set -euo pipefail

TRAIN="../traces/mosquitto/mqtt_preprocessed_v2.csv"
CNN="../oracle/best_cnn_binary_model.pth"
GRU="../oracle/best_gru_multiclass_model.pth"
OUT="../results/tables"
SUBSET="-1,3,4,5,6,8,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"

python binary_tag_metrics_pipeline.py \
  --train-data "$TRAIN" \
  --test-data ../traces/emqx/EMQX_preprocessed.csv \
  --pretrained-model "$CNN" \
  --subset-tags "$SUBSET" \
  --output-prefix "$OUT/binary_tag_metrics_EMQX_preprocessed"

python binary_tag_metrics_pipeline.py \
  --train-data "$TRAIN" \
  --test-data ../traces/nanomq/Final-NanSET_preprocessed.csv \
  --pretrained-model "$CNN" \
  --subset-tags "$SUBSET" \
  --output-prefix "$OUT/binary_tag_metrics_Final-NanSET_preprocessed"

python multiclass_tag_metrics_pipeline.py \
  --train-data "$TRAIN" \
  --test-data ../traces/emqx/EMQX_preprocessed.csv \
  --pretrained-model "$GRU" \
  --subset-tags "$SUBSET" \
  --output-prefix "$OUT/multiclass_tag_metrics_EMQX_preprocessed"

python multiclass_tag_metrics_pipeline.py \
  --train-data "$TRAIN" \
  --test-data ../traces/nanomq/Final-NanSET_preprocessed.csv \
  --pretrained-model "$GRU" \
  --subset-tags "$SUBSET" \
  --output-prefix "$OUT/multiclass_tag_metrics_Final-NanSET_preprocessed"

echo "Done. Summaries written under $OUT"
