"""
Binary tag-aware evaluation pipeline (no model retraining required).

This script runs the existing pretrained CNN binary oracle and exports:
1) Per-session predictions (with stream id + tag)
2) Overall binary metrics
3) Optional tag-subset binary metrics

Usage:
  python binary_tag_metrics_pipeline.py \
    --test-data EMQX_preprocessed.csv \
    --pretrained-model best_cnn_binary_model.pth \
    --subset-tags 1,4,6
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from evaluate_cnn_on_new_data import (
    BEST_HYPERPARAMETERS,
    FEATURE_COLUMNS,
    MQTTSequenceDataset,
    CNNClassifier,
    set_seed,
    evaluate,
    normalize_sequences,
    load_data,
)


class BinaryTagMetricsPipeline:
    """Evaluate pretrained binary model and export tag-aware reporting artifacts."""

    def __init__(self, train_data_file, test_data_file, pretrained_model_path, output_prefix, subset_tags=None):
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.pretrained_model_path = pretrained_model_path
        self.output_prefix = output_prefix
        self.subset_tags = subset_tags or []

    @staticmethod
    def _compute_binary_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "counts": {
                "total": int(len(y_true)),
                "normal_true": int((np.array(y_true) == 0).sum()),
                "violation_true": int((np.array(y_true) == 1).sum()),
            },
        }

    @staticmethod
    def _extract_session_metadata(filepath):
        df = pd.read_csv(filepath)
        sort_col = "seq_position" if "seq_position" in df.columns else "frame.number"
        if sort_col not in df.columns:
            raise ValueError("Missing both 'seq_position' and 'frame.number' required for sequence ordering.")

        tag_col = None
        if "Tag" in df.columns:
            tag_col = "Tag"
        elif "TAG" in df.columns:
            tag_col = "TAG"

        records = []
        for stream_id, group in df.groupby("tcp.stream"):
            group = group.sort_values(sort_col)
            target_val = group["Target"].iloc[0] if "Target" in group.columns else 0
            target_val = 0 if pd.isna(target_val) else int(target_val)

            tag_val = None
            if tag_col is not None:
                raw = group[tag_col].iloc[0]
                tag_val = None if pd.isna(raw) else int(raw)

            records.append(
                {
                    "tcp.stream": int(stream_id),
                    "Tag": tag_val,
                    "true_binary": target_val,
                }
            )
        return records

    def run(self):
        if not os.path.exists(self.train_data_file):
            raise FileNotFoundError(f"Training data file not found: {self.train_data_file}")
        if not os.path.exists(self.test_data_file):
            raise FileNotFoundError(f"Test data file not found: {self.test_data_file}")
        if not os.path.exists(self.pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found: {self.pretrained_model_path}")

        set_seed(BEST_HYPERPARAMETERS["random_seed"])

        # Keep preprocessing aligned with existing binary pipeline.
        sequences_train, labels_train, features_train = load_data(self.train_data_file, FEATURE_COLUMNS)
        sequences_test, labels_test, features_test = load_data(self.test_data_file, FEATURE_COLUMNS)
        _, sequences_test_norm, _, aligned_features = normalize_sequences(
            sequences_train, sequences_test, features_train, features_test
        )

        metadata = self._extract_session_metadata(self.test_data_file)
        if len(metadata) != len(sequences_test_norm):
            raise RuntimeError("Metadata/session count mismatch between test CSV and sequence construction.")

        test_dataset = MQTTSequenceDataset(sequences_test_norm, labels_test, BEST_HYPERPARAMETERS["max_seq_length"])
        test_loader = DataLoader(
            test_dataset,
            batch_size=BEST_HYPERPARAMETERS["batch_size"],
            shuffle=False,
        )

        model = CNNClassifier(
            input_size=len(aligned_features),
            cnn_num_filters=BEST_HYPERPARAMETERS["cnn_num_filters"],
            cnn_kernel_size=BEST_HYPERPARAMETERS["cnn_kernel_size"],
            cnn_pool_size=2,
            cnn_pooling_type=BEST_HYPERPARAMETERS["cnn_pooling_type"],
            num_classes=2,
            dropout=BEST_HYPERPARAMETERS["dropout"],
        )
        model.load_state_dict(torch.load(self.pretrained_model_path, map_location=BEST_HYPERPARAMETERS["device"]))

        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, criterion, BEST_HYPERPARAMETERS["device"], return_probs=True
        )

        rows = []
        for i, meta in enumerate(metadata):
            rows.append(
                {
                    "dataset_name": os.path.basename(self.test_data_file),
                    "tcp.stream": meta["tcp.stream"],
                    "Tag": meta["Tag"],
                    "true_binary": int(test_labels[i]),
                    "pred_binary": int(test_preds[i]),
                    "pred_prob_violation": float(test_probs[i]),
                    "correct": int(test_labels[i] == test_preds[i]),
                }
            )

        per_session_df = pd.DataFrame(rows)
        per_session_csv = f"{self.output_prefix}_per_session_binary_predictions.csv"
        per_session_df.to_csv(per_session_csv, index=False)

        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "train_data_file": self.train_data_file,
            "test_data_file": self.test_data_file,
            "pretrained_model_path": self.pretrained_model_path,
            "overall_metrics": {
                "loss": float(test_loss),
                "accuracy": float(test_acc),
                "precision": float(test_prec),
                "recall": float(test_rec),
                "f1": float(test_f1),
            },
            "overall_metrics_recomputed": self._compute_binary_metrics(
                per_session_df["true_binary"].tolist(), per_session_df["pred_binary"].tolist()
            ),
            "subset_metrics": {},
            "artifacts": {
                "per_session_csv": per_session_csv,
            },
        }

        if self.subset_tags:
            subset = per_session_df[per_session_df["Tag"].isin(self.subset_tags)]
            if len(subset) > 0:
                summary["subset_metrics"]["selected_tags"] = self.subset_tags
                summary["subset_metrics"]["metrics"] = self._compute_binary_metrics(
                    subset["true_binary"].tolist(), subset["pred_binary"].tolist()
                )
            else:
                summary["subset_metrics"]["selected_tags"] = self.subset_tags
                summary["subset_metrics"]["warning"] = "No rows matched selected tags."

        summary_json = f"{self.output_prefix}_binary_tag_metrics_summary.json"
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        summary["artifacts"]["summary_json"] = summary_json

        print("\nBinary tag-aware pipeline complete.")
        print(f"Per-session predictions: {per_session_csv}")
        print(f"Summary JSON: {summary_json}")
        print(
            "Overall: "
            f"acc={summary['overall_metrics']['accuracy']:.4f}, "
            f"prec={summary['overall_metrics']['precision']:.4f}, "
            f"rec={summary['overall_metrics']['recall']:.4f}, "
            f"f1={summary['overall_metrics']['f1']:.4f}"
        )
        if self.subset_tags and "metrics" in summary["subset_metrics"]:
            m = summary["subset_metrics"]["metrics"]
            print(
                "Subset tags: "
                f"{self.subset_tags} -> "
                f"acc={m['accuracy']:.4f}, prec={m['precision']:.4f}, rec={m['recall']:.4f}, f1={m['f1']:.4f}"
            )

        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Binary tag-aware metrics pipeline.")
    parser.add_argument("--train-data", default="../traces/mosquitto/mqtt_preprocessed_v2.csv")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--pretrained-model", default="../oracle/best_cnn_binary_model.pth")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output files.")
    parser.add_argument(
        "--subset-tags",
        default="-1,3,4,5,6,8,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
        help="Comma-separated integer tags for subset metrics (e.g., 1,4,6).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subset_tags = [int(x.strip()) for x in args.subset_tags.split(",") if x.strip()]
    output_prefix = args.output_prefix or f"binary_tag_metrics_{os.path.splitext(os.path.basename(args.test_data))[0]}"

    pipeline = BinaryTagMetricsPipeline(
        train_data_file=args.train_data,
        test_data_file=args.test_data,
        pretrained_model_path=args.pretrained_model,
        output_prefix=output_prefix,
        subset_tags=subset_tags,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
