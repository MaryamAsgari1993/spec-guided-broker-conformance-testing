"""
Multiclass tag-aware evaluation pipeline (no model retraining required).

Exports:
1) Per-session multiclass predictions (stream id, true/pred tag + class)
2) Overall multiclass metrics
3) Optional subset-tag metrics
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from evaluate_gru_multiclass_on_new_data import (
    BEST_HYPERPARAMETERS,
    FEATURE_COLUMNS,
    MQTTSequenceDataset,
    GRUClassifier,
    set_seed,
    evaluate,
    load_data,
    normalize_sequences,
    align_class_mappings,
)


class MulticlassTagMetricsPipeline:
    def __init__(self, train_data_file, test_data_file, pretrained_model_path, output_prefix, subset_tags=None):
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.pretrained_model_path = pretrained_model_path
        self.output_prefix = output_prefix
        self.subset_tags = subset_tags or []

    @staticmethod
    def _compute_multiclass_metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    @staticmethod
    def _compute_per_class_metrics(y_true, y_pred, class_ids, class_names, class_to_tag):
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_ids,
            average=None,
            zero_division=0,
        )
        out = {}
        for i, cid in enumerate(class_ids):
            tag = int(class_to_tag[cid])
            out[str(tag)] = {
                "class_id": int(cid),
                "class_name": class_names[cid],
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
        return out

    @staticmethod
    def _extract_session_metadata(filepath):
        df = pd.read_csv(filepath)
        sort_col = "seq_position" if "seq_position" in df.columns else "frame.number"
        if sort_col not in df.columns:
            raise ValueError("Missing both 'seq_position' and 'frame.number' required for sequence ordering.")

        tag_col = "Tag" if "Tag" in df.columns else ("TAG" if "TAG" in df.columns else None)

        rows = []
        for stream_id, group in df.groupby("tcp.stream"):
            group = group.sort_values(sort_col)
            raw_tag = group[tag_col].iloc[0] if tag_col is not None else -1
            true_tag = -1 if pd.isna(raw_tag) else int(raw_tag)
            rows.append({"tcp.stream": int(stream_id), "true_tag": true_tag})
        return rows

    def run(self):
        if not os.path.exists(self.train_data_file):
            raise FileNotFoundError(f"Training data file not found: {self.train_data_file}")
        if not os.path.exists(self.test_data_file):
            raise FileNotFoundError(f"Test data file not found: {self.test_data_file}")
        if not os.path.exists(self.pretrained_model_path):
            raise FileNotFoundError(f"Pretrained model not found: {self.pretrained_model_path}")

        set_seed(BEST_HYPERPARAMETERS["random_seed"])

        (
            sequences_train,
            labels_train,
            features_train,
            tag_to_class_train,
            class_names_train,
            num_classes_train,
        ) = load_data(self.train_data_file, FEATURE_COLUMNS)
        (
            sequences_test,
            labels_test,
            features_test,
            tag_to_class_test,
            class_names_test,
            num_classes_test,
        ) = load_data(self.test_data_file, FEATURE_COLUMNS)

        (
            unified_tag_to_class,
            unified_class_names,
            unified_num_classes,
            train_class_to_unified,
            test_class_to_unified,
        ) = align_class_mappings(
            tag_to_class_train,
            class_names_train,
            num_classes_train,
            tag_to_class_test,
            class_names_test,
            num_classes_test,
        )

        labels_test_unified = [test_class_to_unified[label] for label in labels_test]
        _, sequences_test_norm, _, aligned_features = normalize_sequences(
            sequences_train, sequences_test, features_train, features_test
        )

        metadata = self._extract_session_metadata(self.test_data_file)
        if len(metadata) != len(sequences_test_norm):
            raise RuntimeError("Metadata/session count mismatch between test CSV and sequence construction.")

        test_dataset = MQTTSequenceDataset(sequences_test_norm, labels_test_unified, BEST_HYPERPARAMETERS["max_seq_length"])
        test_loader = DataLoader(test_dataset, batch_size=BEST_HYPERPARAMETERS["batch_size"], shuffle=False)

        model = GRUClassifier(
            input_size=len(aligned_features),
            hidden_size=BEST_HYPERPARAMETERS["gru_hidden_size"],
            num_layers=BEST_HYPERPARAMETERS["gru_num_layers"],
            bidirectional=BEST_HYPERPARAMETERS["gru_bidirectional"],
            num_classes=unified_num_classes,
            dropout=BEST_HYPERPARAMETERS["dropout"],
        )
        model.load_state_dict(torch.load(self.pretrained_model_path, map_location=BEST_HYPERPARAMETERS["device"]))

        criterion = nn.CrossEntropyLoss()
        (
            test_loss,
            test_acc,
            prec_macro,
            rec_macro,
            f1_macro,
            prec_micro,
            rec_micro,
            f1_micro,
            prec_weighted,
            rec_weighted,
            f1_weighted,
            test_preds,
            test_labels,
            test_probs,
        ) = evaluate(
            model,
            test_loader,
            criterion,
            BEST_HYPERPARAMETERS["device"],
            unified_num_classes,
            return_probs=True,
        )

        class_to_tag = {v: k for k, v in unified_tag_to_class.items()}
        per_rows = []
        for i, meta in enumerate(metadata):
            pred_class = int(test_preds[i])
            true_class = int(test_labels[i])
            pred_tag = int(class_to_tag[pred_class])
            true_tag = int(class_to_tag[true_class])

            probs = np.array(test_probs[i], dtype=float)
            pred_prob = float(probs[pred_class]) if probs.size > pred_class else None

            per_rows.append(
                {
                    "dataset_name": os.path.basename(self.test_data_file),
                    "tcp.stream": meta["tcp.stream"],
                    "true_tag": true_tag,
                    "pred_tag": pred_tag,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "true_class_name": unified_class_names[true_class],
                    "pred_class_name": unified_class_names[pred_class],
                    "pred_prob_class": pred_prob,
                    "correct": int(true_class == pred_class),
                }
            )

        per_df = pd.DataFrame(per_rows)
        per_csv = f"{self.output_prefix}_per_session_multiclass_predictions.csv"
        per_df.to_csv(per_csv, index=False)

        cm_all = confusion_matrix(test_labels, test_preds, labels=list(range(unified_num_classes)))

        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "train_data_file": self.train_data_file,
            "test_data_file": self.test_data_file,
            "pretrained_model_path": self.pretrained_model_path,
            "num_classes": int(unified_num_classes),
            "class_names": {str(i): unified_class_names[i] for i in range(unified_num_classes)},
            "tag_to_class": {str(k): int(v) for k, v in unified_tag_to_class.items()},
            "overall_metrics": {
                "loss": float(test_loss),
                "accuracy": float(test_acc),
                "precision_macro": float(prec_macro),
                "recall_macro": float(rec_macro),
                "f1_macro": float(f1_macro),
                "precision_micro": float(prec_micro),
                "recall_micro": float(rec_micro),
                "f1_micro": float(f1_micro),
                "precision_weighted": float(prec_weighted),
                "recall_weighted": float(rec_weighted),
                "f1_weighted": float(f1_weighted),
            },
            "overall_metrics_recomputed": self._compute_multiclass_metrics(test_labels, test_preds),
            "per_tag_metrics": self._compute_per_class_metrics(
                test_labels,
                test_preds,
                list(range(unified_num_classes)),
                unified_class_names,
                class_to_tag,
            ),
            "confusion_matrix": cm_all.tolist(),
            "subset_metrics": {},
            "artifacts": {"per_session_csv": per_csv},
        }

        if self.subset_tags:
            subset = per_df[per_df["true_tag"].isin(self.subset_tags)].copy()
            if len(subset) > 0:
                subset_true = subset["true_class"].astype(int).tolist()
                subset_pred = subset["pred_class"].astype(int).tolist()

                summary["subset_metrics"]["selected_tags"] = self.subset_tags
                summary["subset_metrics"]["num_samples"] = int(len(subset))
                summary["subset_metrics"]["metrics"] = self._compute_multiclass_metrics(subset_true, subset_pred)

                # confusion over only classes represented by selected tags
                subset_class_ids = sorted(
                    set(unified_tag_to_class[t] for t in self.subset_tags if t in unified_tag_to_class)
                )
                if subset_class_ids:
                    cm_subset = confusion_matrix(subset_true, subset_pred, labels=subset_class_ids)
                    summary["subset_metrics"]["class_ids"] = subset_class_ids
                    summary["subset_metrics"]["class_names"] = {
                        str(cid): unified_class_names[cid] for cid in subset_class_ids
                    }
                    summary["subset_metrics"]["confusion_matrix"] = cm_subset.tolist()
                    summary["subset_metrics"]["per_tag_metrics"] = self._compute_per_class_metrics(
                        subset_true,
                        subset_pred,
                        subset_class_ids,
                        unified_class_names,
                        class_to_tag,
                    )
            else:
                summary["subset_metrics"]["selected_tags"] = self.subset_tags
                summary["subset_metrics"]["warning"] = "No rows matched selected tags."

        summary_json = f"{self.output_prefix}_multiclass_tag_metrics_summary.json"
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        summary["artifacts"]["summary_json"] = summary_json

        print("\nMulticlass tag-aware pipeline complete.")
        print(f"Per-session predictions: {per_csv}")
        print(f"Summary JSON: {summary_json}")
        print(
            "Overall: "
            f"acc={summary['overall_metrics']['accuracy']:.4f}, "
            f"macro_f1={summary['overall_metrics']['f1_macro']:.4f}, "
            f"weighted_f1={summary['overall_metrics']['f1_weighted']:.4f}"
        )
        if self.subset_tags and "metrics" in summary["subset_metrics"]:
            m = summary["subset_metrics"]["metrics"]
            print(
                "Subset tags: "
                f"{self.subset_tags} -> "
                f"acc={m['accuracy']:.4f}, macro_f1={m['f1_macro']:.4f}, weighted_f1={m['f1_weighted']:.4f}"
            )

        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Multiclass tag-aware metrics pipeline.")
    parser.add_argument("--train-data", default="../traces/mosquitto/mqtt_preprocessed_v2.csv")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--pretrained-model", default="../oracle/best_gru_multiclass_model.pth")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--subset-tags",
        default="-1,3,4,5,6,8,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
        help="Comma-separated tags for subset metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subset_tags = [int(x.strip()) for x in args.subset_tags.split(",") if x.strip()]
    output_prefix = args.output_prefix or f"multiclass_tag_metrics_{os.path.splitext(os.path.basename(args.test_data))[0]}"

    pipeline = MulticlassTagMetricsPipeline(
        train_data_file=args.train_data,
        test_data_file=args.test_data,
        pretrained_model_path=args.pretrained_model,
        output_prefix=output_prefix,
        subset_tags=subset_tags,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
