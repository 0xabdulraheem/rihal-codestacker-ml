#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anomaly import AnomalyDetector, build_feature_vector

UNIFIED_DIR = ROOT / "data" / "unified"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    print("=" * 60)
    print("DocFusion — Cross-Validation Evaluation")
    print("=" * 60)

    train_jsonl = UNIFIED_DIR / "train" / "train.jsonl"
    records = load_jsonl(train_jsonl)

    fia_records = [r for r in records if r.get("source") == "finditagain"]
    print(f"\nFind-It-Again records: {len(fia_records)}")
    forged = sum(1 for r in fia_records if r["label"]["is_forged"] == 1)
    print(f"  Genuine: {len(fia_records) - forged}, Forged: {forged}")

    print("\nExtracting features...")
    features_list = []
    labels = []
    valid_records = []

    for rec in fia_records:
        image_path = str(UNIFIED_DIR / "train" / rec.get("image_path", ""))
        if not os.path.exists(image_path):
            continue

        gt_fields = rec.get("fields", {})
        ocr_text = rec.get("_ocr_gt", "")
        feat = build_feature_vector(image_path, gt_fields, ocr_text)
        features_list.append(feat)
        labels.append(int(rec["label"]["is_forged"]))
        valid_records.append(rec)

    print(f"Valid feature vectors: {len(features_list)}")

    X = pd.DataFrame(features_list).fillna(0.0)
    y = np.array(labels)
    feature_cols = sorted(X.columns.tolist())
    X = X[feature_cols]

    print("\n--- 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        detector = AnomalyDetector()
        train_feats = [features_list[i] for i in train_idx]
        detector.train(train_feats, y_train.tolist())

        test_feats = [features_list[i] for i in test_idx]
        preds = detector.predict(test_feats)
        probas = detector.predict_proba(test_feats)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, probas)
        except ValueError:
            auc = 0.0

        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "f1": f1, "auc": auc})
        print(f"  Fold {fold + 1}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n--- Cross-Validation Summary ---")
    print(f"  Accuracy: {metrics_df['accuracy'].mean():.4f} +/- {metrics_df['accuracy'].std():.4f}")
    print(f"  F1 Score: {metrics_df['f1'].mean():.4f} +/- {metrics_df['f1'].std():.4f}")
    print(f"  AUC:      {metrics_df['auc'].mean():.4f} +/- {metrics_df['auc'].std():.4f}")

    test_jsonl = UNIFIED_DIR / "test" / "test.jsonl"
    if test_jsonl.exists():
        test_records = load_jsonl(test_jsonl)
        fia_test = [r for r in test_records if r.get("source") == "finditagain"]
        if fia_test:
            print(f"\n--- Hold-Out Test Set ({len(fia_test)} FIA records) ---")

            detector = AnomalyDetector()
            detector.train(features_list, labels)

            test_feats = []
            test_labels = []
            for rec in fia_test:
                image_path = str(UNIFIED_DIR / "test" / rec.get("image_path", ""))
                if not os.path.exists(image_path):
                    continue
                gt_fields = rec.get("fields", {})
                ocr_text = rec.get("_ocr_gt", "")
                feat = build_feature_vector(image_path, gt_fields, ocr_text)
                test_feats.append(feat)
                label = rec.get("label", {}).get("is_forged", 0)
                test_labels.append(int(label))

            if test_feats:
                preds = detector.predict(test_feats)
                probas = detector.predict_proba(test_feats)
                acc = accuracy_score(test_labels, preds)
                f1 = f1_score(test_labels, preds, zero_division=0)
                try:
                    auc = roc_auc_score(test_labels, probas)
                except ValueError:
                    auc = 0.0

                print(f"  Accuracy: {acc:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  AUC:      {auc:.4f}")
                print(f"\n{classification_report(test_labels, preds, target_names=['Genuine', 'Forged'])}")


if __name__ == "__main__":
    main()
