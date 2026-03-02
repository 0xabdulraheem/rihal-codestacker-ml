#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anomaly import AnomalyDetector, build_feature_vector
from src.extraction import extract_fields

UNIFIED_DIR = ROOT / "data" / "unified"
MODEL_DIR = ROOT / "tmp_work" / "model"


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
    print("DocFusion — Training on Real Unified Dataset")
    print("=" * 60)

    train_jsonl = UNIFIED_DIR / "train" / "train.jsonl"
    if not train_jsonl.exists():
        print(f"ERROR: {train_jsonl} not found. Run prepare_cord.py and prepare_finditagain.py first.")
        return

    records = load_jsonl(train_jsonl)
    print(f"\nLoaded {len(records)} training records")

    labeled_records = [r for r in records if "label" in r and "is_forged" in r.get("label", {})]
    print(f"Records with forgery labels: {len(labeled_records)}")
    
    forged_count = sum(1 for r in labeled_records if r["label"]["is_forged"] == 1)
    genuine_count = len(labeled_records) - forged_count
    print(f"  Genuine: {genuine_count}, Forged: {forged_count}")

    print("\nExtracting features...")
    features_list = []
    labels = []
    skipped = 0
    start_time = time.time()

    for i, rec in enumerate(labeled_records):
        image_path = str(UNIFIED_DIR / "train" / rec.get("image_path", ""))
        
        if not os.path.exists(image_path):
            skipped += 1
            continue

        gt_fields = rec.get("fields", {})
        ocr_text = rec.get("_ocr_gt", "")

        feat = build_feature_vector(image_path, gt_fields, ocr_text)
        features_list.append(feat)
        labels.append(int(rec["label"]["is_forged"]))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Processed {i + 1}/{len(labeled_records)} ({rate:.1f} records/sec)")

    elapsed = time.time() - start_time
    print(f"\nFeature extraction complete: {len(features_list)} vectors in {elapsed:.1f}s")
    print(f"  Skipped (no image): {skipped}")

    if len(features_list) < 10:
        print("ERROR: Not enough training data")
        return

    print("\nTraining anomaly detector...")
    detector = AnomalyDetector()
    detector.train(features_list, labels)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    detector.save(str(MODEL_DIR))
    print(f"Model saved to: {MODEL_DIR}")

    print("\nQuick validation on training data...")
    predictions = detector.predict(features_list)
    probas = detector.predict_proba(features_list)
    
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\n{classification_report(labels, predictions, target_names=['Genuine', 'Forged'])}")

    if hasattr(detector.model, "feature_importances_"):
        importances = list(zip(detector.feature_columns, detector.model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        print("Top 15 features:")
        for name, imp in importances[:15]:
            print(f"  {name:30s} {imp:.4f}")

    print(f"\nDone! Model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
