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
GIT_MODEL_DIR = ROOT / "models"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    train_jsonl = UNIFIED_DIR / "train" / "train.jsonl"
    records = load_jsonl(train_jsonl)

    fia_only = [r for r in records if r.get("source") == "finditagain"]
    print(f"FIA records: {len(fia_only)}")
    forged = sum(1 for r in fia_only if r["label"]["is_forged"] == 1)
    print(f"  Genuine: {len(fia_only) - forged}, Forged: {forged}")

    features_list = []
    labels = []
    start = time.time()

    for i, rec in enumerate(fia_only):
        image_path = str(UNIFIED_DIR / "train" / rec.get("image_path", ""))
        if not os.path.exists(image_path):
            continue

        ocr_text = rec.get("_ocr_gt", "")
        fields = extract_fields(ocr_text)
        feat = build_feature_vector(image_path, fields, ocr_text)
        features_list.append(feat)
        labels.append(int(rec["label"]["is_forged"]))

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(fia_only)} ({(i + 1) / (time.time() - start):.1f}/s)")

    print(f"Extracted {len(features_list)} features in {time.time() - start:.0f}s")

    detector = AnomalyDetector()
    detector.train(features_list, labels)
    os.makedirs(str(MODEL_DIR), exist_ok=True)
    detector.save(str(MODEL_DIR))
    print(f"Model saved to {MODEL_DIR}")

    os.makedirs(str(GIT_MODEL_DIR), exist_ok=True)
    detector.save(str(GIT_MODEL_DIR))
    print(f"Model saved to {GIT_MODEL_DIR} (git-tracked)")

    preds = detector.predict(features_list)
    from sklearn.metrics import accuracy_score, f1_score
    print(f"Train Acc: {accuracy_score(labels, preds):.3f}")
    print(f"Train F1:  {f1_score(labels, preds, zero_division=0):.3f}")


if __name__ == "__main__":
    main()
