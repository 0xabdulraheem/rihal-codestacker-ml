#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "cord"
UNIFIED_DIR = Path(__file__).resolve().parent.parent / "data" / "unified"


def parse_cord_ground_truth(gt_str: str) -> dict:
    try:
        gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
    except (json.JSONDecodeError, TypeError):
        return {}

    fields = {"vendor": None, "date": None, "total": None}
    
    if "gt_parse" not in gt:
        return fields

    gt_parse = gt["gt_parse"]

    total_info = gt_parse.get("total", {})
    if isinstance(total_info, dict):
        total_str = total_info.get("total_price") or total_info.get("total", "")
        if total_str:
            clean = total_str.strip().replace(",", "").replace(".", "").rstrip(".")
            try:
                fields["total"] = clean
            except ValueError:
                fields["total"] = total_str.strip()

    sub_info = gt_parse.get("sub_total", {})
    if isinstance(sub_info, dict):
        fields["_subtotal"] = sub_info.get("subtotal_price")
        fields["_tax"] = sub_info.get("tax_price")
        fields["_service"] = sub_info.get("service_price")

    menu = gt_parse.get("menu", [])
    if isinstance(menu, list):
        fields["_item_count"] = len(menu)

    return fields


def main():
    from datasets import load_dataset

    print("[CORD] Loading dataset from HuggingFace...")
    ds = load_dataset("naver-clova-ix/cord-v2")

    os.makedirs(UNIFIED_DIR / "train" / "images", exist_ok=True)
    os.makedirs(UNIFIED_DIR / "test" / "images", exist_ok=True)

    train_records = []
    train_split = ds.get("train", [])
    print(f"[CORD] Processing {len(train_split)} training samples...")
    
    for i, sample in enumerate(train_split):
        record_id = f"cord_train_{i:05d}"
        img_filename = f"{record_id}.png"
        img_path = UNIFIED_DIR / "train" / "images" / img_filename

        img = sample.get("image")
        if img is not None:
            img.save(str(img_path))

        gt_str = sample.get("ground_truth", "{}")
        fields = parse_cord_ground_truth(gt_str)

        record = {
            "id": record_id,
            "image_path": f"images/{img_filename}",
            "fields": {
                "vendor": fields.get("vendor"),
                "date": fields.get("date"),
                "total": fields.get("total"),
            },
            "label": {
                "is_forged": 0,
                "fraud_type": "none",
            },
            "source": "cord",
        }
        train_records.append(record)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(train_split)} train samples")

    val_split = ds.get("validation", [])
    print(f"[CORD] Processing {len(val_split)} validation samples...")

    for i, sample in enumerate(val_split):
        record_id = f"cord_val_{i:05d}"
        img_filename = f"{record_id}.png"
        img_path = UNIFIED_DIR / "train" / "images" / img_filename

        img = sample.get("image")
        if img is not None:
            img.save(str(img_path))

        gt_str = sample.get("ground_truth", "{}")
        fields = parse_cord_ground_truth(gt_str)

        record = {
            "id": record_id,
            "image_path": f"images/{img_filename}",
            "fields": {
                "vendor": fields.get("vendor"),
                "date": fields.get("date"),
                "total": fields.get("total"),
            },
            "label": {
                "is_forged": 0,
                "fraud_type": "none",
            },
            "source": "cord",
        }
        train_records.append(record)

    test_records = []
    test_split = ds.get("test", [])
    print(f"[CORD] Processing {len(test_split)} test samples...")

    for i, sample in enumerate(test_split):
        record_id = f"cord_test_{i:05d}"
        img_filename = f"{record_id}.png"
        img_path = UNIFIED_DIR / "test" / "images" / img_filename

        img = sample.get("image")
        if img is not None:
            img.save(str(img_path))

        gt_str = sample.get("ground_truth", "{}")
        fields = parse_cord_ground_truth(gt_str)

        record = {
            "id": record_id,
            "image_path": f"images/{img_filename}",
            "fields": {
                "vendor": fields.get("vendor"),
                "date": fields.get("date"),
                "total": fields.get("total"),
            },
            "source": "cord",
        }
        test_records.append(record)

    train_jsonl = UNIFIED_DIR / "train" / "train.jsonl"
    with open(train_jsonl, "w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    test_jsonl = UNIFIED_DIR / "test" / "test.jsonl"
    with open(test_jsonl, "w") as f:
        for rec in test_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[CORD] Done!")
    print(f"  Train records: {len(train_records)} -> {train_jsonl}")
    print(f"  Test records:  {len(test_records)} -> {test_jsonl}")

    vendors_found = sum(1 for r in train_records if r["fields"].get("vendor"))
    dates_found = sum(1 for r in train_records if r["fields"].get("date"))
    totals_found = sum(1 for r in train_records if r["fields"].get("total"))
    print(f"  Vendors: {vendors_found}/{len(train_records)}")
    print(f"  Dates:   {dates_found}/{len(train_records)}")
    print(f"  Totals:  {totals_found}/{len(train_records)}")


if __name__ == "__main__":
    main()
