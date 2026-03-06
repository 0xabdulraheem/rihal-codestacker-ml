#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import os
import re
import shutil
from pathlib import Path

FINDIT_DIR = Path(__file__).resolve().parent.parent / "data" / "finditagain" / "findit2"
UNIFIED_DIR = Path(__file__).resolve().parent.parent / "data" / "unified"


def extract_fields_from_ocr_text(text: str) -> dict:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    
    fields = {"vendor": None, "date": None, "total": None}
    
    if not lines:
        return fields

    vendor_candidates = []
    for line in lines[:5]:
        lower = line.lower()
        if re.match(r"^[\d\s\-\+\(\)\.]{7,}$", line):
            continue
        if re.match(r"^\(?\w{2}\d{5,}\)?$", line):
            continue
        if any(kw in lower for kw in ["tel", "fax", "gst", "tax invoice", "cash", "doc no", "date", "time"]):
            continue
        if re.match(r"^\d+.*(?:jalan|road|street|avenue|blvd|lane)", lower):
            continue
        if re.match(r"^\d{5}\s", line):
            continue
        
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio > 0.4 and len(line) > 2:
            vendor_candidates.append(line)
            if len(vendor_candidates) >= 2:
                break
    
    if vendor_candidates:
        fields["vendor"] = vendor_candidates[0]

    date_patterns = [
        r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b",
        r"\b(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b",
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})\b",
    ]
    for line in lines:
        for pattern in date_patterns:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                fields["date"] = m.group(1)
                break
        if fields["date"]:
            break

    total_patterns = [
        r"(?:total\s*(?:sales?)?\s*(?:\(inclusive[^)]*\))?\s*:?\s*)[RM$]*\s*([\d,]+\.?\d*)",
        r"(?:grand\s*total|amount\s*due|total\s*amount|net\s*total)\s*:?\s*[RM$]*\s*([\d,]+\.?\d*)",
        r"(?:total)\s*:?\s*[RM$]*\s*([\d,]+\.?\d*)",
    ]
    
    full_text = "\n".join(lines)
    best_total = None
    
    for pattern in total_patterns:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        for match in matches:
            clean = match.replace(",", "")
            try:
                val = float(clean)
                if val > 0:
                    if best_total is None or val > best_total:
                        best_total = val
            except ValueError:
                continue
    
    if best_total is not None:
        fields["total"] = f"{best_total:.2f}"

    return fields


def parse_forgery_annotations(ann_str: str) -> dict:
    if not ann_str or ann_str == "0":
        return {"fraud_type": "none", "regions": []}
    
    try:
        ann = ast.literal_eval(ann_str)
        if isinstance(ann, dict):
            regions = ann.get("regions", [])
            fraud_types = set()
            for region in regions:
                attrs = region.get("region_attributes", {})
                modified = attrs.get("Modified area", {})
                if isinstance(modified, dict):
                    for key in modified:
                        if key != "None" and modified[key]:
                            fraud_types.add(key)
            
            fraud_type = ",".join(sorted(fraud_types)) if fraud_types else "unknown_edit"
            return {
                "fraud_type": fraud_type,
                "regions": len(regions),
                "software": ann.get("file_attributes", {}).get("Software used", ""),
            }
    except (ValueError, SyntaxError):
        pass
    
    return {"fraud_type": "unknown_edit", "regions": 0}


def process_split(split_name: str, csv_path: Path, images_dir: Path, out_split: str) -> list[dict]:
    records = []
    
    if not csv_path.exists():
        print(f"  [WARN] {csv_path} not found, skipping")
        return records
    
    out_images = UNIFIED_DIR / out_split / "images"
    os.makedirs(out_images, exist_ok=True)
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            image_name = row.get("image", "").strip()
            if not image_name:
                continue
            
            is_forged = int(row.get("forged", "0"))
            
            ann_str = row.get("forgery annotations", "0")
            ann_info = parse_forgery_annotations(ann_str)
            
            txt_name = image_name.replace(".png", ".txt").replace(".jpg", ".txt")
            txt_path = images_dir / txt_name
            ocr_text = ""
            if txt_path.exists():
                ocr_text = txt_path.read_text(encoding="utf-8", errors="replace")
            
            fields = extract_fields_from_ocr_text(ocr_text)
            
            src_img = images_dir / image_name
            record_id = f"fia_{split_name}_{i:05d}"
            dst_img = out_images / f"{record_id}.png"
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            record = {
                "id": record_id,
                "image_path": f"images/{record_id}.png",
                "fields": {
                    "vendor": fields.get("vendor"),
                    "date": fields.get("date"),
                    "total": fields.get("total"),
                },
                "label": {
                    "is_forged": is_forged,
                    "fraud_type": ann_info["fraud_type"] if is_forged else "none",
                },
                "source": "finditagain",
                "_ocr_gt": ocr_text[:500],
            }
            records.append(record)
    
    return records


def main():
    print(f"[Find-It-Again] Source: {FINDIT_DIR}")
    print(f"[Find-It-Again] Output: {UNIFIED_DIR}")
    
    if not FINDIT_DIR.exists():
        print("[Find-It-Again] ERROR: Dataset not found. Extract findit2.zip first.")
        return
    
    all_train = []
    all_test = []
    
    train_recs = process_split(
        "train",
        FINDIT_DIR / "train.txt",
        FINDIT_DIR / "train",
        "train",
    )
    all_train.extend(train_recs)
    print(f"  Train: {len(train_recs)} records ({sum(1 for r in train_recs if r['label']['is_forged'])} forged)")
    
    val_recs = process_split(
        "val",
        FINDIT_DIR / "val.txt",
        FINDIT_DIR / "val",
        "train",
    )
    all_train.extend(val_recs)
    print(f"  Val:   {len(val_recs)} records ({sum(1 for r in val_recs if r['label']['is_forged'])} forged)")
    
    test_recs = process_split(
        "test",
        FINDIT_DIR / "test.txt",
        FINDIT_DIR / "test",
        "test",
    )
    all_test.extend(test_recs)
    print(f"  Test:  {len(test_recs)} records ({sum(1 for r in test_recs if r['label']['is_forged'])} forged)")
    
    train_jsonl = UNIFIED_DIR / "train" / "train.jsonl"
    with open(train_jsonl, "a") as f:
        for rec in all_train:
            f.write(json.dumps(rec) + "\n")
    
    test_jsonl = UNIFIED_DIR / "test" / "test.jsonl"
    with open(test_jsonl, "a") as f:
        for rec in all_test:
            f.write(json.dumps(rec) + "\n")
    
    total_train = len(all_train)
    total_forged = sum(1 for r in all_train if r["label"]["is_forged"])
    vendors_found = sum(1 for r in all_train if r["fields"].get("vendor"))
    dates_found = sum(1 for r in all_train if r["fields"].get("date"))
    totals_found = sum(1 for r in all_train if r["fields"].get("total"))
    
    print(f"\n[Find-It-Again] Done!")
    print(f"  Added to train: {total_train} ({total_forged} forged)")
    print(f"  Added to test:  {len(all_test)}")
    print(f"  Vendors: {vendors_found}/{total_train}")
    print(f"  Dates:   {dates_found}/{total_train}")
    print(f"  Totals:  {totals_found}/{total_train}")


if __name__ == "__main__":
    main()
