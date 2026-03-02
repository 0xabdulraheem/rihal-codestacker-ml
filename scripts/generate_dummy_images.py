#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DUMMY_DATA = Path(__file__).resolve().parent.parent / "dummy_data"


def generate_receipt_image(record: dict, out_path: str) -> None:
    fields = record.get("fields", {})
    label = record.get("label", {})
    
    width, height = 400, 550
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_large = ImageFont.truetype("arial.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        font_large = font
        font_small = font

    y = 20
    vendor = fields.get("vendor", "UNKNOWN STORE")
    draw.text((width // 2 - len(vendor) * 6, y), vendor, fill="black", font=font_large)
    y += 40

    draw.line([(20, y), (width - 20, y)], fill="gray", width=1)
    y += 15

    date = fields.get("date", "2024-01-01")
    draw.text((30, y), f"Date: {date}", fill="black", font=font)
    y += 25

    draw.text((30, y), f"Receipt #{record['id'].upper()}", fill="black", font=font_small)
    y += 25

    draw.line([(20, y), (width - 20, y)], fill="gray", width=1)
    y += 15

    items = [
        ("Item A", round(random.uniform(5, 50), 2)),
        ("Item B", round(random.uniform(5, 50), 2)),
        ("Item C", round(random.uniform(5, 50), 2)),
    ]
    total = float(fields.get("total", sum(p for _, p in items)))
    
    item_total = sum(p for _, p in items)
    if item_total > 0:
        scale = total / item_total
        items = [(name, round(price * scale, 2)) for name, price in items]

    for name, price in items:
        draw.text((30, y), name, fill="black", font=font)
        price_str = f"${price:.2f}"
        draw.text((width - 30 - len(price_str) * 9, y), price_str, fill="black", font=font)
        y += 22

    y += 10
    draw.line([(20, y), (width - 20, y)], fill="black", width=2)
    y += 15

    total_str = f"${total:.2f}"
    draw.text((30, y), "TOTAL:", fill="black", font=font_large)
    draw.text((width - 30 - len(total_str) * 12, y), total_str, fill="black", font=font_large)
    y += 35

    draw.line([(20, y), (width - 20, y)], fill="gray", width=1)
    y += 15

    draw.text((width // 2 - 60, y), "Thank you!", fill="gray", font=font)

    if label.get("is_forged", 0) == 1:
        fraud_type = label.get("fraud_type", "")
        if fraud_type == "price_change":
            draw.rectangle([(180, y - 60), (350, y - 35)], fill=(255, 255, 248))
            draw.text((width - 30 - len(total_str) * 12, y - 50), total_str, fill="black", font=font_large)
        elif fraud_type == "text_edit":
            for _ in range(50):
                x = random.randint(30, width - 30)
                ry = random.randint(40, 80)
                draw.point((x, ry), fill=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
        elif fraud_type == "layout_edit":
            draw.text((32, 22), vendor, fill=(30, 30, 30), font=font_large)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def main():
    random.seed(42)

    for split in ("train", "test"):
        jsonl_name = f"{split}.jsonl"
        jsonl_path = DUMMY_DATA / split / jsonl_name
        if not jsonl_path.exists():
            print(f"Skipping {split}: {jsonl_path} not found")
            continue

        images_dir = DUMMY_DATA / split / "images"
        os.makedirs(images_dir, exist_ok=True)

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_rel = record.get("image_path", f"images/{record['id']}.png")
                image_path = DUMMY_DATA / split / image_rel
                generate_receipt_image(record, str(image_path))
                print(f"  Generated: {image_path}")

    print("Done generating dummy images.")


if __name__ == "__main__":
    main()
