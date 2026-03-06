#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_sroie():
    out_dir = DATA_DIR / "sroie"
    if out_dir.exists() and any(out_dir.iterdir()):
        print("[SROIE] Already exists, skipping.")
        return

    print("[SROIE] Downloading from Kaggle...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("urbikn/sroie-datasetv2")
        print(f"[SROIE] Downloaded to: {path}")
        os.makedirs(out_dir, exist_ok=True)
        src = Path(path)
        if src.is_dir():
            for item in src.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(src)
                    dest = out_dir / rel
                    os.makedirs(dest.parent, exist_ok=True)
                    shutil.copy2(item, dest)
        print(f"[SROIE] Prepared at: {out_dir}")
    except Exception as e:
        print(f"[SROIE] Error: {e}")
        print("[SROIE] Manual download: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2")
        print(f"[SROIE] Extract to: {out_dir}")


def download_cord():
    out_dir = DATA_DIR / "cord"
    if out_dir.exists() and any(out_dir.iterdir()):
        print("[CORD] Already exists, skipping.")
        return

    print("[CORD] Downloading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("naver-clova-ix/cord-v2")
        os.makedirs(out_dir, exist_ok=True)
        
        for split_name in ds:
            split_dir = out_dir / split_name
            os.makedirs(split_dir / "images", exist_ok=True)

            split = ds[split_name]
            for i, sample in enumerate(split):
                img = sample.get("image")
                if img is not None:
                    img_path = split_dir / "images" / f"{i:05d}.png"
                    img.save(str(img_path))
                
                gt = sample.get("ground_truth", "")
                gt_path = split_dir / f"{i:05d}_gt.json"
                with open(gt_path, "w") as f:
                    f.write(gt if isinstance(gt, str) else str(gt))

        print(f"[CORD] Prepared at: {out_dir}")
    except Exception as e:
        print(f"[CORD] Error: {e}")
        print("[CORD] Install: pip install datasets")
        print("[CORD] Or browse: https://huggingface.co/datasets/naver-clova-ix/cord-v2")


def download_finditagain():
    out_dir = DATA_DIR / "finditagain"
    if out_dir.exists() and any(out_dir.iterdir()):
        print("[Find-It-Again] Already exists, skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)
    print("[Find-It-Again] This dataset requires manual download:")
    print("  URL: https://l3i-share.univ-lr.fr/2023Finditagain/index.html")
    print(f"  Extract to: {out_dir}")
    print("  The dataset contains genuine SROIE receipts + realistic forgeries.")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}\n")
    
    download_sroie()
    print()
    download_cord()
    print()
    download_finditagain()
    print("\nDone.")


if __name__ == "__main__":
    main()
