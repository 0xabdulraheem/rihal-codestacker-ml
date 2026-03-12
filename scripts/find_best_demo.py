import sys, json, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.anomaly import AnomalyDetector, build_feature_vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
UNIFIED_TRAIN = os.path.join(ROOT, "data", "unified", "train")
UNIFIED_TEST = os.path.join(ROOT, "data", "unified", "test")

detector = AnomalyDetector()
detector.load(MODEL_DIR)
print(f"Model loaded: {len(detector.feature_columns)} features, threshold={detector.threshold:.2f}")

results = []
for split_name, split_dir in [("train", UNIFIED_TRAIN), ("test", UNIFIED_TEST)]:
    jsonl_path = os.path.join(split_dir, f"{split_name}.jsonl")
    if not os.path.exists(jsonl_path):
        continue
    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    fia_records = [r for r in records if r.get("source") == "finditagain"]
    print(f"\n{split_name}: {len(fia_records)} FIA records")
    
    for i, rec in enumerate(fia_records):
        img_path = os.path.join(split_dir, rec.get("image_path", ""))
        if not os.path.exists(img_path):
            continue
        label = rec.get("label", {}).get("is_forged", 0)
        feat = build_feature_vector(img_path, {}, "")
        proba = detector.predict_proba([feat])[0]
        results.append({
            "id": rec["id"],
            "split": split_name,
            "label": label,
            "proba": proba,
            "path": img_path,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(fia_records)}")

results.sort(key=lambda x: x["proba"])

print("\n=== TOP 5 LOWEST (most genuine-looking) ===")
for r in results[:5]:
    print(f"  {r['id']} label={r['label']} proba={r['proba']:.4f} {r['path'][-40:]}")

print("\n=== TOP 5 HIGHEST (most forged-looking) ===")
for r in results[-5:]:
    print(f"  {r['id']} label={r['label']} proba={r['proba']:.4f} {r['path'][-40:]}")

print("\n=== BEST GENUINE (lowest proba, label=0) ===")
genuine = [r for r in results if r["label"] == 0]
for r in genuine[:3]:
    print(f"  {r['id']} proba={r['proba']:.4f} {r['path'][-40:]}")

print("\n=== BEST FORGED (highest proba, label=1) ===")
forged = [r for r in results if r["label"] == 1]
for r in forged[-3:]:
    print(f"  {r['id']} proba={r['proba']:.4f} {r['path'][-40:]}")

DEMO_DIR = os.path.join(ROOT, "demo_receipts")
os.makedirs(DEMO_DIR, exist_ok=True)

best_genuine = genuine[0]
best_forged = forged[-1]
shutil.copy2(best_genuine["path"], os.path.join(DEMO_DIR, f"genuine_best_{best_genuine['proba']:.0%}.png".replace("%", "pct")))
shutil.copy2(best_forged["path"], os.path.join(DEMO_DIR, f"forged_best_{best_forged['proba']:.0%}.png".replace("%", "pct")))
print(f"\nCopied best genuine ({best_genuine['proba']:.4f}) and best forged ({best_forged['proba']:.4f}) to demo_receipts/")
