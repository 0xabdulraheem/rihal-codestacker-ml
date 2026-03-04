from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.anomaly import AnomalyDetector, build_feature_vector
from src.extraction import extract_fields
from src.ocr import OCREngine, group_into_lines


class DocFusionSolution:

    def __init__(self):
        self._ocr: OCREngine | None = None

    @property
    def ocr(self) -> OCREngine:
        if self._ocr is None:
            gpu = _gpu_available()
            self._ocr = OCREngine(gpu=gpu)
        return self._ocr

    def train(self, train_dir: str, work_dir: str) -> str:
        model_dir = os.path.join(work_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        train_jsonl = os.path.join(train_dir, "train.jsonl")
        records = _load_jsonl(train_jsonl)

        features_list: list[dict] = []
        labels: list[int] = []

        for rec in records:
            image_path = os.path.join(train_dir, rec.get("image_path", ""))
            if not os.path.exists(image_path):
                image_path = None

            ocr_text = ""
            ocr_results = []
            if image_path:
                try:
                    ocr_results = self.ocr.extract_text(image_path)
                    ocr_text = "\n".join(
                        group_into_lines(ocr_results)
                    )
                except Exception:
                    pass

            fields = extract_fields(ocr_text, ocr_results)
            feat = build_feature_vector(image_path, fields, ocr_text, ocr_results)
            features_list.append(feat)

            label_info = rec.get("label", {})
            labels.append(int(label_info.get("is_forged", 0)))

        detector = AnomalyDetector()
        detector.train(features_list, labels)
        detector.save(model_dir)

        return model_dir

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        detector = AnomalyDetector()
        model_path = os.path.join(model_dir, "anomaly_model.joblib")
        if os.path.exists(model_path):
            try:
                detector.load(model_dir)
            except Exception:
                pass

        test_jsonl = os.path.join(data_dir, "test.jsonl")
        records = _load_jsonl(test_jsonl)

        features_list: list[dict] = []
        extracted_fields: list[dict] = []

        for rec in records:
            image_path = os.path.join(data_dir, rec.get("image_path", ""))
            if not os.path.exists(image_path):
                image_path = None

            ocr_text = ""
            ocr_results = []
            if image_path:
                try:
                    ocr_results = self.ocr.extract_text(image_path)
                    ocr_text = "\n".join(group_into_lines(ocr_results))
                except Exception:
                    pass

            fields = extract_fields(ocr_text, ocr_results)

            feat = build_feature_vector(image_path, fields, ocr_text, ocr_results)
            features_list.append(feat)
            extracted_fields.append(fields)

        if features_list:
            predictions = detector.predict(features_list)
        else:
            predictions = []

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            for idx, (rec, fields, is_forged) in enumerate(zip(records, extracted_fields, predictions)):
                prediction = {
                    "id": rec.get("id") or f"unknown_{idx}",
                    "vendor": fields.get("vendor"),
                    "date": fields.get("date"),
                    "total": fields.get("total"),
                    "is_forged": int(is_forged),
                }
                f.write(json.dumps(prediction) + "\n")


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
