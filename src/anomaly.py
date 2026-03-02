from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .preprocessing import compute_image_features, load_image


def compute_text_features(fields: dict[str, Any], ocr_text: str) -> dict[str, float]:
    features: dict[str, float] = {}

    features["has_vendor"] = 1.0 if fields.get("vendor") else 0.0
    features["has_date"] = 1.0 if fields.get("date") else 0.0
    features["has_total"] = 1.0 if fields.get("total") else 0.0
    features["field_completeness"] = sum([
        features["has_vendor"], features["has_date"], features["has_total"]
    ]) / 3.0

    total = fields.get("total")
    if total:
        try:
            total_val = float(total)
            features["total_value"] = total_val
            features["total_log"] = float(np.log1p(total_val))
            features["total_is_round"] = 1.0 if total_val == int(total_val) else 0.0
            features["total_decimal_part"] = total_val - int(total_val)
        except (ValueError, TypeError):
            features["total_value"] = 0.0
            features["total_log"] = 0.0
            features["total_is_round"] = 0.0
            features["total_decimal_part"] = 0.0
    else:
        features["total_value"] = 0.0
        features["total_log"] = 0.0
        features["total_is_round"] = 0.0
        features["total_decimal_part"] = 0.0

    features["text_length"] = float(len(ocr_text))
    features["num_lines"] = float(ocr_text.count("\n") + 1) if ocr_text else 0.0
    features["num_words"] = float(len(ocr_text.split())) if ocr_text else 0.0
    features["avg_word_length"] = (
        float(np.mean([len(w) for w in ocr_text.split()])) if ocr_text.split() else 0.0
    )

    if ocr_text:
        features["alpha_ratio"] = sum(c.isalpha() for c in ocr_text) / max(len(ocr_text), 1)
        features["digit_ratio"] = sum(c.isdigit() for c in ocr_text) / max(len(ocr_text), 1)
        features["special_ratio"] = sum(not c.isalnum() and not c.isspace() for c in ocr_text) / max(len(ocr_text), 1)
        features["upper_ratio"] = sum(c.isupper() for c in ocr_text) / max(len(ocr_text), 1)
    else:
        features["alpha_ratio"] = 0.0
        features["digit_ratio"] = 0.0
        features["special_ratio"] = 0.0
        features["upper_ratio"] = 0.0

    vendor = fields.get("vendor") or ""
    features["vendor_length"] = float(len(vendor))
    features["vendor_word_count"] = float(len(vendor.split())) if vendor else 0.0

    return features


FEATURE_COLUMNS: list[str] | None = None


def build_feature_vector(
    image_path: str | None,
    fields: dict[str, Any],
    ocr_text: str,
) -> dict[str, float]:
    features: dict[str, float] = {}

    if image_path and os.path.exists(image_path):
        try:
            img = load_image(image_path)
            img_feats = compute_image_features(img)
            features.update(img_feats)
        except Exception:
            pass

    text_feats = compute_text_features(fields, ocr_text)
    features.update(text_feats)

    return features


class AnomalyDetector:

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []
        self._fitted = False

    def train(
        self,
        features_list: list[dict[str, float]],
        labels: list[int],
    ) -> None:
        df = pd.DataFrame(features_list)
        self.feature_columns = sorted(df.columns.tolist())
        df = df[self.feature_columns].fillna(0.0)

        X = pd.DataFrame(self.scaler.fit_transform(df.values), columns=self.feature_columns)
        y = np.array(labels)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            scale_pos_weight = n_neg / n_pos
        else:
            scale_pos_weight = 1.0

        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.7,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1,
            )
            self.model.fit(X, y)
        except ImportError:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight("balanced", y)
            self.model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                min_samples_leaf=10,
                random_state=42,
            )
            self.model.fit(X, y, sample_weight=sample_weights)

        self._fitted = True

    def predict(self, features_list: list[dict[str, float]]) -> list[int]:
        if not self._fitted:
            return [0] * len(features_list)

        df = pd.DataFrame(features_list)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_columns].fillna(0.0)

        X = pd.DataFrame(self.scaler.transform(df.values), columns=self.feature_columns)
        predictions = self.model.predict(X)
        return [int(p) for p in predictions]

    def predict_proba(self, features_list: list[dict[str, float]]) -> list[float]:
        if not self._fitted:
            return [0.0] * len(features_list)

        df = pd.DataFrame(features_list)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_columns].fillna(0.0)

        X = pd.DataFrame(self.scaler.transform(df.values), columns=self.feature_columns)
        proba = self.model.predict_proba(X)
        return [float(p[1]) for p in proba]

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "anomaly_model.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, "anomaly_scaler.joblib"))
        with open(os.path.join(model_dir, "anomaly_features.json"), "w") as f:
            json.dump(self.feature_columns, f)
        self._fitted = True

    def load(self, model_dir: str) -> None:
        self.model = joblib.load(os.path.join(model_dir, "anomaly_model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "anomaly_scaler.joblib"))
        with open(os.path.join(model_dir, "anomaly_features.json")) as f:
            self.feature_columns = json.load(f)
        self._fitted = True
