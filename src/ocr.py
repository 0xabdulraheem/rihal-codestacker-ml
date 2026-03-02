from __future__ import annotations

import os
from typing import Any

import numpy as np

from .preprocessing import load_image, preprocess_for_ocr


class OCREngine:

    def __init__(self, languages: list[str] | None = None, gpu: bool = True):
        self._reader = None
        self._languages = languages or ["en"]
        self._gpu = gpu

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        return self._reader

    def extract_text(self, image_path: str) -> list[dict[str, Any]]:
        img = load_image(image_path)
        img = preprocess_for_ocr(img)
        results = self.reader.readtext(img)
        
        extracted = []
        for bbox, text, conf in results:
            extracted.append({
                "text": text.strip(),
                "confidence": float(conf),
                "bbox": bbox,
            })
        return extracted

    def extract_full_text(self, image_path: str) -> str:
        results = self.extract_text(image_path)
        lines = self._group_into_lines(results)
        return "\n".join(lines)

    def _group_into_lines(self, results: list[dict], y_threshold: int = 15) -> list[str]:
        if not results:
            return []
        
        sorted_results = sorted(results, key=lambda r: (
            min(p[1] for p in r["bbox"]),
            min(p[0] for p in r["bbox"]),
        ))

        lines: list[list[dict]] = []
        current_line: list[dict] = [sorted_results[0]]
        current_y = min(p[1] for p in sorted_results[0]["bbox"])

        for result in sorted_results[1:]:
            y = min(p[1] for p in result["bbox"])
            if abs(y - current_y) < y_threshold:
                current_line.append(result)
            else:
                lines.append(current_line)
                current_line = [result]
                current_y = y
        lines.append(current_line)

        text_lines = []
        for line in lines:
            line.sort(key=lambda r: min(p[0] for p in r["bbox"]))
            text_lines.append(" ".join(r["text"] for r in line))
        
        return text_lines


_engine: OCREngine | None = None


def get_ocr_engine(gpu: bool = True) -> OCREngine:
    global _engine
    if _engine is None:
        _engine = OCREngine(gpu=gpu)
    return _engine
