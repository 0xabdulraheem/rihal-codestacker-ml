from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from rapidfuzz import fuzz


DATE_PATTERNS = [
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
    r"\b(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4})\b",
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
    r"\b(20\d{2}[01]\d[0-3]\d)\b",
]

TOTAL_PATTERNS = [
    r"(?:total|grand\s*total|amount\s*due|balance\s*due|net\s*total|total\s*amount|sum|payment)\s*[:\s]*[\$£€RM]?\s*(\d+[.,]\d{2})\b",
    r"(?:total|grand\s*total|amount\s*due|balance\s*due)\s*[:\s]*[\$£€RM]?\s*(\d+)\b",
    r"[\$£€RM]\s*(\d+[.,]\d{2})\b",
    r"\b(\d+[.,]\d{2})\s*(?:total|due|paid)\b",
]

SKIP_KEYWORDS = {
    "receipt", "invoice", "tax", "total", "subtotal", "change", "cash",
    "credit", "debit", "visa", "mastercard", "thank", "date", "time",
    "qty", "quantity", "item", "price", "amount", "discount", "vat",
    "gst", "tel", "phone", "fax", "email", "address", "no.", "number",
}


def normalize_date(raw: str) -> str | None:
    raw = raw.strip().replace(",", "")
    
    formats = [
        "%Y-%m-%d", "%Y/%m/%d",
        "%m/%d/%Y", "%m-%d-%Y",
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d %b %Y", "%d %B %Y",
        "%b %d %Y", "%B %d %Y",
        "%d%b%Y",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if 2000 <= dt.year <= 2030:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_date(text: str) -> str | None:
    for pattern in DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            normalized = normalize_date(match)
            if normalized:
                return normalized
    return None


def extract_total(text: str) -> str | None:
    candidates: list[tuple[float, str]] = []
    
    for priority, pattern in enumerate(TOTAL_PATTERNS):
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean = match.replace(",", ".")
            try:
                value = float(clean)
                if 0.01 <= value <= 99999.99:
                    candidates.append((priority, f"{value:.2f}"))
            except ValueError:
                continue

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    all_amounts = re.findall(r"\b(\d+[.,]\d{2})\b", text)
    if all_amounts:
        values = []
        for amt in all_amounts:
            try:
                values.append(float(amt.replace(",", ".")))
            except ValueError:
                continue
        if values:
            return f"{max(values):.2f}"

    return None


def extract_vendor(text: str, ocr_results: list[dict] | None = None) -> str | None:
    lines = text.strip().split("\n")
    
    for line in lines[:5]:
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 2:
            continue

        lower = line_clean.lower()
        
        alpha_ratio = sum(c.isalpha() for c in line_clean) / max(len(line_clean), 1)
        if alpha_ratio < 0.3:
            continue

        words = set(lower.split())
        if words & SKIP_KEYWORDS:
            if len(words) <= 2:
                continue

        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", line_clean):
            continue

        if re.match(r"^[\d\s\-\+\(\)]{7,}$", line_clean):
            continue

        return line_clean

    return None


def extract_fields(text: str, ocr_results: list[dict] | None = None) -> dict[str, Any]:
    return {
        "vendor": extract_vendor(text, ocr_results),
        "date": extract_date(text),
        "total": extract_total(text),
    }


def extract_fields_from_image(image_path: str, ocr_engine=None) -> dict[str, Any]:
    if ocr_engine is None:
        from .ocr import get_ocr_engine
        ocr_engine = get_ocr_engine()

    ocr_results = ocr_engine.extract_text(image_path)
    full_text = "\n".join(
        line for line in _group_text_lines(ocr_results)
    )
    
    fields = extract_fields(full_text, ocr_results)
    fields["_ocr_text"] = full_text
    fields["_ocr_results"] = ocr_results
    return fields


def _group_text_lines(results: list[dict], y_threshold: int = 15) -> list[str]:
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
