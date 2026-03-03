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

FINAL_TOTAL_KEYWORDS = [
    r"total\s*payable",
    r"net\s*am(?:oun)?t",
    r"amount\s*due",
    r"balance\s*due",
    r"grand\s*total",
    r"total\s*rounded",
    r"total\s*sales?\s*\(?inclusive",
    r"total\s*(?:\(incl|inc)",
]

GENERIC_TOTAL_RE = re.compile(r"\btotal\b", re.IGNORECASE)

NOT_TOTAL_KEYWORDS = [
    "cash", "change", "received", "tender", "rounding", "round adj",
    "subtotal", "sub total", "sub-total",
    "gst", "tax", "discount", "saving", "service",
    "total 0%", "total 6%", "total sr", "total zr",
    "supplies (excl", "supplies(excl",
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
        "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",
        "%m-%d-%y", "%m/%d/%y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.year < 100:
                dt = dt.replace(year=dt.year + 2000)
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


def _find_amount_in_line(line: str) -> str | None:
    m = re.search(r"(?:RM|USD|\$|£|€)\s*(\d+[.,]\d{2})", line, re.IGNORECASE)
    if m:
        try:
            return f"{float(m.group(1).replace(',', '.')):.2f}"
        except ValueError:
            pass
    amounts = re.findall(r"\b(\d+[.,]\d{2})\b", line)
    if amounts:
        try:
            return f"{float(amounts[-1].replace(',', '.')):.2f}"
        except ValueError:
            pass
    m2 = re.search(r"(?:RM|USD|\$|£|€)\s*(\d+)", line, re.IGNORECASE)
    if m2:
        try:
            val = float(m2.group(1))
            if val >= 1:
                return f"{val:.2f}"
        except ValueError:
            pass
    return None


def extract_total(text: str) -> str | None:
    lines = text.strip().split("\n")

    for kw_pattern in FINAL_TOTAL_KEYWORDS:
        for i, line in enumerate(lines):
            if re.search(kw_pattern, line, re.IGNORECASE):
                amt = _find_amount_in_line(line)
                if amt:
                    return amt
                if i + 1 < len(lines):
                    amt = _find_amount_in_line(lines[i + 1])
                    if amt:
                        return amt

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        lower = line.lower()

        if not GENERIC_TOTAL_RE.search(line):
            continue

        if any(excl in lower for excl in NOT_TOTAL_KEYWORDS):
            continue

        amt = _find_amount_in_line(line)
        if amt:
            return amt
        if i + 1 < len(lines):
            amt = _find_amount_in_line(lines[i + 1])
            if amt:
                return amt

    currency_amounts = re.findall(r"(?:RM|USD|\$|£|€)\s*(\d+[.,]\d{2})", text, re.IGNORECASE)
    if currency_amounts:
        values = []
        for amt in currency_amounts:
            try:
                values.append(float(amt.replace(",", ".")))
            except ValueError:
                continue
        if values:
            return f"{max(values):.2f}"

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
