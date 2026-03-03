from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .ocr import group_into_lines

DATE_PATTERNS = [
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
    r"\b(\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4})\b",
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
    r"\b(20\d{2}[01]\d[0-3]\d)\b",
]

DATE_CONTEXT_KEYWORDS = [
    "date", "dated", "issued", "transaction", "receipt",
    "invoice", "bill", "sold", "purchased",
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


def _ocr_fix_digits(s: str) -> str:
    return s.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1").replace("S", "5").replace("B", "8").replace("Z", "2")


def _normalize_amount(raw: str) -> float | None:
    raw = _ocr_fix_digits(raw.strip())
    if re.match(r"^\d{1,3}(\.\d{3})+(,\d{2})$", raw):
        raw = raw.replace(".", "").replace(",", ".")
    elif re.match(r"^\d{1,3}(,\d{3})+(\.\d{2})?$", raw):
        raw = raw.replace(",", "")
    else:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def normalize_date(raw: str) -> str | None:
    raw = raw.strip().replace(",", "")

    parts = re.split(r"[-/.]" , raw)
    if len(parts) == 3:
        try:
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            a, b, c = 0, 0, 0

        if a > 31 and 1 <= b <= 12 and 1 <= c <= 31:
            formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]
        elif a > 12 and 1 <= b <= 12:
            formats = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y"]
        elif b > 12 and 1 <= a <= 12:
            formats = ["%m/%d/%Y", "%m-%d-%Y",
                        "%m/%d/%y", "%m-%d-%y"]
        else:
            formats = [
                "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                "%m/%d/%Y", "%m-%d-%Y",
                "%Y-%m-%d", "%Y/%m/%d",
                "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
                "%m/%d/%y", "%m-%d-%y",
            ]
    else:
        formats = [
            "%Y-%m-%d", "%Y/%m/%d",
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%m/%d/%Y", "%m-%d-%Y",
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
            if 2000 <= dt.year <= 2040:
                return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_date(text: str) -> str | None:
    lines = text.strip().split("\n")
    candidates: list[tuple[str, int]] = []

    for line_idx, line in enumerate(lines):
        for pattern in DATE_PATTERNS:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                raw = match.group(1)
                normalized = normalize_date(raw)
                if normalized:
                    lower_line = line.lower()
                    has_context = any(kw in lower_line for kw in DATE_CONTEXT_KEYWORDS)
                    score = 10 if has_context else 0
                    score -= line_idx
                    candidates.append((normalized, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return None


def _find_amount_in_line(line: str) -> str | None:
    m = re.search(r"(?:RM|USD|\$|£|€)\s*([\d,.OolISBZ]+)", line, re.IGNORECASE)
    if m:
        val = _normalize_amount(m.group(1))
        if val is not None and val > 0:
            return f"{val:.2f}"
    amounts = re.findall(r"\b(\d{1,3}(?:[,.]\d{3})*[.,]\d{2})\b", line)
    if not amounts:
        amounts = re.findall(r"\b(\d+[.,]\d{2})\b", line)
    if amounts:
        val = _normalize_amount(amounts[-1])
        if val is not None:
            return f"{val:.2f}"
    m2 = re.search(r"(?:RM|USD|\$|£|€)\s*(\d+)", line, re.IGNORECASE)
    if m2:
        val = _normalize_amount(m2.group(1))
        if val is not None and val >= 1:
            return f"{val:.2f}"
    if re.search(r"\btotal\b", line, re.IGNORECASE):
        m3 = re.search(r"\b(\d{1,6})\b(?!\s*\d)", line)
        if m3:
            val = _normalize_amount(m3.group(1))
            if val is not None and val >= 1:
                return f"{val:.2f}"
    return None


def extract_total(text: str) -> str | None:
    lines = text.strip().split("\n")

    for kw_pattern in FINAL_TOTAL_KEYWORDS:
        for i, line in enumerate(lines):
            if re.search(kw_pattern, line, re.IGNORECASE):
                for offset in range(3):
                    if i + offset < len(lines):
                        amt = _find_amount_in_line(lines[i + offset])
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

    for i in range(len(lines) - 1, -1, -1):
        amt = _find_amount_in_line(lines[i])
        if amt:
            return amt

    return None


def extract_vendor(
    text: str,
    ocr_results: list[dict] | None = None,
) -> str | None:
    lines = text.strip().split("\n")

    candidates: list[tuple[str, float]] = []
    for idx, line in enumerate(lines[:10]):
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 2:
            continue

        lower = line_clean.lower()

        alpha_ratio = sum(c.isalpha() for c in line_clean) / max(len(line_clean), 1)
        if alpha_ratio < 0.3:
            continue

        if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", line_clean):
            continue
        if re.match(r"^[\d\s\-\+\(\)]{7,}$", line_clean):
            continue
        if re.match(r"^\d+.*(?:jalan|road|street|avenue|blvd|lane)", lower):
            continue
        if re.match(r"^\d{5}\s", line_clean):
            continue
        if re.match(r"^(?:tel|fax|phone|email|www\.|http)\b", lower):
            continue
        if re.search(r"\b(?:reg|registration|gst|sst|vat|company|co\.)\s*no\.?\b", lower):
            continue
        if re.match(r"^[\dA-Z]{6,15}[-/]?\d{0,4}[A-Z]?$", line_clean):
            continue

        words = set(lower.split())
        if words & SKIP_KEYWORDS and len(words) <= 2:
            continue

        score = 100.0 - idx * 5
        if line_clean.isupper():
            score += 20
        elif line_clean.istitle():
            score += 10
        if alpha_ratio > 0.7:
            score += 10
        if len(line_clean) > 5:
            score += 5

        if ocr_results:
            for r in ocr_results:
                if r["text"].strip() in line_clean:
                    bbox = r["bbox"]
                    width = max(p[0] for p in bbox) - min(p[0] for p in bbox)
                    if width > 100:
                        score += 5
                    score += r.get("confidence", 0) * 5
                    break

        candidates.append((line_clean, score))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


def extract_fields(
    text: str,
    ocr_results: list[dict] | None = None,
    confidence_threshold: float = 0.15,
) -> dict[str, Any]:
    if ocr_results:
        filtered = [r for r in ocr_results if r.get("confidence", 1.0) >= confidence_threshold]
        if filtered:
            filtered_text = "\n".join(group_into_lines(filtered))
        else:
            filtered_text = text
    else:
        filtered_text = text

    return {
        "vendor": extract_vendor(filtered_text, ocr_results),
        "date": extract_date(filtered_text),
        "total": extract_total(filtered_text),
    }


def extract_fields_from_image(image_path: str, ocr_engine=None) -> dict[str, Any]:
    if ocr_engine is None:
        from .ocr import get_ocr_engine
        ocr_engine = get_ocr_engine()

    ocr_results = ocr_engine.extract_text(image_path)
    full_text = "\n".join(group_into_lines(ocr_results))

    fields = extract_fields(full_text, ocr_results)
    fields["_ocr_text"] = full_text
    fields["_ocr_results"] = ocr_results
    return fields
