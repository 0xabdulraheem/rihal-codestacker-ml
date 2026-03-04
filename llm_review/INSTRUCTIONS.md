# LLM Code Review Instructions

You are reviewing a submission for the **Rihal CodeStacker 2026 Machine Learning Challenge** called **DocFusion**.

**IMPORTANT: Do NOT make any changes to any files. Only read and analyze, then write your suggestions in a new file called `suggestions_[your_name].md` in this same folder.**

## What This Project Does

DocFusion is an intelligent document processing pipeline that:
1. Extracts structured fields (vendor, date, total) from scanned receipt images using OCR
2. Detects forged/tampered documents using image forensics + ML
3. Provides a Streamlit web dashboard for interactive analysis
4. Integrates with an autograder harness via `solution.py`

## Repository

**GitHub**: https://github.com/0xabdulraheem/rihal-codestacker-ml

Please clone or browse the repo to review all code.

## Challenge Requirements

Read the full challenge spec here: https://github.com/rihal-om/rihal-codestacker/blob/main/ML/README.md

### Key Requirements Summary

**Level 1 - EDA**: Jupyter notebook exploring the datasets (SROIE, Find-It-Again, CORD)
**Level 2 - Extraction**: Extract `vendor`, `date`, `total` from receipts via OCR
**Level 3A - Anomaly Detection**: Predict `is_forged` (0 or 1) using image forensics
**Level 3B - Web UI**: Streamlit dashboard with upload, extracted fields, anomaly status, visual highlights
**Level 4A - Harness**: `solution.py` with `DocFusionSolution` class (`train()` and `predict()`)
**Level 4B - Performance**: Optimize inference latency, memory usage, model size
**Level 4C - Reproducibility**: Clear structure, dependency management, deterministic execution

**Bonus Tasks**:
- Dockerfile for containerization
- Cloud deployment with public demo link
- LLM-generated human-readable anomaly summaries

**Submission Checklist**:
- `solution.py` at root with `DocFusionSolution`
- `requirements.txt` or `pyproject.toml` (Python 3.13+)
- Jupyter notebooks for EDA
- Web UI code (`app.py`)
- Comprehensive README

## What To Review

Please analyze the following and provide specific, actionable suggestions:

1. **Code Quality**: Is the code clean, well-structured, and production-ready?
2. **Extraction Accuracy**: Can `src/extraction.py` be improved for more receipt formats?
3. **Anomaly Detection**: Can `src/anomaly.py` and `src/preprocessing.py` features be improved?
4. **Web UI**: Can `app.py` be more polished or have better UX?
5. **Performance**: Is `solution.py` efficient enough? Any bottlenecks?
6. **Missing Features**: What are we missing that could make this a winning submission?
7. **README**: Is the documentation comprehensive enough?
8. **Bonus**: Ideas for cloud deployment, LLM summaries, or other creative additions?
9. **Edge Cases**: Any potential failures the autograder might trigger?
10. **Competition Strategy**: What would make this stand out from other submissions?

## Key Files To Review

| File | Purpose |
|------|---------|
| `solution.py` | Autograder harness interface |
| `app.py` | Streamlit web UI |
| `src/preprocessing.py` | Image preprocessing, ELA, feature extraction |
| `src/ocr.py` | EasyOCR wrapper |
| `src/extraction.py` | Field extraction (vendor/date/total) |
| `src/anomaly.py` | LightGBM anomaly detector |
| `src/summarizer.py` | LLM-style anomaly summary generator |
| `notebooks/eda.ipynb` | EDA notebook |
| `Dockerfile` | Containerization |
| `README.md` | Documentation |
| `pyproject.toml` | Dependencies and metadata |
| `requirements.txt` | Python dependencies |

## Output Format

Create a file called `suggestions_[your_name].md` in this folder with:

```
# [Your Name] Review Suggestions

## Critical Issues (must fix)
- ...

## High Priority Improvements
- ...

## Nice To Have
- ...

## What's Already Good
- ...
```

Thank you for your review.
