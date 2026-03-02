# DocFusion: Operation Intelligent Documents

**rihal CodeStacker 2026 — ML Challenge**

An end-to-end intelligent document processing pipeline that extracts structured fields from scanned receipts and detects forged/tampered documents using OCR + ML-based anomaly detection.

## Architecture

```
Receipt Image → Preprocessing → OCR (EasyOCR) → Field Extraction (Regex + Heuristics)
                                                        ↓
                                    Image Features (ELA, edge density, etc.)
                                                        ↓
                                         Anomaly Detector (LightGBM)
                                                        ↓
                                    {vendor, date, total, is_forged}
```

### Pipeline Components

| Module | Description |
|--------|-------------|
| `src/preprocessing.py` | Image preprocessing: deskew, denoise, binarize, ELA, feature extraction |
| `src/ocr.py` | EasyOCR wrapper with line grouping and confidence filtering |
| `src/extraction.py` | Regex-based field extraction for vendor, date, total |
| `src/anomaly.py` | LightGBM anomaly detector with image + text features |
| `solution.py` | DocFusionSolution harness interface (train + predict) |
| `app.py` | Streamlit web UI for interactive analysis |

## Quick Start

### 1. Setup

```bash
# Clone
git clone https://github.com/0xabdulraheem/rihal-codestacker-ml.git
cd rihal-codestacker-ml

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dummy Images

```bash
python scripts/generate_dummy_images.py
```

### 3. Run Local Validation

```bash
python check_submission.py --submission . --verbose
```

Expected output:
```
[check] Loaded submission: ...
[check] PASSED
[check] test records: 10
[check] predictions:  10
```

### 4. Launch Web UI

```bash
streamlit run app.py
```

### 5. Download Real Datasets

```bash
pip install datasets kagglehub
python scripts/download_datasets.py
```

## Challenge Levels

### Level 1: EDA
Jupyter notebook exploring dataset distributions, fraud types, and image features.
→ `notebooks/eda.ipynb`

### Level 2: Structured Information Extraction
Regex-based extraction of `vendor`, `date`, `total` from OCR text with fallback heuristics.
→ `src/extraction.py`

### Level 3: Anomaly Detection + Web UI
- **3A:** LightGBM classifier using 25+ features (ELA, edge density, text statistics, etc.)
- **3B:** Streamlit dashboard with upload, extraction, ELA visualization, and forgery scoring.
→ `src/anomaly.py`, `app.py`

### Level 4: Harness Integration
`DocFusionSolution` class with `train()` and `predict()` methods, optimized for inference speed and memory.
→ `solution.py`

### Bonus
- **Dockerfile** for containerized deployment
- **Cloud-ready** Streamlit app

## Docker

```bash
docker build -t docfusion .
docker run -p 8501:8501 docfusion
```

## Project Structure

```
rihal-codestacker-ml/
├── solution.py              # Harness interface (DocFusionSolution)
├── app.py                   # Streamlit Web UI
├── check_submission.py      # Local validation script
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata
├── Dockerfile               # Containerization
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Image preprocessing + ELA + features
│   ├── ocr.py               # EasyOCR engine
│   ├── extraction.py        # Field extraction (vendor/date/total)
│   └── anomaly.py           # LightGBM anomaly detector
├── notebooks/
│   └── eda.ipynb            # Level 1 EDA
├── scripts/
│   ├── generate_dummy_images.py
│   └── download_datasets.py
├── dummy_data/
│   ├── train/
│   │   ├── train.jsonl
│   │   └── images/
│   └── test/
│       ├── test.jsonl
│       └── images/
└── data/                    # Real datasets (gitignored)
```

## Tech Stack

- **OCR:** EasyOCR
- **ML:** LightGBM, scikit-learn
- **Image Analysis:** OpenCV, PIL (ELA, edge detection)
- **Web UI:** Streamlit
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly

## Author

**0xabdulraheem** — rihal CodeStacker 2026
