from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw as PilDraw

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.anomaly import AnomalyDetector, build_feature_vector
from src.extraction import extract_fields
from src.ocr import OCREngine, group_into_lines
from src.preprocessing import error_level_analysis
from src.summarizer import generate_anomaly_summary

st.set_page_config(
    page_title="DocFusion",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stDeployButton"] { display: none; }
    footer { visibility: hidden; }
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a73e8, #4285f4, #1a73e8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        color: #5f6368;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .status-genuine {
        background-color: #e6f4ea;
        color: #137333;
        padding: 1rem 1.4rem;
        border-radius: 10px;
        border-left: 5px solid #34a853;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .status-forged {
        background-color: #fce8e6;
        color: #c5221f;
        padding: 1rem 1.4rem;
        border-radius: 10px;
        border-left: 5px solid #ea4335;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .status-uncertain {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem 1.4rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .field-box {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        text-align: center;
    }
    .field-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .field-value {
        font-size: 1.1rem;
        font-weight: 600;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    [data-testid="stExpander"] {
        border: 1px solid #e8eaed;
        border-radius: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .stFileUploader > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_ocr_engine():
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    return OCREngine(gpu=gpu)


@st.cache_resource(show_spinner=False)
def load_detector():
    detector = AnomalyDetector()
    for model_dir in [_ROOT / "models", _ROOT / "tmp_work" / "model"]:
        if (model_dir / "anomaly_model.joblib").exists():
            try:
                detector.load(str(model_dir))
                return detector, True
            except Exception:
                continue
    return detector, False


def _run_analysis(file_bytes: bytes, file_name: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        img_array = np.array(pil_img)

        ocr_engine = load_ocr_engine()
        ocr_img = img_array
        if len(ocr_img.shape) == 2:
            pass
        elif ocr_img.shape[2] == 4:
            ocr_img = cv2.cvtColor(ocr_img, cv2.COLOR_RGBA2RGB)
        ocr_results_raw = ocr_engine.reader.readtext(ocr_img)
        ocr_results = []
        for bbox, text, conf in ocr_results_raw:
            ocr_results.append({
                "text": text.strip(),
                "confidence": float(conf),
                "bbox": [[float(p[0]), float(p[1])] for p in bbox],
            })

        full_ocr_text = "\n".join(group_into_lines(ocr_results))
        fields = extract_fields(full_ocr_text, ocr_results, confidence_threshold=0.3)

        detector, detector_loaded = load_detector()
        feat = build_feature_vector(tmp_path, fields, full_ocr_text, ocr_results)

        prediction = 0
        proba = 0.0
        if detector_loaded:
            prediction = detector.predict([feat])[0]
            proba = detector.predict_proba([feat])[0]

        if proba >= 0.6:
            verdict_label = "SUSPICIOUS"
        elif proba >= 0.3:
            verdict_label = "UNCERTAIN"
        else:
            verdict_label = "GENUINE"

        summary = generate_anomaly_summary(fields, feat, prediction, proba) if detector_loaded else ""

        return {
            "file_name": file_name,
            "fields": fields,
            "ocr_results": ocr_results,
            "full_ocr_text": full_ocr_text,
            "feat": feat,
            "prediction": prediction,
            "proba": proba,
            "verdict": verdict_label,
            "summary": summary,
            "detector_loaded": detector_loaded,
        }
    finally:
        os.unlink(tmp_path)


def _render_result(result: dict, file_bytes: bytes, show_ela: bool, show_bboxes: bool,
                   show_ocr_text: bool, confidence_threshold: float):
    pil_img = Image.open(io.BytesIO(file_bytes))
    img_array = np.array(pil_img)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    file_name = result["file_name"]
    fields = result["fields"]
    ocr_results = result["ocr_results"]
    full_ocr_text = result["full_ocr_text"]
    feat = result["feat"]
    prediction = result["prediction"]
    proba = result["proba"]
    verdict_label = result["verdict"]
    summary = result["summary"]
    detector_loaded = result["detector_loaded"]

    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.markdown(f"#### {file_name}")

        if show_bboxes and ocr_results:
            draw_img = pil_img.copy()
            draw = PilDraw.Draw(draw_img)
            for r in ocr_results:
                if r["confidence"] >= confidence_threshold:
                    bbox = r["bbox"]
                    pts = [(int(p[0]), int(p[1])) for p in bbox]
                    draw.polygon(pts, outline="blue", width=2)
            st.image(draw_img, use_container_width=True)
        else:
            st.image(pil_img, use_container_width=True)

        if show_ela:
            st.markdown("#### Error Level Analysis")
            try:
                ela = error_level_analysis(img_bgr, amplify=True)
                ela_rgb = cv2.cvtColor(ela, cv2.COLOR_BGR2RGB)
                st.image(ela_rgb, use_container_width=True, caption="ELA: brighter regions indicate potential tampering")
            except Exception as e:
                st.warning(f"ELA failed: {e}")

    with col_results:
        st.markdown("#### Extracted Fields")

        vendor = fields.get("vendor") or "Not detected"
        date = fields.get("date") or "Not detected"
        total = fields.get("total") or "Not detected"
        total_display = f"${total}" if total != "Not detected" else total

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="field-box"><div class="field-label">Vendor</div>'
                f'<div class="field-value">{vendor}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="field-box"><div class="field-label">Date</div>'
                f'<div class="field-value">{date}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="field-box"><div class="field-label">Total</div>'
                f'<div class="field-value">{total_display}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### Forgery Analysis")

        if detector_loaded:
            if verdict_label == "SUSPICIOUS":
                recommendation = "Reject / escalate for manual review"
                st.markdown(
                    f'<div class="status-forged">SUSPICIOUS: Forgery probability {proba:.1%}</div>',
                    unsafe_allow_html=True,
                )
            elif verdict_label == "UNCERTAIN":
                recommendation = "Manual review recommended"
                st.markdown(
                    f'<div class="status-uncertain">UNCERTAIN: Forgery probability {proba:.1%}</div>',
                    unsafe_allow_html=True,
                )
            else:
                recommendation = "Auto-approve"
                st.markdown(
                    f'<div class="status-genuine">GENUINE: Forgery probability {proba:.1%}</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"Recommendation: {recommendation}")

            with st.expander("Anomaly Summary"):
                st.markdown(summary)

            with st.expander("Feature Details"):
                feat_df = {k: f"{v:.4f}" for k, v in sorted(feat.items())}
                st.json(feat_df)
        else:
            st.info(
                "No trained model found. Run `python check_submission.py --submission .` "
                "to train the model first, then restart the app."
            )

        if show_ocr_text:
            st.markdown("---")
            st.markdown("#### Raw OCR Output")
            st.text(full_ocr_text if full_ocr_text else "(No text detected)")

        st.markdown("---")
        st.markdown("#### Prediction JSON")
        output = {
            "vendor": fields.get("vendor"),
            "date": fields.get("date"),
            "total": fields.get("total"),
            "is_forged": int(prediction),
        }
        st.code(json.dumps(output, indent=2), language="json")
        st.download_button(
            "Download Prediction JSON",
            json.dumps(output, indent=2),
            file_name="prediction.json",
            mime="application/json",
            key=f"dl_{file_name}",
        )


with st.sidebar:
    st.markdown("### Settings")
    show_ocr_text = st.checkbox("Show raw OCR text", value=False)
    show_ela = st.checkbox("Show Error Level Analysis", value=True)
    show_bboxes = st.checkbox("Show OCR bounding boxes", value=False)
    confidence_threshold = st.slider("OCR confidence threshold", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("**DocFusion** analyzes scanned receipts to extract structured data and detect potential forgeries using OCR + ML.")
    st.markdown("Built for **Rihal CodeStacker 2026 ML challenge**")

st.markdown('<div class="main-header">DocFusion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Document Analysis and Forgery Detection</div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "cached_results" not in st.session_state:
    st.session_state.cached_results = {}

uploaded_files = st.file_uploader(
    "Upload receipt image(s)",
    type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    help="Supported formats: PNG, JPG, TIFF, BMP, WebP. Upload multiple files for batch analysis.",
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
)

files_to_process = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        files_to_process.append((uploaded_file.getvalue(), uploaded_file.name))

if files_to_process:
    for file_bytes, file_name in files_to_process:
        cache_key = f"{file_name}_{len(file_bytes)}"
        if cache_key not in st.session_state.cached_results:
            with st.spinner(f"Analyzing {file_name}..."):
                result = _run_analysis(file_bytes, file_name)
            st.session_state.cached_results[cache_key] = result

            already_in_history = any(e["file"] == file_name for e in st.session_state.history)
            if not already_in_history:
                st.session_state.history.append({
                    "file": file_name,
                    "vendor": result["fields"].get("vendor") or "Not detected",
                    "date": result["fields"].get("date") or "Not detected",
                    "total": result["fields"].get("total") or "Not detected",
                    "verdict": result["verdict"],
                    "probability": f"{result['proba']:.1%}",
                })

        result = st.session_state.cached_results[cache_key]
        st.markdown("---")
        _render_result(result, file_bytes, show_ela, show_bboxes, show_ocr_text, confidence_threshold)
else:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Extract")
        st.markdown("Automatically extract **vendor**, **date**, and **total** from scanned receipts using OCR.")

    with col2:
        st.markdown("### Detect")
        st.markdown("Identify **forged** or **tampered** documents using image analysis and ML-based anomaly detection.")

    with col3:
        st.markdown("### Analyze")
        st.markdown("Visualize **Error Level Analysis** and OCR confidence to understand document authenticity.")

with st.sidebar:
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Session History")
        for entry in reversed(st.session_state.history[-10:]):
            icon = "!!" if entry["verdict"] == "SUSPICIOUS" else "OK"
            st.markdown(f"**[{icon}]** {entry['file']}: {entry['verdict']} ({entry['probability']})")
        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.cached_results = {}
            st.session_state.uploader_key += 1
            st.rerun()
