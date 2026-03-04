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
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.anomaly import AnomalyDetector, build_feature_vector
from src.extraction import extract_fields
from src.ocr import OCREngine
from src.preprocessing import error_level_analysis, load_image
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
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 10px;
        padding: 0.8rem;
        background-color: #f8f9fa;
        color: #1a1a1a;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #5f6368;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1a1a;
    }
    .status-genuine, .status-forged {
        margin-bottom: 1rem;
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

@st.cache_resource
def load_ocr_engine():
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False
    return OCREngine(gpu=gpu)

@st.cache_resource
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

with st.sidebar:
    st.markdown("### Settings")
    show_ocr_text = st.checkbox("Show raw OCR text", value=False)
    show_ela = st.checkbox("Show Error Level Analysis", value=True)
    show_bboxes = st.checkbox("Show OCR bounding boxes", value=False)
    confidence_threshold = st.slider("OCR confidence threshold", 0.0, 1.0, 0.3, 0.05)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("**DocFusion** analyzes scanned receipts to extract structured data and detect potential forgeries using OCR + ML.")
    st.markdown("Built for **Rihal CodeStacker 2026 ML challenge**")

st.markdown('<div class="main-header">DocFusion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Document Analysis and Forgery Detection</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a receipt image",
    type=["png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    help="Supported formats: PNG, JPG, TIFF, BMP, WebP",
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        pil_img = Image.open(uploaded_file)
        img_array = np.array(pil_img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        col_img, col_results = st.columns([1, 1])

        with col_img:
            st.markdown("#### Uploaded Receipt")
            
            with st.spinner("Running OCR..."):
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
                        "bbox": bbox,
                    })

            if show_bboxes and ocr_results:
                from PIL import ImageDraw as PilDraw
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
                    ela = error_level_analysis(img_bgr)
                    ela_rgb = cv2.cvtColor(ela, cv2.COLOR_BGR2RGB)
                    st.image(ela_rgb, use_container_width=True, caption="ELA — brighter regions indicate potential tampering")
                except Exception as e:
                    st.warning(f"ELA failed: {e}")

        with col_results:
            st.markdown("#### Extracted Fields")
            
            all_lines = []
            display_lines = []
            for r in sorted(ocr_results, key=lambda x: min(p[1] for p in x["bbox"])):
                all_lines.append(r["text"])
                if r["confidence"] >= confidence_threshold:
                    display_lines.append(r["text"])
            full_ocr_text = "\n".join(all_lines)
            ocr_text = "\n".join(display_lines)

            fields = extract_fields(full_ocr_text, ocr_results)

            vendor = fields.get("vendor") or "Not detected"
            date = fields.get("date") or "Not detected"
            total = fields.get("total") or "Not detected"

            c1, c2, c3 = st.columns(3)
            c1.metric("Vendor", vendor)
            c2.metric("Date", date)
            c3.metric("Total", f"${total}" if total != "Not detected" else total)

            st.markdown("---")
            st.markdown("#### Forgery Analysis")

            detector, detector_loaded = load_detector()
            with st.spinner("Analyzing image forensics..."):
                feat = build_feature_vector(tmp_path, fields, full_ocr_text)

            if detector_loaded:
                prediction = detector.predict([feat])[0]
                proba = detector.predict_proba([feat])[0]

                if prediction == 1:
                    st.markdown(
                        f'<div class="status-forged">SUSPICIOUS — Forgery probability: {proba:.1%}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="status-genuine">GENUINE — Forgery probability: {proba:.1%}</div>',
                        unsafe_allow_html=True,
                    )

                summary = generate_anomaly_summary(fields, feat, prediction, proba)
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
                with st.expander("Image Features (raw)"):
                    feat_df = {k: f"{v:.4f}" for k, v in sorted(feat.items())}
                    st.json(feat_df)

            if show_ocr_text:
                st.markdown("---")
                st.markdown("#### Raw OCR Output")
                st.text(ocr_text if ocr_text else "(No text detected)")

            st.markdown("---")
            st.markdown("#### Prediction JSON")
            output = {
                "vendor": fields.get("vendor"),
                "date": fields.get("date"),
                "total": fields.get("total"),
                "is_forged": int(prediction) if detector_loaded else 0,
            }
            st.code(json.dumps(output, indent=2), language="json")
            st.download_button(
                "Download Prediction JSON",
                json.dumps(output, indent=2),
                file_name="prediction.json",
                mime="application/json",
            )

    finally:
        os.unlink(tmp_path)

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
