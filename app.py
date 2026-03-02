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

st.set_page_config(
    page_title="DocFusion — Intelligent Document Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #4285f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #5f6368;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .status-genuine {
        background-color: #e6f4ea;
        color: #137333;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #34a853;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .status-forged {
        background-color: #fce8e6;
        color: #c5221f;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #ea4335;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .field-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e8eaed;
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
    model_dir = _ROOT / "tmp_work" / "model"
    if model_dir.exists():
        try:
            detector.load(str(model_dir))
            return detector, True
        except Exception:
            pass
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
    st.markdown("Built for **rihal CodeStacker 2026**")

st.markdown('<div class="main-header">DocFusion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Document Analysis & Forgery Detection</div>', unsafe_allow_html=True)

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
                ocr_results = ocr_engine.extract_text(tmp_path)

            display_img = img_array.copy()
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
            
            lines = []
            for r in sorted(ocr_results, key=lambda x: min(p[1] for p in x["bbox"])):
                if r["confidence"] >= confidence_threshold:
                    lines.append(r["text"])
            ocr_text = "\n".join(lines)

            fields = extract_fields(ocr_text, ocr_results)

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
            feat = build_feature_vector(tmp_path, fields, ocr_text)

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
                "is_forged": int(detector.predict([feat])[0]) if detector_loaded else 0,
            }
            st.code(json.dumps(output, indent=2), language="json")

    finally:
        os.unlink(tmp_path)

else:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄 Extract")
        st.markdown("Automatically extract **vendor**, **date**, and **total** from scanned receipts using OCR.")

    with col2:
        st.markdown("### 🔍 Detect")
        st.markdown("Identify **forged** or **tampered** documents using image analysis and ML-based anomaly detection.")

    with col3:
        st.markdown("### 📊 Analyze")
        st.markdown("Visualize **Error Level Analysis** and OCR confidence to understand document authenticity.")
