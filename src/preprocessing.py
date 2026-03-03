from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def binarize(img: np.ndarray, method: str = "otsu") -> np.ndarray:
    gray = to_grayscale(img)
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def denoise(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def deskew(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def resize_for_ocr(img: np.ndarray, target_height: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= target_height:
        return img
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    img = resize_for_ocr(img)
    img = deskew(img)
    return img


def error_level_analysis(img: np.ndarray, quality: int = 90) -> np.ndarray:
    import io

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer).convert("RGB"))
    compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR)

    h_orig, w_orig = img.shape[:2]
    h_comp, w_comp = compressed.shape[:2]
    if (h_orig, w_orig) != (h_comp, w_comp):
        compressed = cv2.resize(compressed, (w_orig, h_orig))

    ela = cv2.absdiff(img, compressed)
    ela = (ela * 10).clip(0, 255).astype(np.uint8)
    return ela


def compute_image_features(img: np.ndarray) -> dict:
    gray = to_grayscale(img)
    features = {}

    features["mean_intensity"] = float(np.mean(gray))
    features["std_intensity"] = float(np.std(gray))
    features["min_intensity"] = float(np.min(gray))
    features["max_intensity"] = float(np.max(gray))

    h, w = gray.shape[:2]
    features["height"] = h
    features["width"] = w
    features["aspect_ratio"] = w / max(h, 1)

    edges = cv2.Canny(gray, 50, 150)
    features["edge_density"] = float(np.sum(edges > 0) / (h * w))

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    features["hist_entropy"] = float(-np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0])))

    try:
        ela = error_level_analysis(img)
        ela_gray = to_grayscale(ela)
        features["ela_mean"] = float(np.mean(ela_gray))
        features["ela_std"] = float(np.std(ela_gray))
        features["ela_max"] = float(np.max(ela_gray))
        features["ela_q95"] = float(np.percentile(ela_gray, 95))
        features["ela_q99"] = float(np.percentile(ela_gray, 99))
        features["ela_q75"] = float(np.percentile(ela_gray, 75))
        features["ela_skew"] = float(_safe_skew(ela_gray.flatten()))
        features["ela_kurtosis"] = float(_safe_kurtosis(ela_gray.flatten()))

        patch_stats = _patch_ela_features(ela_gray, grid=(4, 4))
        features.update(patch_stats)
    except Exception:
        for k in ["ela_mean", "ela_std", "ela_max", "ela_q95", "ela_q99",
                   "ela_q75", "ela_skew", "ela_kurtosis",
                   "ela_patch_std", "ela_patch_max_ratio", "ela_patch_range",
                   "ela_patch_cv"]:
            features[k] = 0.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features["blur_score"] = float(laplacian.var())

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blurred)
    features["noise_mean"] = float(np.mean(noise))
    features["noise_std"] = float(np.std(noise))
    features["noise_max"] = float(np.max(noise))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    features["gradient_mean"] = float(np.mean(gradient_mag))
    features["gradient_std"] = float(np.std(gradient_mag))
    features["gradient_skew"] = float(_safe_skew(gradient_mag.flatten()))

    try:
        ela_lo = error_level_analysis(img, quality=75)
        ela_lo_gray = to_grayscale(ela_lo)
        features["ela_lo_mean"] = float(np.mean(ela_lo_gray))
        features["ela_diff_mean"] = float(np.mean(ela_lo_gray) - features.get("ela_mean", 0))
    except Exception:
        features["ela_lo_mean"] = 0.0
        features["ela_diff_mean"] = 0.0

    if len(img.shape) == 3:
        for i, channel in enumerate(["b", "c_g", "r"]):
            ch = img[:, :, i]
            features[f"{channel}_mean"] = float(np.mean(ch))
            features[f"{channel}_std"] = float(np.std(ch))

        features.update(_color_patch_features(img, grid=(4, 4)))

    return features


def _safe_skew(arr: np.ndarray) -> float:
    from scipy import stats
    try:
        return float(stats.skew(arr, nan_policy="omit"))
    except Exception:
        return 0.0


def _safe_kurtosis(arr: np.ndarray) -> float:
    from scipy import stats
    try:
        return float(stats.kurtosis(arr, nan_policy="omit"))
    except Exception:
        return 0.0


def _patch_ela_features(ela_gray: np.ndarray, grid: tuple = (4, 4)) -> dict:
    h, w = ela_gray.shape[:2]
    rows, cols = grid
    ph, pw = h // max(rows, 1), w // max(cols, 1)
    
    patch_means = []
    for r in range(rows):
        for c in range(cols):
            patch = ela_gray[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            if patch.size > 0:
                patch_means.append(float(np.mean(patch)))

    if not patch_means:
        return {"ela_patch_std": 0.0, "ela_patch_max_ratio": 0.0,
                "ela_patch_range": 0.0, "ela_patch_cv": 0.0}

    arr = np.array(patch_means)
    global_mean = np.mean(arr) if np.mean(arr) > 0 else 1.0
    
    return {
        "ela_patch_std": float(np.std(arr)),
        "ela_patch_max_ratio": float(np.max(arr) / global_mean) if global_mean > 0 else 0.0,
        "ela_patch_range": float(np.max(arr) - np.min(arr)),
        "ela_patch_cv": float(np.std(arr) / global_mean) if global_mean > 0 else 0.0,
    }


def _color_patch_features(img: np.ndarray, grid: tuple = (4, 4)) -> dict:
    h, w = img.shape[:2]
    rows, cols = grid
    ph, pw = h // max(rows, 1), w // max(cols, 1)

    channel_patch_means = {0: [], 1: [], 2: []}
    for r in range(rows):
        for c in range(cols):
            patch = img[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            if patch.size > 0:
                for ch in range(3):
                    channel_patch_means[ch].append(float(np.mean(patch[:, :, ch])))

    features = {}
    for ch, name in enumerate(["b", "c_g", "r"]):
        vals = np.array(channel_patch_means[ch])
        if len(vals) > 1:
            features[f"{name}_patch_std"] = float(np.std(vals))
            features[f"{name}_patch_range"] = float(np.max(vals) - np.min(vals))
        else:
            features[f"{name}_patch_std"] = 0.0
            features[f"{name}_patch_range"] = 0.0

    return features
