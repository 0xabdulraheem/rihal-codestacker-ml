from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.array(Image.open(path).convert("RGB"))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
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
    if np.mean(gray) < 128:
        gray = cv2.bitwise_not(gray)
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


def resize_for_ocr(img: np.ndarray, target_height: int = 1024, min_height: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        return cv2.resize(img, (new_w, min_height), interpolation=cv2.INTER_CUBIC)
    if h <= target_height:
        return img
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    img = resize_for_ocr(img)
    img = deskew(img)
    gray = to_grayscale(img)
    noise_level = float(np.std(cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))))
    if noise_level > 15:
        img = denoise(img)
    return img


def error_level_analysis(img: np.ndarray, quality: int = 90, amplify: bool = True) -> np.ndarray:
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
    if amplify:
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
        ela = error_level_analysis(img, amplify=False)
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
        ela_lo = error_level_analysis(img, quality=75, amplify=False)
        ela_lo_gray = to_grayscale(ela_lo)
        features["ela_lo_mean"] = float(np.mean(ela_lo_gray))
        features["ela_diff_mean"] = float(np.mean(ela_lo_gray) - features.get("ela_mean", 0))
    except Exception:
        features["ela_lo_mean"] = 0.0
        features["ela_diff_mean"] = 0.0

    try:
        ela_mid = error_level_analysis(img, quality=85, amplify=False)
        ela_mid_gray = to_grayscale(ela_mid)
        features["ela_mid_mean"] = float(np.mean(ela_mid_gray))
        features["ela_diff_hi_mid"] = float(features.get("ela_mean", 0) - np.mean(ela_mid_gray))
    except Exception:
        features["ela_mid_mean"] = 0.0
        features["ela_diff_hi_mid"] = 0.0

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)
    features["noise_residual_mean"] = float(np.mean(lap_abs))
    features["noise_residual_std"] = float(np.std(lap_abs))
    features["noise_residual_kurtosis"] = float(_safe_kurtosis(lap_abs.flatten()))

    f = np.fft.fft2(gray.astype(float))
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log1p(np.abs(fshift))
    fh, fw = magnitude.shape
    center_energy = float(np.mean(magnitude[fh // 4:3 * fh // 4, fw // 4:3 * fw // 4]))
    total_energy = float(np.mean(magnitude))
    features["fft_center_energy"] = center_energy
    features["fft_edge_ratio"] = float((total_energy - center_energy) / (center_energy + 1e-6))

    if len(img.shape) == 3:
        for i, channel in enumerate(["b", "g", "r"]):
            ch = img[:, :, i]
            features[f"{channel}_mean"] = float(np.mean(ch))
            features[f"{channel}_std"] = float(np.std(ch))

        features.update(_color_patch_features(img, grid=(4, 4)))

    features.update(_dct_blockiness_features(gray))
    features.update(_regional_noise_features(gray, grid=(4, 4)))

    try:
        features.update(_lbp_features(ela_gray if "ela_mean" in features else gray))
    except Exception:
        features["lbp_entropy"] = 0.0
        features["lbp_uniformity"] = 0.0
        features["lbp_max_bin"] = 0.0

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
    for ch, name in enumerate(["b", "g", "r"]):
        vals = np.array(channel_patch_means[ch])
        if len(vals) > 1:
            features[f"{name}_patch_std"] = float(np.std(vals))
            features[f"{name}_patch_range"] = float(np.max(vals) - np.min(vals))
        else:
            features[f"{name}_patch_std"] = 0.0
            features[f"{name}_patch_range"] = 0.0

    return features


def _dct_blockiness_features(gray: np.ndarray) -> dict:
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 16 or w8 < 16:
        return {"dct_blockiness": 0.0, "dct_horiz_ratio": 0.0, "dct_vert_ratio": 0.0}
    g = gray[:h8, :w8].astype(float)
    left_cols = g[:, 7::8]
    right_cols = g[:, 8::8]
    n = min(left_cols.shape[1], right_cols.shape[1])
    horiz = float(np.abs(left_cols[:, :n] - right_cols[:, :n]).mean()) if n > 0 else 0.0
    top_rows = g[7::8, :]
    bot_rows = g[8::8, :]
    m = min(top_rows.shape[0], bot_rows.shape[0])
    vert = float(np.abs(top_rows[:m, :] - bot_rows[:m, :]).mean()) if m > 0 else 0.0
    interior_h = float(np.abs(np.diff(g, axis=1)).mean())
    interior_v = float(np.abs(np.diff(g, axis=0)).mean())
    ratio_h = horiz / (interior_h + 1e-6)
    ratio_v = vert / (interior_v + 1e-6)
    return {
        "dct_blockiness": float((ratio_h + ratio_v) / 2),
        "dct_horiz_ratio": float(ratio_h),
        "dct_vert_ratio": float(ratio_v),
    }


def _regional_noise_features(gray: np.ndarray, grid: tuple = (4, 4)) -> dict:
    h, w = gray.shape
    rows, cols = grid
    ph, pw = h // max(rows, 1), w // max(cols, 1)
    if ph < 8 or pw < 8:
        return {"noise_regional_std": 0.0, "noise_regional_max_ratio": 0.0, "noise_regional_cv": 0.0}
    noise_levels = []
    for r in range(rows):
        for c in range(cols):
            patch = gray[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            blur = cv2.GaussianBlur(patch, (5, 5), 0)
            noise = cv2.absdiff(patch, blur)
            noise_levels.append(float(np.std(noise)))
    arr = np.array(noise_levels)
    mean_val = float(np.mean(arr))
    return {
        "noise_regional_std": float(np.std(arr)),
        "noise_regional_max_ratio": float(np.max(arr) / (mean_val + 1e-6)),
        "noise_regional_cv": float(np.std(arr) / (mean_val + 1e-6)),
    }


def _lbp_features(gray: np.ndarray) -> dict:
    h, w = gray.shape
    if h < 4 or w < 4:
        return {"lbp_entropy": 0.0, "lbp_uniformity": 0.0, "lbp_max_bin": 0.0}
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp = (lbp << 1) | (shifted >= gray).astype(np.uint8)
    hist, _ = np.histogram(lbp[1:-1, 1:-1].ravel(), bins=32, range=(0, 256), density=True)
    entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-9)))
    uniformity = float(np.sum(hist ** 2))
    return {
        "lbp_entropy": entropy,
        "lbp_uniformity": uniformity,
        "lbp_max_bin": float(np.max(hist)),
    }
