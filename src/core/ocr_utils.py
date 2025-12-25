"""
OCR utilities and preprocessing for crossword puzzle text extraction.

Functions:
- build_clues_config(dpi=350) - Tesseract config for multi-word clue text
- build_digits_config(dpi=350) - Tesseract config for digit-only recognition
- ensure_readable_scale(image) - Intelligent upscaling based on x-height
- prepare_digits_image(image) - Preprocessing for small digits
- ocr_digit_roi(roi_image, config) - Multi-variant ensemble digit OCR
- ocr_text_region(region, config) - OCR multi-word text regions
"""
from typing import Tuple
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from .models import OCRResult
from .config import Settings

# Constants (ALL CAPS)
DEFAULT_DPI = 350
TARGET_X_HEIGHT = 14  # desired median x-height in pixels for reliable OCR
MAX_UPSCALE = 3.0     # don't blow up too large


def build_clues_config(dpi: int = DEFAULT_DPI) -> str:
    """Return Tesseract config string tuned for clue text (multi-word lines)."""
    cfg = (
        f"--oem 1 --psm 6 -l eng "
        f"-c preserve_interword_spaces=1 "
        f"-c user_defined_dpi={dpi}"
    )
    return cfg


def build_digits_config(dpi: int = DEFAULT_DPI) -> str:
    """Return Tesseract config string tuned for small digit-only OCR."""
    cfg = (
        f"--oem 1 --psm 11 -l eng "
        f"-c tessedit_char_whitelist=0123456789 "
        f"-c user_defined_dpi={dpi}"
    )
    return cfg


def ensure_readable_scale(image: np.ndarray,
                          target_x_height: int = TARGET_X_HEIGHT,
                          max_scale: float = MAX_UPSCALE) -> Tuple[np.ndarray, float, float]:
    """
    Estimate median 'x-height' (proxy = median contour height) and upscale image
    so median component height ~= target_x_height px.

    Returns: (upscaled_image, scale_factor, median_height_px)
    - scale_factor == 1.0 means no change.
    Notes:
      - Works on grayscale / color images.
      - Uses light adaptive threshold and connected component heights as a proxy.
    """
    if image is None:
        raise ValueError("image is None")

    gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # quick denoise to reduce halftone noise that fragments components
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # small-window adaptive threshold to preserve small strokes
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 8)

    # find connected components contours and collect heights
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = []
    img_h, img_w = gray.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # ignore very large regions (non-text) and very small noise
        if 2 <= h <= img_h * 0.5 and 2 <= w <= img_w * 0.9:
            heights.append(h)

    if not heights:
        # fallback: try Otsu and look for blob heights
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            _, _, _, h = cv2.boundingRect(c)
            if 2 <= h <= img_h * 0.5:
                heights.append(h)

    if not heights:
        # Cannot estimate; return original
        return image, 1.0, 0.0

    median_h = float(np.median(heights))

    if median_h <= 0:
        return image, 1.0, median_h

    scale = target_x_height / median_h
    if scale <= 1.0:
        return image, 1.0, median_h

    scale = min(scale, max_scale)
    new_w = int(round(img_w * scale))
    new_h = int(round(img_h * scale))
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return upscaled, float(scale), median_h


def prepare_digits_image(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Prepare a crop likely containing small digits for digit OCR:
    - upscale (INTER_CUBIC)
    - denoise
    - morphological top-hat to boost light numerals on darker background
    - adaptive threshold

    Caller may then pass the result to pytesseract with build_digits_config().
    """
    if image is None:
        raise ValueError("image is None")

    gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        w = int(round(gray.shape[1] * scale))
        h = int(round(gray.shape[0] * scale))
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_CUBIC)

    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)

    # top-hat: enhance bright details (digits on gray newsprint)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)

    # adaptive threshold tuned for small strokes
    thresh = cv2.adaptiveThreshold(tophat, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, 6)
    # small closing to connect thin strokes
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k2)

    return closed


def ocr_digit_roi(roi_image: np.ndarray, config: Settings, timeout: float = 2.0) -> OCRResult:
    """
    OCR a single digit ROI with multi-variant ensemble for maximum accuracy.

    Applies triangular mask to focus on clue-number region, then tries multiple
    preprocessing variants (Otsu, CLAHE+Otsu, Adaptive) combined with multiple
    PSM configs (7, 10, 6) for robust digit detection.

    Args:
        roi_image: Grayscale ROI image containing potential digit
        config: Settings object with OCR parameters
        timeout: Timeout per OCR attempt in seconds

    Returns:
        OCRResult with best detected digit and confidence
    """
    if roi_image is None or roi_image.size == 0:
        return OCRResult(text="", confidence=0.0)

    # Apply triangular mask to focus on top-left corner (clue number region)
    hR, wR = roi_image.shape[:2]
    mask = np.zeros((hR, wR), dtype=np.uint8)
    tri = np.array([[0, 0], [wR - 1, 0], [0, hR - 1]], dtype=np.int32)
    cv2.fillConvexPoly(mask, tri, 255)
    roi = cv2.bitwise_and(roi_image, roi_image, mask=mask)

    # Upscale to target height for better OCR
    target_h = 80
    scale = target_h / max(roi.shape[0], 1)
    roi_rs = cv2.resize(roi, (max(8, int(roi.shape[1] * scale)), target_h),
                       interpolation=cv2.INTER_CUBIC)

    # Build threshold variants
    variants = []

    # Variant 1: Base Otsu binary
    _, roi_th = cv2.threshold(roi_rs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(roi_th)

    # Variant 2: CLAHE + Otsu + slight dilation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(roi_rs)
    _, th_eq = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_eq = cv2.dilate(th_eq, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    variants.append(th_eq)

    # Variant 3: Adaptive Gaussian binary
    th_ad = cv2.adaptiveThreshold(roi_rs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 21, 5)
    variants.append(th_ad)

    # Try multiple PSM configs
    ocr_cfgs = [
        "--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789",  # Single text line
        "--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789",  # Single character
        "--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789",   # Uniform block of text
    ]

    best_pred = ""
    best_conf = -1

    for img_bin in variants:
        # Skip blank/solid images to save time
        black_ratio = float((img_bin < 128).mean())
        if black_ratio < 0.02 or black_ratio > 0.70:
            continue

        for ocr_config in ocr_cfgs:
            try:
                data = pytesseract.image_to_data(
                    img_bin,
                    config=ocr_config,
                    output_type=pytesseract.Output.DICT,
                    timeout=timeout
                )
            except Exception:
                data = {"text": [], "conf": []}

            # Find best digit result from this pass
            for i, txt in enumerate(data.get("text", [])):
                t = (txt or "").strip()
                try:
                    conf_i = int(data.get("conf", [])[i])
                except (ValueError, IndexError, TypeError):
                    try:
                        conf_i = int(float(data.get("conf", [])[i]))
                    except (ValueError, IndexError, TypeError):
                        conf_i = -1

                # Valid 1-3 digit number
                if t and all(ch.isdigit() for ch in t) and 1 <= len(t) <= 3:
                    if conf_i > best_conf:
                        best_conf = conf_i
                        best_pred = t

            # Early exit if we found high-confidence result
            if best_conf >= 75:
                break

        if best_conf >= 75:
            break

    # Normalize confidence to 0.0-1.0
    confidence = max(0.0, min(1.0, best_conf / 100.0)) if best_conf >= 0 else 0.0

    return OCRResult(text=best_pred, confidence=confidence)


def ocr_text_region(region: np.ndarray, config: Settings, timeout: float = 5.0) -> OCRResult:
    """
    OCR multi-word text region (for clue text).

    Applies ensure_readable_scale() then runs Tesseract with PSM 6 (uniform block of text).

    Args:
        region: Grayscale or color image region containing text
        config: Settings object with OCR parameters
        timeout: Timeout for OCR in seconds

    Returns:
        OCRResult with detected text and confidence
    """
    if region is None or region.size == 0:
        return OCRResult(text="", confidence=0.0)

    # Ensure text is large enough for OCR
    scaled, scale_factor, median_h = ensure_readable_scale(region)

    # Build Tesseract config
    tesseract_config = build_clues_config(dpi=config.OCR_DPI)

    try:
        # Get OCR result with confidence data
        data = pytesseract.image_to_data(
            scaled,
            config=tesseract_config,
            output_type=pytesseract.Output.DICT,
            timeout=timeout
        )

        # Combine all text and calculate average confidence
        texts = []
        confs = []

        for i, txt in enumerate(data.get("text", [])):
            t = (txt or "").strip()
            if t:
                texts.append(t)
                try:
                    conf = int(data.get("conf", [])[i])
                    if conf >= 0:
                        confs.append(conf)
                except (ValueError, IndexError, TypeError):
                    pass

        combined_text = " ".join(texts)
        avg_confidence = sum(confs) / len(confs) if confs else 0.0

        # Normalize to 0.0-1.0
        normalized_conf = max(0.0, min(1.0, avg_confidence / 100.0))

        return OCRResult(text=combined_text, confidence=normalized_conf)

    except Exception as e:
        # Return empty result on timeout or error
        return OCRResult(text="", confidence=0.0)
