"""
OCR presets and a utility to ensure text is large/readable before running Tesseract.

Functions:
- build_clues_config(dpi=350)
- build_digits_config(dpi=350)
- ensure_readable_scale(image, target_x_height=14, max_scale=3.0)
- prepare_digits_image(image, scale=2.0)

Place in src/pipeline and import from preprocess.py or your OCR module.
"""
from typing import Tuple
from pathlib import Path

import cv2
import numpy as np

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