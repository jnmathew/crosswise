# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crosswise is an OCR pipeline project focused on extracting text from crossword puzzles and similar printed materials. The project uses Tesseract OCR with OpenCV preprocessing to handle challenging scanned images with small text and varying quality.

## Dependencies

**Tesseract Binary Required**: Install separately via `brew install tesseract` (macOS) before installing Python dependencies.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Core dependencies:
- pytesseract (Tesseract wrapper)
- opencv-python (image preprocessing)
- numpy (array operations)

## Architecture

### OCR Pipeline (`src/pipeline/`)

The pipeline consists of preprocessing utilities designed to improve OCR accuracy on challenging inputs:

**ocr_presets.py** - Core OCR configuration and preprocessing:
- `build_clues_config(dpi)` - Tesseract config for multi-word clue text (PSM 6)
- `build_digits_config(dpi)` - Tesseract config for digit-only recognition (PSM 11, whitelist "0-9")
- `ensure_readable_scale(image)` - Intelligent upscaling based on estimated x-height to ensure text is readable (minimum 14px median x-height)
- `prepare_digits_image(image)` - Specialized preprocessing for small digits using top-hat morphology and adaptive thresholding

**Key Preprocessing Strategies**:
- Adaptive upscaling based on connected component analysis (not fixed scaling)
- Denoising with fastNlMeansDenoising and bilateral filtering to handle halftone/newsprint artifacts
- Morphological top-hat transform to enhance light text on darker backgrounds
- Adaptive thresholding tuned for small text strokes
- DPI defaults to 350 for Tesseract (adjustable per use case)

## Development Notes

- Constants follow ALL_CAPS naming convention (e.g., `DEFAULT_DPI`, `TARGET_X_HEIGHT`)
- Image processing uses grayscale conversion with careful handling of both color and grayscale inputs
- Scale factors capped at 3.0× maximum to prevent excessive upscaling
- Fallback strategies implemented (adaptive → Otsu thresholding) when component detection fails
