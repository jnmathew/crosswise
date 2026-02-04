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

### Grid Detection (`src/core/`)

**grid_detection.py** - Crossword grid extraction from newspaper images:
- Adaptive threshold selection with multiple fallback strategies (gap, percentile, Otsu)
- Quad detection and perspective transformation
- Black cell classification for grid structure analysis
- `assign_clue_numbers()` - Compute clue numbers from grid structure algorithmically
- `compute_clue_slots()` - Derive all clue slots with positions and answer lengths

**image_preprocessing.py** - General image preprocessing utilities:
- Four-point perspective transform for grid warping
- Rotation correction (detects and fixes grid alignment using Hough line detection)
- Contour analysis and quadrilateral extraction

**clue_column_detector.py** - Multi-column layout detection for clue extraction:
- Hybrid column detection combining vertical projection and text clustering
- Handles both full-height and partial-height columns
- Automatic separator line placement between detected columns
- Optional non-text region masking

**clue_extraction.py** - OCR parsing and puzzle verification:
- `parse_ocr_markdown()` - Parse Mistral OCR markdown output into structured clue data
- `verify_puzzle()` - Match OCR clues against grid slots, ensure 100% correspondence
- `match_clues_to_slots()` - Pair each OCR clue with its grid position and answer length
- `build_puzzle_clues()` - Create structured Clue objects for the Puzzle model

### Interactive Tools (`src/examples/`)

**interactive_masker.py** - GUI tool for manual preprocessing:
- MASK mode: Draw white rectangles over unwanted areas (grids, ads, acrostics)
- SEPARATOR mode: Click two points to draw tilted separator lines matching column angles
- Save/load coordinates as JSON for reproducible preprocessing
- Press 'v' to toggle modes, 's' to save, 'u' to undo, 'c' to clear
- Separator width: 8px for clear OCR guidance

**process_masked_image.py** - Automated post-masking pipeline:
- Loads manually masked images
- Applies automatic column detection if needed
- Draws separator lines and saves preprocessed images
- Optional Mistral OCR integration with `--run-ocr` flag

See `src/examples/COLUMN_DETECTION_WORKFLOW.md` for complete usage guide.

### OCR Integration

**Tesseract OCR** (`src/pipeline/`) - Traditional OCR for grid digit extraction:
- Used for extracting clue numbers from grid cells
- Requires local Tesseract installation

**Mistral OCR API** (`src/examples/mistralv2.py`) - Modern OCR for clue text extraction:
- Better accuracy for multi-column newspaper layouts
- Structured output using Pydantic models
- Handles complex layouts with separator guidance
- Requires MISTRAL_API_KEY environment variable

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

## Complete Workflow

For extracting crossword clues from newspaper images:

1. **Grid Detection**: Use `grid_detection.py` to locate and extract the crossword grid
2. **Manual Masking**: Run `interactive_masker.py` to:
   - Draw white rectangles over grids, ads, and irrelevant content
   - Place tilted separator lines between clue columns (matching text angle)
   - Save preprocessing coordinates for reuse
3. **OCR Extraction**: Use Mistral OCR on the masked image to extract structured clues
4. **Puzzle Verification**: Use `verify_puzzle()` to match OCR clues against grid slots
   - Every OCR clue must match a grid slot
   - Every grid slot must have an OCR clue
   - Verification must pass 100% before proceeding
5. **Build Puzzle**: Use `build_puzzle_clues()` to create structured Clue objects with answer lengths
6. **Output**: Verified puzzle saved as JSON with grid structure and clue data

The tilted separator approach (matching actual column angles) achieved 100% OCR accuracy on test images.

### Key Insight: Tilted Separators

Original vertical separators caused OCR errors when text columns were slightly angled. Using two-click tilted separators that follow the actual column angle achieved perfect (138/138) clue extraction.

## Development Notes

- Constants follow ALL_CAPS naming convention (e.g., `DEFAULT_DPI`, `TARGET_X_HEIGHT`)
- Image processing uses grayscale conversion with careful handling of both color and grayscale inputs
- Scale factors capped at 3.0× maximum to prevent excessive upscaling
- Fallback strategies implemented (adaptive → Otsu thresholding) when component detection fails
- Aqua/cyan (BGR: 255, 255, 0) used for separator lines - visible over white masks and gray newspaper
- Separator width: 8px for clear OCR visibility
- Grid clue numbers are computed algorithmically (not OCR'd) - more reliable
- Puzzle verification requires 100% match between OCR clues and grid slots

## Next Steps (Future Sessions)

The pipeline from image → verified puzzle is complete. Remaining work:

1. **Puzzle Solving**
   - Option A: LLM-based solving (send clue + answer length, get answer)
   - Option B: External crossword solver API/library
   - Need to fill in `answer` field for each Clue

2. **AI-Generated Hints**
   - Pre-generate one hint + one explanation per clue
   - Use solved answers to create targeted hints
   - Store alongside puzzle data (not real-time chatbot)

3. **Frontend Rendering**
   - Display interactive crossword grid
   - Show clues with hint system
   - User fills in answers

4. **Image Quality Check** (optional)
   - Add upfront blur/resolution detection
   - Fail fast with "please upload clearer image" message

### Data Model

Verified puzzle JSON stored at `data/output/<name>_puzzle.json`:
```json
{
  "metadata": { "source_image", "grid_size", "verification" },
  "grid": { "rows", "cols", "cells": [[{row, col, is_block, clue_number}]] },
  "clues": {
    "across": [{ "number", "text", "start", "length", "answer"? }],
    "down": [...]
  }
}
```
