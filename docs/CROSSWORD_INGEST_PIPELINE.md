# Crossword Ingest Pipeline — Overview

For an end-to-end, detailed guide, see `preprocess.md`. This document provides a compact overview and references the code entry points.

## Stages (summary)
1. Intake & quality check (brightness, blur, skew)
2. Preprocessing (grayscale, adaptive threshold, dewarp, optional CLAHE)
3. Grid lattice detection (lines → lattice → cell crops; black cell mask)
4. Cell metadata extraction (Tesseract `--psm 11` for clue numbers)
5. Clue detection & OCR (mask grid, detect text blocks, Tesseract `--psm 6`)
6. Post-OCR correction (spell-check, glyph normalization, enumerations)
7. Puzzle model assembly (canonical JSON)
8. Validation & feedback (counts, confidence heatmap, manual edits)

## Code entry points
- `src/main.py` — end-to-end runner (WIP)
- `src/pipeline/image_preprocessing.py`
- `src/pipeline/grid_detection.py`
- `src/pipeline/ocr_utils.py`
- `src/pipeline/clue_extraction.py`
- `src/pipeline/postprocess.py`
- `src/pipeline/puzzle_model.py`
- `src/pipeline/exporter.py`
- `src/pipeline/validator.py`

## Outputs
- Debug artifacts: `data/logs/`, `data/temp/`
- Canonical JSON: `data/output_json/`
- Examples: `data/examples/`
