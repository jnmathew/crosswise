# 🧩 Crossword Ingest Pipeline

**Last Updated:** 2025-11-02  
**Maintainer:** Joseph Mathew  
**Related:** `vision.md`, `requirements.txt`

This guide describes an end-to-end workflow to turn a crossword photo into a structured puzzle model consumable by a reasoning layer.

---

## 🔧 Setup (macOS)
- Install system OCR: `brew install tesseract`
- Create a venv and install Python deps (OpenCV + pytesseract) from `requirements.txt`.

---

## 🥇 Stage 1 — Image Intake & Quality Check

Inputs
- Camera capture or gallery import

Process
- Show adjustable crop overlay (corners can be dragged)
- Auto-assess quality with quick heuristics:
  - brightness > threshold
  - blur metric (Laplacian variance)
  - skew angle < 3°

Outputs
- Accepted image or user prompt to retake

User feedback
- If poor, prompt: “Retake in better light / straighten grid.”

---

## 🥈 Stage 2 — Preprocessing

Process
1) Convert to grayscale  
2) Adaptive threshold (`cv2.ADAPTIVE_THRESH_GAUSSIAN_C`)  
3) Dewarp via four‑point perspective transform (`cv2.getPerspectiveTransform`)  
4) Optional CLAHE (contrast-limited adaptive histogram equalization)

Artifacts
- Save binary and contrast-boosted versions for debugging

---

## 🥉 Stage 3 — Grid Lattice Detection

Process
1) Morphological closing → Hough lines (vertical + horizontal)  
2) Intersections → derive N×N lattice  
3) Snap lattice to equal spacing (median cell width/height)  
4) Extract each cell as a sub-image  
5) Detect black cells by fill ratio; record coordinates

Outputs
- Grid geometry, cell crops, black/white mask

---

## 🏅 Stage 4 — Cell Metadata Extraction

Process
- Run Tesseract (`--psm 11`) on each non-black cell to detect small digits (clue numbers)
- Optionally detect circled/shaded cells (mean brightness threshold)

Output schema (per cell)

```json
{
  "row": 0,
  "col": 0,
  "is_block": false,
  "clue_number": 1
}
```

---

## 🌟 Stage 5 — Clue Detection & OCR (single image)

Goal
- Find Across/Down clues in the same photo as the grid without assuming position.

5.0 Mask the grid
- Use the grid bbox from Stage 3 and white it out so lines don’t confuse text detection.

5.1 Locate text blocks
- Downscale + binarize  
- Morphology to merge text:
  - horizontal dilate → join words into lines
  - vertical dilate → join lines into blocks
- Connected components → candidate rectangles
- Filter by area/aspect to drop tiny scraps

5.2 Auto-pick clue regions
- Score each block: `score = text_density * area * proximity_bonus`
  - text_density = black_pixels / area
  - area = prefer larger blocks
  - proximity_bonus = boost if near grid edges (~10–40 px)
- Keep top 1–2 blocks
- Fallback: if no block scores above a threshold, let the user drag one rectangle over the clues

5.3 OCR regions
- Tesseract: `--oem 1 --psm 6` (uniform text block)  
- Keep per-line OCR confidence

5.4 Split Across vs Down without headers
- If “ACROSS” / “DOWN” found (case-insensitive), split there  
- Else, pattern-split by numbered lines:  
  Regex: `^\s*(\d+)[\.\)]?\s+(.*)` → number + clue text  
- Optional later: infer split when numbering restarts after a long run

5.5 Handle wraps and hyphens
- Merge wrapped lines (no punctuation + next line lowercase/indented)  
- Join hyphen breaks: `end-` + `start` → `endstart`

5.6 Parse enumerations
- Extract `( … )` forms like `(6)`, `(6,5)`, `(3-4)` and compute lengths

5.7 Store results

```json
{
  "clues": [
    {"num": 1, "text": "Kitchen VIP", "enum": [4], "raw_conf": 0.93},
    {"num": 2, "text": "Garden tool", "enum": [7], "raw_conf": 0.89}
  ],
  "had_headers": false
}
```

5.8 UX fallback
- If fewer than N_min clues (e.g., < 8) parsed, show overlay with a “Refine Box” handle and retry OCR

---

## 🥮 Stage 6 — Post‑OCR Correction

Process
- Spell-check against English + crossword-abbreviation dictionary  
- Normalize confusions (O↔0, I↔l)  
- Enforce enumerations (e.g., `(6,5)` → 6 + 5)  
- Flag any line with OCR confidence < 80% for review

---

## 🥡 Stage 7 — Puzzle Model Assembly

Build a structured JSON object as the single source of truth.

```json
{
  "grid_size": 15,
  "cells": [],
  "clues": {
    "across": [
      { "num": 1, "text": "Kitchen VIP", "len": 4 },
      { "num": 5, "text": "Garden tool", "len": 7 }
    ],
    "down": []
  }
}
```

---

## ⚙️ Stage 8 — Validation & Feedback

Checks
- Count of Across/Down clues roughly matches number of starting cells  
- Highlight mismatches and allow inline edits  
- OCR-confidence heatmap to focus user review  
- If no headers found, allow user to draw clue region and persist preference

---

## 🧄 Stage 9 — Next Steps

- Feed the verified model into the reasoning/solver module  
- Cache original image + JSON for training/debugging

---

## 💡 Implementation Tips

- Wrap OpenCV + Tesseract calls in helpers (`extract_grid()`, `ocr_clues()`)  
- Maintain a central `Puzzle` class to store grid + clue lists  
- Log/save artifacts per stage to simplify debugging  
- Include Apache 2.0 attribution for OpenCV and Tesseract

---

## 🤾 License Notices (Example)

```
This product includes software developed by:
• The OpenCV project (https://opencv.org/)
• The Tesseract OCR engine (https://github.com/tesseract-ocr/tesseract)
Both licensed under the Apache License, Version 2.0:
http://www.apache.org/licenses/LICENSE-2.0
```
