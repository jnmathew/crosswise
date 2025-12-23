# 🧩 Crossword Solver App — Project Vision

**Last Updated:** 2025-11-02  
**Maintainer:** Joseph Mathew (solo developer)  
**License:** Proprietary app code; uses Apache-2.0 licensed components (Tesseract OCR + OpenCV)

---

## 🚀 Vision

Let users take a photo of any printed crossword, automatically parse it into a playable digital puzzle, and optionally get AI-based hints or explanations.

Long-term goal:
> A mobile crossword assistant that feels like a mentor — it understands your puzzle from a photo, can solve clues intelligently, and explain how.

---

## 🧠 Current Status

MVP: building the end-to-end ingest pipeline (image → structured JSON puzzle).  
Design is documented in `preprocess.md`.

Approach:
- Single image input (no separate image for clues).
- OpenCV for grid detection and cleaning.
- Tesseract for text recognition (grid numbers + clue text).
- Export a structured JSON model to feed into a reasoning layer (LLM or fine-tuned model).

---

## 🧩 Data Flow Overview
1. Capture crossword photo  
2. Preprocess image (grayscale, threshold, dewarp)  
3. Detect grid & extract cells  
4. OCR cell numbers (Tesseract `--psm 11`)  
5. Mask grid → find clue text region(s)  
6. OCR clues (Tesseract `--psm 6`)  
7. Parse clues → JSON structure  
8. Validate, correct, export  
9. Feed to reasoning layer (ChatGPT / model)

---

## 🧱 Core Modules (planned / in progress)

### 1) `image_preprocessing.py`
Grayscale conversion, adaptive thresholding, and perspective correction.

### 2) `grid_detection.py`
- Detect outer crossword boundary.
- Find vertical & horizontal lines (Hough transforms).
- Split grid into N×N cells and identify black cells.
- Extract small number crops for OCR.

### 3) `ocr_utils.py`
- Run Tesseract with tuned page segmentation modes:  
  `--psm 11` for scattered digits (cell numbers), `--psm 6` for uniform text blocks (clues).
- Dictionary-based corrections and confidence filtering.

### 4) `clue_extraction.py`
- Detect text regions adjacent to the grid (left/right/top/bottom).
- Score blocks by text density and proximity; run OCR on selected region(s).
- Split lines into clues (e.g., regex `^\s*(\d+)[\.\)]?\s+(.*)`).
- Extract enumerations (e.g., `(6,5)` → `[6,5]`).
- Return objects like `{ "num": X, "text": "...", "enum": [...] }`.

### 5) `postprocess.py`
- Spell-check and abbreviation normalization.
- Merge hyphenated/wrapped lines.
- Enforce enumeration integrity.
- Build `Puzzle` object with grid + clues.

### 6) `puzzle_model.py`
Conceptual structure:

```python
class Puzzle:
    grid_size: int
    cells: list  # Each cell = {row, col, is_block, clue_number}
    clues: dict  # {'across': [...], 'down': [...]}
```

---

## 🧰 Tech Stack
- Language: Python 3.10+
- Image Processing: OpenCV (cv2)
- OCR: Tesseract OCR (pytesseract)
- ML / AI Layer (planned): OpenAI GPT API / fine-tuned model
- UI (future): React Native / Flutter (TBD)
- Data Model: JSON / SQLite

---

## ⚙️ Key Parameters and Defaults
- `cell_threshold`: 0.5 — percent black pixels to mark a block.
- `ocr_confidence_cutoff`: 80 — minimum confidence to accept OCR result.
- `psm_digits`: 11 — Tesseract mode for clue numbers.
- `psm_clues`: 6 — Tesseract mode for clue text.
- `grid_snap_tolerance`: ±3 px — allowed alignment variance.
- `blur_limit`: 100 — Laplacian variance threshold for re-take suggestion.

---

## 🧾 Licensing & Attribution
This project integrates two Apache 2.0 licensed components. Includes software developed by:
- The OpenCV project (https://opencv.org/)
- The Tesseract OCR engine (https://github.com/tesseract-ocr/tesseract)

Licensed under the Apache License, Version 2.0:  
http://www.apache.org/licenses/LICENSE-2.0

You must include this text in the distributed app’s “Legal / About” section.

---

## 🧠 Future Phases
- Phase 1 (Current): Ingest pipeline MVP — fully working OCR + grid parser with structured output.
- Phase 2: Reasoning integration — add LLM / fine-tuned clue‑answer model (e.g., T5 or GPT).
- Phase 3: Explanation UI — users tap a clue to get reasoning (“why the answer fits”).
- Phase 4: Offline mode — on-device OCR and small transformer for edge use.
- Phase 5: Generation — create new crosswords automatically from word lists.

---

## 📂 Key Files
- `CROSSWORD_INGEST_PIPELINE.md` — step-by-step design document for the image pipeline.
- `PROJECT_CONTEXT_AI_AGENT.md` — context file defining the project state for takeover.
- `requirements.txt` — Python dependencies.
- `puzzle_model.py` — core data structure.
- `crossword_ingest.py` — entry script to run end-to-end ingestion.
- `tests/` — OCR and grid parsing test cases (future).

---

## 🧩 Notes for an AI / Future Maintainer
- Assumes one image per puzzle — grid and clues in the same frame.
- Prefer heuristics for clue segmentation first; consider ML later with data.
- Keep modules small and composable so the reasoning layer can evolve independently.
- Logging is crucial: save intermediate images (after each stage) for debugging OCR.
- Future integration target: small on-device transformer for clue → answer predictions.
