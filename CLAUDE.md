# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crosswise is a full-stack crossword puzzle app: upload a newspaper photo, automatically extract the grid and clues via OCR, solve with AI, and play interactively with hints. The pipeline uses OpenCV for grid detection, Mistral OCR for clue extraction, Claude Opus for solving, and a React frontend for the interactive player.

## Environment

**Virtual Environment**: Always use the project venv:
```bash
.venv/bin/python3 -m src.solver.solve_puzzle ...
```

## Dependencies

**Tesseract Binary Required**: Install separately via `brew install tesseract` (macOS) before installing Python dependencies.

Install Python dependencies:
```bash
.venv/bin/pip install -r requirements.txt
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

### Crossword Solver (`src/solver/`)

**solve_puzzle.py** - Main solver script:
```bash
# With TSV database (recommended)
.venv/bin/python3 -m src.solver.solve_puzzle data/output/puzzle.json --use-database

# Database only (no LLM fallback)
.venv/bin/python3 -m src.solver.solve_puzzle data/output/puzzle.json --database-only

# LLM only (original behavior)
.venv/bin/python3 -m src.solver.solve_puzzle data/output/puzzle.json
```

**clue_database.py** - SQLite-backed clue database:
- Loads `data/xd 2/clues.tsv` (7.5M clue/answer pairs from historical crosswords)
- Converts to SQLite on first use (`data/clues.db`)
- Provides fast lookup by clue text, pattern matching, and answer length
- Pattern matching uses GLOB (e.g., `C_T` matches `CAT`, `COT`, `CUT`)

**word_index.py** - Unified crossword word index:
- Loads multiple word lists (Broda, Crossword Nexus, Spread the Wordlist)
- Fast `contains()` membership testing and `match_pattern()` for pattern matching
- Quality scores for value ordering in the solver

**candidate_generator.py** - Candidate generation:
- `generate_candidates_with_database()` - Database lookup with Claude Opus fallback
- `generate_with_claude()` - Claude Opus candidate generation (replaced OpenAI)
- `regenerate_with_patterns()` - Pattern-based refinement from crossing letters
- `bouncer_filter()` - Score candidates by DB/word-index verification (0.3–1.0)
- `ensure_minimum_candidates()` - Guarantee >= 5 candidates per clue

**csp.py** - Constraint satisfaction solver:
- MAC (Maintaining Arc Consistency) with `mac_mode="search-only"` (skip AC-3 preprocessing)
- MRV heuristic for variable selection, score-based value ordering
- Conflict-Directed Backjumping (CDBJ)
- 50 random starting points, best-of-N runs to combat nondeterminism

**generate_hints.py** - AI hint generation:
- Batch hint + explanation generation using Claude
- One hint and one explanation per solved clue

**Solving Strategy** (achieves 100% on test puzzle):
1. Database lookup finds ~70% of clues from 7.5M historical pairs
2. Claude Opus fallback generates candidates for remaining clues
3. Bouncer filter scores all candidates by DB/word-index verification
4. Best-of-3 CSP solve with score-based value ordering
5. Multi-pass pattern refinement: extract crossing letters → regenerate via DB + Claude → re-solve
6. Hint generation runs in parallel after solve

### FastAPI Backend (`src/api/`)

**server.py** - API endpoints:
- `POST /api/upload` — Photo upload, grid detection, perspective warp
- `POST /api/{id}/grid-edit` — User grid corrections (toggle black cells)
- `POST /api/{id}/mask` — Apply masks/separators, Mistral OCR, verification
- `POST /api/{id}/solve` — Trigger background solve with SSE progress
- `GET /api/{id}/progress` — SSE stream for solve/hint progress
- `GET /api/puzzles` — List available puzzles
- `PATCH /api/puzzles/{id}` — Update puzzle metadata (e.g. name)

**pipeline.py** - Orchestration wrapping core functions:
- `run_grid_detection()` — Preprocess + detect grid
- `run_ocr_and_verify()` — Mask application + Mistral OCR + verification
- `run_solve_background()` — Background solve with progress callbacks
- `_run_solve()` — Multi-pass CSP solver with Claude candidate generation

**session_manager.py** — Session directory management under `data/sessions/{id}/`

**models.py** — Pydantic schemas for API requests/responses

### React Frontend (`web/`)

**Components:**
- `CrosswordPlayer.tsx` — Main player with react-crossword, auto-scroll via CrosswordContext, editable name, togglable correct counter, photo reference modal (original + masked tabs), Check Word, background solve banner with SSE
- `HintPanel.tsx` — Progressive hint reveal (hint → explanation → answer), Check Word button
- `PuzzleSelector.tsx` — Dynamic puzzle list from `/api/puzzles`
- `UploadPage.tsx` — Multi-step wizard (upload → preview → grid edit → mask → solve)
- `ImageMasker.tsx` — Canvas mask/separator tool with help overlay and example image
- `GridEditor.tsx` — Toggle black cells to fix grid detection errors
- `GridPreview.tsx` — Confirm detected grid before proceeding

**Hooks:** `usePuzzle`, `useHints`, `useSSE`, `useUploadPipeline`

**Running the app:**
```bash
# Terminal 1: Backend
.venv/bin/python3 -m src.api.server

# Terminal 2: Frontend
cd web && npm run dev
# Open http://localhost:5173
```

### Data Model

Puzzle JSON stored at `web/public/puzzles/{id}.json`:
```json
{
  "metadata": { "source_image", "grid_size", "verification", "name?" },
  "grid": { "rows", "cols", "cells": [[{row, col, is_block, clue_number}]] },
  "clues": {
    "across": [{ "number", "text", "start", "length", "answer?", "hint?", "explanation?" }],
    "down": [...]
  }
}
```

### Environment Variables

- `ANTHROPIC_API_KEY` — Required for Claude Opus candidate generation and hints
- `MISTRAL_API_KEY` — Required for Mistral OCR clue extraction
- `OPENAI_API_KEY` — Optional, used by legacy candidate generation functions
