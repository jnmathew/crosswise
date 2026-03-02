# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crosswise is a full-stack crossword puzzle app: upload a newspaper photo, automatically extract the grid and clues via OCR, solve with AI, and play interactively with hints. The pipeline uses OpenCV for grid detection, Gemini 3 Flash for clue extraction (with Mistral as fallback), Claude Opus for solving, and a React frontend for the interactive player.

## Environment

**Virtual Environment**: Always use the project venv:
```bash
.venv/bin/python3 -m crosswise.solver.solve_puzzle ...
```

## Dependencies

Install Python dependencies:
```bash
.venv/bin/pip install -r requirements.txt
```

Key dependencies:
- opencv-python, numpy (image processing, grid detection)
- pydantic, pydantic-settings (data models, config)
- fastapi, uvicorn (API server)
- anthropic (Claude candidate generation + hints)
- google-genai (Gemini 3 Flash OCR — default provider)
- mistralai (Mistral OCR — fallback provider)
- openai (legacy/fallback candidate generation)
- loguru (logging)

## Architecture

### Grid Detection (`crosswise/core/`)

**grid_detection.py** - Crossword grid extraction from newspaper images:
- Adaptive threshold selection with multiple fallback strategies (gap, percentile, Otsu)
- Quad detection and perspective transformation
- Black cell classification for grid structure analysis
- `assign_clue_numbers()` - Compute clue numbers from grid structure algorithmically
- `compute_clue_slots()` - Derive all clue slots with positions and answer lengths

**image_preprocessing.py** - General image preprocessing utilities:
- Four-point perspective transform for grid warping
- Contour analysis and quadrilateral extraction

**clue_column_detector.py** - Multi-column layout detection for clue extraction:
- Hybrid column detection combining vertical projection and text clustering
- Handles both full-height and partial-height columns
- Automatic separator line placement between detected columns
- Optional non-text region masking

**clue_extraction.py** - OCR parsing and puzzle verification:
- `parse_ocr_markdown()` - Parse OCR markdown output into structured clue data
- `verify_puzzle()` - Match OCR clues against grid slots, ensure 100% correspondence
- `match_clues_to_slots()` - Pair each OCR clue with its grid position and answer length
- `build_puzzle_clues()` - Create structured Clue objects for the Puzzle model

### OCR Integration (`crosswise/ocr/`)

Pluggable OCR provider system — switch with `OCR_PROVIDER=gemini|mistral` in `.env`.

**base.py** — `OCRProvider` protocol and `create_ocr_provider()` factory.

**gemini.py** — Gemini 3 Flash provider (default):
- One-shots clue extraction from raw newspaper photos, no preprocessing needed
- Handles multi-column Sunday-size layouts (142 clues) without masking or separators
- Uses `google-genai` SDK with structured extraction prompt
- Requires `GEMINI_API_KEY` environment variable

**mistral.py** — Mistral OCR provider (fallback):
- Structured output using Pydantic models
- Works best with preprocessed images (masking + separator lines)
- Requires `MISTRAL_API_KEY` environment variable

## Complete Workflow

For extracting crossword clues from newspaper images:

1. **Grid Detection**: Use `grid_detection.py` to locate and extract the crossword grid
2. **OCR Extraction**: Gemini 3 Flash extracts clues directly from raw photos — no preprocessing needed
3. **Puzzle Verification**: Use `verify_puzzle()` to match OCR clues against grid slots
   - Every OCR clue must match a grid slot
   - Every grid slot must have an OCR clue
   - Verification must pass 100% before proceeding
4. **Build Puzzle**: Use `build_puzzle_clues()` to create structured Clue objects with answer lengths
5. **Output**: Verified puzzle saved as JSON with grid structure and clue data

### Fallback: Manual Masking (for Mistral provider)

If Gemini struggles with a particular image, switch to Mistral (`OCR_PROVIDER=mistral`) and use the preprocessing tools:
- Run `interactive_masker.py` to draw white rectangles over grids/ads and tilted separator lines between columns
- The tilted separator approach (matching actual column angles) achieved 100% Mistral OCR accuracy on test images

## Development Notes

- **Git history was scrubbed** with `git filter-repo` to remove ~53MB of binary images (v1 debug PNGs, cell crops, example JPGs). `.git/` went from 49MB → 608K. A pre-filter backup exists locally outside the repo.
- Image processing uses grayscale conversion with careful handling of both color and grayscale inputs
- Fallback strategies implemented (adaptive → Otsu thresholding) when component detection fails
- Aqua/cyan (BGR: 255, 255, 0) used for separator lines — visible over white masks and gray newspaper
- Interactive masker draws separators at 8px width for clear OCR visibility
- Grid clue numbers are computed algorithmically (not OCR'd) — more reliable
- Puzzle verification requires 100% match between OCR clues and grid slots

### Crossword Solver (`crosswise/solver/`)

**solve_puzzle.py** - Main solver script:
```bash
# With TSV database (recommended)
.venv/bin/python3 -m crosswise.solver.solve_puzzle data/output/puzzle.json --use-database

# Database only (no LLM fallback)
.venv/bin/python3 -m crosswise.solver.solve_puzzle data/output/puzzle.json --database-only

# LLM only (original behavior)
.venv/bin/python3 -m crosswise.solver.solve_puzzle data/output/puzzle.json
```

**clue_database.py** - SQLite-backed clue database:
- Two data sources: xd TSV (7.5M pairs) + CrosswordQA from HuggingFace (6.8M pairs)
- CrosswordQA deduplicated against xd on (answer, clue_normalized) — ~9-11M total after dedup
- Download all data sources: `bash scripts/setup_data.sh` (or `make setup`)
- Converts to SQLite on first use (`data/clues.db`); delete DB to force rebuild from sources
- Simplified schema: `(id, answer, clue_normalized, length)` — no pubid/year/raw clue
- Provides fast lookup by clue text, pattern matching, and answer length
- Pattern matching uses GLOB (e.g., `C_T` matches `CAT`, `COT`, `CUT`)

**word_index.py** - Unified crossword word index:
- Loads multiple word lists (Broda, Crossword Nexus, Spread the Wordlist)
- Fast `contains()` membership testing and `match_pattern()` for pattern matching
- Quality scores for value ordering in the solver

**candidates/** - Candidate generation package (split from candidate_generator.py):
- `database.py` - `generate_candidates_with_database()`, `regenerate_with_patterns()` — SQLite lookup
- `claude.py` - `generate_with_claude()`, `ensure_minimum_candidates()`, `generate_with_extended_thinking()` — Claude Opus/Sonnet generation
- `scoring.py` - `bouncer_filter()`, `categorize_clue()`, `compute_target_domain_size()` — candidate scoring (0.3–1.0)
- `web_prepass.py` - `web_search_prepass()` — Haiku web search for pop culture clues in parallel
- `escalation_legacy.py` - `sniper_escalation()` — multi-level fallback (legacy CLI only)
- `openai_legacy.py` - `generate_candidates_batch()`, `generate_candidates()` — OpenAI functions (legacy CLI only)
- `models.py` - `ClueInput`, `ScoredCandidate`, `_matches_pattern()` — shared data types
- `prompts.py` - `_build_prompt()`, `_parse_response()` — shared LLM prompt logic

**csp.py** - Constraint satisfaction solver:
- MAC (Maintaining Arc Consistency) with `mac_mode="search-only"` (skip AC-3 preprocessing)
- MRV heuristic for variable selection, score-based value ordering
- Conflict-Directed Backjumping (CDBJ)
- 50 random starting points, best-of-N runs to combat nondeterminism

**llm_solver.py** - LLM-based iterative solver (primary solver):
- Multi-pass architecture: commit high-confidence answers first, propagate crossing letters
- `solve_pass()` — single-turn Opus call (no web search, no multi-turn continuation)
- `find_conflict_clusters()` — detects dead-end patterns (crossing letters match no valid word), traces blame to wrong committed answers, groups into connected clusters
- `resolve_conflict_cluster()` — removes blamed answers from grid, asks LLM to re-solve the cluster jointly with web search available (Anthropic `web_search_20250305`)
- `propagate_constraints()` — zero-cost logic: auto-commits clues where crossing patterns eliminate all but one candidate; handles fully-constrained patterns via word index + dictionary API + Haiku verification
- Post-resolution follow-up pass picks up newly-unblocked clues after conflict resolution frees crossing letters

**cost_tracker.py** - API cost tracking:
- Thread-safe `CostTracker` accumulates costs across all API calls during a solve
- Tracks input/output tokens, cache write (1.25x) and read (0.1x) tokens, web search ($0.01/query)
- Pricing: Opus 4 ($15/$75), Sonnet 4 ($3/$15), Haiku 4.5 ($1/$5) per MTok
- Per-call logging and phase-grouped summary at end of solve

**generate_hints.py** - AI hint generation:
- Batch hint + explanation generation using Claude
- One hint and one explanation per solved clue

**Solving Strategy** (LLM solver — default):
1. Database lookup finds ~70% of clues from ~9-11M historical pairs (xd + CrosswordQA)
2. Haiku web search pre-pass for pop culture clues (~$0.01-0.03/clue)
3. Claude Opus fallback generates candidates for remaining clues
4. Bouncer filter scores all candidates by DB/word-index verification (+0.1 web confirmation bonus)
5. LLM iterative solve: 6 single-turn Opus passes of commit → propagate crossing letters → re-solve
6. Constraint propagation: auto-commit clues where crossing patterns leave one candidate (zero API cost)
7. Conflict resolution: detect dead-end patterns → trace blamed crossings → LLM re-solves clusters (with web search)
8. Post-resolution constraint propagation + follow-up pass
9. Fully-constrained pattern handling: dictionary API + Haiku verification for words not in candidate list
10. CSP cleanup for any remaining unsolved clues
11. Hint generation runs in parallel after solve
- **Cost**: ~$1.20/puzzle (down from $5.85), 69/69 solve rate

**Solving Strategy** (CSP solver — legacy):
1. Database lookup finds ~70% of clues
2. Claude Opus fallback generates candidates for remaining clues
3. Bouncer filter scores all candidates by DB/word-index verification
4. Best-of-3 CSP solve with score-based value ordering
5. Multi-pass pattern refinement: extract crossing letters → regenerate via DB + Claude → re-solve
6. Hint generation runs in parallel after solve

### FastAPI Backend (`crosswise/api/`)

**server.py** - API endpoints:
- `POST /api/upload` — Photo upload, grid detection, perspective warp
- `POST /api/{id}/grid-edit` — User grid corrections (toggle black cells)
- `POST /api/{id}/mask` — Apply masks/separators, OCR (via configured provider), verification
- `POST /api/{id}/solve` — Trigger background solve with SSE progress
- `GET /api/{id}/progress` — SSE stream for solve/hint progress
- `GET /api/puzzles` — List available puzzles
- `PATCH /api/puzzles/{id}` — Update puzzle metadata (e.g. name)

**pipeline.py** - Orchestration wrapping core functions:
- `run_grid_detection()` — Preprocess + detect grid
- `run_ocr_and_verify()` — Mask application + OCR (via configured provider) + verification
- `run_solve_background()` — Background solve with progress callbacks
- `_run_solve()` — Multi-pass CSP solver with Claude candidate generation

**session_manager.py** — Session directory management under `data/sessions/{id}/`

**models.py** — Pydantic schemas for API requests/responses

### React Frontend (`web/`)

**Components:**
- `CrosswordPlayer.tsx` — Main player with react-crossword, auto-scroll via CrosswordContext, editable name, togglable correct counter, photo reference modal (original + masked tabs), Check Word, background solve banner with SSE, checks session status API for re-solve SSE connection
- `HintPanel.tsx` — Progressive hint reveal (hint → explanation → answer), Check Word button
- `PuzzleSelector.tsx` — Dynamic puzzle list from `/api/puzzles`, re-solve button for partially-solved puzzles
- `UploadPage.tsx` — Multi-step wizard (upload → preview → grid edit → mask → solve)
- `ImageMasker.tsx` — Canvas mask/separator tool with help overlay and example image
- `GridEditor.tsx` — Toggle black cells to fix grid detection errors
- `GridPreview.tsx` — Confirm detected grid before proceeding

**Hooks:** `usePuzzle`, `useHints`, `useSSE`, `useUploadPipeline`

**Running the app:**
```bash
# Terminal 1: Backend
.venv/bin/python3 -m crosswise.api.server

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

- `GEMINI_API_KEY` — Required for Gemini 3 Flash OCR (default provider)
- `ANTHROPIC_API_KEY` — Required for Claude Opus candidate generation and hints
- `MISTRAL_API_KEY` — Required if using Mistral OCR provider
- `OPENAI_API_KEY` — Optional, used by legacy candidate generation functions
- `OCR_PROVIDER` — `"gemini"` (default) or `"mistral"`
