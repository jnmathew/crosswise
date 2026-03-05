# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crosswise is a full-stack crossword puzzle app: upload a newspaper photo, automatically extract the grid and clues via OCR, solve with AI, and play interactively with hints. The pipeline uses OpenCV for grid detection, Gemini 3 Flash for clue extraction, Claude Opus for solving, and a React frontend for the interactive player.

## Environment

**Package manager**: [uv](https://docs.astral.sh/uv/) — use `uv run` to execute commands in the project venv:
```bash
uv run python -m crosswise.api.server
```

## Dependencies

Install Python dependencies:
```bash
uv sync              # core deps
uv sync --extra test # include test deps
```

Key dependencies:
- opencv-python, numpy (image processing, grid detection)
- pydantic, pydantic-settings (data models, config, .env loading)
- fastapi, uvicorn (API server)
- anthropic (Claude candidate generation, solving, hints)
- google-genai (Gemini 3 Flash OCR)
- loguru (logging)
- python-multipart (file upload handling)

## Architecture

### Grid Detection (`crosswise/vision/`)

**grid_detection.py** - Crossword grid extraction from newspaper images:
- Adaptive threshold selection with multiple fallback strategies (gap, percentile, Otsu)
- Quad detection and perspective transformation
- Black cell classification for grid structure analysis
- `assign_clue_numbers()` - Compute clue numbers from grid structure algorithmically
- `compute_clue_slots()` - Derive all clue slots with positions and answer lengths

**image_preprocessing.py** - General image preprocessing utilities:
- Four-point perspective transform for grid warping
- Contour analysis and quadrilateral extraction

**clue_extraction.py** - OCR parsing and puzzle verification:
- `parse_ocr_markdown()` - Parse OCR markdown output into structured clue data
- `verify_puzzle()` - Match OCR clues against grid slots, ensure 100% correspondence
- `match_clues_to_slots()` - Pair each OCR clue with its grid position and answer length
- `build_puzzle_clues()` - Create structured Clue objects for the Puzzle model

### OCR Integration (`crosswise/ocr/`)

Pluggable OCR provider system with `OCRProvider` protocol for future extensibility.

**base.py** — `OCRProvider` protocol and `create_ocr_provider()` factory.

**gemini.py** — Gemini 3 Flash provider:
- One-shots clue extraction from raw newspaper photos, no preprocessing needed
- Handles multi-column Sunday-size layouts (142 clues) without masking or separators
- Uses `google-genai` SDK with structured extraction prompt
- Requires `GEMINI_API_KEY` environment variable

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

## Development Notes

- **Git history was scrubbed** with `git filter-repo` to remove ~53MB of binary images (v1 debug PNGs, cell crops, example JPGs). `.git/` went from 49MB → 608K. A pre-filter backup exists locally outside the repo.
- Image processing uses grayscale conversion with careful handling of both color and grayscale inputs
- Fallback strategies implemented (adaptive → Otsu thresholding) when component detection fails
- Grid clue numbers are computed algorithmically (not OCR'd) — more reliable
- Puzzle verification requires 100% match between OCR clues and grid slots

### Crossword Solver (`crosswise/solver/`)

**models.py** - Solver data models and JSON builders:
- `SolverInput`, `SolveResult` — core solver types
- `build_solver_input_from_json()` — build solver input from puzzle JSON
- `build_clue_inputs_from_json()` — build candidate generation inputs from puzzle JSON

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

**candidates/** - Candidate generation package:
- `database.py` - `generate_candidates_with_database()`, `regenerate_with_patterns()` — SQLite lookup
- `claude.py` - `generate_with_claude()`, `ensure_minimum_candidates()`, `generate_with_extended_thinking()` — Claude Opus/Sonnet generation
- `scoring.py` - `bouncer_filter()`, `categorize_clue()`, `compute_target_domain_size()` — candidate scoring (0.3–1.0)
- `web_prepass.py` - `web_search_prepass()` — Haiku web search for pop culture clues in parallel
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

### FastAPI Backend (`crosswise/api/`)

**server.py** - API endpoints (15 total):
- `GET /api/config` — Frontend configuration
- `POST /api/upload` — Photo upload, grid detection, perspective warp
- `POST /api/{id}/grid-edit` — User grid corrections (toggle black cells)
- `POST /api/{id}/resize-grid` — Re-detect grid at new dimensions
- `POST /api/{id}/manual-crop` — Manual crop coordinates
- `POST /api/{id}/mask` — Apply masks/separators, OCR (via configured provider), verification
- `POST /api/{id}/start-pipeline` — Start full OCR+solve pipeline
- `POST /api/{id}/solve` — Trigger background solve with SSE progress
- `POST /api/{id}/cancel` — Cancel running solve
- `GET /api/{id}/progress` — SSE stream for solve/hint progress
- `GET /api/{id}/diagnostics` — Session diagnostic info
- `GET /api/{id}/status` — Session status (solving/complete/failed)
- `GET /api/puzzles` — List available puzzles
- `PATCH /api/puzzles/{id}` — Update puzzle metadata (e.g. name)
- `DELETE /api/puzzles/{id}` — Delete a puzzle

**pipeline.py** - Orchestration wrapping vision/solver functions:
- `run_grid_detection()` — Preprocess + detect grid
- `run_ocr_and_verify()` — Mask application + OCR (via configured provider) + verification
- `run_solve_background()` — Background solve with progress callbacks
- `_run_solve()` — Multi-pass CSP solver with Claude candidate generation

**session_manager.py** — Session directory management under `data/sessions/{id}/`

**models.py** — Pydantic schemas for API requests/responses

### React Frontend (`web/`)

**Components:**
- `CrosswordPlayer.tsx` — Main player with react-crossword, pencil mode (handwriting font), undo (Ctrl+Z), dark mode, keyboard shortcuts popover, editable name, togglable correct counter, photo reference modal, background solve banner with SSE
- `HintPanel.tsx` — Check/Reveal/Clear actions for Letter/Word/Puzzle, progressive hint reveal (hint → explanation → answer)
- `PuzzleSelector.tsx` — Dynamic puzzle list from `/api/puzzles`, skeleton loading, upload card tile, re-solve button for partially-solved puzzles
- `UploadPage.tsx` — Multi-step wizard (upload → preview → grid edit → mask → solve), clipboard paste support
- `ImageMasker.tsx` — Canvas mask/separator tool with help overlay and example image
- `GridEditor.tsx` — Toggle black cells to fix grid detection errors
- `GridPreview.tsx` — Confirm detected grid before proceeding

**Hooks:** `usePuzzle`, `useHints`, `useSSE`, `useTheme`, `useUploadPipeline`

**Styles:** `styles/theme.ts` — light and dark crossword grid themes for react-crossword

**Running the app:**
```bash
# Terminal 1: Backend
uv run python -m crosswise.api.server

# Terminal 2: Frontend
cd web && npm run dev
# Open http://localhost:5173

# Or both at once:
make run
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

- `GEMINI_API_KEY` — Required for Gemini 3 Flash OCR
- `ANTHROPIC_API_KEY` — Required for Claude candidate generation, solving, and hints
