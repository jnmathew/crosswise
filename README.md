# Crosswise

Upload a newspaper crossword photo, extract the grid and clues via OCR, solve with AI, and play interactively with progressive hints.

## How it works

1. **Upload** a photo of a printed crossword
2. **Grid detection** — OpenCV finds the grid, corrects perspective, classifies black/white cells, and computes clue numbers algorithmically
3. **Masking** — User draws masks over ads/irrelevant content and tilted separator lines between clue columns
4. **OCR** — Mistral OCR extracts structured clues from the masked image
5. **Verification** — Every OCR clue must match a grid slot and vice versa (100% correspondence required)
6. **Solve** — Database lookup (7.5M historical clue/answer pairs) + Claude Opus fallback + constraint satisfaction solver
7. **Play** — Interactive crossword player with progressive hints, explanations, and answer reveal

## Quick start

```bash
# Python backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Frontend
cd web && npm install && cd ..

# Environment
cp .env.example .env
# Set ANTHROPIC_API_KEY and MISTRAL_API_KEY

# Run (two terminals)
.venv/bin/python3 -m src.api.server    # http://localhost:8000
cd web && npm run dev                   # http://localhost:5173
```

## Project structure

```
src/
  core/               Grid detection, image preprocessing, clue extraction
  api/                FastAPI server, SSE progress, session management
  solver/             CSP solver, candidate generation, clue database, hints
  tools/              Interactive masker, post-masking pipeline
  examples/           Reference scripts (Mistral OCR demos, column detection)
web/
  src/components/     CrosswordPlayer, UploadPage, GridEditor, ImageMasker, HintPanel
  src/hooks/          usePuzzle, useHints, useSSE, useUploadPipeline
tests/
docs/
```

## Tech stack

**Backend:** FastAPI, OpenCV, Mistral OCR, Claude Opus, SQLite (clue database)
**Frontend:** React, TypeScript, Vite, react-crossword, Server-Sent Events
**Solver:** Database lookup (7.5M pairs) + LLM candidate generation + MAC arc consistency + conflict-directed backjumping, best-of-N with multi-pass pattern refinement

## Solver approach

1. **Database lookup** finds ~70% of answers from 7.5M historical crossword clue/answer pairs
2. **Claude Opus** generates candidates for remaining clues
3. **Bouncer filter** scores candidates by database/word-index verification
4. **CSP solver** uses MAC with MRV heuristic and score-based value ordering
5. **Multi-pass refinement** extracts crossing letters from partial solutions, regenerates candidates via database + Claude, and re-solves
6. **Hint generation** produces one hint and one explanation per clue via Claude
