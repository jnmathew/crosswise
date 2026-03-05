# Crosswise

<p align="center">
  <img src="assets/logo/crosswise_v1_clean.svg" alt="Crosswise" width="300">
</p>

<!-- TODO: CI badge — uncomment when repo is public -->
<!-- ![CI](https://github.com/jnmathew/crosswise/actions/workflows/ci.yml/badge.svg) -->

Crosswise is a high-precision crossword digitizer and autonomous solver. It transforms raw newspaper photographs into accurate, solved, and playable digital puzzles using a multi-stage AI orchestration pipeline.

<!-- TODO: demo GIF — record the upload-to-solve flow -->

## How it works

1. **Upload** a photo of a printed crossword
2. **Grid detection** — OpenCV finds the grid, corrects perspective, and computes clue numbers algorithmically from the grid geometry (no need to OCR tiny numbers inside grid)
3. **OCR** — Gemini 3 Flash extracts clues directly from the raw photo, no preprocessing needed
4. **Verification** — every OCR clue must match a grid slot and vice versa (100% correspondence required before solving)
5. **Solve** — Multi-stage AI solver: database lookup (10.1M clue pairs), web search for pop culture, Claude Opus iterative solving with constraint propagation, and conflict backtracking.
6. **Play** — interactive player with Check/Reveal functionality, timer, and hints + explanations to facilitate learning.

## Key numbers

- **XX% solve rate** on XX tested puzzles (mix of newspapers, 14x13 to 21x21)
- **10.1M** historical clue/answer pairs in SQLite from 2 data sources + **605K** unique words across 2 curated crossword word lists for pattern matching and validation (see [DATASETS.md](docs/DATASETS.md))
- **97 tests** (91 unit, 6 integration) with GitHub Actions CI on every push
- **~$XX cost in API calls per puzzle** (based on current Anthropic + Gemini pricing, typical grid size)

## Demo

A pre-solved sample puzzle is included so you can try the interactive player without API keys or the clue database:

```bash
make install
make run-demo
# Open http://localhost:5173
```

## Getting started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js 20+
- API keys: [Anthropic](https://console.anthropic.com/) (required), [Google Gemini](https://aistudio.google.com/apikey) (required)

### Setup

```bash
make install

# Set ANTHROPIC_API_KEY and GEMINI_API_KEY
cp .env.example .env

# Download clue database (~1.5GB peak, ~1GB after cleanup)
make setup
```

### Run

```bash
make run
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
# API docs: http://localhost:8000/docs
```

## Solve pipeline

The solver uses a tiered strategy that minimizes API cost while maintaining accuracy:

1. **Database lookup** — instant SQLite query finds ~70% of answers from 9-11M historical pairs
2. **Web pre-pass** — Haiku web search identifies pop culture, celebrity, and current-event clues (~$0.01/clue)
3. **Candidate generation** — parallel Claude Opus + Sonnet calls generate candidates for remaining clues
4. **Bouncer scoring** — cross-references all candidates against the database and word index (0.3-1.0 confidence)
5. **LLM iterative solving** — 6 Opus passes: commit high-confidence answers, propagate crossing letters, re-solve
6. **Constraint propagation** — auto-commits clues where crossing patterns eliminate all but one candidate (zero API cost)
7. **Conflict resolution** — detects dead-end patterns, traces blame to wrong committed answers, re-solves clusters with web search
8. **CSP cleanup** — constraint satisfaction solver (MAC + MRV + backjumping) handles any remaining clues
9. **Hint generation** — parallel Claude calls produce one hint and one explanation per solved clue

## Architecture

```
crosswise/
  vision/           OpenCV grid detection, image preprocessing, clue extraction
  ocr/              Pluggable OCR provider (Gemini 3 Flash)
  solver/           LLM iterative solver, CSP, cost tracking, hint generation
  solver/candidates/ Database lookup, Claude generation, web pre-pass, scoring
  api/              FastAPI backend, SSE progress streaming, session management
web/                React + TypeScript + Vite interactive player
```

**Where to start reading**: `crosswise/api/server.py` (14 endpoints) delegates to `crosswise/api/pipeline.py` (orchestration), which calls into `crosswise/vision/` (image processing) and `crosswise/solver/` (solving).

## Tech stack

**Backend**: Python 3.11, FastAPI, OpenCV, Gemini 3 Flash, Claude Opus/Sonnet/Haiku, SQLite
**Frontend**: React 18, TypeScript, Vite, react-crossword, Server-Sent Events

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | Yes | Gemini 3 Flash OCR |
| `ANTHROPIC_API_KEY` | Yes | Claude candidate generation, solving, and hints |

## Data sources

The clue database combines ~9-11M deduplicated pairs from 4 sources (xd archive, CrosswordQA, Crossword Nexus, Peter Broda). See [DATASETS.md](docs/DATASETS.md) for licensing, URLs, and details. Run `make setup` to download everything.
