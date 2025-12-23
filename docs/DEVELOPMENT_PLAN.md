# Crosswise — Development Plan and Scaffold

Last Updated: 2025-11-02
Owner: Joseph Mathew
Related docs: `vision.md`, `preprocess.md`, `CROSSWORD_INGEST_PIPELINE.md`, `PROJECT_CONTEXT_AI_AGENT.md`

## Goal
Ship an MVP that converts a single photo of a printed crossword into a canonical JSON puzzle suitable for a reasoning layer. Keep modules small, testable, and swappable.

## Scope (MVP)
- Single image input containing grid + clues in one frame
- Grid detection, cell extraction, block detection
- OCR for cell numbers and clues
- Clue parsing and enumeration extraction
- Canonical JSON export + basic validation
- Save intermediate artifacts for debugging

Out of scope (MVP): on-device models, advanced UI, multi-photo workflows, heavy ML for layout.

## Proposed project structure (target)
```
crosswise/
│
├── README.md
├── PROJECT_CONTEXT_AI_AGENT.md
├── CROSSWORD_INGEST_PIPELINE.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point for the full pipeline
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── image_preprocessing.py  # Grayscale, threshold, dewarp
│   │   ├── grid_detection.py       # Find grid, slice cells, detect blocks
│   │   ├── ocr_utils.py            # Tesseract wrapper, config presets
│   │   ├── clue_extraction.py      # Detect and OCR clue text
│   │   ├── postprocess.py          # Clean-up, merge lines, spellcheck
│   │   ├── puzzle_model.py         # Data structure for Puzzle object
│   │   ├── exporter.py             # Convert to canonical JSON schema
│   │   └── validator.py            # Sanity checks, confidence validation
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── reasoning.py            # Clue→answer logic (ChatGPT / fine-tuned model)
│   │   ├── explanation.py          # “Mentor” explanations for each clue
│   │   ├── embeddings.py           # (future) local embedding models for clues
│   │   └── trainer.py              # (future) dataset fine-tuning code
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_io.py              # Image I/O, save/load helpers
│   │   ├── config.py               # Global constants, PSM modes, paths
│   │   ├── logger.py               # Logging and debug image snapshots
│   │   └── viz.py                  # Debug visual overlays (grid lines, boxes)
│
├── data/
│   ├── examples/                   # Sample input images
│   ├── output_json/                # Parsed canonical puzzle JSONs
│   ├── logs/                       # OCR debug outputs
│   ├── temp/                       # Intermediate cache (auto-cleanable)
│   └── dictionaries/               # Custom wordlists, abbreviations
│
├── tests/
│   ├── test_grid_detection.py
│   ├── test_ocr_accuracy.py
│   ├── test_clue_extraction.py
│   └── test_schema_validation.py
│
└── docs/
    ├── schema/
    │   └── crossword_puzzle.schema.json
    ├── design_diagrams/
    │   ├── data_flow.png
    │   └── pipeline_layers.png
    └── LICENSES/
        ├── OpenCV_LICENSE.txt
        └── Tesseract_LICENSE.txt
```

## Module contracts (tiny “what/returns”)
- image_preprocessing.preprocess(input_path) → {image, binary, meta}
  - Errors: unreadable image; too blurry/skewed (flag in meta)
- grid_detection.detect_grid(preprocessed) → {grid_bbox, cells[], black_mask, meta}
  - cell: {row, col, crop, is_block?}
- ocr_utils.ocr_cells(cells) → updates cell.clue_number if present
- clue_extraction.extract_regions(image, grid_bbox) → [regions]
- clue_extraction.ocr_and_parse(regions) → {across[], down[], had_headers?}
  - clue: {num, text, enum?, raw_conf?}
- postprocess.apply(puzzle) → puzzle (normalized text, merged wraps, enforced enums)
- exporter.to_json(puzzle) → dict (canonical)
- validator.validate(puzzle|dict) → [errors]

## Canonical puzzle JSON (example)
```json
{
  "grid_size": 15,
  "cells": [
    {"row": 0, "col": 0, "is_block": false, "clue_number": 1}
  ],
  "clues": {
    "across": [
      {"num": 1, "text": "Capital of France", "enum": [5]}
    ],
    "down": []
  },
  "metadata": {
    "source_image": "data/examples/img_001.jpg",
    "processed_at": "<iso8601>",
    "debug_artifacts": ["warp.jpg", "grid_mask.png"]
  }
}
```

## Milestones
1) Skeleton + environment
- README, vision/preprocess docs, requirements
- Tesseract via Homebrew, venv instructions
- Directory scaffold or plan-only (this doc)

2) Grid MVP
- Preprocess → grid detection → cells + black mask
- Save debug overlays

3) OCR MVP
- Cell numbers (`--psm 11`), clue regions, OCR clues (`--psm 6`)
- Basic clue parsing (numbered lines, enumerations)

4) Model assembly + export
- Build Puzzle, export canonical JSON, validator checks
- 10–20 sample images processed end-to-end

5) Reliability pass
- Simple unit tests for each stage
- Integration test over sample set

## Acceptance criteria (MVP)
- Given 10 varied crossword photos, at least 7 produce valid JSON with:
  - Correct grid size and >90% correct block mask by visual inspection
  - ≥70% clue lines parsed with correct numbering
  - JSON passes schema validation; manual fixups allowed for low-confidence lines
- All runs produce debug artifacts for review

## Risks & mitigations
- OCR brittleness → denoise, CLAHE, alternative PSMs, user crop fallback
- Multi-column clues → heuristic block scoring + manual refine-box overlay
- Low-res photos → prompt retake if blur/skew exceeds threshold

## Setup (macOS quick start)
- System: `brew install tesseract`
- Python: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Next steps (when ready to code)
- Option A: create folder skeleton only (no code)
- Option B: add minimal stubs for each module (no OpenCV import at import-time)
- Option C: tests-first: write schema + minimal tests, then fill stubs

Preference? Reply with A, B, or C and I’ll proceed.
