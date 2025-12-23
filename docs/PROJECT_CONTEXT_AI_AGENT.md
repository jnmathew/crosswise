# Project Context for Automated Agent

Last Updated: 2025-11-02

This repository implements a pipeline to parse a crossword photo into a canonical JSON, with future AI reasoning and explanation layers.

Key docs:
- `vision.md` — product vision and roadmap
- `preprocess.md` — detailed ingest pipeline
- `CROSSWORD_INGEST_PIPELINE.md` — concise pipeline overview and entry points

Key modules:
- `src/pipeline/*` — image processing, OCR, parsing, exporting, validation
- `src/ai/*` — clue reasoning, explanations, embeddings, training (future)
- `src/utils/*` — config, logging, I/O helpers, visualization

Environment:
- macOS, Python 3.10+
- Homebrew Tesseract is required: `brew install tesseract`
- Python deps in `requirements.txt`

Operational conventions:
- Save intermediate debug artifacts under `data/logs/` and `data/temp/`
- Canonical puzzle JSONs under `data/output_json/`
- Unit tests with `pytest` in `tests/`
