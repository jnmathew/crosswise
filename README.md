# Crosswise — Crossword OCR & Reasoning

A pipeline that turns a photo of a printed crossword into a structured, playable digital puzzle and (eventually) provides AI hints and explanations.

- Primary docs: `vision.md`, `preprocess.md`
- Code entry point: `src/main.py`

## Quick start (macOS)

```bash
# 1) System dependency
brew install tesseract

# 2) Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) Environment (OpenAI, optional for Vision OCR)
cp .env.example .env
# edit .env and set OPENAI_API_KEY (and OPENAI_MODEL if desired)

# 4) Run (placeholder)
python -m src.main --input data/examples/sample.jpg --out data/output_json/sample.json
```

## Project structure

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
│   ├── main.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── image_preprocessing.py
│   │   ├── grid_detection.py
│   │   ├── ocr_utils.py
│   │   ├── clue_extraction.py
│   │   ├── postprocess.py
│   │   ├── puzzle_model.py
│   │   ├── exporter.py
│   │   └── validator.py
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── reasoning.py
│   │   ├── explanation.py
│   │   ├── embeddings.py
│   │   └── trainer.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_io.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── viz.py
│
├── data/
│   ├── examples/
│   ├── output_json/
│   ├── logs/
│   ├── temp/
│   └── dictionaries/
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

## Notes
- Install Tesseract system binary via Homebrew.
- Python deps include `opencv-python`, `pytesseract`, and `openai` (plus `python-dotenv` to load `.env`).
- See `preprocess.md` for pipeline stages and `vision.md` for the high-level plan.
