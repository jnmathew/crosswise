"""Google Gemini vision provider for crossword clue extraction."""

import base64
import json
import os
import re
from pathlib import Path

import requests

EXTRACTION_PROMPT = """\
You are a precision OCR system for extracting crossword puzzle clues from newspaper photographs.

The image shows crossword clues arranged in newspaper columns.

Extract every clue exactly as printed. Rules:
1. Read columns LEFT to RIGHT, top to bottom within each column
2. ACROSS clues appear first, then DOWN clues (there will be a header or transition)
3. Each clue starts with a number followed by the clue text
4. Preserve exact punctuation, capitalization, and any italicized/quoted words
5. If a clue wraps across multiple lines, join it into one line
6. Do NOT guess, infer, or correct clue text — transcribe exactly what is printed
7. Do NOT skip any clues, even if partially obscured

Output strictly as JSON:
{"ACROSS": [{"num": 1, "clue": "Exact clue text"}, ...], "DOWN": [{"num": 1, "clue": "Exact clue text"}, ...]}"""


class GeminiOCRProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model

    def extract_clues(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # Detect mime type from extension
        suffix = image_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": mime_type, "data": img_b64}},
                {"text": EXTRACTION_PROMPT},
            ]}],
            "generationConfig": {"temperature": 0},
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()

        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        # Parse JSON from response (may be wrapped in markdown code fences)
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL
        )
        if json_match:
            raw_json = json_match.group(1)
        else:
            raw_json = raw_text.strip()

        clues = json.loads(raw_json)

        lines = ["## ACROSS\n"]
        for clue in clues.get("ACROSS", []):
            lines.append(f"{clue['num']}. {clue['clue']}")
        lines.append("\n## DOWN\n")
        for clue in clues.get("DOWN", []):
            lines.append(f"{clue['num']}. {clue['clue']}")

        return "\n".join(lines)
