"""Google Gemini vision provider for crossword clue extraction."""

import json
import os
import re
from pathlib import Path

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
        from google import genai

        client = genai.Client(api_key=self.api_key)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Detect mime type from extension
        suffix = image_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        response = client.models.generate_content(
            model=self.model,
            contents=[
                genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                EXTRACTION_PROMPT,
            ],
            config=genai.types.GenerateContentConfig(temperature=0),
        )

        raw_text = response.text

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
