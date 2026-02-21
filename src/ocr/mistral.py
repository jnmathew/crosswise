"""Mistral OCR provider for crossword clue extraction."""

import base64
import json
import os
from pathlib import Path


class MistralOCRProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-ocr-latest",
    ):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self.model = model

    def extract_clues(self, image_path: Path) -> str:
        from typing import List as TypingList

        from mistralai import Mistral, ImageURLChunk
        from mistralai.extra import response_format_from_pydantic_model
        from pydantic import BaseModel as PydanticBaseModel

        class Clue(PydanticBaseModel):
            num: int
            clue: str

        class CrosswordClues(PydanticBaseModel):
            ACROSS: TypingList[Clue]
            DOWN: TypingList[Clue]

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        with Mistral(api_key=self.api_key) as mistral:
            res = mistral.ocr.process(
                model=self.model,
                document=ImageURLChunk(
                    image_url=f"data:image/jpeg;base64,{img_b64}"
                ),
                document_annotation_format=response_format_from_pydantic_model(
                    CrosswordClues
                ),
                include_image_base64=False,
            )

        if not hasattr(res, "document_annotation") or not res.document_annotation:
            raise ValueError("Mistral OCR returned no structured data")

        clues = (
            json.loads(res.document_annotation)
            if isinstance(res.document_annotation, str)
            else res.document_annotation
        )

        lines = ["## ACROSS\n"]
        for clue in (
            clues.get("ACROSS", []) if isinstance(clues, dict) else clues.ACROSS
        ):
            c = clue if isinstance(clue, dict) else {"num": clue.num, "clue": clue.clue}
            lines.append(f"{c['num']}. {c['clue']}")
        lines.append("\n## DOWN\n")
        for clue in (
            clues.get("DOWN", []) if isinstance(clues, dict) else clues.DOWN
        ):
            c = clue if isinstance(clue, dict) else {"num": clue.num, "clue": clue.clue}
            lines.append(f"{c['num']}. {c['clue']}")

        return "\n".join(lines)
