"""OCR provider abstraction for pluggable clue extraction."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from crosswise.config import Settings


@runtime_checkable
class OCRProvider(Protocol):
    """Any OCR provider must implement this single method."""

    def extract_clues(self, image_path: Path) -> str:
        """Return markdown-formatted clue text from a crossword image.

        Expected output format:
            ## ACROSS

            1. Clue text
            7. Another clue
            ...

            ## DOWN

            1. Down clue
            ...
        """
        ...


def create_ocr_provider(config: Settings) -> OCRProvider:
    """Instantiate an OCR provider based on config."""
    provider = config.OCR_PROVIDER

    if provider == "mistral":
        from crosswise.ocr.mistral import MistralOCRProvider

        return MistralOCRProvider(
            api_key=config.MISTRAL_API_KEY,
            model=config.MISTRAL_OCR_MODEL,
        )
    elif provider == "gemini":
        from crosswise.ocr.gemini import GeminiOCRProvider

        return GeminiOCRProvider(
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_OCR_MODEL,
        )
    else:
        raise ValueError(
            f"Unknown OCR provider: {provider!r}. Choose 'gemini' or 'mistral'."
        )
