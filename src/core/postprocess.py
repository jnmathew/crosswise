"""
Postprocessing utilities for puzzle data.

Cleans and normalizes OCR output.
"""
from loguru import logger

from .config import Settings
from .models import Puzzle, Direction


def apply(puzzle: Puzzle, config: Settings) -> Puzzle:
    """
    Apply postprocessing to puzzle data.

    Current operations:
    - Normalize whitespace in clue text
    - Strip extra spaces

    Future enhancements:
    - Fix common OCR errors (O→0, l→1 in enumerations)
    - Merge hyphenated clue text across lines
    - LLM-based spell-checking and cleanup

    Args:
        puzzle: Puzzle object to process
        config: Settings object

    Returns:
        Updated Puzzle object (modified in place)
    """
    logger.debug("Applying postprocessing")

    # Normalize clue text whitespace
    for direction in [Direction.ACROSS, Direction.DOWN]:
        for clue in puzzle.clues[direction]:
            # Strip and normalize whitespace
            clue.text = " ".join(clue.text.split())

    logger.debug("Postprocessing complete")
    return puzzle
