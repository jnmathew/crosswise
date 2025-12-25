"""
Export utilities for puzzle data.

Saves puzzle to JSON format.
"""
from pathlib import Path
import json

from loguru import logger

from .config import Settings
from .models import Puzzle


def save_json(puzzle: Puzzle, output_path: Path, config: Settings) -> None:
    """
    Export Puzzle to canonical JSON format.

    Uses the Puzzle.to_json() method for serialization, ensuring all
    data is properly formatted and validated by Pydantic.

    Args:
        puzzle: Puzzle object to export
        output_path: Path to save JSON file
        config: Settings object (for future configuration options)

    Raises:
        IOError: If file cannot be written
    """
    logger.info(f"Exporting puzzle to {output_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize puzzle using Pydantic
    json_data = puzzle.to_json()

    # Write to file with pretty printing
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.success(f"Saved puzzle JSON ({puzzle.rows}x{puzzle.cols} grid)")
