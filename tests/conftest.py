"""Pytest configuration and fixtures."""
import pytest
from pathlib import Path
import cv2
import numpy as np
from src.core.models import Direction


@pytest.fixture
def sample_image_path():
    """Path to sample crossword image."""
    return Path(__file__).parent.parent / "data" / "examples" / "IMG_5526.JPG"


@pytest.fixture
def sample_image(sample_image_path):
    """Load sample image."""
    if not sample_image_path.exists():
        pytest.skip(f"Sample image not found at {sample_image_path}")

    img = cv2.imread(str(sample_image_path))
    assert img is not None, f"Failed to load {sample_image_path}"
    return img


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def sample_puzzle_data():
    """Sample puzzle data for testing."""
    return {
        "grid": [
            [
                {"row": 0, "col": 0, "is_block": False, "clue_number": 1},
                {"row": 0, "col": 1, "is_block": False},
                {"row": 0, "col": 2, "is_block": False},
                {"row": 0, "col": 3, "is_block": False},
                {"row": 0, "col": 4, "is_block": False},
            ]
        ],
        "clues": {
            "across": [
                {
                    "number": 1,
                    "direction": "across",
                    "text": "Capital of France",
                    "answer_length": [5],
                }
            ],
            "down": [],
        },
        "metadata": {
            "source_image": "test.jpg",
            "processed_at": "2025-01-01T00:00:00Z",
            "grid_size": [1, 5],
            "total_clues": 1,
        },
    }
