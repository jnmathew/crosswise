"""Pytest configuration and fixtures."""
import pytest
from pathlib import Path
import cv2
import numpy as np


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


# Fixtures for black cell classification tests

@pytest.fixture
def uniform_white_cells():
    """All white cells (intensities 150-200)."""
    np.random.seed(42)  # For reproducibility
    return np.random.randint(150, 200, size=100).astype(np.float32)


@pytest.fixture
def uniform_black_cells():
    """All black cells (intensities 20-50)."""
    np.random.seed(42)
    return np.random.randint(20, 50, size=100).astype(np.float32)


@pytest.fixture
def bimodal_clean_separation():
    """Black cells (30-50) + white cells (120-180), clean gap."""
    np.random.seed(42)
    black = np.random.randint(30, 50, size=36).astype(np.float32)
    white = np.random.randint(120, 180, size=120).astype(np.float32)
    return np.concatenate([black, white])


@pytest.fixture
def bimodal_multiple_clusters():
    """Multiple black cell clusters (30-50, 95-115)."""
    np.random.seed(42)
    cluster1 = np.random.randint(30, 50, size=30).astype(np.float32)
    cluster2 = np.random.randint(95, 115, size=6).astype(np.float32)
    white = np.random.randint(120, 180, size=120).astype(np.float32)
    return np.concatenate([cluster1, cluster2, white])


@pytest.fixture
def borderline_intensities():
    """Cells near typical threshold (80-100)."""
    np.random.seed(42)
    return np.random.randint(80, 100, size=50).astype(np.float32)


@pytest.fixture
def mock_warped_grid():
    """Create synthetic warped grid image with known black cells (checkerboard pattern)."""
    # 5x5 grid, cells 50x50 pixels, some black (intensity 30), some white (160)
    grid = np.ones((250, 250), dtype=np.uint8) * 160  # Start white

    # Make some cells black (checkerboard pattern)
    for r in range(5):
        for c in range(5):
            if (r + c) % 2 == 0:  # Checkerboard: 13 black cells
                grid[r*50:(r+1)*50, c*50:(c+1)*50] = 30

    return grid
