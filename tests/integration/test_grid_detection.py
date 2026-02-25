"""
Integration tests for grid detection on real crossword images.

These tests verify that the full preprocessing + grid detection pipeline
produces correct grid dimensions for known test images.
"""
import pytest
import numpy as np
from pathlib import Path

from src.core.config import Settings
from src.core.image_preprocessing import preprocess
from src.core.grid_detection import detect_grid


# Test image parameters: (image_path_relative, expected_rows, expected_cols)
TEST_IMAGES = [
    ("data/temp/IMG_5526/original.jpg", 14, 13),
    ("data/temp/IMG_5527/original.jpg", 21, 21),
]

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _get_image_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


@pytest.fixture
def config():
    return Settings()


@pytest.mark.parametrize(
    "image_rel_path, expected_rows, expected_cols",
    TEST_IMAGES,
    ids=["IMG_5526_14x13", "IMG_5527_21x21"],
)
class TestGridDimensions:
    """Test that grid detection produces correct dimensions."""

    def test_correct_dimensions(
        self, config, image_rel_path, expected_rows, expected_cols
    ):
        image_path = _get_image_path(image_rel_path)
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        prep = preprocess(image_path, config)
        result = detect_grid(prep["warped_gray"], config)

        num_rows, num_cols = result["grid_size"]
        assert num_rows == expected_rows, (
            f"Expected {expected_rows} rows, got {num_rows}"
        )
        assert num_cols == expected_cols, (
            f"Expected {expected_cols} cols, got {num_cols}"
        )

    def test_spacing_jitter(
        self, config, image_rel_path, expected_rows, expected_cols
    ):
        """Verify detected lines have regular spacing (jitter < 0.35)."""
        image_path = _get_image_path(image_rel_path)
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        prep = preprocess(image_path, config)
        result = detect_grid(prep["warped_gray"], config)

        xs = result["grid_lines"]["xs"]
        ys = result["grid_lines"]["ys"]

        max_jitter = 0.35

        if len(xs) > 2:
            x_spacings = np.diff(xs)
            x_median = np.median(x_spacings)
            x_jitter = float(np.std(x_spacings) / x_median) if x_median > 0 else 0
            assert x_jitter < max_jitter, (
                f"X-axis jitter {x_jitter:.3f} exceeds {max_jitter}"
            )

        if len(ys) > 2:
            y_spacings = np.diff(ys)
            y_median = np.median(y_spacings)
            y_jitter = float(np.std(y_spacings) / y_median) if y_median > 0 else 0
            assert y_jitter < max_jitter, (
                f"Y-axis jitter {y_jitter:.3f} exceeds {max_jitter}"
            )

    def test_grid_has_cells(
        self, config, image_rel_path, expected_rows, expected_cols
    ):
        """Verify the cell grid has the correct shape."""
        image_path = _get_image_path(image_rel_path)
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")

        prep = preprocess(image_path, config)
        result = detect_grid(prep["warped_gray"], config)

        cells = result["cells"]
        assert len(cells) == expected_rows
        assert all(len(row) == expected_cols for row in cells)
