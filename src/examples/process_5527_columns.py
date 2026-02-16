"""
Test script for automatic column detection on IMG_5527.

This script demonstrates the clue column detection pipeline:
1. Load the crossword clues image
2. Detect column boundaries automatically
3. Draw vertical separator lines
4. Save the preprocessed image
5. Run OCR with Mistral to extract clues
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from loguru import logger

from src.core.clue_column_detector import preprocess_clues_for_ocr, detect_column_boundaries


def main():
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    # Load image
    image_path = Path(__file__).parent / "IMG_5527 copy.JPG"
    logger.info(f"Loading image: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    h, w = image.shape[:2]
    logger.info(f"Image size: {w}x{h}")

    # Estimate minimum column width based on image size
    # For newspaper-style crossword clues, columns are typically 1/6 to 1/4 of page width
    min_column_width = w // 8  # Conservative estimate
    logger.info(f"Using min_column_width: {min_column_width}")

    # Preprocess with column detection
    logger.info("Running column detection and preprocessing...")
    preprocessed = preprocess_clues_for_ocr(
        image,
        min_column_width=min_column_width,
        max_columns=12,  # Newspaper pages can have many columns
        add_separators=True,
        mask_non_text=False,  # Try without masking first
        separator_color=(180, 180, 180),  # Gray separators
        separator_width=3
    )

    # Save result
    output_path = Path(__file__).parent / "IMG_5527_masked_divided.JPG"
    cv2.imwrite(str(output_path), preprocessed)
    logger.success(f"Saved preprocessed image to: {output_path}")

    # Also save a visualization with column boundaries highlighted
    vis = image.copy()
    boundaries = detect_column_boundaries(image, min_column_width=min_column_width, max_columns=12)

    # Draw boundaries in bright color for visualization
    for x in boundaries:
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 2)

    vis_path = Path(__file__).parent / "IMG_5527_column_boundaries_vis.JPG"
    cv2.imwrite(str(vis_path), vis)
    logger.success(f"Saved boundary visualization to: {vis_path}")

    logger.info(f"Detected {len(boundaries) + 1} columns at boundaries: {boundaries}")


if __name__ == "__main__":
    main()
