"""
Create test masks for IMG_5527 to demonstrate the pipeline.

This creates predefined white rectangles over:
- Left crossword grid
- Right crossword grid
- Acrostic puzzle area
"""
import json
from pathlib import Path
import cv2
import numpy as np
from loguru import logger


def create_masks_for_img5527():
    """Create test masks for IMG_5527."""

    # Load image to get dimensions
    image_path = Path(__file__).parent / "IMG_5527 copy.JPG"
    image = cv2.imread(str(image_path))

    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return False

    h, w = image.shape[:2]
    logger.info(f"Image size: {w}x{h}")

    # Define mask rectangles (x1, y1, x2, y2)
    # These are more precise - only covering grids and non-clue areas
    rectangles = [
        # Left crossword grid - smaller, just the grid itself
        (120, 150, 550, 850),

        # Right crossword grid - just the grid
        (2350, 150, 2750, 850),

        # Acrostic puzzle header area (right side)
        (2350, 900, 2950, 1100),

        # Comics/ads on left edge (top)
        (0, 0, 100, 2500),

        # Comics/ads on left edge (bottom)
        (0, 2700, 600, 4032),

        # Ads on right edge
        (2850, 0, 3024, 2500),
    ]

    # Apply masks to create masked image
    masked_image = image.copy()
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
        logger.info(f"Added mask: ({x1}, {y1}) to ({x2}, {y2})")

    # Save masked image
    output_image_path = Path(__file__).parent / "IMG_5527 copy_masked.JPG"
    cv2.imwrite(str(output_image_path), masked_image)
    logger.success(f"Saved masked image to: {output_image_path}")

    # Save coordinates
    coords_data = {
        "image_path": str(image_path),
        "image_size": {
            "width": w,
            "height": h
        },
        "rectangles": [
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            for x1, y1, x2, y2 in rectangles
        ]
    }

    output_coords_path = Path(__file__).parent / "IMG_5527 copy_mask_coords.json"
    with open(output_coords_path, 'w') as f:
        json.dump(coords_data, f, indent=2)
    logger.success(f"Saved coordinates to: {output_coords_path}")

    # Create a visualization showing the masked areas
    vis = image.copy()
    for x1, y1, x2, y2 in rectangles:
        # Draw semi-transparent rectangles
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

    vis_path = Path(__file__).parent / "IMG_5527_mask_preview.JPG"
    cv2.imwrite(str(vis_path), vis)
    logger.success(f"Saved mask preview to: {vis_path}")

    return True


if __name__ == "__main__":
    logger.remove()
    logger.add(__import__('sys').stderr, level="INFO")

    logger.info("Creating test masks for IMG_5527...")
    if create_masks_for_img5527():
        logger.success("\nTest masks created successfully!")
        logger.info("\nNext step: Run the auto-processing script:")
        logger.info('  python src/examples/process_masked_image.py "src/examples/IMG_5527 copy_masked.JPG" --run-ocr')
    else:
        logger.error("Failed to create test masks")
