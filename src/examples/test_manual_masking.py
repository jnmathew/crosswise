"""
Test script to simulate manual masking and separator placement.

This demonstrates the interactive masking workflow without GUI interaction.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from loguru import logger


def create_manual_masks_and_separators():
    """Create test masks and separators to demonstrate the manual workflow."""

    # Load image
    image_path = Path(__file__).parent / "IMG_5527 copy.JPG"
    image = cv2.imread(str(image_path))

    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return False

    h, w = image.shape[:2]
    logger.info(f"Image size: {w}x{h}")

    # Define mask rectangles (covering grids, ads, acrostic)
    rectangles = [
        # Left crossword grid - just the grid
        (120, 150, 550, 850),

        # Right crossword grid - just the grid
        (2350, 150, 2750, 850),

        # Acrostic puzzle area
        (2350, 900, 2950, 1600),

        # Left edge ads/comics
        (0, 0, 100, 2500),
        (0, 2700, 600, 4032),

        # Right edge ads
        (2850, 0, 3024, 2500),
    ]

    # Define separator x-positions (manually chosen column boundaries)
    # Looking at the clue layout, let's place separators between columns
    separators = [
        650,   # After first column of clues
        1300,  # After second column of clues
        1950,  # After third column of clues
    ]

    logger.info(f"Creating {len(rectangles)} masks and {len(separators)} separators")

    # Apply masks and separators
    result = image.copy()

    # Draw white rectangle masks
    for x1, y1, x2, y2 in rectangles:
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), -1)
        logger.debug(f"Added mask: ({x1}, {y1}) to ({x2}, {y2})")

    # Draw aqua separators
    for x in separators:
        cv2.line(result, (x, 0), (x, h), (255, 255, 0), 4)  # Aqua/cyan, 4px wide
        logger.debug(f"Added separator at x={x}")

    # Save final image
    output_image_path = Path(__file__).parent / "IMG_5527 copy_masked_divided.JPG"
    cv2.imwrite(str(output_image_path), result)
    logger.success(f"Saved image to: {output_image_path}")

    # Save coordinates JSON
    coords_data = {
        "image_path": str(image_path),
        "image_size": {
            "width": w,
            "height": h
        },
        "rectangles": [
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            for x1, y1, x2, y2 in rectangles
        ],
        "separators": separators
    }

    output_coords_path = Path(__file__).parent / "IMG_5527 copy_mask_coords.json"
    with open(output_coords_path, 'w') as f:
        json.dump(coords_data, f, indent=2)
    logger.success(f"Saved coordinates to: {output_coords_path}")

    # Create a visualization
    vis = image.copy()

    # Draw masks with transparency
    for x1, y1, x2, y2 in rectangles:
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Draw separators in bright green
    for x in separators:
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 3)

    vis_path = Path(__file__).parent / "IMG_5527_manual_masking_preview.JPG"
    cv2.imwrite(str(vis_path), vis)
    logger.success(f"Saved preview to: {vis_path}")

    logger.info(f"\nCreated:")
    logger.info(f"  - {len(rectangles)} white rectangle masks")
    logger.info(f"  - {len(separators)} aqua vertical separators at x={separators}")

    return True


if __name__ == "__main__":
    logger.remove()
    logger.add(__import__('sys').stderr, level="INFO")

    logger.info("Testing manual masking with separators...")
    if create_manual_masks_and_separators():
        logger.success("\nManual masking test complete!")
        logger.info("\nThis simulates what you would create with the interactive tool:")
        logger.info("  1. Draw white rectangles over grids/ads (MASK mode)")
        logger.info("  2. Press 'v' to switch to SEPARATOR mode")
        logger.info("  3. Click to place aqua vertical lines between columns")
        logger.info("  4. Press 's' to save")
        logger.info("\nThe result shows masks + separators ready for OCR!")
    else:
        logger.error("Test failed")
