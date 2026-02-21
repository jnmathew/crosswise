"""
Post-masking processing script for crossword clues.

This script runs after interactive_masker.py to:
1. Load the masked image (with white rectangles over grids/irrelevant areas)
2. Apply automatic column detection (handles partial-height columns)
3. Draw aqua/cyan vertical separator lines between columns
4. Save the final preprocessed image
5. Optionally run OCR with Mistral

Usage:
    python process_masked_image.py <masked_image_path> [--run-ocr]

Example:
    python process_masked_image.py "IMG_5527 copy_masked.JPG" --run-ocr
"""
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from loguru import logger

from src.core.clue_column_detector import preprocess_clues_for_ocr, detect_column_boundaries


def main():
    parser = argparse.ArgumentParser(description="Process masked crossword image with column detection")
    parser.add_argument("image_path", help="Path to masked image")
    parser.add_argument("--run-ocr", action="store_true", help="Run Mistral OCR after preprocessing")
    parser.add_argument("--min-column-width", type=int, default=None, help="Minimum column width (auto if not set)")
    parser.add_argument("--max-columns", type=int, default=12, help="Maximum number of columns")
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    # Load masked image
    image_path = Path(args.image_path)
    if not image_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        image_path = script_dir / args.image_path

    if not image_path.exists():
        logger.error(f"Image not found: {args.image_path}")
        return 1

    logger.info(f"Loading masked image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return 1

    h, w = image.shape[:2]
    logger.info(f"Image size: {w}x{h}")

    # Auto-determine min_column_width if not specified
    if args.min_column_width is None:
        # For newspaper-style layouts, columns are typically 1/8 to 1/4 of page width
        # Use a more conservative estimate to avoid false detections within columns
        min_column_width = max(300, w // 8)
        logger.info(f"Auto-determined min_column_width: {min_column_width}")
    else:
        min_column_width = args.min_column_width
        logger.info(f"Using specified min_column_width: {min_column_width}")

    # Apply column detection and draw separators
    logger.info("Running improved column detection (handles partial-height columns)...")
    preprocessed = preprocess_clues_for_ocr(
        image,
        min_column_width=min_column_width,
        max_columns=args.max_columns,
        add_separators=True,
        mask_non_text=False,  # Already masked manually
        separator_color=(255, 255, 0),  # Aqua/cyan in BGR
        separator_width=4  # Slightly thicker for visibility
    )

    # Generate output paths
    output_dir = image_path.parent
    stem = image_path.stem.replace("_masked", "")  # Remove _masked suffix if present

    final_output_path = output_dir / f"{stem}_masked_divided.JPG"
    vis_output_path = output_dir / f"{stem}_column_boundaries_vis.JPG"

    # Save final preprocessed image
    cv2.imwrite(str(final_output_path), preprocessed)
    logger.success(f"Saved final preprocessed image to: {final_output_path}")

    # Create visualization with just the boundaries highlighted
    vis = image.copy()
    boundaries = detect_column_boundaries(
        image,
        min_column_width=min_column_width,
        max_columns=args.max_columns
    )

    # Draw boundaries in bright green for visualization
    for x in boundaries:
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 3)

    cv2.imwrite(str(vis_output_path), vis)
    logger.success(f"Saved boundary visualization to: {vis_output_path}")

    logger.info(f"\nDetected {len(boundaries) + 1} columns")
    logger.info(f"Boundaries at x-coordinates: {boundaries}")

    # Run OCR if requested
    if args.run_ocr:
        logger.info("\nRunning OCR...")
        try:
            from src.core.config import settings
            from src.ocr.base import create_ocr_provider

            provider = create_ocr_provider(settings)
            ocr_markdown = provider.extract_clues(final_output_path)

            output_md_path = output_dir / f"{stem}_final_ocr_results.md"
            with open(output_md_path, "w") as f:
                f.write(f"# OCR Results - {final_output_path.name}\n\n")
                f.write(ocr_markdown)

            logger.success(f"OCR results saved to: {output_md_path}")

        except ImportError as e:
            logger.error(f"OCR dependencies not available: {e}")
        except Exception as e:
            logger.error(f"OCR failed: {e}")

    logger.success("\nProcessing complete!")
    logger.info(f"Final image: {final_output_path}")
    if args.run_ocr:
        logger.info(f"OCR results: {output_dir / f'{stem}_final_ocr_results.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
