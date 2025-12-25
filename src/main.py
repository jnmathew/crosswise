"""
Main CLI entry point for crossword OCR pipeline.

Usage:
    python -m src.main <input_image> [options]
"""
from pathlib import Path
from datetime import datetime
import sys

import click
import cv2
import numpy as np
from loguru import logger

from src.core.config import Settings
from src.core import (
    image_preprocessing,
    grid_detection,
    clue_extraction,
    postprocess,
    validator,
    exporter,
)
from src.core.models import Puzzle, PuzzleMetadata, Direction


def setup_logging(config: Settings, verbose: bool):
    """Configure loguru logging."""
    logger.remove()  # Remove default handler

    level = "DEBUG" if verbose else "INFO"

    # Console handler with colors
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    )

    # File handler (always DEBUG)
    log_file = config.LOGS_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    logger.info(f"Logging to {log_file}")


def save_debug_artifacts(preprocess_result: dict, grid_result: dict, debug_path: Path):
    """Save debug images and metadata."""
    debug_path.mkdir(parents=True, exist_ok=True)

    # Save preprocessing outputs
    cv2.imwrite(str(debug_path / "original.jpg"), preprocess_result["original"])
    cv2.imwrite(str(debug_path / "warped.jpg"), preprocess_result["warped"])
    cv2.imwrite(str(debug_path / "warped_gray.jpg"), preprocess_result["warped_gray"])
    cv2.imwrite(str(debug_path / "warped_thresh.jpg"), preprocess_result["warped_thresh"])

    # Save grid overlay
    if "cells" in grid_result:
        overlay = preprocess_result["warped"].copy()
        xs = grid_result["grid_lines"]["xs"]
        ys = grid_result["grid_lines"]["ys"]

        # Draw grid lines (clipped to grid boundaries)
        x_min, x_max = xs[0], xs[-1]
        y_min, y_max = ys[0], ys[-1]

        # Draw vertical lines (clipped to grid height)
        for x in xs:
            cv2.line(overlay, (x, y_min), (x, y_max), (0, 255, 0), 1)

        # Draw horizontal lines (clipped to grid width)
        for y in ys:
            cv2.line(overlay, (x_min, y), (x_max, y), (0, 255, 0), 1)

        # Annotate cell numbers
        for row in grid_result["cells"]:
            for cell in row:
                if cell.clue_number:
                    r, c = cell.row, cell.col
                    x0 = xs[c]
                    y0 = ys[r]
                    x1 = xs[c + 1] if c + 1 < len(xs) else overlay.shape[1]
                    y1 = ys[r + 1] if r + 1 < len(ys) else overlay.shape[0]
                    wcell = x1 - x0
                    hcell = y1 - y0

                    # Draw clue number
                    text = str(cell.clue_number)
                    font_scale = 0.4
                    thickness = 1
                    color = (0, 0, 255)  # Red
                    cv2.putText(
                        overlay,
                        text,
                        (x0 + 4, y0 + int(hcell * 0.25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness,
                        cv2.LINE_AA
                    )

        cv2.imwrite(str(debug_path / "grid_overlay.jpg"), overlay)

    logger.info(f"Debug artifacts saved to {debug_path}")


@click.command()
@click.argument('input_image', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output JSON path')
@click.option('--debug-dir', type=click.Path(path_type=Path), help='Debug artifacts directory')
@click.option('--skip-clues', is_flag=True, help='Skip clue OCR (grid only)')
@click.option('--corners', type=str, help='Manual quad corners as "x1,y1,x2,y2,x3,y3,x4,y4" (TL,TR,BR,BL)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
def main(input_image: Path, output: Path, debug_dir: Path, skip_clues: bool, corners: str, verbose: bool):
    """
    Process crossword photo to structured JSON.

    Extracts grid structure and clue numbers from a crossword puzzle image.

    Example:
        python -m src.main src/examples/IMG_5527.JPG

        python -m src.main src/examples/IMG_5527.JPG -o output/puzzle.json --verbose
    """
    # Setup
    config = Settings()
    setup_logging(config, verbose)

    # Determine output paths
    output_path = output if output else config.OUTPUT_DIR / f"{input_image.stem}.json"
    debug_path = debug_dir if debug_dir else config.OUTPUT_DIR / input_image.stem

    logger.info("=" * 60)
    logger.info(f"Crosswise OCR Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input:  {input_image}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Debug:  {debug_path}")
    logger.info("=" * 60)

    start_time = datetime.now()

    try:
        # Parse manual corners if provided
        manual_quad = None
        if corners:
            try:
                coords = [float(x) for x in corners.split(',')]
                if len(coords) != 8:
                    raise ValueError("Corners must be 8 comma-separated numbers")
                manual_quad = np.array([
                    [coords[0], coords[1]],  # TL
                    [coords[2], coords[3]],  # TR
                    [coords[4], coords[5]],  # BR
                    [coords[6], coords[7]]   # BL
                ], dtype="float32")
                logger.info(f"Using manual corners: TL({coords[0]},{coords[1]}) TR({coords[2]},{coords[3]}) BR({coords[4]},{coords[5]}) BL({coords[6]},{coords[7]})")
            except Exception as e:
                logger.error(f"Invalid corners format: {e}")
                sys.exit(1)

        # Stage 1: Preprocess
        logger.info("Stage 1: Image Preprocessing")
        preprocess_result = image_preprocessing.preprocess(input_image, config, manual_quad=manual_quad)

        # Stage 2: Grid Detection
        logger.info("Stage 2: Grid Detection")
        grid_result = grid_detection.detect_grid(preprocess_result['warped_gray'], config)

        # Stage 3: OCR Cell Numbers
        logger.info("Stage 3: OCR Cell Numbers")
        cells_with_numbers = clue_extraction.ocr_cell_numbers(
            grid_result['cells'],
            preprocess_result['warped_gray'],
            grid_result['grid_lines'],
            config
        )

        # Stage 4: Extract Clues (MVP: stubbed)
        logger.info("Stage 4: Clue Extraction")
        clues = {Direction.ACROSS: [], Direction.DOWN: []}
        if not skip_clues:
            clue_regions = clue_extraction.extract_clue_regions(
                preprocess_result['original'],
                preprocess_result['quad_points'],
                None  # grid_bbox not needed for MVP stub
            )
            clues = clue_extraction.ocr_and_parse_clues(clue_regions, config)
        else:
            logger.info("Skipping clue extraction (--skip-clues)")

        # Stage 5: Build Puzzle Object
        logger.info("Stage 5: Building Puzzle Model")
        processing_time = (datetime.now() - start_time).total_seconds()

        metadata = PuzzleMetadata(
            source_image=str(input_image),
            processed_at=datetime.now().isoformat(),
            grid_size=grid_result['grid_size'],
            total_clues=len(clues[Direction.ACROSS]) + len(clues[Direction.DOWN]),
            processing_time_seconds=processing_time,
            debug_artifacts=[str(debug_path)]
        )

        puzzle = Puzzle(
            grid=cells_with_numbers,
            clues=clues,
            metadata=metadata
        )

        logger.info(f"Puzzle: {puzzle.rows}x{puzzle.cols} grid, {len(clues[Direction.ACROSS])+len(clues[Direction.DOWN])} clues")

        # Stage 6: Postprocess
        logger.info("Stage 6: Postprocessing")
        puzzle = postprocess.apply(puzzle, config)

        # Stage 7: Validate
        logger.info("Stage 7: Validation")
        validation_errors = validator.validate(puzzle, config)
        if validation_errors:
            logger.warning(f"Found {len(validation_errors)} validation issues (non-fatal)")

        # Stage 8: Export
        logger.info("Stage 8: Export JSON")
        exporter.save_json(puzzle, output_path, config)

        # Save debug artifacts
        logger.info("Stage 9: Save Debug Artifacts")
        save_debug_artifacts(preprocess_result, grid_result, debug_path)

        # Summary
        logger.info("=" * 60)
        logger.success("Pipeline Complete!")
        logger.info(f"Processing time: {processing_time:.2f}s")
        logger.info(f"Output: {output_path}")
        logger.info(f"Debug:  {debug_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Pipeline failed: {e}")
        logger.error("=" * 60)

        # Try to save debug artifacts even on failure
        try:
            if 'preprocess_result' in locals():
                debug_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_path / "original.jpg"), preprocess_result["original"])
                cv2.imwrite(str(debug_path / "warped.jpg"), preprocess_result["warped"])
                logger.info(f"Partial debug artifacts saved to {debug_path}")
        except Exception:
            pass

        logger.info(f"Check logs and debug artifacts in {debug_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
