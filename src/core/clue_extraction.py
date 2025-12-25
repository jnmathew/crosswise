"""
Clue extraction utilities for crossword puzzles.

Pipeline:
1. OCR cell numbers (digits in grid cells)
2. Extract clue text regions (stubbed for MVP)
3. Parse clues into structured format (stubbed for MVP)
"""
from typing import Dict, List
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .config import Settings
from .models import Cell, Clue, Direction
from .ocr_utils import ocr_digit_roi


def ocr_cell_numbers(
    cells: List[List[Cell]],
    warped_gray: np.ndarray,
    grid_lines: Dict[str, List[int]],
    config: Settings
) -> List[List[Cell]]:
    """
    OCR clue numbers in grid cells.

    Only attempts OCR on white cells that start a clue (where left or above
    cell is black or is at grid boundary). Uses multi-variant ensemble OCR
    for robust digit detection.

    Args:
        cells: 2D grid of Cell objects
        warped_gray: Grayscale warped grid image
        grid_lines: Dictionary with 'xs' and 'ys' lists of grid line positions
        config: Settings object

    Returns:
        Updated cells grid with clue_number and confidence set
    """
    logger.info("OCR cell numbers")

    xs = grid_lines['xs']
    ys = grid_lines['ys']
    num_rows = len(cells)
    num_cols = len(cells[0]) if cells else 0

    if num_rows == 0 or num_cols == 0:
        logger.warning("Empty cell grid")
        return cells

    # Statistics
    stats = {
        "white_cells": 0,
        "black_cells": 0,
        "start_candidates": 0,
        "ocr_attempts": 0,
        "successful_ocr": 0,
        "low_confidence": 0
    }

    # Process each cell
    for r in range(num_rows):
        for c in range(num_cols):
            cell = cells[r][c]

            # Skip black cells
            if cell.is_block:
                stats["black_cells"] += 1
                continue

            stats["white_cells"] += 1

            # Only OCR if this cell starts a clue
            # (left or above is black, or at grid border)
            starts_across = (c == 0) or (c > 0 and cells[r][c - 1].is_block)
            starts_down = (r == 0) or (r > 0 and cells[r - 1][c].is_block)

            if not (starts_across or starts_down):
                # Cell doesn't start a clue - skip OCR
                continue

            stats["start_candidates"] += 1

            # Get cell boundaries
            x0 = xs[c]
            y0 = ys[r]
            x1 = xs[c + 1] if c + 1 < len(xs) else warped_gray.shape[1]
            y1 = ys[r + 1] if r + 1 < len(ys) else warped_gray.shape[0]
            wcell = max(3, x1 - x0)
            hcell = max(3, y1 - y0)

            # Extract top-left triangular ROI (inset from grid lines)
            inset = max(2, int(min(wcell, hcell) * 0.06))
            xi = x0 + inset
            yi = y0 + inset
            roi_w = max(8, int(wcell * 0.34))
            roi_h = max(8, int(hcell * 0.34))

            # Extract ROI
            roi_rect = warped_gray[yi:min(yi + roi_h, y1), xi:min(xi + roi_w, x1)]

            if roi_rect.size == 0:
                continue

            # OCR the digit ROI
            stats["ocr_attempts"] += 1
            ocr_result = ocr_digit_roi(roi_rect, config, timeout=2.0)

            # Update cell if confidence meets threshold
            if ocr_result.text and ocr_result.confidence >= (config.OCR_CONFIDENCE_MIN / 100.0):
                try:
                    clue_num = int(ocr_result.text)
                    cell.clue_number = clue_num
                    cell.confidence = ocr_result.confidence
                    stats["successful_ocr"] += 1
                    logger.debug(f"Cell ({r},{c}): clue #{clue_num} (conf={ocr_result.confidence:.2f})")
                except ValueError:
                    # Not a valid integer
                    stats["low_confidence"] += 1
            else:
                stats["low_confidence"] += 1

    # Log statistics
    logger.info(
        f"Cell number OCR complete: "
        f"{stats['successful_ocr']}/{stats['start_candidates']} detected "
        f"({stats['white_cells']} white, {stats['black_cells']} black, "
        f"{stats['low_confidence']} low-conf)"
    )

    return cells


def extract_clue_regions(
    original: np.ndarray,
    quad_points: np.ndarray,
    grid_bbox: tuple
) -> Dict[str, List[np.ndarray]]:
    """
    Detect clue text regions outside grid.

    **MVP STUB**: Returns empty regions. Clue text OCR not implemented yet.

    Future implementation will:
    - Mask out the grid area
    - Detect text regions above/below grid
    - Separate into "Across" and "Down" sections
    - Return cropped regions for each clue

    Args:
        original: Original image
        quad_points: Puzzle quad corner points
        grid_bbox: Grid bounding box

    Returns:
        Dictionary with 'across_regions' and 'down_regions' (empty for MVP)
    """
    logger.debug("Clue text extraction stubbed (MVP) - returning empty")

    return {
        "across_regions": [],
        "down_regions": []
    }


def ocr_and_parse_clues(
    regions: Dict[str, List[np.ndarray]],
    config: Settings
) -> Dict[Direction, List[Clue]]:
    """
    OCR clue regions and parse into structured Clue objects.

    **MVP STUB**: Returns empty clue lists. Clue text OCR not implemented yet.

    Future implementation will:
    - OCR each text region
    - Parse clue format: "1. Clue text (5,3)"
    - Extract clue number, text, and answer length
    - Return structured Clue objects

    Args:
        regions: Dictionary with text region images
        config: Settings object

    Returns:
        Dictionary with Direction.ACROSS and Direction.DOWN clue lists (empty for MVP)
    """
    logger.debug("Clue parsing stubbed (MVP) - returning empty")

    return {
        Direction.ACROSS: [],
        Direction.DOWN: []
    }
