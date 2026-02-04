"""
Clue extraction utilities for crossword puzzles.

Pipeline:
1. OCR cell numbers (digits in grid cells)
2. Extract clue text regions (stubbed for MVP)
3. Parse clues into structured format
"""
import re
from typing import Dict, List, Optional, Tuple
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


def parse_ocr_markdown(text: str) -> Tuple[List[Dict], List[str]]:
    """
    Parse OCR markdown output into structured clue data.

    Expected format:
        ## ACROSS
        1. Clue text here
        7. Another clue
        ...
        ## DOWN
        1. Down clue text
        ...

    Args:
        text: Raw markdown text from OCR

    Returns:
        Tuple of:
        - List of parsed clues: [{"number": int, "direction": str, "text": str}, ...]
        - List of warnings/errors encountered during parsing
    """
    clues = []
    warnings = []

    lines = text.strip().split('\n')
    current_direction: Optional[str] = None

    # Patterns for section headers (flexible matching)
    across_pattern = re.compile(r'^#{1,3}\s*ACROSS\s*:?\s*$', re.IGNORECASE)
    down_pattern = re.compile(r'^#{1,3}\s*DOWN\s*:?\s*$', re.IGNORECASE)

    # Pattern for clue lines: number followed by period/dot, then text
    # Handles: "1. text", "1.text", "1 . text", "1- text"
    clue_pattern = re.compile(r'^(\d+)\s*[.\-\)]\s*(.+)$')

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check for section headers
        if across_pattern.match(line):
            current_direction = "across"
            continue

        if down_pattern.match(line):
            current_direction = "down"
            continue

        # Skip title/header lines (starts with #)
        if line.startswith('#'):
            continue

        # Try to parse as clue line
        match = clue_pattern.match(line)
        if match:
            if current_direction is None:
                warnings.append(f"Line {line_num}: Clue found before ACROSS/DOWN header: '{line[:50]}...'")
                continue

            number = int(match.group(1))
            clue_text = match.group(2).strip()

            # Basic validation
            if number <= 0:
                warnings.append(f"Line {line_num}: Invalid clue number {number}")
                continue

            if len(clue_text) < 2:
                warnings.append(f"Line {line_num}: Clue text too short: '{clue_text}'")
                continue

            clues.append({
                "number": number,
                "direction": current_direction,
                "text": clue_text
            })
        else:
            # Line doesn't match expected format
            # Could be continuation of previous clue, or garbage
            if line and not line.startswith('#') and current_direction:
                warnings.append(f"Line {line_num}: Unparseable line: '{line[:50]}...'")

    # Summary logging
    across_count = sum(1 for c in clues if c["direction"] == "across")
    down_count = sum(1 for c in clues if c["direction"] == "down")
    logger.info(f"Parsed {len(clues)} clues ({across_count} across, {down_count} down)")

    if warnings:
        logger.warning(f"Parsing encountered {len(warnings)} warnings")
        for w in warnings[:5]:  # Log first 5
            logger.debug(w)

    return clues, warnings


def check_duplicate_clues(clues: List[Dict]) -> List[str]:
    """
    Check for duplicate clue numbers within each direction.

    Args:
        clues: List of parsed clue dicts

    Returns:
        List of error messages for duplicates found
    """
    errors = []
    seen = {"across": set(), "down": set()}

    for clue in clues:
        direction = clue["direction"]
        number = clue["number"]

        if number in seen[direction]:
            errors.append(f"Duplicate {direction} clue: {number}")
        else:
            seen[direction].add(number)

    return errors


def match_clues_to_slots(
    ocr_clues: List[Dict],
    grid_slots: List[Dict]
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Match OCR-parsed clues to grid-detected slots.

    Verification rules:
    - Every OCR clue MUST have a matching grid slot (error if not)
    - Every grid slot MUST have a matching OCR clue (error if not)

    Args:
        ocr_clues: List of {"number": int, "direction": str, "text": str}
        grid_slots: List of {"number": int, "direction": str, "start": tuple, "length": int}

    Returns:
        Tuple of:
        - matched_clues: List of merged clue dicts with grid info added
        - errors: Critical errors (mismatches)
        - warnings: Non-critical issues
    """
    matched = []
    errors = []
    warnings = []

    # Build lookup for grid slots: (number, direction) -> slot
    slot_lookup = {}
    for slot in grid_slots:
        key = (slot["number"], slot["direction"])
        slot_lookup[key] = slot

    # Track which slots were matched
    matched_keys = set()

    # Match each OCR clue to a grid slot
    for clue in ocr_clues:
        key = (clue["number"], clue["direction"])

        if key in slot_lookup:
            slot = slot_lookup[key]
            matched_keys.add(key)

            # Merge clue text with grid slot info
            matched.append({
                "number": clue["number"],
                "direction": clue["direction"],
                "text": clue["text"],
                "start": slot["start"],
                "length": slot["length"]
            })
        else:
            # Critical error: OCR found a clue that doesn't exist in grid
            errors.append(
                f"OCR clue {clue['number']}-{clue['direction'].upper()} "
                f"has no matching grid slot"
            )

    # Check for unmatched grid slots (also errors - every slot needs a clue)
    for slot in grid_slots:
        key = (slot["number"], slot["direction"])
        if key not in matched_keys:
            errors.append(
                f"Grid slot {slot['number']}-{slot['direction'].upper()} "
                f"(length={slot['length']}) has no OCR clue"
            )

    # Log summary
    logger.info(
        f"Matched {len(matched)}/{len(ocr_clues)} OCR clues to grid slots"
    )
    if errors:
        logger.error(f"Found {len(errors)} verification errors")

    return matched, errors, warnings


def verify_puzzle(
    ocr_clues: List[Dict],
    grid_slots: List[Dict]
) -> Tuple[bool, List[Dict], Dict]:
    """
    Full puzzle verification: match clues to slots and validate.

    Args:
        ocr_clues: Parsed OCR clues
        grid_slots: Computed grid slots

    Returns:
        Tuple of:
        - success: True if verification passed
        - matched_clues: Merged clue data (empty if failed)
        - report: Verification report dict
    """
    report = {
        "ocr_clue_count": len(ocr_clues),
        "grid_slot_count": len(grid_slots),
        "ocr_across": sum(1 for c in ocr_clues if c["direction"] == "across"),
        "ocr_down": sum(1 for c in ocr_clues if c["direction"] == "down"),
        "grid_across": sum(1 for s in grid_slots if s["direction"] == "across"),
        "grid_down": sum(1 for s in grid_slots if s["direction"] == "down"),
        "errors": [],
        "warnings": [],
        "duplicate_errors": []
    }

    # Check for duplicate clues in OCR
    report["duplicate_errors"] = check_duplicate_clues(ocr_clues)

    # Match clues to slots
    matched, errors, warnings = match_clues_to_slots(ocr_clues, grid_slots)

    report["errors"] = errors
    report["warnings"] = warnings
    report["matched_count"] = len(matched)

    # Verification passes if no errors and no duplicates
    success = len(errors) == 0 and len(report["duplicate_errors"]) == 0

    if success:
        logger.info("✓ Puzzle verification PASSED")
    else:
        logger.error("✗ Puzzle verification FAILED")
        for err in report["duplicate_errors"]:
            logger.error(f"  Duplicate: {err}")
        for err in errors[:10]:  # Show first 10
            logger.error(f"  {err}")

    return success, matched, report


def build_puzzle_clues(
    matched_clues: List[Dict]
) -> Dict[Direction, List[Clue]]:
    """
    Convert matched clue data into structured Clue objects.

    Args:
        matched_clues: List of verified clue dicts from match_clues_to_slots

    Returns:
        Dictionary with Direction.ACROSS and Direction.DOWN lists of Clue objects
    """
    result = {
        Direction.ACROSS: [],
        Direction.DOWN: []
    }

    for clue_data in matched_clues:
        direction = Direction.ACROSS if clue_data["direction"] == "across" else Direction.DOWN

        clue = Clue(
            number=clue_data["number"],
            direction=direction,
            text=clue_data["text"],
            answer_length=[clue_data["length"]]  # Single word for now
        )
        result[direction].append(clue)

    # Sort by clue number
    result[Direction.ACROSS].sort(key=lambda c: c.number)
    result[Direction.DOWN].sort(key=lambda c: c.number)

    logger.info(
        f"Built {len(result[Direction.ACROSS])} ACROSS and "
        f"{len(result[Direction.DOWN])} DOWN clues"
    )

    return result
