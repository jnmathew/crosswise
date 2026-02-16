"""
Clue extraction utilities for crossword puzzles.

Parses OCR markdown output into structured clue data and verifies
correspondence between OCR clues and grid-detected slots.
"""
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .models import Clue, Direction


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

    # Build reverse lookup: number -> list of available directions in grid
    number_to_dirs = {}
    for slot in grid_slots:
        number_to_dirs.setdefault(slot["number"], []).append(slot["direction"])

    # Match each OCR clue to a grid slot
    for clue in ocr_clues:
        key = (clue["number"], clue["direction"])

        if key in slot_lookup:
            slot = slot_lookup[key]
            matched_keys.add(key)

            matched.append({
                "number": clue["number"],
                "direction": clue["direction"],
                "text": clue["text"],
                "start": slot["start"],
                "length": slot["length"]
            })
        else:
            # Direction-swap fallback: trust grid for direction assignment.
            # If OCR says "2-ACROSS" but grid only has "2-DOWN", remap it.
            flip = "down" if clue["direction"] == "across" else "across"
            flip_key = (clue["number"], flip)
            if flip_key in slot_lookup and flip_key not in matched_keys:
                slot = slot_lookup[flip_key]
                matched_keys.add(flip_key)
                warnings.append(
                    f"OCR clue {clue['number']}-{clue['direction'].upper()} "
                    f"remapped to {flip.upper()} (grid has no {clue['direction'].upper()} slot)"
                )
                matched.append({
                    "number": clue["number"],
                    "direction": flip,
                    "text": clue["text"],
                    "start": slot["start"],
                    "length": slot["length"]
                })
            else:
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
