"""
Validation utilities for puzzle data.

Checks puzzle completeness and consistency.
"""
from typing import List

from loguru import logger

from .config import Settings
from .models import Puzzle, Direction


def validate(puzzle: Puzzle, config: Settings) -> List[str]:
    """
    Validate puzzle completeness and consistency.

    Checks:
    - Grid size is reasonable (5-25 rows/cols)
    - Clue numbering starts at 1 (if clues exist)
    - Grid clue numbers match clue list (if clues exist)
    - Optional: Grid symmetry (typical crossword property)

    Args:
        puzzle: Puzzle object to validate
        config: Settings object

    Returns:
        List of validation warning/error messages (empty if valid)
    """
    logger.debug("Validating puzzle")

    warnings = []

    # Check grid size
    if not (5 <= puzzle.rows <= 25):
        warnings.append(f"Unusual grid size: {puzzle.rows} rows (expected 5-25)")

    if not (5 <= puzzle.cols <= 25):
        warnings.append(f"Unusual grid size: {puzzle.cols} cols (expected 5-25)")

    # Check clue numbering (if clues exist)
    clue_nums = []
    for direction in [Direction.ACROSS, Direction.DOWN]:
        for clue in puzzle.clues[direction]:
            clue_nums.append(clue.number)

    if clue_nums:
        min_num = min(clue_nums)
        if min_num != 1:
            warnings.append(f"Clue numbering doesn't start at 1 (starts at {min_num})")

    # Check grid clue numbers match clue list (if clues exist)
    grid_nums = {
        cell.clue_number
        for row in puzzle.grid
        for cell in row
        if cell.clue_number is not None
    }

    if clue_nums and grid_nums:
        clue_nums_set = set(clue_nums)
        missing_in_grid = clue_nums_set - grid_nums
        extra_in_grid = grid_nums - clue_nums_set

        if missing_in_grid:
            warnings.append(
                f"Clue numbers in list but not in grid: {sorted(missing_in_grid)}"
            )

        if extra_in_grid:
            warnings.append(
                f"Clue numbers in grid but not in list: {sorted(extra_in_grid)}"
            )

    # Symmetry check (optional - many crosswords have rotational symmetry)
    # Uncomment to enable:
    # symmetry_errors = 0
    # for r in range(puzzle.rows):
    #     for c in range(puzzle.cols):
    #         sym_r = puzzle.rows - 1 - r
    #         sym_c = puzzle.cols - 1 - c
    #         cell = puzzle.get_cell(r, c)
    #         sym_cell = puzzle.get_cell(sym_r, sym_c)
    #         if cell and sym_cell and cell.is_block != sym_cell.is_block:
    #             symmetry_errors += 1
    #
    # if symmetry_errors > 0:
    #     warnings.append(f"Grid lacks rotational symmetry ({symmetry_errors} asymmetric cells)")

    if warnings:
        logger.warning(f"Validation found {len(warnings)} issues")
        for w in warnings:
            logger.warning(f"  - {w}")
    else:
        logger.success("Validation passed")

    return warnings
