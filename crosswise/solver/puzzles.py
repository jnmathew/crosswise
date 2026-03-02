"""
Toy puzzles for testing crossword solvers.
"""

from typing import Dict, List, Tuple
from crosswise.solver.models import SolverInput


def get_tiny_3x3() -> Tuple[SolverInput, Dict[str, List[str]]]:
    """
    Return a tiny 3x3 puzzle for quick solver testing.

    Grid (# = block):
        C A T
        A # A
        B A D

    Clues:
        1-across: CAT (row 0)
        3-across: BAD (row 2)
        1-down: CAB (col 0)
        2-down: TAD (col 2)

    Returns:
        (SolverInput, candidates_by_clue) tuple
    """
    clue_cells = {
        "1-across": [(0, 0), (0, 1), (0, 2)],
        "3-across": [(2, 0), (2, 1), (2, 2)],
        "1-down": [(0, 0), (1, 0), (2, 0)],
        "2-down": [(0, 2), (1, 2), (2, 2)],
    }

    cell_to_clues: Dict[Tuple[int, int], List[str]] = {}
    for clue_id, cells in clue_cells.items():
        for cell in cells:
            if cell not in cell_to_clues:
                cell_to_clues[cell] = []
            cell_to_clues[cell].append(clue_id)

    solver_input = SolverInput(
        clue_cells=clue_cells,
        cell_to_clues=cell_to_clues,
        grid_size=(3, 3),
    )

    # Candidates: correct answer plus distractors
    candidates = {
        "1-across": ["CAT", "COW", "CUP", "CAR", "CAN", "CAP"],
        "3-across": ["BAD", "BAT", "BAG", "BED", "BIG", "BUS"],
        "1-down": ["CAB", "COB", "CUB", "CAD", "COD", "CUD"],
        "2-down": ["TAD", "TED", "TOD", "TAB", "TUB", "TIP"],
    }

    return solver_input, candidates


def get_mini_5x5() -> Tuple[SolverInput, Dict[str, List[str]]]:
    """
    Return a mini 4x4 puzzle (named 5x5 for legacy reasons).

    This is a symmetric puzzle where rows equal columns:

    Grid:
        C A R S
        A R E A
        R E A L
        S A L E

    Clues:
        1-across: CARS, 2-across: AREA, 3-across: REAL, 4-across: SALE
        1-down: CARS, 5-down: AREA, 6-down: REAL, 7-down: SALE

    Returns:
        (SolverInput, candidates_by_clue) tuple
    """
    clue_cells = {
        "1-across": [(0, 0), (0, 1), (0, 2), (0, 3)],  # CARS
        "2-across": [(1, 0), (1, 1), (1, 2), (1, 3)],  # AREA
        "3-across": [(2, 0), (2, 1), (2, 2), (2, 3)],  # REAL
        "4-across": [(3, 0), (3, 1), (3, 2), (3, 3)],  # SALE
        "1-down": [(0, 0), (1, 0), (2, 0), (3, 0)],  # CARS
        "5-down": [(0, 1), (1, 1), (2, 1), (3, 1)],  # AREA
        "6-down": [(0, 2), (1, 2), (2, 2), (3, 2)],  # REAL
        "7-down": [(0, 3), (1, 3), (2, 3), (3, 3)],  # SALE
    }

    cell_to_clues: Dict[Tuple[int, int], List[str]] = {}
    for clue_id, cells in clue_cells.items():
        for cell in cells:
            if cell not in cell_to_clues:
                cell_to_clues[cell] = []
            cell_to_clues[cell].append(clue_id)

    solver_input = SolverInput(
        clue_cells=clue_cells,
        cell_to_clues=cell_to_clues,
        grid_size=(4, 4),
    )

    # Candidates: correct answers plus distractors
    candidates = {
        "1-across": ["CARS", "CATS", "CAPS", "CABS", "CANS", "CUPS"],
        "2-across": ["AREA", "ARIA", "ANNA", "AQUA", "AURA", "ALGA"],
        "3-across": ["REAL", "READ", "REAM", "REAP", "REAR", "REEL"],
        "4-across": ["SALE", "SAFE", "SAGE", "SAKE", "SAME", "SANE"],
        "1-down": ["CARS", "CATS", "CAPS", "CABS", "CANS", "CUPS"],
        "5-down": ["AREA", "ARIA", "ANNA", "AQUA", "AURA", "ALGA"],
        "6-down": ["REAL", "READ", "REAM", "REAP", "REAR", "REEL"],
        "7-down": ["SALE", "SAFE", "SAGE", "SAKE", "SAME", "SANE"],
    }

    return solver_input, candidates


def get_t_intersection() -> Tuple[SolverInput, Dict[str, List[str]]]:
    """
    Return a simple T-shaped intersection puzzle.

    Grid (# = block):
        # C A T #
        # A # # #
        # R # # #

    Clues:
        1-across: CAT (row 0, cols 1-3)
        1-down: CAR (col 1, rows 0-2)

    The letter 'C' at (0,1) is shared by both clues.

    Returns:
        (SolverInput, candidates_by_clue) tuple
    """
    clue_cells = {
        "1-across": [(0, 1), (0, 2), (0, 3)],  # CAT
        "1-down": [(0, 1), (1, 1), (2, 1)],  # CAR
    }

    cell_to_clues: Dict[Tuple[int, int], List[str]] = {}
    for clue_id, cells in clue_cells.items():
        for cell in cells:
            if cell not in cell_to_clues:
                cell_to_clues[cell] = []
            cell_to_clues[cell].append(clue_id)

    solver_input = SolverInput(
        clue_cells=clue_cells,
        cell_to_clues=cell_to_clues,
        grid_size=(3, 5),
    )

    candidates = {
        "1-across": ["CAT", "CAR", "CAN", "CAP", "CUP", "COW"],
        "1-down": ["CAR", "CAT", "CAN", "CAP", "CUB", "COB"],
    }

    return solver_input, candidates
