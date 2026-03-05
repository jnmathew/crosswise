"""
Solver data models for crossword puzzle solving.
"""

from typing import Any, Dict, List, Tuple, Optional

# Try to import pydantic; provide lightweight fallbacks if unavailable
try:
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # ImportError or runtime resolution issues
    class _FieldSpec:
        def __init__(self, default=None, *, default_factory=None, description: Optional[str] = None):
            self.default = default
            self.default_factory = default_factory
            self.description = description

    def Field(default=None, *, default_factory=None, description: Optional[str] = None):
        return _FieldSpec(default, default_factory=default_factory, description=description)

    class ConfigDict(dict):
        pass

    class BaseModel:
        def __init__(self, **kwargs):
            # Assign provided values
            for k, v in kwargs.items():
                setattr(self, k, v)
            # Apply defaults from Field specs when not provided
            for name, value in self.__class__.__dict__.items():
                if isinstance(value, _FieldSpec) and not hasattr(self, name):
                    if value.default_factory is not None:
                        setattr(self, name, value.default_factory())
                    else:
                        setattr(self, name, value.default)


class SolverInput(BaseModel):
    """Input structure for crossword solvers."""

    model_config = ConfigDict(frozen=True)

    clue_cells: Dict[str, List[Tuple[int, int]]] = Field(
        description="Mapping from clue_id (e.g., '1-across') to list of (row, col) cells"
    )
    cell_to_clues: Dict[Tuple[int, int], List[str]] = Field(
        description="Mapping from (row, col) to list of clue_ids that pass through it"
    )
    grid_size: Tuple[int, int] = Field(description="(rows, cols)")

    def clue_length(self, clue_id: str) -> int:
        """Return the length of a clue's answer."""
        return len(self.clue_cells[clue_id])

    def crossing_count(self, clue_id: str) -> int:
        """Count how many other clues cross this one."""
        count = 0
        for cell in self.clue_cells.get(clue_id, []):
            if len(self.cell_to_clues.get(cell, [])) > 1:
                count += 1
        return count


class SolveResult(BaseModel):
    """Result of a crossword solve attempt."""

    solved: bool = Field(description="True if puzzle was fully solved")
    assignment: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from clue_id to assigned word",
    )
    time_ms: float = Field(default=0.0, description="Solve time in milliseconds")
    nodes_expanded: int = Field(default=0, description="Number of search nodes expanded")
    backtracks: int = Field(default=0, description="Number of backtracks during search")
    max_recursion_depth: int = Field(default=0, description="Maximum recursion depth reached")

    # CSP-specific metrics
    prune_operations: Optional[int] = Field(
        default=None, description="Forward checking prune operations"
    )
    domain_wipeouts: Optional[int] = Field(
        default=None, description="Domain wipeouts during forward checking"
    )
    arc_consistency_pruned: Optional[int] = Field(
        default=None, description="Values pruned by AC-3 preprocessing"
    )
    backjumps: Optional[int] = Field(
        default=None, description="Conflict-directed backjumps"
    )
    sniper_calls: Optional[int] = Field(
        default=None, description="Sniper escalation API calls"
    )
    theme_detected: Optional[str] = Field(
        default=None, description="Detected theme pattern, if any"
    )


def build_solver_input_from_json(data: Dict[str, Any]) -> SolverInput:
    """
    Build SolverInput directly from puzzle JSON structure.

    The JSON has:
    - grid.rows, grid.cols, grid.cells (nested list)
    - clues.across/down with number, text, start, length
    """
    grid_data = data["grid"]
    rows = grid_data["rows"]
    cols = grid_data["cols"]
    cells = grid_data["cells"]

    clue_cells: Dict[str, List[Tuple[int, int]]] = {}
    cell_to_clues: Dict[Tuple[int, int], List[str]] = {}

    for clue in data["clues"].get("across", []):
        clue_id = f"{clue['number']}-across"
        start_row, start_col = clue["start"]
        length = clue["length"]

        clue_cell_list = []
        for i in range(length):
            c = start_col + i
            if c < cols:
                cell = cells[start_row][c]
                if not cell.get("is_block", False):
                    clue_cell_list.append((start_row, c))
                else:
                    break
        clue_cells[clue_id] = clue_cell_list

    for clue in data["clues"].get("down", []):
        clue_id = f"{clue['number']}-down"
        start_row, start_col = clue["start"]
        length = clue["length"]

        clue_cell_list = []
        for i in range(length):
            r = start_row + i
            if r < rows:
                cell = cells[r][start_col]
                if not cell.get("is_block", False):
                    clue_cell_list.append((r, start_col))
                else:
                    break
        clue_cells[clue_id] = clue_cell_list

    for clue_id, cell_list in clue_cells.items():
        for cell in cell_list:
            if cell not in cell_to_clues:
                cell_to_clues[cell] = []
            cell_to_clues[cell].append(clue_id)

    return SolverInput(
        clue_cells=clue_cells,
        cell_to_clues=cell_to_clues,
        grid_size=(rows, cols),
    )


def build_clue_inputs_from_json(
    data: Dict[str, Any],
    solver_input: Optional[SolverInput] = None,
) -> list:
    """Build ClueInput list from puzzle JSON for candidate generation."""
    from crosswise.solver.candidates import ClueInput, categorize_clue

    clue_inputs = []

    for direction in ["across", "down"]:
        for clue in data["clues"].get(direction, []):
            clue_id = f"{clue['number']}-{direction}"
            text = clue["text"]
            num_crossings = 0
            if solver_input is not None:
                num_crossings = solver_input.crossing_count(clue_id)
            clue_inputs.append(
                ClueInput(
                    clue_id=clue_id,
                    text=text,
                    length=clue["length"],
                    category=categorize_clue(text),
                    num_crossings=num_crossings,
                )
            )

    return clue_inputs
