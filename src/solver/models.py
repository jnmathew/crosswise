"""
Solver data models for crossword puzzle solving.
"""

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    from src.core.models import Puzzle


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

    @classmethod
    def from_puzzle(cls, puzzle: "Puzzle") -> "SolverInput":
        """
        Build SolverInput from a Puzzle instance.

        Scans the grid for clue starts and builds clue_cells and cell_to_clues mappings.

        Args:
            puzzle: Puzzle instance with grid and clues

        Returns:
            SolverInput ready for solving
        """
        from src.core.models import Direction

        clue_cells: Dict[str, List[Tuple[int, int]]] = {}
        cell_to_clues: Dict[Tuple[int, int], List[str]] = {}

        rows = puzzle.rows
        cols = puzzle.cols

        # Build a map from clue_number -> (row, col) of the cell with that number
        clue_number_to_cell: Dict[int, Tuple[int, int]] = {}
        for r in range(rows):
            for c in range(cols):
                cell = puzzle.grid[r][c]
                if cell.clue_number is not None:
                    clue_number_to_cell[cell.clue_number] = (r, c)

        # Process across clues
        for clue in puzzle.clues.get(Direction.ACROSS, []):
            clue_id = f"{clue.number}-across"
            start_pos = clue_number_to_cell.get(clue.number)
            if start_pos is None:
                continue

            r, c = start_pos
            length = clue.total_length
            cells = []
            for i in range(length):
                if c + i < cols:
                    cell = puzzle.grid[r][c + i]
                    if not cell.is_block:
                        cells.append((r, c + i))
                    else:
                        break
            clue_cells[clue_id] = cells

        # Process down clues
        for clue in puzzle.clues.get(Direction.DOWN, []):
            clue_id = f"{clue.number}-down"
            start_pos = clue_number_to_cell.get(clue.number)
            if start_pos is None:
                continue

            r, c = start_pos
            length = clue.total_length
            cells = []
            for i in range(length):
                if r + i < rows:
                    cell = puzzle.grid[r + i][c]
                    if not cell.is_block:
                        cells.append((r + i, c))
                    else:
                        break
            clue_cells[clue_id] = cells

        # Build cell_to_clues mapping
        for clue_id, cells in clue_cells.items():
            for cell in cells:
                if cell not in cell_to_clues:
                    cell_to_clues[cell] = []
                cell_to_clues[cell].append(clue_id)

        return cls(
            clue_cells=clue_cells,
            cell_to_clues=cell_to_clues,
            grid_size=(rows, cols),
        )


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
