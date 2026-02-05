"""Tests for solver models."""

import pytest
from src.solver.models import SolverInput, SolveResult
from src.core.models import Puzzle, Cell, Clue, Direction, PuzzleMetadata


class TestSolverInput:
    """Test SolverInput model."""

    def test_clue_length(self):
        """clue_length should return number of cells."""
        clue_cells = {
            "1-across": [(0, 0), (0, 1), (0, 2)],
            "1-down": [(0, 0), (1, 0)],
        }
        cell_to_clues = {
            (0, 0): ["1-across", "1-down"],
            (0, 1): ["1-across"],
            (0, 2): ["1-across"],
            (1, 0): ["1-down"],
        }
        solver_input = SolverInput(
            clue_cells=clue_cells,
            cell_to_clues=cell_to_clues,
            grid_size=(2, 3),
        )

        assert solver_input.clue_length("1-across") == 3
        assert solver_input.clue_length("1-down") == 2

    def test_frozen_model(self):
        """SolverInput should be immutable (frozen)."""
        solver_input = SolverInput(
            clue_cells={"1-across": [(0, 0)]},
            cell_to_clues={(0, 0): ["1-across"]},
            grid_size=(1, 1),
        )

        # Pydantic frozen models raise error on modification
        with pytest.raises(Exception):
            solver_input.grid_size = (2, 2)


class TestSolverInputFromPuzzle:
    """Test SolverInput.from_puzzle() method."""

    def test_simple_puzzle(self):
        """from_puzzle should correctly build SolverInput from a simple puzzle."""
        # Create a simple 3x3 grid:
        #   C . .
        #   . # .
        #   . . .
        # With 1-across (3 letters) and 1-down (3 letters)
        grid = [
            [
                Cell(row=0, col=0, clue_number=1),
                Cell(row=0, col=1),
                Cell(row=0, col=2),
            ],
            [
                Cell(row=1, col=0),
                Cell(row=1, col=1, is_block=True),
                Cell(row=1, col=2),
            ],
            [
                Cell(row=2, col=0),
                Cell(row=2, col=1),
                Cell(row=2, col=2),
            ],
        ]

        clues = {
            Direction.ACROSS: [
                Clue(number=1, direction=Direction.ACROSS, text="Clue 1A", answer_length=[3]),
            ],
            Direction.DOWN: [
                Clue(number=1, direction=Direction.DOWN, text="Clue 1D", answer_length=[3]),
            ],
        }

        metadata = PuzzleMetadata(
            source_image="test.png",
            processed_at="2024-01-01",
            grid_size=(3, 3),
            total_clues=2,
        )

        puzzle = Puzzle(grid=grid, clues=clues, metadata=metadata)
        solver_input = SolverInput.from_puzzle(puzzle)

        # Check structure
        assert solver_input.grid_size == (3, 3)
        assert "1-across" in solver_input.clue_cells
        assert "1-down" in solver_input.clue_cells

        # 1-across should have 3 cells in row 0
        assert solver_input.clue_cells["1-across"] == [(0, 0), (0, 1), (0, 2)]

        # 1-down should have 3 cells in col 0
        assert solver_input.clue_cells["1-down"] == [(0, 0), (1, 0), (2, 0)]

        # Cell (0,0) should be in both clues
        assert "1-across" in solver_input.cell_to_clues[(0, 0)]
        assert "1-down" in solver_input.cell_to_clues[(0, 0)]

    def test_multi_word_answer(self):
        """from_puzzle should handle multi-word answers (sum of lengths)."""
        # Create a 5-cell row for a (2,2) answer = 4 total letters
        grid = [[Cell(row=0, col=c, clue_number=1 if c == 0 else None) for c in range(5)]]

        clues = {
            Direction.ACROSS: [
                Clue(number=1, direction=Direction.ACROSS, text="Two words", answer_length=[2, 2]),
            ],
            Direction.DOWN: [],
        }

        metadata = PuzzleMetadata(
            source_image="test.png",
            processed_at="2024-01-01",
            grid_size=(1, 5),
            total_clues=1,
        )

        puzzle = Puzzle(grid=grid, clues=clues, metadata=metadata)
        solver_input = SolverInput.from_puzzle(puzzle)

        # total_length = 2 + 2 = 4
        assert len(solver_input.clue_cells["1-across"]) == 4

    def test_down_clue_stops_at_block(self):
        """from_puzzle should stop down clue at block."""
        # Create a 3x1 column with block in middle
        grid = [
            [Cell(row=0, col=0, clue_number=1)],
            [Cell(row=1, col=0, is_block=True)],
            [Cell(row=2, col=0)],
        ]

        clues = {
            Direction.ACROSS: [],
            Direction.DOWN: [
                Clue(number=1, direction=Direction.DOWN, text="Short", answer_length=[1]),
            ],
        }

        metadata = PuzzleMetadata(
            source_image="test.png",
            processed_at="2024-01-01",
            grid_size=(3, 1),
            total_clues=1,
        )

        puzzle = Puzzle(grid=grid, clues=clues, metadata=metadata)
        solver_input = SolverInput.from_puzzle(puzzle)

        # Should only include cell (0,0) before the block
        assert solver_input.clue_cells["1-down"] == [(0, 0)]


class TestSolveResult:
    """Test SolveResult model."""

    def test_default_values(self):
        """SolveResult should have sensible defaults."""
        result = SolveResult(solved=True)

        assert result.solved is True
        assert result.assignment == {}
        assert result.time_ms == 0.0
        assert result.nodes_expanded == 0
        assert result.backtracks == 0
        assert result.max_recursion_depth == 0
        assert result.prune_operations is None
        assert result.domain_wipeouts is None

    def test_csp_specific_metrics(self):
        """SolveResult should accept CSP-specific metrics."""
        result = SolveResult(
            solved=True,
            prune_operations=100,
            domain_wipeouts=5,
        )

        assert result.prune_operations == 100
        assert result.domain_wipeouts == 5

    def test_full_result(self):
        """SolveResult should hold all fields."""
        result = SolveResult(
            solved=True,
            assignment={"1-across": "CAT", "1-down": "CAB"},
            time_ms=12.5,
            nodes_expanded=42,
            backtracks=7,
            max_recursion_depth=4,
            prune_operations=100,
            domain_wipeouts=2,
        )

        assert result.solved is True
        assert result.assignment["1-across"] == "CAT"
        assert result.time_ms == 12.5
        assert result.nodes_expanded == 42
        assert result.backtracks == 7
        assert result.max_recursion_depth == 4
