"""Tests for solver models."""

import pytest
from src.solver.models import SolverInput, SolveResult


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
