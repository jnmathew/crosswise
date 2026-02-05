"""Tests for CSP solver."""

import pytest
from src.solver.csp import solve_csp, CSPSolver
from src.solver.puzzles import get_tiny_3x3, get_mini_5x5, get_t_intersection
from src.solver.models import SolverInput


class TestCSPSolverTiny3x3:
    """Test CSP solver on tiny 3x3 puzzle."""

    def test_solves_tiny_puzzle(self):
        """CSP solver should find a valid solution for tiny puzzle."""
        solver_input, candidates = get_tiny_3x3()
        result = solve_csp(solver_input, candidates)

        assert result.solved is True
        # Check all clues are assigned
        assert len(result.assignment) == 4
        assert all(clue_id in result.assignment for clue_id in solver_input.clue_cells)

    def test_crossing_constraints_satisfied(self):
        """Solution should satisfy all crossing constraints."""
        solver_input, candidates = get_tiny_3x3()
        result = solve_csp(solver_input, candidates)

        assert result.solved is True

        # Check crossings:
        # (0,0): 1-across[0] == 1-down[0] -> C == C
        assert result.assignment["1-across"][0] == result.assignment["1-down"][0]

        # (0,2): 1-across[2] == 2-down[0] -> T == T
        assert result.assignment["1-across"][2] == result.assignment["2-down"][0]

        # (2,0): 3-across[0] == 1-down[2] -> B == B
        assert result.assignment["3-across"][0] == result.assignment["1-down"][2]

        # (2,2): 3-across[2] == 2-down[2] -> D == D
        assert result.assignment["3-across"][2] == result.assignment["2-down"][2]

    def test_statistics_recorded(self):
        """Solver should record meaningful statistics."""
        solver_input, candidates = get_tiny_3x3()
        result = solve_csp(solver_input, candidates)

        assert result.nodes_expanded > 0
        assert result.time_ms >= 0
        assert result.max_recursion_depth >= 1
        # CSP-specific metrics should be set
        assert result.prune_operations is not None
        assert result.domain_wipeouts is not None


class TestCSPSolverMini5x5:
    """Test CSP solver on mini 5x5 (actually 4x4) puzzle."""

    def test_solves_mini_puzzle(self):
        """CSP solver should find a valid solution for mini puzzle."""
        solver_input, candidates = get_mini_5x5()
        result = solve_csp(solver_input, candidates)

        assert result.solved is True
        # Check all clues are assigned
        assert len(result.assignment) == 8  # 4 across + 4 down
        assert all(clue_id in result.assignment for clue_id in solver_input.clue_cells)

    def test_all_crossings_valid(self):
        """All crossing cells should have matching letters."""
        solver_input, candidates = get_mini_5x5()
        result = solve_csp(solver_input, candidates)

        assert result.solved is True

        # Verify every cell that has multiple clues has matching letters
        for cell, clue_ids in solver_input.cell_to_clues.items():
            if len(clue_ids) < 2:
                continue

            letters = []
            for clue_id in clue_ids:
                word = result.assignment[clue_id]
                pos = solver_input.clue_cells[clue_id].index(cell)
                letters.append(word[pos])

            # All letters at this cell should match
            assert len(set(letters)) == 1, f"Mismatch at {cell}: {letters}"


class TestCSPSolverTIntersection:
    """Test CSP solver on T-intersection puzzle."""

    def test_solves_t_intersection(self):
        """CSP solver should find solution with matching first letter."""
        solver_input, candidates = get_t_intersection()
        result = solve_csp(solver_input, candidates)

        assert result.solved is True
        # Both clues share cell (0,1) which is position 0 for both
        assert result.assignment["1-across"][0] == result.assignment["1-down"][0]


class TestCSPSolverEdgeCases:
    """Test CSP solver edge cases."""

    def test_no_solution(self):
        """CSP solver should handle unsolvable puzzles."""
        # Create a puzzle with incompatible candidates
        clue_cells = {
            "1-across": [(0, 0), (0, 1)],
            "1-down": [(0, 0), (1, 0)],
        }
        cell_to_clues = {
            (0, 0): ["1-across", "1-down"],
            (0, 1): ["1-across"],
            (1, 0): ["1-down"],
        }
        solver_input = SolverInput(
            clue_cells=clue_cells,
            cell_to_clues=cell_to_clues,
            grid_size=(2, 2),
        )
        # Candidates that can't work together (no common first letter)
        candidates = {
            "1-across": ["AB", "AC"],  # Start with A
            "1-down": ["XY", "XZ"],  # Start with X - incompatible!
        }

        result = solve_csp(solver_input, candidates)

        assert result.solved is False
        assert result.assignment == {}

    def test_single_clue(self):
        """CSP solver should handle puzzle with single clue."""
        clue_cells = {"1-across": [(0, 0), (0, 1), (0, 2)]}
        cell_to_clues = {
            (0, 0): ["1-across"],
            (0, 1): ["1-across"],
            (0, 2): ["1-across"],
        }
        solver_input = SolverInput(
            clue_cells=clue_cells,
            cell_to_clues=cell_to_clues,
            grid_size=(1, 3),
        )
        candidates = {"1-across": ["CAT", "DOG", "BAT"]}

        result = solve_csp(solver_input, candidates)

        assert result.solved is True
        assert result.assignment["1-across"] in ["CAT", "DOG", "BAT"]

    def test_empty_domain(self):
        """CSP solver should handle empty candidate list gracefully."""
        clue_cells = {"1-across": [(0, 0), (0, 1)]}
        cell_to_clues = {(0, 0): ["1-across"], (0, 1): ["1-across"]}
        solver_input = SolverInput(
            clue_cells=clue_cells,
            cell_to_clues=cell_to_clues,
            grid_size=(1, 2),
        )
        candidates = {"1-across": []}  # Empty!

        result = solve_csp(solver_input, candidates)

        assert result.solved is False


class TestCSPSolverInternals:
    """Test CSP solver internal methods."""

    def test_precompute_crossings(self):
        """Crossing precomputation should find all intersections."""
        solver_input, candidates = get_tiny_3x3()
        solver = CSPSolver(solver_input, candidates)

        # 1-across crosses 1-down at (0,0) and 2-down at (0,2)
        crossings_1a = solver.crossings["1-across"]
        assert len(crossings_1a) == 2

        # Find the crossing with 1-down
        crossing_1d = next(c for c in crossings_1a if c[0] == "1-down")
        assert crossing_1d[1] == 0  # Position 0 in 1-across
        assert crossing_1d[2] == 0  # Position 0 in 1-down

    def test_mrv_selects_smallest_domain(self):
        """MRV heuristic should select variable with smallest domain."""
        clue_cells = {
            "1-across": [(0, 0), (0, 1)],
            "2-across": [(1, 0), (1, 1)],
        }
        cell_to_clues = {
            (0, 0): ["1-across"],
            (0, 1): ["1-across"],
            (1, 0): ["2-across"],
            (1, 1): ["2-across"],
        }
        solver_input = SolverInput(
            clue_cells=clue_cells,
            cell_to_clues=cell_to_clues,
            grid_size=(2, 2),
        )
        candidates = {
            "1-across": ["AB"],  # 1 candidate
            "2-across": ["CD", "EF", "GH"],  # 3 candidates
        }

        solver = CSPSolver(solver_input, candidates)
        selected = solver._select_unassigned_variable({})

        # Should select 1-across (smaller domain)
        assert selected == "1-across"

    def test_forward_check_prunes_domains(self):
        """Forward checking should prune inconsistent candidates."""
        solver_input, candidates = get_tiny_3x3()
        solver = CSPSolver(solver_input, candidates)

        # Assign CAT to 1-across
        assignment = {"1-across": "CAT"}
        pruned = solver._forward_check("1-across", "CAT", assignment)

        assert pruned is not None
        # 1-down should be pruned to only words starting with C
        # 2-down should be pruned to only words starting with T
        assert all(w.startswith("C") for w in solver.domains["1-down"])
        assert all(w.startswith("T") for w in solver.domains["2-down"])
