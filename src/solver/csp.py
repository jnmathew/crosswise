"""
CSP solver for crossword puzzles using backtracking with forward checking and MRV heuristic.
"""

import time
from typing import Dict, List, Tuple, Optional, Set
from src.solver.models import SolverInput, SolveResult


# Type aliases for clarity
ClueId = str
Word = str
CellPos = Tuple[int, int]


class CSPSolver:
    """
    Constraint Satisfaction Problem solver for crosswords.

    Uses backtracking search with:
    - Forward checking: prune domains of neighboring variables after assignment
    - MRV (Minimum Remaining Values) heuristic: select variable with smallest domain
    """

    def __init__(
        self,
        solver_input: SolverInput,
        candidates_by_clue: Dict[ClueId, List[Word]],
    ):
        self.solver_input = solver_input
        self.clue_cells = solver_input.clue_cells
        self.cell_to_clues = solver_input.cell_to_clues

        # Initialize domains (copy candidates to allow pruning)
        self.domains: Dict[ClueId, Set[Word]] = {
            clue_id: set(words) for clue_id, words in candidates_by_clue.items()
        }

        # Precompute crossings between clues
        self.crossings = self._precompute_crossings()

        # Statistics
        self.nodes_expanded = 0
        self.backtracks = 0
        self.max_depth = 0
        self.prune_operations = 0
        self.domain_wipeouts = 0

    def _precompute_crossings(self) -> Dict[ClueId, List[Tuple[ClueId, int, int]]]:
        """
        Build a map from each clue to its crossing clues.

        Returns:
            Dict mapping clue_id -> list of (other_clue_id, self_position, other_position)
            where position is the index within the clue's cells where they intersect.
        """
        crossings: Dict[ClueId, List[Tuple[ClueId, int, int]]] = {
            clue_id: [] for clue_id in self.clue_cells
        }

        for cell, clue_ids in self.cell_to_clues.items():
            if len(clue_ids) < 2:
                continue

            # For each pair of clues that share this cell
            for i, clue_a in enumerate(clue_ids):
                for clue_b in clue_ids[i + 1 :]:
                    # Find position of cell in each clue
                    pos_a = self.clue_cells[clue_a].index(cell)
                    pos_b = self.clue_cells[clue_b].index(cell)

                    crossings[clue_a].append((clue_b, pos_a, pos_b))
                    crossings[clue_b].append((clue_a, pos_b, pos_a))

        return crossings

    def _select_unassigned_variable(
        self, assignment: Dict[ClueId, Word]
    ) -> Optional[ClueId]:
        """
        Select next variable using MRV (Minimum Remaining Values) heuristic.

        Returns:
            Unassigned clue_id with smallest domain, or None if all assigned.
        """
        min_size = float("inf")
        best_clue = None

        for clue_id in self.clue_cells:
            if clue_id in assignment:
                continue

            domain_size = len(self.domains[clue_id])
            if domain_size < min_size:
                min_size = domain_size
                best_clue = clue_id

        return best_clue

    def _is_consistent(
        self, clue_id: ClueId, word: Word, assignment: Dict[ClueId, Word]
    ) -> bool:
        """
        Check if assigning word to clue_id is consistent with current assignment.

        Args:
            clue_id: The clue to assign
            word: The candidate word
            assignment: Current partial assignment

        Returns:
            True if consistent (letters match at all crossing cells)
        """
        for other_id, self_pos, other_pos in self.crossings[clue_id]:
            if other_id not in assignment:
                continue

            other_word = assignment[other_id]
            if word[self_pos] != other_word[other_pos]:
                return False

        return True

    def _forward_check(
        self, clue_id: ClueId, word: Word, assignment: Dict[ClueId, Word]
    ) -> Optional[Dict[ClueId, Set[Word]]]:
        """
        Prune domains of neighboring clues based on new assignment.

        Args:
            clue_id: The clue just assigned
            word: The word assigned to it
            assignment: Current assignment (including clue_id -> word)

        Returns:
            Dict of pruned values (for undo), or None if a domain was wiped out.
        """
        pruned: Dict[ClueId, Set[Word]] = {}

        for other_id, self_pos, other_pos in self.crossings[clue_id]:
            if other_id in assignment:
                continue

            required_letter = word[self_pos]
            to_remove = set()

            for candidate in self.domains[other_id]:
                if candidate[other_pos] != required_letter:
                    to_remove.add(candidate)
                    self.prune_operations += 1

            if to_remove:
                if len(to_remove) == len(self.domains[other_id]):
                    # Domain wipeout - restore already-pruned domains before failing
                    self.domain_wipeouts += 1
                    self._restore_domains(pruned)
                    return None

                pruned[other_id] = to_remove
                self.domains[other_id] -= to_remove

        return pruned

    def _restore_domains(self, pruned: Dict[ClueId, Set[Word]]) -> None:
        """Restore pruned values to domains."""
        for clue_id, words in pruned.items():
            self.domains[clue_id] |= words

    def _backtrack(
        self, assignment: Dict[ClueId, Word], depth: int
    ) -> Optional[Dict[ClueId, Word]]:
        """
        Recursive backtracking search.

        Args:
            assignment: Current partial assignment
            depth: Current recursion depth

        Returns:
            Complete assignment if solution found, None otherwise.
        """
        self.max_depth = max(self.max_depth, depth)

        # Check if complete
        if len(assignment) == len(self.clue_cells):
            return assignment

        # Select next variable (MRV)
        clue_id = self._select_unassigned_variable(assignment)
        if clue_id is None:
            return None

        self.nodes_expanded += 1

        # Try each value in domain
        for word in list(self.domains[clue_id]):
            if not self._is_consistent(clue_id, word, assignment):
                continue

            # Assign
            assignment[clue_id] = word

            # Forward check
            pruned = self._forward_check(clue_id, word, assignment)

            if pruned is not None:
                # No wipeout, recurse
                result = self._backtrack(assignment, depth + 1)
                if result is not None:
                    return result

                # Restore domains
                self._restore_domains(pruned)

            # Unassign
            del assignment[clue_id]
            self.backtracks += 1

        return None

    def solve(self) -> SolveResult:
        """
        Solve the crossword puzzle.

        Returns:
            SolveResult with solution and statistics.
        """
        start_time = time.perf_counter()

        result = self._backtrack({}, 0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SolveResult(
            solved=result is not None,
            assignment=result if result else {},
            time_ms=elapsed_ms,
            nodes_expanded=self.nodes_expanded,
            backtracks=self.backtracks,
            max_recursion_depth=self.max_depth,
            prune_operations=self.prune_operations,
            domain_wipeouts=self.domain_wipeouts,
        )


def solve_csp(
    solver_input: SolverInput,
    candidates_by_clue: Dict[ClueId, List[Word]],
) -> SolveResult:
    """
    Solve a crossword puzzle using CSP with forward checking.

    Args:
        solver_input: The puzzle structure (clue cells and crossings)
        candidates_by_clue: Candidate words for each clue

    Returns:
        SolveResult with solution and statistics.
    """
    solver = CSPSolver(solver_input, candidates_by_clue)
    return solver.solve()
