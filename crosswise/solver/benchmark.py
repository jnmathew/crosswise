"""
Benchmark harness for testing crossword solver.
"""

import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from crosswise.solver.models import SolverInput
from crosswise.solver.csp import solve_csp


ClueId = str
Word = str


@dataclass
class BenchmarkResult:
    """Result of benchmarking the solver on a puzzle."""

    solver_name: str
    puzzle_name: str
    times_ms: List[float] = field(default_factory=list)
    all_solved: bool = True
    correct: bool = True
    assignment: Dict[str, str] = field(default_factory=dict)

    # Aggregated metrics from last run
    nodes_expanded: int = 0
    backtracks: int = 0
    max_recursion_depth: int = 0

    # CSP-specific
    prune_operations: Optional[int] = None
    domain_wipeouts: Optional[int] = None

    @property
    def median_ms(self) -> float:
        """Median solve time in milliseconds."""
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def p95_ms(self) -> float:
        """95th percentile solve time in milliseconds."""
        if not self.times_ms:
            return 0.0
        sorted_times = sorted(self.times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def min_ms(self) -> float:
        """Minimum solve time in milliseconds."""
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum solve time in milliseconds."""
        return max(self.times_ms) if self.times_ms else 0.0


def verify_solution(solver_input: SolverInput, assignment: Dict[str, str]) -> bool:
    """
    Verify that a solution has no crossing conflicts.

    Args:
        solver_input: The puzzle structure
        assignment: The proposed solution (clue_id -> word)

    Returns:
        True if solution is valid (all crossings match), False otherwise.
    """
    if not assignment:
        return False

    # Check that all clues are assigned
    if set(assignment.keys()) != set(solver_input.clue_cells.keys()):
        return False

    # Check each cell with multiple clues
    for cell, clue_ids in solver_input.cell_to_clues.items():
        if len(clue_ids) < 2:
            continue

        letters = []
        for clue_id in clue_ids:
            word = assignment.get(clue_id)
            if word is None:
                return False

            cells = solver_input.clue_cells[clue_id]
            try:
                pos = cells.index(cell)
            except ValueError:
                return False

            if pos >= len(word):
                return False

            letters.append(word[pos])

        # All letters at this cell must match
        if len(set(letters)) != 1:
            return False

    return True


def run_benchmark(
    puzzle_name: str,
    solver_input: SolverInput,
    candidates: Dict[ClueId, List[Word]],
    runs: int = 5,
) -> BenchmarkResult:
    """
    Benchmark CSP solver on a puzzle.

    Args:
        puzzle_name: Name for the puzzle (for reporting)
        solver_input: The puzzle structure
        candidates: Candidate words for each clue
        runs: Number of runs (for timing statistics)

    Returns:
        BenchmarkResult with timing and metrics.
    """
    benchmark = BenchmarkResult(solver_name="csp", puzzle_name=puzzle_name)

    for _ in range(runs):
        result = solve_csp(solver_input, candidates)
        benchmark.times_ms.append(result.time_ms)

        if not result.solved:
            benchmark.all_solved = False

    # Run one more time to capture final result and metrics
    final_result = solve_csp(solver_input, candidates)
    benchmark.assignment = final_result.assignment
    benchmark.correct = verify_solution(solver_input, final_result.assignment)
    benchmark.nodes_expanded = final_result.nodes_expanded
    benchmark.backtracks = final_result.backtracks
    benchmark.max_recursion_depth = final_result.max_recursion_depth
    benchmark.prune_operations = final_result.prune_operations
    benchmark.domain_wipeouts = final_result.domain_wipeouts

    return benchmark


def print_comparison_table(results: List[BenchmarkResult]) -> None:
    """
    Print a table of benchmark results.

    Args:
        results: List of BenchmarkResult to display.
    """
    if not results:
        print("No results to display.")
        return

    # Header
    print("\n" + "=" * 80)
    print(f"{'Solver':<15} {'Puzzle':<12} {'Solved':<8} {'Correct':<8} "
          f"{'Median':<10} {'P95':<10} {'Nodes':<8}")
    print("=" * 80)

    for r in results:
        solved_str = "Yes" if r.all_solved else "No"
        correct_str = "Yes" if r.correct else "No"
        median_str = f"{r.median_ms:.3f}ms"
        p95_str = f"{r.p95_ms:.3f}ms"

        print(f"{r.solver_name:<15} {r.puzzle_name:<12} {solved_str:<8} {correct_str:<8} "
              f"{median_str:<10} {p95_str:<10} {r.nodes_expanded:<8}")

    print("=" * 80)

    # Additional metrics
    print("\nAdditional Metrics:")
    print("-" * 60)

    for r in results:
        print(f"\n{r.solver_name} on {r.puzzle_name}:")
        print(f"  Backtracks: {r.backtracks}")
        print(f"  Max Depth: {r.max_recursion_depth}")

        if r.prune_operations is not None:
            print(f"  Prune Operations: {r.prune_operations}")
            print(f"  Domain Wipeouts: {r.domain_wipeouts}")

    print()
