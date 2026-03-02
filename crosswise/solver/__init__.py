"""
Crossword solver modules.

This package provides a CSP solver and benchmark harness:
- CSP solver with forward checking and MRV heuristic
- Benchmark harness for testing solver performance

Example usage:
    from crosswise.solver import get_tiny_3x3, solve_csp

    solver_input, candidates = get_tiny_3x3()
    result = solve_csp(solver_input, candidates)
    print(result.assignment)

    # Or run benchmark
    from crosswise.solver import run_benchmark, print_comparison_table
    results = run_benchmark("tiny", solver_input, candidates)
    print_comparison_table(results)
"""

# Models
from crosswise.solver.models import SolverInput, SolveResult

# Solvers
from crosswise.solver.csp import solve_csp, CSPSolver

# Benchmark
from crosswise.solver.benchmark import (
    run_benchmark,
    verify_solution,
    print_comparison_table,
    BenchmarkResult,
)

# Toy puzzles
from crosswise.solver.puzzles import get_tiny_3x3, get_mini_5x5, get_t_intersection

# Candidate generation
from crosswise.solver.candidates import (
    generate_with_extended_thinking,
    ClueInput,
    CandidateResult,
    ScoredCandidate,
    to_plain_candidates,
    to_score_map,
)

# Word index
from crosswise.solver.word_index import WordIndex

__all__ = [
    # Models
    "SolverInput",
    "SolveResult",
    # Solvers
    "solve_csp",
    "CSPSolver",
    # Benchmark
    "run_benchmark",
    "verify_solution",
    "print_comparison_table",
    "BenchmarkResult",
    # Puzzles
    "get_tiny_3x3",
    "get_mini_5x5",
    "get_t_intersection",
    # Candidate generation
    "ClueInput",
    "CandidateResult",
    "ScoredCandidate",
    "to_plain_candidates",
    "to_score_map",
    "generate_with_extended_thinking",
    # Word index
    "WordIndex",
]
