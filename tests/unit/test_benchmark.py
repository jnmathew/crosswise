"""Tests for benchmark harness."""

from crosswise.solver.benchmark import run_benchmark, verify_solution, print_comparison_table
from crosswise.solver.puzzles import get_tiny_3x3, get_mini_5x5, get_t_intersection


class TestVerifySolution:
    """Test solution verification."""

    def test_valid_solution(self):
        """Valid solution should pass verification."""
        solver_input, _ = get_tiny_3x3()
        assignment = {
            "1-across": "CAT",
            "3-across": "BAD",
            "1-down": "CAB",
            "2-down": "TAD",
        }

        assert verify_solution(solver_input, assignment) is True

    def test_crossing_mismatch(self):
        """Solution with crossing mismatch should fail."""
        solver_input, _ = get_tiny_3x3()
        # CAT and DOG don't share first letter
        assignment = {
            "1-across": "CAT",
            "3-across": "BAD",
            "1-down": "DOG",  # D != C at (0,0)
            "2-down": "TAD",
        }

        assert verify_solution(solver_input, assignment) is False

    def test_missing_clue(self):
        """Solution missing a clue should fail."""
        solver_input, _ = get_tiny_3x3()
        assignment = {
            "1-across": "CAT",
            "3-across": "BAD",
            "1-down": "CAB",
            # Missing 2-down
        }

        assert verify_solution(solver_input, assignment) is False

    def test_empty_assignment(self):
        """Empty assignment should fail."""
        solver_input, _ = get_tiny_3x3()
        assert verify_solution(solver_input, {}) is False

    def test_extra_clue(self):
        """Solution with extra clue should fail."""
        solver_input, _ = get_tiny_3x3()
        assignment = {
            "1-across": "CAT",
            "3-across": "BAD",
            "1-down": "CAB",
            "2-down": "TAD",
            "99-across": "EXTRA",  # Doesn't exist
        }

        assert verify_solution(solver_input, assignment) is False


class TestRunBenchmark:
    """Test benchmark runner."""

    def test_runs_csp_solver(self):
        """Benchmark should run CSP solver."""
        solver_input, candidates = get_tiny_3x3()
        result = run_benchmark("tiny", solver_input, candidates, runs=2)

        assert result.solver_name == "csp"

    def test_collects_timing_data(self):
        """Benchmark should collect timing data for each run."""
        solver_input, candidates = get_tiny_3x3()
        result = run_benchmark("tiny", solver_input, candidates, runs=3)

        assert len(result.times_ms) == 3
        assert all(t >= 0 for t in result.times_ms)

    def test_verifies_solutions(self):
        """Benchmark should verify solutions."""
        solver_input, candidates = get_tiny_3x3()
        result = run_benchmark("tiny", solver_input, candidates, runs=1)

        assert result.all_solved is True
        assert result.correct is True

    def test_timing_statistics(self):
        """Benchmark should compute timing statistics."""
        solver_input, candidates = get_tiny_3x3()
        result = run_benchmark("tiny", solver_input, candidates, runs=5)

        assert result.median_ms >= 0
        assert result.p95_ms >= 0
        assert result.min_ms <= result.median_ms
        assert result.median_ms <= result.max_ms

    def test_captures_metrics(self):
        """Benchmark should capture CSP-specific metrics."""
        solver_input, candidates = get_tiny_3x3()
        result = run_benchmark("tiny", solver_input, candidates, runs=1)

        assert result.prune_operations is not None
        assert result.domain_wipeouts is not None


class TestRunBenchmarkOnDifferentPuzzles:
    """Test benchmark on different puzzle types."""

    def test_mini_5x5(self):
        """Benchmark should work on mini 5x5 puzzle."""
        solver_input, candidates = get_mini_5x5()
        result = run_benchmark("mini", solver_input, candidates, runs=1)

        assert result.all_solved is True
        assert result.correct is True

    def test_t_intersection(self):
        """Benchmark should work on T-intersection puzzle."""
        solver_input, candidates = get_t_intersection()
        result = run_benchmark("t-int", solver_input, candidates, runs=1)

        assert result.all_solved is True
        assert result.correct is True


class TestPrintComparisonTable:
    """Test comparison table printing."""

    def test_prints_without_error(self, capsys):
        """Print function should execute without error."""
        solver_input, candidates = get_tiny_3x3()
        result = run_benchmark("tiny", solver_input, candidates, runs=1)

        # Should not raise
        print_comparison_table([result])

        captured = capsys.readouterr()
        assert "csp" in captured.out
        assert "tiny" in captured.out

    def test_prints_empty_results(self, capsys):
        """Print function should handle empty results."""
        print_comparison_table([])

        captured = capsys.readouterr()
        assert "No results" in captured.out
