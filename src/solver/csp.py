"""
CSP solver for crossword puzzles.

Supports two propagation modes:
- Forward checking (legacy): prune immediate neighbors only
- MAC (Maintaining Arc Consistency): full transitive propagation via AC-3

And two backtracking strategies:
- Chronological: standard depth-first backtracking
- CDBJ (Conflict-Directed Backjumping): jump to causal variable on failure
"""

import time
from collections import deque
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
    - AC-3 preprocessing to prune initial domains
    - MAC (Maintaining Arc Consistency) during search
    - MRV (Minimum Remaining Values) heuristic for variable selection
    - Conflict-Directed Backjumping (CDBJ) to skip irrelevant variables
    """

    def __init__(
        self,
        solver_input: SolverInput,
        candidates_by_clue: Dict[ClueId, List[Word]],
        use_mac: bool = True,
        use_cdbj: bool = True,
        candidate_scores: Optional[Dict[ClueId, Dict[Word, float]]] = None,
        mac_mode: str = "full",
    ):
        self.solver_input = solver_input
        self.clue_cells = solver_input.clue_cells
        self.cell_to_clues = solver_input.cell_to_clues
        # mac_mode: "full" (AC-3 preprocessing + MAC), "search-only" (MAC during search, no AC-3 preprocessing), "off" (forward checking)
        self.mac_mode = mac_mode
        self.use_mac = use_mac if mac_mode != "off" else False
        self.use_cdbj = use_cdbj

        # Score map for value ordering: clue_id -> {word: score}
        self.candidate_scores = candidate_scores or {}

        # Store initial candidates for domain reset
        self.initial_candidates = candidates_by_clue

        # Initialize domains (copy candidates to allow pruning)
        self.domains: Dict[ClueId, Set[Word]] = {
            clue_id: set(words) for clue_id, words in candidates_by_clue.items()
        }

        # Track clues that started with empty domains (can't be solved in this pass)
        self.initially_empty: Set[ClueId] = {
            clue_id for clue_id in self.clue_cells
            if clue_id not in candidates_by_clue or not candidates_by_clue[clue_id]
        }

        # Precompute crossings between clues
        self.crossings = self._precompute_crossings()

        # Best partial solution found so far
        self.best_partial: Dict[ClueId, Word] = {}

        # Statistics
        self.nodes_expanded = 0
        self.backtracks = 0
        self.backjumps = 0
        self.max_depth = 0
        self.prune_operations = 0
        self.domain_wipeouts = 0
        self.ac3_pruned = 0

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
        Select next variable using modified MRV heuristic.

        Prefers clues with 2+ candidates over single-candidate clues,
        since single-candidate clues have no fallback if they fail.

        Returns:
            Unassigned clue_id with smallest domain (preferring 2+), or None if all assigned/empty.
        """
        best_clue = None
        best_size = float("inf")
        # Fallback for single-candidate clues
        single_candidate_clue = None

        for clue_id in self.clue_cells:
            if clue_id in assignment:
                continue

            domain_size = len(self.domains[clue_id])
            # Skip empty domains - they can't be assigned yet
            if domain_size == 0:
                continue

            if domain_size == 1:
                # Save single-candidate clues for last
                if single_candidate_clue is None:
                    single_candidate_clue = clue_id
            elif domain_size < best_size:
                best_size = domain_size
                best_clue = clue_id

        # Prefer 2+ candidates, fall back to single-candidate
        return best_clue if best_clue else single_candidate_clue

    def _sorted_domain(self, clue_id: ClueId) -> List[Word]:
        """Return domain values sorted by candidate score (highest first).

        Value ordering heuristic: try high-confidence candidates before low-confidence ones.
        Falls back to arbitrary order if no scores available.
        """
        domain = list(self.domains[clue_id])
        scores = self.candidate_scores.get(clue_id)
        if scores:
            domain.sort(key=lambda w: scores.get(w, 0.0), reverse=True)
        return domain

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
            # Skip clues with no candidates - they can't be assigned anyway
            if other_id in self.initially_empty:
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

    def _restore_saved_domains(self, saved: Dict[ClueId, Set[Word]]) -> None:
        """Restore domains from full snapshots (used by MAC)."""
        for clue_id, domain in saved.items():
            self.domains[clue_id] = domain

    # ── AC-3 and MAC ──────────────────────────────────────────────

    def _revise(self, xi: ClueId, xj: ClueId, pos_i: int, pos_j: int) -> bool:
        """
        Remove values from xi's domain that have no support in xj's domain.

        Returns True if any value was removed.
        """
        # Build set of letters at pos_j in xj's domain for fast lookup
        supported_letters = {w[pos_j] for w in self.domains[xj]}

        to_remove = set()
        for word_i in self.domains[xi]:
            if word_i[pos_i] not in supported_letters:
                to_remove.add(word_i)
                self.prune_operations += 1

        if to_remove:
            self.domains[xi] -= to_remove
            return True
        return False

    def _ac3(
        self,
        arcs: Optional[List[Tuple[ClueId, ClueId, int, int]]] = None,
    ) -> bool:
        """
        Arc Consistency 3 algorithm.

        Propagates constraints across the full crossing graph. For every pair
        of intersecting clues (Xi, Xj), removes any value from Xi's domain
        that has no compatible value in Xj's domain.

        Args:
            arcs: Optional initial arc queue. If None, initializes with all arcs.
                  Each arc is (clue_i, clue_j, pos_i, pos_j).

        Returns:
            True if all domains are non-empty after propagation.
            False if any domain was wiped out (inconsistency detected).
        """
        if arcs is None:
            queue = deque()
            for clue_id, crossings_list in self.crossings.items():
                if clue_id in self.initially_empty:
                    continue
                for other_id, self_pos, other_pos in crossings_list:
                    if other_id in self.initially_empty:
                        continue
                    queue.append((clue_id, other_id, self_pos, other_pos))
        else:
            queue = deque(arcs)

        while queue:
            xi, xj, pos_i, pos_j = queue.popleft()

            if self._revise(xi, xj, pos_i, pos_j):
                if len(self.domains[xi]) == 0:
                    self.domain_wipeouts += 1
                    return False

                # Add all arcs (xk, xi) where xk != xj
                for other_id, xi_pos, other_pos in self.crossings[xi]:
                    if other_id != xj and other_id not in self.initially_empty:
                        queue.append((other_id, xi, other_pos, xi_pos))

        return True

    def _mac(
        self,
        clue_id: ClueId,
        word: Word,
        assignment: Dict[ClueId, Word],
    ) -> Optional[Dict[ClueId, Set[Word]]]:
        """
        Maintaining Arc Consistency after assigning clue_id = word.

        Prunes immediate neighbors, then runs AC-3 on affected arcs
        to propagate transitively.

        Returns:
            Dict of saved domain snapshots (for undo), or None if wipeout.
        """
        saved: Dict[ClueId, Set[Word]] = {}
        affected_arcs: List[Tuple[ClueId, ClueId, int, int]] = []

        for other_id, self_pos, other_pos in self.crossings[clue_id]:
            if other_id in assignment or other_id in self.initially_empty:
                continue

            # Save domain before pruning
            saved[other_id] = self.domains[other_id].copy()

            # Immediate pruning: remove candidates inconsistent with placed word
            required_letter = word[self_pos]
            new_domain = set()
            for w in self.domains[other_id]:
                if w[other_pos] == required_letter:
                    new_domain.add(w)
                else:
                    self.prune_operations += 1

            self.domains[other_id] = new_domain

            if len(new_domain) == 0:
                self.domain_wipeouts += 1
                self._restore_saved_domains(saved)
                return None

            # Collect arcs for transitive propagation
            for xk_id, other_pos2, xk_pos in self.crossings[other_id]:
                if xk_id != clue_id and xk_id not in assignment and xk_id not in self.initially_empty:
                    affected_arcs.append((xk_id, other_id, xk_pos, other_pos2))

        # Transitive propagation via AC-3
        if affected_arcs:
            # Save domains of variables that might be affected
            for arc in affected_arcs:
                xk = arc[0]
                if xk not in saved:
                    saved[xk] = self.domains[xk].copy()

            if not self._ac3(affected_arcs):
                self._restore_saved_domains(saved)
                return None

        return saved

    # ── Backtracking ──────────────────────────────────────────────

    def _propagate(
        self,
        clue_id: ClueId,
        word: Word,
        assignment: Dict[ClueId, Word],
    ) -> Optional[Dict[ClueId, Set[Word]]]:
        """Dispatch to MAC or forward checking based on configuration."""
        if self.use_mac:
            return self._mac(clue_id, word, assignment)
        return self._forward_check(clue_id, word, assignment)

    def _undo_propagation(self, saved: Dict[ClueId, Set[Word]]) -> None:
        """Undo propagation by restoring saved domains."""
        if self.use_mac:
            self._restore_saved_domains(saved)
        else:
            self._restore_domains(saved)

    def _backtrack_with_restarts(
        self, assignment: Dict[ClueId, Word], depth: int
    ) -> Optional[Dict[ClueId, Word]]:
        """
        Backtracking with restarts at root level.

        If the first starting clue fails, try other starting clues instead of giving up.
        """
        # Get all assignable clues sorted by domain size (smallest first, but 2+ before 1)
        candidates = []
        for clue_id in self.clue_cells:
            if clue_id in self.initially_empty:
                continue
            size = len(self.domains[clue_id])
            if size > 0:
                # Sort key: (is_single, size) so 2+ come before 1s
                candidates.append((size == 1, size, clue_id))
        candidates.sort()

        max_restarts = 50  # Cap restarts to avoid infinite loops
        for i, (_, _, start_clue) in enumerate(candidates):
            if i >= max_restarts:
                print(f"  Tried {max_restarts} starting points, best partial: {len(self.best_partial)} clues")
                break
            # Reset domains for fresh start
            self._reset_domains()
            # Run AC-3 preprocessing only in "full" MAC mode
            if self.use_mac and self.mac_mode == "full":
                if not self._ac3():
                    continue  # Inconsistent from this starting config
            if self.use_cdbj:
                result, _ = self._backtrack_cdbj(start_clue, {}, 0)
            else:
                result = self._backtrack_from(start_clue, {}, 0)
            if result is not None:
                return result
            # Track that this start didn't work
            self.tried_starts.add(start_clue)

        return None

    def _reset_domains(self) -> None:
        """Reset domains to initial state."""
        self.domains = {
            clue_id: set(words)
            for clue_id, words in self.initial_candidates.items()
        }

    def _backtrack_from(
        self, forced_start: Optional[ClueId], assignment: Dict[ClueId, Word], depth: int
    ) -> Optional[Dict[ClueId, Word]]:
        """Backtracking with optional forced starting clue (chronological)."""
        self.max_depth = max(self.max_depth, depth)

        if len(assignment) > len(self.best_partial):
            self.best_partial = assignment.copy()

        assignable = len(self.clue_cells) - len(self.initially_empty)
        if len(assignment) == assignable:
            return assignment

        # Use forced start at depth 0, otherwise MRV
        if depth == 0 and forced_start:
            clue_id = forced_start
        else:
            clue_id = self._select_unassigned_variable(assignment)

        if clue_id is None:
            return None

        self.nodes_expanded += 1

        for word in self._sorted_domain(clue_id):
            if not self._is_consistent(clue_id, word, assignment):
                continue

            assignment[clue_id] = word
            saved = self._propagate(clue_id, word, assignment)

            if saved is not None:
                result = self._backtrack_from(None, assignment, depth + 1)
                if result is not None:
                    return result
                self._undo_propagation(saved)

            del assignment[clue_id]
            self.backtracks += 1

        return None

    # ── Conflict-Directed Backjumping ─────────────────────────────

    def _backtrack_cdbj(
        self,
        forced_start: Optional[ClueId],
        assignment: Dict[ClueId, Word],
        depth: int,
    ) -> Tuple[Optional[Dict[ClueId, Word]], Set[ClueId]]:
        """
        Backtracking with conflict-directed backjumping.

        Returns:
            (solution, conflict_set) -- solution if found, otherwise the
            conflict set that caused failure (for the caller to decide
            whether to continue or backjump).
        """
        self.max_depth = max(self.max_depth, depth)

        if len(assignment) > len(self.best_partial):
            self.best_partial = assignment.copy()

        assignable = len(self.clue_cells) - len(self.initially_empty)
        if len(assignment) == assignable:
            return assignment, set()

        if depth == 0 and forced_start:
            clue_id = forced_start
        else:
            clue_id = self._select_unassigned_variable(assignment)

        if clue_id is None:
            # Dead end -- conflict is with all currently assigned crossing vars
            return None, self._crossing_assigned(clue_id or "", assignment)

        self.nodes_expanded += 1

        # Accumulate conflict set across all failed values
        combined_conflict: Set[ClueId] = set()

        for word in self._sorted_domain(clue_id):
            if not self._is_consistent(clue_id, word, assignment):
                # Conflict with whichever assigned neighbors mismatch
                for other_id, self_pos, other_pos in self.crossings[clue_id]:
                    if other_id in assignment and word[self_pos] != assignment[other_id][other_pos]:
                        combined_conflict.add(other_id)
                continue

            assignment[clue_id] = word
            saved = self._propagate(clue_id, word, assignment)

            if saved is not None:
                result, child_conflict = self._backtrack_cdbj(None, assignment, depth + 1)
                if result is not None:
                    return result, set()

                self._undo_propagation(saved)

                # Absorb child's conflict set
                combined_conflict |= child_conflict

                # Backjump: if current variable is NOT in child's conflict set,
                # skip remaining values and propagate conflict upward
                if clue_id not in child_conflict:
                    del assignment[clue_id]
                    self.backjumps += 1
                    return None, combined_conflict
            else:
                # Wipeout -- conflict with assigned crossing neighbors
                combined_conflict |= self._crossing_assigned(clue_id, assignment)

            del assignment[clue_id]
            self.backtracks += 1

        # All values exhausted
        # Remove self from conflict set before passing up
        combined_conflict.discard(clue_id)
        return None, combined_conflict

    def _crossing_assigned(
        self, clue_id: ClueId, assignment: Dict[ClueId, Word]
    ) -> Set[ClueId]:
        """Return the set of assigned variables that cross clue_id."""
        result = set()
        for other_id, _, _ in self.crossings.get(clue_id, []):
            if other_id in assignment:
                result.add(other_id)
        return result

    def solve_greedy(self) -> SolveResult:
        """
        Greedy solver - assign clues without backtracking.

        Picks clues in MRV order and assigns first consistent candidate.
        Fast but may miss solutions that require backtracking.
        """
        start_time = time.perf_counter()
        assignment: Dict[ClueId, Word] = {}

        # Keep trying until no more progress
        made_progress = True
        while made_progress:
            made_progress = False
            for clue_id in self.clue_cells:
                if clue_id in assignment or clue_id in self.initially_empty:
                    continue
                if len(self.domains[clue_id]) == 0:
                    continue

                # Try each candidate (sorted by score for value ordering)
                for word in self._sorted_domain(clue_id):
                    if self._is_consistent(clue_id, word, assignment):
                        assignment[clue_id] = word
                        self.nodes_expanded += 1
                        # Prune crossing domains
                        saved = self._propagate(clue_id, word, assignment)
                        if saved is None:
                            # Wipeout - undo and try next candidate
                            del assignment[clue_id]
                            continue
                        made_progress = True
                        break
                else:
                    # No candidate worked for this clue, skip it
                    continue

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        assignable = len(self.clue_cells) - len(self.initially_empty)

        return SolveResult(
            solved=len(assignment) == assignable,
            assignment=assignment,
            time_ms=elapsed_ms,
            nodes_expanded=self.nodes_expanded,
            backtracks=0,
            max_recursion_depth=0,
            prune_operations=self.prune_operations,
            domain_wipeouts=self.domain_wipeouts,
            arc_consistency_pruned=self.ac3_pruned,
            backjumps=0,
        )

    def _apply_seed(self, seed: Dict[ClueId, Word]) -> Dict[ClueId, Word]:
        """
        Apply a seed assignment: validate each seed entry against current domains,
        pre-assign valid ones, and propagate constraints.

        Returns the validated seed assignment (entries with missing domain words are skipped).
        """
        assignment: Dict[ClueId, Word] = {}
        for clue_id, word in seed.items():
            if clue_id not in self.domains:
                continue
            # Ensure the word is still in the domain
            if word not in self.domains[clue_id]:
                continue
            # Ensure consistency with already-applied seed entries
            if not self._is_consistent(clue_id, word, assignment):
                continue
            assignment[clue_id] = word
            # Propagate constraints (prune crossing domains)
            saved = self._propagate(clue_id, word, assignment)
            if saved is None:
                # Wipeout -- this seed entry causes problems, undo
                del assignment[clue_id]

        return assignment

    def solve(
        self,
        return_partial: bool = False,
        seed_assignment: Optional[Dict[ClueId, Word]] = None,
    ) -> SolveResult:
        """
        Solve the crossword puzzle.

        Args:
            return_partial: If True, return best partial solution when full solve fails
            seed_assignment: Optional partial assignment to seed greedy phase

        Returns:
            SolveResult with solution and statistics.
        """
        start_time = time.perf_counter()

        # AC-3 preprocessing to prune initial domains (only in "full" mode)
        if self.use_mac and self.mac_mode == "full":
            prune_before = self.prune_operations
            ac3_ok = self._ac3()
            self.ac3_pruned = self.prune_operations - prune_before
            if not ac3_ok:
                # Inconsistency detected before search
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return SolveResult(
                    solved=False,
                    assignment={},
                    time_ms=elapsed_ms,
                    nodes_expanded=0,
                    backtracks=0,
                    max_recursion_depth=0,
                    prune_operations=self.prune_operations,
                    domain_wipeouts=self.domain_wipeouts,
                    arc_consistency_pruned=self.ac3_pruned,
                )

        # If seeded, use seed as warm start for greedy (don't skip greedy)
        if seed_assignment:
            validated_seed = self._apply_seed(seed_assignment)
            if validated_seed:
                self.best_partial = validated_seed.copy()

                assignable = len(self.clue_cells) - len(self.initially_empty)
                if len(validated_seed) == assignable:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    return SolveResult(
                        solved=True,
                        assignment=validated_seed,
                        time_ms=elapsed_ms,
                        nodes_expanded=0,
                        backtracks=0,
                        max_recursion_depth=0,
                        prune_operations=self.prune_operations,
                        domain_wipeouts=self.domain_wipeouts,
                        arc_consistency_pruned=0,
                        backjumps=0,
                    )

            # Reset domains after seed validation (seed was applied to check validity)
            self._reset_domains()

        # Try greedy first - fast and often good enough (50 random starts)
        greedy_result = self.solve_greedy()
        if greedy_result.solved:
            return greedy_result

        # Use greedy result as best partial (compare with seed if available)
        if len(greedy_result.assignment) >= len(self.best_partial):
            self.best_partial = greedy_result.assignment.copy()

        # Reset domains and try backtracking for full solution
        self._reset_domains()
        self.tried_starts = set()
        result = self._backtrack_with_restarts({}, 0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Use best partial if full solve failed and partial requested
        assignment = result if result else (self.best_partial if return_partial else {})

        return SolveResult(
            solved=result is not None,
            assignment=assignment,
            time_ms=elapsed_ms,
            nodes_expanded=self.nodes_expanded,
            backtracks=self.backtracks,
            max_recursion_depth=self.max_depth,
            prune_operations=self.prune_operations,
            domain_wipeouts=self.domain_wipeouts,
            arc_consistency_pruned=self.ac3_pruned,
            backjumps=self.backjumps,
        )


def solve_csp(
    solver_input: SolverInput,
    candidates_by_clue: Dict[ClueId, List[Word]],
    return_partial: bool = False,
    use_mac: bool = True,
    use_cdbj: bool = True,
    candidate_scores: Optional[Dict[ClueId, Dict[Word, float]]] = None,
    seed_assignment: Optional[Dict[ClueId, Word]] = None,
    mac_mode: str = "full",
) -> SolveResult:
    """
    Solve a crossword puzzle using CSP.

    Args:
        solver_input: The puzzle structure (clue cells and crossings)
        candidates_by_clue: Candidate words for each clue
        return_partial: If True, return best partial solution when full solve fails
        use_mac: Use MAC (AC-3 based) instead of forward checking
        use_cdbj: Use conflict-directed backjumping instead of chronological
        candidate_scores: Optional score map for value ordering (clue_id -> {word: score})
        seed_assignment: Optional partial assignment from previous pass to build on
        mac_mode: "full" (AC-3 + MAC), "search-only" (MAC during search only), "off" (FC only)

    Returns:
        SolveResult with solution and statistics.
    """
    solver = CSPSolver(
        solver_input, candidates_by_clue,
        use_mac=use_mac, use_cdbj=use_cdbj,
        candidate_scores=candidate_scores,
        mac_mode=mac_mode,
    )
    return solver.solve(return_partial=return_partial, seed_assignment=seed_assignment)


def extract_letter_patterns(
    solver_input: SolverInput,
    assignment: Dict[ClueId, Word],
    score_map: Optional[Dict[ClueId, Dict[Word, float]]] = None,
    min_confidence: float = 0.0,
) -> Dict[ClueId, str]:
    """
    Extract letter patterns for unsolved clues based on crossing letters.

    For each unsolved clue, creates a pattern like "C_T" where known letters
    are filled in from crossing clues and unknowns are '_'.

    When score_map and min_confidence are provided, only uses crossing letters
    from assignments where the chosen word has confidence >= min_confidence.
    This prevents poisoned crossings from wrong answers feeding bad patterns.

    Args:
        solver_input: The puzzle structure
        assignment: Partial assignment from solving
        score_map: Optional confidence scores {clue_id: {word: score}}
        min_confidence: Minimum confidence to trust a crossing (default: 0.0 = trust all)

    Returns:
        Dict mapping unsolved clue_id to pattern string
    """
    patterns: Dict[ClueId, str] = {}

    # Build a grid of known letters from assignment
    # Only include letters from high-confidence assignments when filtering is active
    grid_letters: Dict[CellPos, str] = {}
    for clue_id, word in assignment.items():
        if score_map and min_confidence > 0.0:
            word_score = score_map.get(clue_id, {}).get(word, 0.0)
            if word_score < min_confidence:
                continue  # Don't trust this assignment's crossing letters
        cells = solver_input.clue_cells[clue_id]
        for i, cell in enumerate(cells):
            grid_letters[cell] = word[i]

    # For each unsolved clue, build pattern from crossing letters
    for clue_id, cells in solver_input.clue_cells.items():
        if clue_id in assignment:
            continue

        pattern = ""
        for cell in cells:
            if cell in grid_letters:
                pattern += grid_letters[cell]
            else:
                pattern += "_"

        patterns[clue_id] = pattern

    return patterns
