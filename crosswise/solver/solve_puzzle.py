"""
Crossword puzzle solver using multi-pass CSP with LLM-generated candidates.

Supports TSV/SQLite database lookup as primary candidate source with LLM fallback.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger

from crosswise.solver.models import SolverInput
from crosswise.solver.candidates import (
    ClueInput,
    ScoredCandidate,
    generate_candidates,
    generate_candidates_batch,
    generate_candidates_with_database,
    regenerate_with_patterns,
    ensure_minimum_candidates,
    bouncer_filter,
    categorize_clue,
    compute_target_domain_size,
    to_plain_candidates,
    to_score_map,
    sniper_escalation,
    generate_synonyms_batch,
    generate_with_extended_thinking,
    DEFAULT_MIN_CANDIDATES,
)
from crosswise.solver.csp import solve_csp, extract_letter_patterns
from crosswise.solver.benchmark import verify_solution


def load_puzzle_json(puzzle_path: str) -> Dict[str, Any]:
    """Load raw puzzle JSON data."""
    with open(puzzle_path) as f:
        return json.load(f)


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

    # Process across clues
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

    # Process down clues
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

    # Build cell_to_clues mapping
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
) -> List[ClueInput]:
    """Build ClueInput list from puzzle JSON for candidate generation."""
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


def detect_theme(
    puzzle_data: Dict[str, Any],
    assignment: Dict[str, str],
) -> Optional[str]:
    """
    Lightweight theme detection.

    1. Identify theme-position entries (longest Across answers, need >= 2)
    2. After first theme answer is placed, probe for pattern
    3. Return pattern string or None

    Uses a single cheap LLM call (Sonnet). Cost: ~$0.001.
    """
    across_clues = puzzle_data["clues"].get("across", [])
    if not across_clues:
        return None

    max_length = max(c["length"] for c in across_clues)
    # Theme answers are typically the longest entries
    theme_clues = [c for c in across_clues if c["length"] == max_length]

    if len(theme_clues) < 2:
        return None

    # Check if any theme answer is already placed
    placed_theme = None
    for tc in theme_clues:
        clue_id = f"{tc['number']}-across"
        if clue_id in assignment:
            placed_theme = (tc, assignment[clue_id])
            break

    if not placed_theme:
        return None

    tc, answer = placed_theme
    other_theme_clues = [t for t in theme_clues if t["number"] != tc["number"]]

    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    others_text = "\n".join(
        f'- "{t["text"]}" ({t["length"]} letters)' for t in other_theme_clues
    )

    prompt = f"""The answer to "{tc['text']}" is {answer}.
The other theme clues are:
{others_text}

Do these share a structural pattern (shared substring, hidden word, common transformation)?
If yes, state the pattern concisely in one sentence. If unclear, say NONE."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        result_text = response.content[0].text.strip()
        if result_text.upper() == "NONE" or "none" in result_text.lower()[:10]:
            return None
        return result_text
    except Exception:
        return None


def save_candidates(candidates: Dict[str, List[str]], output_path: str) -> None:
    """Save generated candidates to JSON file."""
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2)
    logger.info(f"Saved candidates to {output_path}")


def load_candidates(candidates_path: str) -> Dict[str, List[str]]:
    """Load candidates from JSON file."""
    with open(candidates_path) as f:
        return json.load(f)


def solve_puzzle(
    puzzle_path: str,
    candidates_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    candidates_per_clue: int = 8,
    model: str = "gpt-4o-mini",
    max_passes: int = 3,
    use_database: bool = False,
    database_only: bool = False,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
    use_o1_for_wordplay: bool = False,
    use_mac: bool = True,
    use_cdbj: bool = True,
    sniper_max_level: int = 3,
    enable_theme_detection: bool = False,
    mac_mode: str = "full",
    use_thinking: bool = False,
) -> Dict[str, str]:
    """
    Solve a crossword puzzle from JSON file using multi-pass strategy.

    Pass 1: Generate candidates and attempt to solve
    Pass 2+: For unsolved clues, use crossing letters as constraints and regenerate
    Sniper: Escalate through LLM tiers for remaining gaps

    Args:
        puzzle_path: Path to puzzle JSON file
        candidates_path: Path to existing candidates JSON (skip generation if provided)
        output_dir: Directory for output files (defaults to same as puzzle)
        candidates_per_clue: Number of candidates to generate per clue
        model: OpenAI model to use
        max_passes: Maximum number of solve passes (default: 3)
        use_database: Use TSV/SQLite clue database for candidate lookup
        database_only: Only use database (no LLM fallback)
        min_candidates: Minimum candidates per clue (prevents single-candidate blocking)
        use_o1_for_wordplay: Use o1 model for pun/wordplay clues
        use_mac: Use MAC (AC-3) instead of forward checking
        use_cdbj: Use conflict-directed backjumping
        sniper_max_level: Maximum sniper escalation level (0=disabled, 1-4)
        enable_theme_detection: Enable lightweight theme detection
        mac_mode: MAC mode - "full", "search-only", or "off"

    Returns:
        Solution assignment (clue_id -> word)
    """
    puzzle_path = Path(puzzle_path)
    if output_dir is None:
        output_dir = puzzle_path.parent
    else:
        output_dir = Path(output_dir)

    puzzle_stem = puzzle_path.stem.replace("_puzzle", "")

    # Load puzzle JSON
    logger.info(f"Loading puzzle from {puzzle_path}...")
    puzzle_data = load_puzzle_json(str(puzzle_path))
    grid_size = puzzle_data["grid"]["rows"], puzzle_data["grid"]["cols"]
    total_clues = puzzle_data["metadata"]["total_clues"]
    logger.debug(f"  Grid size: {grid_size[0]}x{grid_size[1]}")
    logger.debug(f"  Total clues: {total_clues}")

    # Build solver input directly from JSON
    solver_input = build_solver_input_from_json(puzzle_data)
    logger.debug(f"  Clues for solving: {len(solver_input.clue_cells)}")

    # Build clue text lookup
    clue_text_lookup = {}
    for direction in ["across", "down"]:
        for clue in puzzle_data["clues"].get(direction, []):
            clue_id = f"{clue['number']}-{direction}"
            clue_text_lookup[clue_id] = clue["text"]

    # Initialize database if requested
    db = None
    if use_database:
        from crosswise.solver.clue_database import ClueDatabase
        logger.info("Initializing clue database...")
        db = ClueDatabase()

    # Initialize word index (graceful degradation if files missing)
    word_index = None
    try:
        from crosswise.solver.word_index import WordIndex
        logger.info("Initializing word index...")
        word_index = WordIndex()
        if word_index.size == 0:
            word_index = None
            logger.debug("  Word index empty (no word list files found)")
    except Exception as e:
        logger.debug(f"  Word index unavailable: {e}")

    # Get or generate initial candidates
    if candidates_path and Path(candidates_path).exists():
        logger.debug(f"Loading existing candidates from {candidates_path}...")
        candidates = load_candidates(str(candidates_path))
    else:
        logger.info(f"=== PASS 1: Initial candidate generation ({candidates_per_clue} per clue) ===")
        clue_inputs = build_clue_inputs_from_json(puzzle_data, solver_input)

        if db:
            # Use database-backed generation
            candidates = generate_candidates_with_database(
                clue_inputs,
                db=db,
                candidates_per_clue=candidates_per_clue,
                batch_size=30,
                model=model,
                use_llm_fallback=not database_only,
            )
        else:
            # LLM-only generation (original behavior)
            def on_batch(batch_num: int, total: int, results: Dict[str, List[str]]) -> None:
                logger.info(f"  Batch {batch_num}/{total}: {len(results)} clues")

            candidates = generate_candidates(
                clue_inputs,
                candidates_per_clue=candidates_per_clue,
                batch_size=30,
                model=model,
                on_batch_complete=on_batch,
            )

        # Ensure minimum candidates per clue (prevents single-candidate blocking)
        if db and min_candidates > 1:
            logger.info(f"=== Ensuring minimum {min_candidates} candidates per clue ===")
            candidates = ensure_minimum_candidates(
                clue_inputs,
                candidates,
                db=db,
                min_candidates=min_candidates,
                model=model,
                use_o1_for_wordplay=use_o1_for_wordplay,
            )

        # Apply Bouncer Filter (cross-reference against DB + word index)
        if db is not None or word_index is not None:
            logger.info("=== Bouncer Filter: verifying candidates ===")
            scored = bouncer_filter(
                candidates, db=db, word_index=word_index,
                clue_text_lookup=clue_text_lookup,
            )
            verified_count = sum(
                sum(1 for sc in scs if sc.verified) for scs in scored.values()
            )
            total_count = sum(len(scs) for scs in scored.values())
            logger.debug(f"  {verified_count}/{total_count} candidates verified")
            # Extract score map for value ordering in CSP
            score_map = to_score_map(scored)
            # Convert back to plain format (sorted by score)
            candidates = to_plain_candidates(scored)
        else:
            score_map = {}

        # Save candidates
        candidates_output = output_dir / f"{puzzle_stem}_candidates.json"
        save_candidates(candidates, str(candidates_output))

    # Build clue_inputs if we loaded from file (needed for ensure_minimum_candidates)
    if candidates_path and Path(candidates_path).exists():
        clue_inputs = build_clue_inputs_from_json(puzzle_data, solver_input)
        # Run bouncer filter on loaded candidates for score map
        if db is not None or word_index is not None:
            scored = bouncer_filter(
                candidates, db=db, word_index=word_index,
                clue_text_lookup=clue_text_lookup,
            )
            score_map = to_score_map(scored)
            # Re-sort candidates by score
            candidates = to_plain_candidates(scored)
        else:
            score_map = {}

    # Multi-pass solving loop
    result = None
    prev_assignment = None  # Seed for next pass
    for pass_num in range(1, max_passes + 1):
        # Report candidate coverage
        missing = [cid for cid in solver_input.clue_cells if cid not in candidates or not candidates[cid]]
        low_candidates = [cid for cid in solver_input.clue_cells
                         if cid in candidates and 0 < len(candidates[cid]) <= 2]

        if missing:
            logger.debug(f"  {len(missing)} clues have no candidates")
        if low_candidates:
            logger.debug(f"  {len(low_candidates)} clues have <=2 candidates")

        # Attempt solve (seed with high-confidence previous answers for pass 2+)
        seed = None
        if pass_num > 1 and prev_assignment and score_map:
            # Only seed answers where the chosen word has high confidence
            seed = {}
            for cid, word in prev_assignment.items():
                word_score = score_map.get(cid, {}).get(word, 0.0)
                if word_score >= 0.6:  # DB-verified or word-index verified
                    seed[cid] = word
            if not seed:
                seed = None
        logger.info(f"=== PASS {pass_num}: Solving with CSP solver ===")
        if seed:
            logger.debug(f"  Seeding with {len(seed)}/{len(prev_assignment)} high-confidence clues")
        result = solve_csp(
            solver_input, candidates, return_partial=True,
            use_mac=use_mac, use_cdbj=use_cdbj,
            candidate_scores=score_map,
            seed_assignment=seed,
            mac_mode=mac_mode,
        )

        logger.debug(f"  Solved: {result.solved}")
        logger.debug(f"  Assigned: {len(result.assignment)}/{len(solver_input.clue_cells)} clues")
        logger.debug(f"  Time: {result.time_ms:.2f}ms")
        stats_line = f"  Nodes: {result.nodes_expanded}, Backtracks: {result.backtracks}"
        if result.backjumps:
            stats_line += f", Backjumps: {result.backjumps}"
        if result.arc_consistency_pruned:
            stats_line += f", AC-3 pruned: {result.arc_consistency_pruned}"
        logger.debug(stats_line)

        # Save assignment for seeding next pass
        prev_assignment = result.assignment.copy() if result.assignment else None

        if result.solved:
            break

        # Theme detection (after first pass with partial solution)
        if enable_theme_detection and pass_num == 1 and len(result.assignment) > 0:
            logger.info("=== Theme Detection ===")
            theme_pattern = detect_theme(puzzle_data, result.assignment)
            if theme_pattern:
                logger.debug(f"  Theme detected: {theme_pattern}")
            else:
                logger.debug("  No theme pattern detected")

        # Not solved - try to improve using letter patterns from partial solution
        if len(result.assignment) == 0:
            logger.debug("  No progress made - cannot extract patterns")
            break

        # Extended thinking round: after pass 1, generate candidates for ALL unsolved clues
        # Uses Claude extended thinking with no crossing patterns (avoids poisoned crossings)
        if use_thinking and pass_num == 1 and not database_only:
            unsolved_ids = set(solver_input.clue_cells.keys()) - set(result.assignment.keys())
            thinking_clues = []
            for clue_id in unsolved_ids:
                text = clue_text_lookup.get(clue_id, "")
                cells = solver_input.clue_cells.get(clue_id, [])
                thinking_clues.append(ClueInput(
                    clue_id=clue_id,
                    text=text,
                    length=len(cells),
                    category=categorize_clue(text),
                    num_crossings=solver_input.crossing_count(clue_id),
                ))

            if thinking_clues:
                logger.info(f"=== Extended Thinking: {len(thinking_clues)} unsolved clues ===")
                thinking_results = generate_with_extended_thinking(
                    thinking_clues, word_index=word_index,
                )
                thinking_added = 0
                for clue_id, scored_cands in thinking_results.items():
                    if scored_cands:
                        existing = set(candidates.get(clue_id, []))
                        new_words = [sc.word for sc in scored_cands if sc.word not in existing]
                        if new_words:
                            candidates[clue_id] = list(existing) + new_words
                            thinking_added += 1
                            if clue_id not in score_map:
                                score_map[clue_id] = {}
                            for sc in scored_cands:
                                if sc.word not in score_map[clue_id]:
                                    score_map[clue_id][sc.word] = sc.confidence
                logger.debug(f"  Extended thinking added candidates for {thinking_added}/{len(thinking_clues)} clues")

                # Save updated candidates after thinking round
                candidates_output = output_dir / f"{puzzle_stem}_candidates.json"
                save_candidates(candidates, str(candidates_output))

        # Extract patterns from partial solution (confidence-filtered when score_map available)
        confidence_threshold = 0.6 if use_thinking else 0.0
        patterns = extract_letter_patterns(
            solver_input, result.assignment,
            score_map=score_map, min_confidence=confidence_threshold,
        )
        unsolved_with_patterns = {
            cid: pattern for cid, pattern in patterns.items()
            if "_" in pattern and pattern.count("_") < len(pattern)  # Has some known letters
        }

        if not unsolved_with_patterns:
            logger.debug("  No useful patterns found - cannot improve")
            break

        logger.info(f"=== PASS {pass_num + 1}: Regenerating {len(unsolved_with_patterns)} clues with patterns ===")

        # Build constrained clue inputs
        constrained_clues = []
        for clue_id, pattern in unsolved_with_patterns.items():
            text = clue_text_lookup.get(clue_id, "")
            constrained_clues.append(ClueInput(
                clue_id=clue_id,
                text=text,
                length=len(pattern),
                pattern=pattern,
                category=categorize_clue(text),
                num_crossings=solver_input.crossing_count(clue_id),
            ))
            logger.debug(f"  {clue_id}: {pattern} - \"{text[:40]}...\"" if len(text) > 40 else f"  {clue_id}: {pattern} - \"{text}\"")

        # Regenerate with patterns (more candidates for constrained clues)
        if db:
            # Use database pattern matching
            new_candidates = regenerate_with_patterns(
                constrained_clues,
                db=db,
                candidates_per_clue=candidates_per_clue * 2,
            )
            # If database returned nothing and LLM fallback enabled, try LLM
            clues_without_candidates = [c for c in constrained_clues if c.clue_id not in new_candidates or not new_candidates[c.clue_id]]
            if clues_without_candidates and not database_only:
                llm_candidates = generate_candidates_batch(
                    clues_without_candidates,
                    candidates_per_clue=candidates_per_clue * 2,
                    model=model,
                )
                new_candidates.update(llm_candidates)
        else:
            new_candidates = generate_candidates_batch(
                constrained_clues,
                candidates_per_clue=candidates_per_clue * 2,
                model=model,
            )

        # Merge new candidates and update score map
        updated = 0
        for clue_id, cands in new_candidates.items():
            if cands:
                existing = set(candidates.get(clue_id, []))
                new_unique = [c for c in cands if c.upper() not in existing and c not in existing]
                if new_unique:
                    candidates[clue_id] = list(existing) + [c.upper() for c in new_unique]
                    updated += 1
                    # Score new pattern-matched candidates
                    if clue_id not in score_map:
                        score_map[clue_id] = {}
                    for c in new_unique:
                        w = c.upper()
                        if w not in score_map[clue_id]:
                            # Pattern-matched DB candidates get medium score
                            is_verified = (word_index and word_index.contains(w))
                            score_map[clue_id][w] = 0.65 if is_verified else 0.35

        logger.debug(f"  Updated {updated} clues with new pattern-matched candidates")

        # Synonym regeneration for definition clues on first regen pass
        if pass_num == 1 and not database_only:
            definition_clues = [
                c for c in constrained_clues
                if c.category == "definition" and len(candidates.get(c.clue_id, [])) < 15
            ]
            if definition_clues:
                logger.info(f"=== Synonym Generation: {len(definition_clues)} definition clues ===")
                synonym_results = generate_synonyms_batch(definition_clues, model=model)
                syn_added = 0
                for clue_id, syns in synonym_results.items():
                    if syns:
                        existing = set(candidates.get(clue_id, []))
                        new_syns = [s.upper() for s in syns if s.upper() not in existing]
                        if new_syns:
                            candidates[clue_id] = list(existing) + new_syns
                            syn_added += 1
                            if clue_id not in score_map:
                                score_map[clue_id] = {}
                            for s in new_syns:
                                is_verified = (word_index and word_index.contains(s))
                                score_map[clue_id][s] = 0.6 if is_verified else 0.4
                logger.debug(f"  Synonyms added for {syn_added}/{len(definition_clues)} clues")

        # Sniper phases: skip when extended thinking is active (thinking replaces sniping)
        if not use_thinking:
            # Identify bottleneck clues (unsolved with very few candidates)
            bottleneck_clues = [
                c for c in constrained_clues
                if len(candidates.get(c.clue_id, [])) <= 5
            ]
            if bottleneck_clues and sniper_max_level > 0 and not database_only:
                logger.info(f"=== Bottleneck Sniper: {len(bottleneck_clues)} clues with <=5 candidates ===")
                for clue in bottleneck_clues:
                    sniper_results = sniper_escalation(
                        clue, db=db, word_index=word_index, max_level=sniper_max_level,
                    )
                    if sniper_results:
                        existing = set(candidates.get(clue.clue_id, []))
                        new_words = [sc.word for sc in sniper_results if sc.word not in existing]
                        if new_words:
                            candidates[clue.clue_id] = list(existing) + new_words
                            if clue.clue_id not in score_map:
                                score_map[clue.clue_id] = {}
                            for sc in sniper_results:
                                if sc.word not in score_map[clue.clue_id]:
                                    score_map[clue.clue_id][sc.word] = 0.7 if sc.verified else 0.5
                            logger.debug(f"    {clue.clue_id}: +{len(new_words)} bottleneck sniper")

            # Sniper escalation for unsolved clues with rich patterns (>30% known letters)
            sniper_count = 0
            sniper_eligible = [
                c for c in constrained_clues
                if c.pattern and sum(1 for ch in c.pattern if ch != "_") / len(c.pattern) > 0.3
            ]
            if sniper_eligible and sniper_max_level > 0:
                logger.info(f"=== Sniper Phase: {len(sniper_eligible)} clues eligible for escalation ===")
                for clue in sniper_eligible:
                    sniper_results = sniper_escalation(
                        clue, db=db, word_index=word_index, max_level=sniper_max_level,
                    )
                    if sniper_results:
                        sniper_count += 1
                        existing = set(candidates.get(clue.clue_id, []))
                        new_words = [sc.word for sc in sniper_results if sc.word not in existing]
                        if new_words:
                            candidates[clue.clue_id] = list(existing) + new_words
                            # Score sniper candidates (0.7 for individually reasoned)
                            if clue.clue_id not in score_map:
                                score_map[clue.clue_id] = {}
                            for sc in sniper_results:
                                if sc.word not in score_map[clue.clue_id]:
                                    score_map[clue.clue_id][sc.word] = 0.7 if sc.verified else 0.5
                            logger.debug(f"    {clue.clue_id}: +{len(new_words)} from sniper (L{sniper_results[0].source[-1]})")
                logger.debug(f"  Sniper added candidates for {sniper_count}/{len(sniper_eligible)} clues")

        # Save updated candidates
        candidates_output = output_dir / f"{puzzle_stem}_candidates.json"
        save_candidates(candidates, str(candidates_output))

    # Final result
    if result and result.solved:
        is_valid = verify_solution(solver_input, result.assignment)
        logger.info(f"SOLVED — {len(result.assignment)} clues, valid={is_valid}")

        # Save solution
        solution_output = output_dir / f"{puzzle_stem}_solution.json"
        solution_data = {
            "puzzle_file": str(puzzle_path),
            "solved_at": datetime.now().isoformat(),
            "solved": result.solved,
            "valid": is_valid,
            "passes": pass_num,
            "metrics": {
                "time_ms": result.time_ms,
                "nodes_expanded": result.nodes_expanded,
                "backtracks": result.backtracks,
                "backjumps": result.backjumps,
                "max_depth": result.max_recursion_depth,
                "prune_operations": result.prune_operations,
                "domain_wipeouts": result.domain_wipeouts,
                "arc_consistency_pruned": result.arc_consistency_pruned,
                "sniper_calls": result.sniper_calls,
                "theme_detected": result.theme_detected,
            },
            "assignment": result.assignment,
        }
        with open(solution_output, "w") as f:
            json.dump(solution_data, f, indent=2)
        logger.info(f"Saved solution to {solution_output}")
    else:
        if result:
            logger.warning(f"FAILED — best partial: {len(result.assignment)}/{len(solver_input.clue_cells)} clues")
        else:
            logger.warning("FAILED — no result")

    return result.assignment if result else {}


