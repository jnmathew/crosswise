"""Database-backed candidate lookup.

Primary candidate source: looks up clues against ~9-11M historical pairs in SQLite.
Optional LLM fallback via OpenAI (legacy CLI path only).
"""

from typing import Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from .models import ClueInput, DEFAULT_BATCH_SIZE, DEFAULT_CANDIDATES_PER_CLUE

if TYPE_CHECKING:
    from crosswise.solver.clue_database import ClueDatabase


def generate_candidates_with_database(
    clues: List[ClueInput],
    db: "ClueDatabase",
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_llm_fallback: bool = True,
    on_progress: Optional[callable] = None,
) -> Dict[str, List[str]]:
    """
    Generate candidates using database lookup first, with optional LLM fallback.

    This is the preferred method when a clue database is available.
    Database lookups are fast and provide historically accurate answers.

    Args:
        clues: List of ClueInput objects
        db: ClueDatabase instance for lookups
        candidates_per_clue: Max candidates to return per clue
        batch_size: Clues per LLM batch (if fallback needed)
        api_key: OpenAI API key (for LLM fallback)
        model: OpenAI model to use (for LLM fallback)
        use_llm_fallback: Whether to use LLM for clues not in database
        on_progress: Optional callback(phase, clue_id, candidates_found)

    Returns:
        Dict mapping clue_id to list of candidate words
    """
    if not clues:
        return {}

    candidates: Dict[str, List[str]] = {}
    llm_needed: List[ClueInput] = []

    # Phase 1: Database lookup
    db_hits = 0
    for clue in clues:
        if clue.pattern and "_" in clue.pattern:
            # Have pattern constraint - use clue + pattern lookup
            answers = db.lookup_by_clue_and_pattern(
                clue.text,
                clue.pattern,
                max_results=candidates_per_clue,
            )
        else:
            # No pattern - use clue text lookup
            answers = db.lookup_by_clue(
                clue.text,
                clue.length,
                max_results=candidates_per_clue,
            )

        if answers:
            candidates[clue.clue_id] = answers
            db_hits += 1
            if on_progress:
                on_progress("database", clue.clue_id, len(answers))
        else:
            llm_needed.append(clue)

    logger.info(f"Database lookup: {db_hits}/{len(clues)} clues found")

    # Phase 2: LLM fallback for clues not in database
    if llm_needed and use_llm_fallback:
        # Lazy import to avoid circular dependency
        from .openai_legacy import generate_candidates

        # Use smaller batches (10 clues) for better LLM reliability
        llm_batch_size = min(batch_size, 10)
        logger.info(f"LLM fallback: generating for {len(llm_needed)} clues (batch size {llm_batch_size})...")

        def on_batch(batch_num: int, total: int, results: Dict[str, List[str]]) -> None:
            if on_progress:
                for clue_id, cands in results.items():
                    on_progress("llm", clue_id, len(cands))

        llm_candidates = generate_candidates(
            llm_needed,
            candidates_per_clue=candidates_per_clue,
            batch_size=llm_batch_size,
            api_key=api_key,
            model=model,
            on_batch_complete=on_batch,
        )

        # Merge LLM results
        candidates.update(llm_candidates)

        llm_hits = sum(1 for cid in llm_candidates if llm_candidates.get(cid))
        logger.info(f"LLM generated: {llm_hits}/{len(llm_needed)} clues")

    elif llm_needed and not use_llm_fallback:
        logger.info(f"{len(llm_needed)} clues have no database candidates (LLM fallback disabled)")

    return candidates


def regenerate_with_patterns(
    clues: List[ClueInput],
    db: "ClueDatabase",
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
) -> Dict[str, List[str]]:
    """
    Regenerate candidates for clues that have pattern constraints.

    Uses database pattern matching which is efficient for finding
    answers matching constraints like "C_T" (CAT, COT, CUT).

    Args:
        clues: List of ClueInput with pattern constraints
        db: ClueDatabase instance
        candidates_per_clue: Max candidates per clue

    Returns:
        Dict mapping clue_id to list of candidates matching pattern
    """
    candidates: Dict[str, List[str]] = {}

    for clue in clues:
        if not clue.pattern or "_" not in clue.pattern:
            # No pattern - skip
            continue

        # Try clue + pattern first
        answers = db.lookup_by_clue_and_pattern(
            clue.text,
            clue.pattern,
            max_results=candidates_per_clue,
        )

        # If no results from clue match, try pattern-only
        if not answers:
            answers = db.lookup_by_pattern(
                clue.pattern,
                max_results=candidates_per_clue,
            )

        if answers:
            candidates[clue.clue_id] = answers

    return candidates
