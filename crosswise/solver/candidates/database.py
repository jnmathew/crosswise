"""Database-backed candidate lookup.

Primary candidate source: looks up clues against ~9-11M historical pairs in SQLite.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from .models import ClueInput, DEFAULT_CANDIDATES_PER_CLUE

if TYPE_CHECKING:
    from crosswise.solver.clue_database import ClueDatabase


def generate_candidates_with_database(
    clues: List[ClueInput],
    db: "ClueDatabase",
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
    on_progress: Optional[callable] = None,
) -> Dict[str, List[str]]:
    """
    Generate candidates using database lookup.

    This is the preferred method when a clue database is available.
    Database lookups are fast and provide historically accurate answers.
    Claude is used as LLM fallback separately in the pipeline.

    Args:
        clues: List of ClueInput objects
        db: ClueDatabase instance for lookups
        candidates_per_clue: Max candidates to return per clue
        on_progress: Optional callback(phase, clue_id, candidates_found)

    Returns:
        Dict mapping clue_id to list of candidate words
    """
    if not clues:
        return {}

    candidates: Dict[str, List[str]] = {}

    db_hits = 0
    for clue in clues:
        if clue.pattern and "_" in clue.pattern:
            answers = db.lookup_by_clue_and_pattern(
                clue.text,
                clue.pattern,
                max_results=candidates_per_clue,
            )
        else:
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

    logger.info(f"Database lookup: {db_hits}/{len(clues)} clues found")

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
