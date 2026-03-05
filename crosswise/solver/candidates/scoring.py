"""Candidate scoring, clue categorization, and domain sizing.

Pure evaluation layer -- no generation responsibility.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .models import ScoredCandidate

if TYPE_CHECKING:
    from crosswise.solver.clue_database import ClueDatabase


def categorize_clue(text: str) -> str:
    """
    Heuristic clue categorization.

    Returns one of: "trivia", "definition", "wordplay", "fillin".
    """
    text_stripped = text.strip()
    # Wordplay: ends with ? or has pun indicators
    if text_stripped.endswith("?") or "perhaps" in text_stripped.lower() or ", say" in text_stripped.lower():
        return "wordplay"
    # Fill-in-the-blank: contains blank markers
    if "___" in text_stripped or "\u2026" in text_stripped or "..." in text_stripped:
        return "fillin"
    return "definition"


def compute_target_domain_size(
    category: str,
    db_hit_count: int,
    num_crossings: int,
) -> int:
    """
    Compute target domain size based on clue category and crossing density.

    Category-based targets:
    - High-confidence trivia (DB match >= 3): 3-5
    - Standard definition/fill-in-blank: 5-10
    - Wordplay/puns: 10-20

    Crossing-density floors:
    - 4-5 crossings: floor 2
    - 2-3 crossings: floor 5
    - 0-1 crossings: floor 10
    """
    # Category-based target
    if db_hit_count >= 3:
        target = 5
    elif category in ("definition", "fillin"):
        target = 10
    else:  # wordplay
        target = 20

    # Crossing-density floor
    if num_crossings >= 4:
        floor = 2
    elif num_crossings >= 2:
        floor = 5
    else:
        floor = 10

    return max(target, floor)


def bouncer_filter(
    candidates: Dict[str, List[str]],
    db: Optional["ClueDatabase"] = None,
    word_index: Optional[Any] = None,
    clue_text_lookup: Optional[Dict[str, str]] = None,
    candidate_sources: Optional[Dict[str, Dict[str, str]]] = None,
    web_candidates: Optional[Dict[str, str]] = None,
) -> Dict[str, List["ScoredCandidate"]]:
    """
    Cross-reference LLM candidates against DB and word index (Bouncer Filter).

    Computes a composite confidence score per candidate:
    - Source: DB-verified (0.8), word-index-verified (0.6), unverified (0.3)
    - Broda bonus: up to +0.15 from word index quality score
    - Clue-text match bonus: +0.1 if candidate was seen as answer to this clue in DB
    - Web confirmation bonus: +0.1 if candidate matches Haiku web pre-pass result

    Candidates sorted by composite score (highest first).

    Args:
        candidates: Dict mapping clue_id to list of candidate words
        db: ClueDatabase for answer verification
        word_index: WordIndex for membership testing and Broda scores
        clue_text_lookup: Optional dict of clue_id -> clue text (for DB clue match bonus)
        candidate_sources: Optional dict of clue_id -> {word -> source_label} for accurate source tracking
        web_candidates: Optional dict of clue_id -> web-verified answer from Haiku pre-pass

    Returns:
        Dict mapping clue_id to list of ScoredCandidate, sorted by score.
    """
    scored: Dict[str, List[ScoredCandidate]] = {}
    for clue_id, words in candidates.items():
        clue_scores: List[ScoredCandidate] = []
        clue_text = (clue_text_lookup or {}).get(clue_id, "")

        for word in words:
            word_upper = word.upper()
            is_in_db = False
            is_in_index = False
            is_clue_match = False

            if db is not None:
                try:
                    cursor = db._conn.execute(
                        "SELECT 1 FROM clues WHERE answer = ? LIMIT 1",
                        (word_upper,),
                    )
                    is_in_db = cursor.fetchone() is not None

                    # Check if this specific clue-answer pair exists in DB
                    if is_in_db and clue_text:
                        cursor2 = db._conn.execute(
                            "SELECT 1 FROM clues WHERE clue_normalized = ? AND answer = ? LIMIT 1",
                            (clue_text.lower(), word_upper),
                        )
                        is_clue_match = cursor2.fetchone() is not None
                except Exception:
                    pass

            if word_index is not None:
                is_in_index = word_index.contains(word_upper)

            # Compute composite score
            # Base: source reliability
            if is_in_db:
                base_score = 0.8
            elif is_in_index:
                base_score = 0.6
            else:
                base_score = 0.3

            # Broda quality bonus (0 to 0.15)
            broda_bonus = 0.0
            if word_index is not None:
                raw_score = word_index.score(word_upper)
                if raw_score > 0:
                    broda_bonus = min(raw_score / 100.0, 1.0) * 0.15

            # Clue-text match bonus
            clue_bonus = 0.1 if is_clue_match else 0.0

            # Web confirmation bonus: candidate matches Haiku web pre-pass result
            web_bonus = 0.0
            if web_candidates and clue_id in web_candidates:
                if word_upper == web_candidates[clue_id]:
                    web_bonus = 0.1

            confidence = min(base_score + broda_bonus + clue_bonus + web_bonus, 1.0)
            verified = is_in_db or is_in_index

            # Use tracked source if available, otherwise infer from verification
            if candidate_sources and clue_id in candidate_sources:
                source = candidate_sources[clue_id].get(word_upper, "llm")
            else:
                source = "llm"

            clue_scores.append(
                ScoredCandidate(word=word_upper, source=source, confidence=confidence, verified=verified)
            )

        # Sort by confidence descending (best candidates first)
        clue_scores.sort(key=lambda sc: sc.confidence, reverse=True)
        scored[clue_id] = clue_scores
    return scored
