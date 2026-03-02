"""Shared data types, helpers, and constants for candidate generation."""

from dataclasses import dataclass
from typing import Dict, List, Optional


# Default batch size for API calls
DEFAULT_BATCH_SIZE = 30
DEFAULT_CANDIDATES_PER_CLUE = 4
DEFAULT_MIN_CANDIDATES = 5


@dataclass
class ClueInput:
    """Input for candidate generation."""

    clue_id: str  # e.g., "1-across"
    text: str  # e.g., "Capital of France"
    length: int  # Total answer length (sum of word lengths)
    pattern: Optional[str] = None  # e.g., "C_T" for constrained generation
    category: Optional[str] = None  # "trivia", "definition", "wordplay", "fillin"
    num_crossings: int = 0  # Number of crossing clues (for domain sizing)


@dataclass
class ScoredCandidate:
    """A candidate answer with source and verification metadata."""

    word: str
    source: str  # "database", "llm", "word_index", "sniper"
    confidence: float  # 0.0-1.0 from source
    verified: bool  # True if found in DB or word index (Bouncer check)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScoredCandidate) and self.word == other.word

    def __hash__(self) -> int:
        return hash(self.word)


@dataclass
class CandidateResult:
    """Result of candidate generation for one clue."""

    clue_id: str
    candidates: List[str]
    error: Optional[str] = None


def to_plain_candidates(
    scored: Dict[str, List["ScoredCandidate"]],
) -> Dict[str, List[str]]:
    """Convert scored candidates to plain word lists (verified first)."""
    return {
        clue_id: [sc.word for sc in candidates]
        for clue_id, candidates in scored.items()
    }


def to_score_map(
    scored: Dict[str, List["ScoredCandidate"]],
) -> Dict[str, Dict[str, float]]:
    """Extract a score map: clue_id -> {word: composite_score}.

    Used by the CSP solver for value ordering (try high-score candidates first).
    """
    result: Dict[str, Dict[str, float]] = {}
    for clue_id, candidates in scored.items():
        result[clue_id] = {sc.word: sc.confidence for sc in candidates}
    return result


def _matches_pattern(word: str, pattern: str) -> bool:
    """Check if a word matches a pattern like 'C_T' where _ is wildcard."""
    if len(word) != len(pattern):
        return False
    for w_char, p_char in zip(word, pattern):
        if p_char != "_" and w_char != p_char:
            return False
    return True
