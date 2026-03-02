"""Candidate generation package for crossword clues.

Primary source: SQLite database (~9-11M historical pairs).
LLM fallback: Claude Opus/Sonnet for batch generation, extended thinking for wordplay.
"""

from dotenv import load_dotenv

load_dotenv()

# Level 0: models — dataclasses, helpers, constants
from .models import (
    ClueInput,
    ScoredCandidate,
    CandidateResult,
    to_plain_candidates,
    to_score_map,
    _matches_pattern,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CANDIDATES_PER_CLUE,
    DEFAULT_MIN_CANDIDATES,
)

# Level 1: prompts — shared LLM template logic
from .prompts import _build_prompt, _parse_response

# Level 2: providers + scoring + database + web
from .claude import (
    generate_with_claude,
    ensure_minimum_candidates,
    generate_with_extended_thinking,
)
from .scoring import (
    bouncer_filter,
    categorize_clue,
    compute_target_domain_size,
)
from .web_prepass import (
    web_search_prepass,
    _is_pop_culture_clue,
    _extract_answer,
)
from .database import (
    generate_candidates_with_database,
    regenerate_with_patterns,
)

__all__ = [
    # models
    "ClueInput",
    "ScoredCandidate",
    "CandidateResult",
    "to_plain_candidates",
    "to_score_map",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CANDIDATES_PER_CLUE",
    "DEFAULT_MIN_CANDIDATES",
    # prompts
    "_build_prompt",
    "_parse_response",
    "_matches_pattern",
    # claude
    "generate_with_claude",
    "ensure_minimum_candidates",
    "generate_with_extended_thinking",
    # scoring
    "bouncer_filter",
    "categorize_clue",
    "compute_target_domain_size",
    # web_prepass
    "web_search_prepass",
    # database
    "generate_candidates_with_database",
    "regenerate_with_patterns",
]
