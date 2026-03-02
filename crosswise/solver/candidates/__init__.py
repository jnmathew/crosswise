"""Candidate generation package for crossword clues.

Primary source: SQLite database (~9-11M historical pairs).
LLM fallback: Claude Opus/Sonnet for batch generation, extended thinking for wordplay.
Legacy: OpenAI functions for solve_puzzle.py CLI + sniper escalation.

All public names are re-exported here for backward compatibility.
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
from .openai_legacy import (
    generate_candidates_batch,
    generate_candidates,
    generate_synonyms_batch,
)
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

# Level 3: escalation (orchestration)
from .escalation_legacy import (
    sniper_escalation,
    is_wordplay_clue,
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
    # openai_legacy
    "generate_candidates_batch",
    "generate_candidates",
    "generate_synonyms_batch",
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
    # escalation
    "sniper_escalation",
    "is_wordplay_clue",
]
