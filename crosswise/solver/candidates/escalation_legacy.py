"""Multi-level escalation ladder for gap-filling when domain hits zero.

Used by solve_puzzle.py (legacy CLI) only -- not by the web pipeline.
Each level tries Claude first, falls back to OpenAI.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from .models import ClueInput, ScoredCandidate, _matches_pattern
from .prompts import _parse_response


def is_wordplay_clue(text: str) -> bool:
    """
    Detect clues that are puns/wordplay (Claude handles these better).

    Indicators:
    - Question mark at end (?) - classic pun indicator
    - Words like "perhaps", "maybe", "say" - uncertainty hints at wordplay
    - Ellipsis (...) - often indicates hidden meaning
    """
    indicators = [
        text.endswith('?'),
        'perhaps' in text.lower(),
        'maybe' in text.lower(),
        ', say' in text.lower(),
        '...' in text,
    ]
    return any(indicators)


def sniper_escalation(
    clue: ClueInput,
    db: Optional["Any"] = None,
    word_index: Optional[Any] = None,
    max_level: int = 3,
) -> List[ScoredCandidate]:
    """
    Escalation ladder for gap-filling when domain hits zero.

    Level 1: Claude Sonnet with pattern (standard clue, >=50% letters known)
    Level 2: Claude Opus with wordplay decomposition (<50% or wordplay clue)
    Level 3: Word index regex + Opus semantic ranking
    Level 4: Stub (web search -- not yet implemented)

    Auto-selects starting level based on clue category and pattern completeness.
    """
    pattern = clue.pattern or ("_" * clue.length)
    known_ratio = sum(1 for c in pattern if c != "_") / len(pattern) if pattern else 0
    is_wp = clue.category == "wordplay" or is_wordplay_clue(clue.text)

    # Auto-select starting level
    start_level = 2 if (is_wp or known_ratio < 0.5) else 1

    for level in range(start_level, max_level + 1):
        results: List[ScoredCandidate] = []

        if level == 1:
            results = _sniper_level_1(clue)
        elif level == 2:
            results = _sniper_level_2(clue)
        elif level == 3:
            results = _sniper_level_3(clue, word_index)
        elif level == 4:
            results = []  # Web search stub

        if results:
            return results

    return []


def _sniper_level_1(clue: ClueInput) -> List[ScoredCandidate]:
    """Level 1: Claude Sonnet with pattern, fallback to gpt-4o. For standard clues with >=50% letters."""
    # Lazy imports to avoid circular dependency
    from .claude import generate_with_claude
    from .openai_legacy import generate_candidates_batch

    candidates = generate_with_claude([clue], candidates_per_clue=8)
    words = candidates.get(clue.clue_id, [])

    # Fallback to OpenAI if Claude returned nothing
    if not words:
        fallback = generate_candidates_batch([clue], candidates_per_clue=8, model="gpt-4o")
        words = fallback.get(clue.clue_id, [])

    return [
        ScoredCandidate(word=w, source="sniper_l1", confidence=0.6, verified=False)
        for w in words
    ]


def _sniper_level_2(clue: ClueInput) -> List[ScoredCandidate]:
    """Level 2: Claude Opus with explicit wordplay decomposition, fallback to gpt-4o."""
    pattern_info = f"\nPattern: {clue.pattern}" if clue.pattern else ""

    prompt = f"""You are an expert crossword solver specializing in wordplay, puns, and cryptic clues.

Clue: "{clue.text}"
Length: {clue.length} letters{pattern_info}

Analyze this clue step by step:
1. Identify the DEFINITION component (straight meaning)
2. Identify the WORDPLAY component if present (anagram, reversal, container, charade, hidden word, double definition, homophone)
3. Identify the wordplay TYPE
4. Work through the wordplay to derive the answer
5. Verify the answer matches the pattern and length

Provide your top 8 candidate answers as a JSON array: ["ANSWER1", "ANSWER2", ...]
Answers must be EXACTLY {clue.length} letters, UPPERCASE, NO SPACES."""

    response_text = None

    # Try Anthropic first
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text
    except Exception as e:
        logger.warning(f"Sniper L2 Claude error: {e}")

    # Fallback to OpenAI gpt-4o
    if response_text is None:
        try:
            from openai import OpenAI
            openai_client = OpenAI()
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.7,
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Sniper L2 OpenAI error: {e}")
            return []

    if not response_text:
        return []

    # Extract JSON array from response
    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if not match:
        return []

    try:
        words = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    results = []
    for w in words:
        w = str(w).upper().strip()
        if len(w) != clue.length:
            continue
        if clue.pattern and not _matches_pattern(w, clue.pattern):
            continue
        results.append(
            ScoredCandidate(word=w, source="sniper_l2", confidence=0.8, verified=False)
        )
    return results


def _sniper_level_3(
    clue: ClueInput,
    word_index: Optional[Any] = None,
) -> List[ScoredCandidate]:
    """Level 3: Word index regex match + Opus semantic ranking."""
    if word_index is None or not clue.pattern:
        return []

    # Get all pattern matches from word index
    regex_matches = word_index.match_pattern(clue.pattern, max_results=30)
    if not regex_matches:
        return []

    # If few matches, return directly without LLM ranking
    if len(regex_matches) <= 5:
        return [
            ScoredCandidate(word=w, source="sniper_l3", confidence=0.5, verified=True)
            for w in regex_matches
        ]

    # Send to LLM for semantic ranking (try Anthropic, fallback to OpenAI)
    ranking_prompt = f"""You are an expert crossword solver. Given this clue, rank the candidate words by likelihood of being the correct answer.

Clue: "{clue.text}" ({clue.length} letters)
Pattern: {clue.pattern}

Candidates: {json.dumps(regex_matches)}

Return ONLY a JSON array of the top 10 candidates, ordered from most to least likely: ["BEST", "SECOND", ...]"""

    response_text = None

    # Try Anthropic first
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": ranking_prompt}],
            )
            response_text = response.content[0].text
    except Exception:
        pass  # Fall through to OpenAI

    # Fallback to OpenAI gpt-4o
    if response_text is None:
        try:
            from openai import OpenAI
            openai_client = OpenAI()
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": ranking_prompt}],
                max_tokens=1024,
                temperature=0.3,
            )
            response_text = response.choices[0].message.content
        except Exception:
            pass

    if not response_text:
        return [
            ScoredCandidate(word=w, source="sniper_l3", confidence=0.4, verified=True)
            for w in regex_matches[:10]
        ]

    match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if not match:
        return [
            ScoredCandidate(word=w, source="sniper_l3", confidence=0.4, verified=True)
            for w in regex_matches[:10]
        ]

    try:
        ranked = json.loads(match.group())
        return [
            ScoredCandidate(word=str(w).upper(), source="sniper_l3", confidence=0.7, verified=True)
            for w in ranked
            if len(str(w)) == clue.length
        ]
    except json.JSONDecodeError:
        return [
            ScoredCandidate(word=w, source="sniper_l3", confidence=0.4, verified=True)
            for w in regex_matches[:10]
        ]
