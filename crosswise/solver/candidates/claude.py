"""Claude/Anthropic candidate generation.

Primary LLM generation used by the web pipeline:
- generate_with_claude: batch candidate generation via Opus or Sonnet
- ensure_minimum_candidates: pad clues with < 5 candidates via Sonnet
- generate_with_extended_thinking: Sonnet 4.5 with thinking for hard clues
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from loguru import logger

from .models import (
    ClueInput,
    ScoredCandidate,
    _matches_pattern,
    DEFAULT_CANDIDATES_PER_CLUE,
    DEFAULT_MIN_CANDIDATES,
)
from .prompts import _parse_response

if TYPE_CHECKING:
    from crosswise.solver.clue_database import ClueDatabase


def generate_with_claude(
    clues: List[ClueInput],
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
    batch_size: int = 15,
    model: str = "claude-opus-4-20250514",
) -> Dict[str, List[str]]:
    """
    Generate candidates using Anthropic Claude API.

    Claude excels at wordplay and pun clues that GPT often misses.

    Args:
        clues: List of ClueInput objects
        candidates_per_clue: Number of candidates per clue
        batch_size: Clues per API call

    Returns:
        Dict mapping clue_id to list of candidate words
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, skipping Claude generation")
        return {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, skipping Claude generation")
        return {}

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0)
    all_candidates: Dict[str, List[str]] = {}

    def _process_batch(batch: List[ClueInput]) -> Dict[str, List[str]]:
        clue_lines = []
        for c in batch:
            if c.pattern and "_" in c.pattern:
                clue_lines.append(f"- {c.clue_id}: \"{c.text}\" ({c.length} letters, pattern: {c.pattern})")
            else:
                clue_lines.append(f"- {c.clue_id}: \"{c.text}\" ({c.length} letters)")

        prompt = f"""You are a crossword puzzle expert specializing in wordplay and puns.

For each clue, provide exactly {candidates_per_clue} candidate answers.

Key rules:
- Each answer must be EXACTLY the specified number of letters
- Answers should be UPPERCASE with NO SPACES
- Question marks (?) indicate puns or wordplay - think creatively!
- Look for double meanings, homophones, hidden words, anagrams
- If a pattern is given (like "C_T"), the answer MUST match it exactly

Clues:
{chr(10).join(clue_lines)}

Respond with ONLY a JSON object mapping clue_id to an array of answers.
Example: {{"1-across": ["PARIS", "LYONS"], "2-down": ["ECHO", "ARIA"]}}"""

        try:
            from crosswise.solver.cost_tracker import get_tracker

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            short = "opus" if "opus" in model else "sonnet" if "sonnet" in model else model
            get_tracker().track(response, f"candidates_{short}")

            response_text = response.content[0].text
            clue_ids = [c.clue_id for c in batch]
            batch_candidates = _parse_response(response_text, clue_ids)

            # Filter by length and pattern
            clue_map = {c.clue_id: c for c in batch}
            for clue_id in batch_candidates:
                clue = clue_map.get(clue_id)
                if not clue:
                    continue
                filtered = []
                for word in batch_candidates[clue_id]:
                    if len(word) != clue.length:
                        continue
                    if clue.pattern and not _matches_pattern(word, clue.pattern):
                        continue
                    filtered.append(word)
                batch_candidates[clue_id] = filtered

            return batch_candidates

        except Exception as e:
            logger.warning(f"Claude API error: {e}")
            return {}

    batches = [clues[i:i + batch_size] for i in range(0, len(clues), batch_size)]

    if len(batches) <= 1:
        for batch in batches:
            all_candidates.update(_process_batch(batch))
    else:
        with ThreadPoolExecutor(max_workers=len(batches)) as executor:
            futures = {executor.submit(_process_batch, batch): i for i, batch in enumerate(batches)}
            for future in as_completed(futures):
                all_candidates.update(future.result())

    return all_candidates


def ensure_minimum_candidates(
    clues: List[ClueInput],
    candidates: Dict[str, List[str]],
    db: "ClueDatabase",
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
    model: str = "gpt-4o",
    use_o1_for_wordplay: bool = True,
    use_structured_outputs: bool = True,
) -> Dict[str, List[str]]:
    """
    Ensure each clue has at least min_candidates options.

    This prevents single-candidate clues from blocking the solver.
    When a clue has fewer than min_candidates:
    1. Use o1 for wordplay clues (better at reasoning through puns)
    2. Use structured outputs for regular clues (guarantees valid JSON)

    NOTE: We intentionally DON'T use db.lookup_by_length() because
    random words of the correct length add noise without meaning.

    Args:
        clues: List of ClueInput objects
        candidates: Current candidates dict
        db: ClueDatabase instance
        min_candidates: Minimum candidates per clue
        model: OpenAI model for non-wordplay clues
        use_o1_for_wordplay: Use o1 model for pun clues
        use_structured_outputs: Use structured outputs for regular clues

    Returns:
        Updated candidates dict with minimum candidates per clue
    """
    # Identify clues needing more candidates
    needs_more: List[ClueInput] = []

    for clue in clues:
        current = candidates.get(clue.clue_id, [])
        if len(current) >= min_candidates:
            continue
        needs_more.append(clue)

    # Generate with Claude Sonnet for all clues needing more candidates
    if needs_more:
        logger.info(f"Padding {len(needs_more)} clues with Claude Sonnet...")
        claude_candidates = generate_with_claude(
            needs_more,
            candidates_per_clue=min_candidates * 3,  # Over-request; length filter drops ~50%
            model="claude-sonnet-4-20250514",
        )
        for clue_id, cands in claude_candidates.items():
            existing = set(candidates.get(clue_id, []))
            for c in cands:
                if c not in existing:
                    existing.add(c)
            candidates[clue_id] = list(existing)

    # Report final counts
    under_min = sum(1 for clue in clues if len(candidates.get(clue.clue_id, [])) < min_candidates)
    if under_min > 0:
        logger.warning(f"{under_min} clues still have < {min_candidates} candidates")

    return candidates


def generate_with_extended_thinking(
    clues: List[ClueInput],
    batch_size: int = 8,
    budget_tokens: int = 10000,
    word_index: Optional[Any] = None,
) -> Dict[str, List[ScoredCandidate]]:
    """
    Generate candidates using Claude extended thinking for hard clues.

    Uses Sonnet 4.5 with thinking enabled -- the model reasons through
    wordplay, puns, and cryptic clue mechanics before answering.

    No crossing patterns are used (avoids poisoned crossing problem).
    Just clue text + answer length.

    Args:
        clues: Unsolved clues to generate candidates for
        batch_size: Clues per API call (default: 8)
        budget_tokens: Thinking token budget per call (default: 10000)
        word_index: Optional word index for verification

    Returns:
        Dict mapping clue_id to list of ScoredCandidate
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, skipping extended thinking")
        return {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, skipping extended thinking")
        return {}

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0)
    all_results: Dict[str, List[ScoredCandidate]] = {}

    for i in range(0, len(clues), batch_size):
        batch = clues[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(clues) + batch_size - 1) // batch_size

        # Build clue list -- no patterns, just clue text + length
        clue_lines = []
        for c in batch:
            clue_lines.append(f'- {c.clue_id}: "{c.text}" ({c.length} letters)')

        prompt = f"""You are an expert crossword puzzle solver. For each clue below, figure out the answer.

Think carefully about each clue:
- Is it a straight definition, wordplay, pun, or cryptic clue?
- For wordplay: identify the definition part and the wordplay mechanism (anagram, hidden word, container, reversal, charade, double definition, homophone)
- For puns (clues ending with ?): what common phrase or compound word is being punned on?
- Consider abbreviations, slang, and crossword-ese

Clues:
{chr(10).join(clue_lines)}

For each clue, provide your top 5 candidate answers ranked by confidence.
Each answer must be EXACTLY the specified number of letters, UPPERCASE, NO SPACES.

Respond with ONLY a JSON object: {{"clue_id": ["BEST", "SECOND", ...], ...}}"""

        try:
            logger.info(f"Extended thinking batch {batch_num}/{total_batches} ({len(batch)} clues)...")
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                },
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text block (skip thinking blocks)
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text = block.text
                    break

            if not response_text:
                logger.warning(f"no text response in batch {batch_num}")
                continue

            # Parse JSON response
            clue_ids = [c.clue_id for c in batch]
            batch_candidates = _parse_response(response_text, clue_ids)

            # Convert to ScoredCandidates with length filtering
            clue_map = {c.clue_id: c for c in batch}
            added = 0
            for clue_id, words in batch_candidates.items():
                clue = clue_map.get(clue_id)
                if not clue:
                    continue
                scored = []
                for rank, word in enumerate(words):
                    w = word.upper().replace(" ", "")
                    if len(w) != clue.length:
                        continue
                    is_verified = word_index and word_index.contains(w)
                    scored.append(ScoredCandidate(
                        word=w,
                        source="thinking",
                        confidence=0.85 if is_verified else 0.75,
                        verified=bool(is_verified),
                    ))
                if scored:
                    all_results[clue_id] = scored
                    added += 1

            logger.debug(f"Added candidates for {added}/{len(batch)} clues")

        except Exception as e:
            logger.warning(f"Extended thinking error: {e}")

    return all_results
