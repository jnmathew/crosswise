"""Legacy OpenAI candidate generation functions.

Used by solve_puzzle.py CLI and as fallback in sniper escalation levels.
Not used by the web pipeline.
"""

import os
from typing import Dict, List, Optional

from loguru import logger
from openai import OpenAI

from .models import ClueInput, _matches_pattern, DEFAULT_BATCH_SIZE, DEFAULT_CANDIDATES_PER_CLUE
from .prompts import _build_prompt, _parse_response


def generate_candidates_batch(
    clues: List[ClueInput],
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> Dict[str, List[str]]:
    """
    Generate candidate answers for a batch of clues using OpenAI.

    Args:
        clues: List of ClueInput objects
        candidates_per_clue: Number of candidates to generate per clue
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: OpenAI model to use

    Returns:
        Dict mapping clue_id to list of candidate words

    Raises:
        ValueError: If API key not found or response parsing fails
        openai.APIError: If API call fails
    """
    if not clues:
        return {}

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set environment variable or pass api_key."
        )

    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(clues, candidates_per_clue)
    clue_ids = [c.clue_id for c in clues]

    # o1 models use different parameters
    is_o1 = model.startswith("o1")
    if is_o1:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=3000,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.7,
        )

    response_text = response.choices[0].message.content
    if not response_text:
        logger.debug(f"Empty response from {model}")
        logger.debug(f"Full response: {response}")
        raise ValueError(f"Empty response from {model}")
    candidates = _parse_response(response_text, clue_ids)

    # Build lookup maps for filtering
    clue_map = {c.clue_id: c for c in clues}

    # Filter candidates to only those with correct length and matching pattern
    for clue_id in candidates:
        clue = clue_map.get(clue_id)
        if not clue:
            continue

        filtered = []
        for word in candidates[clue_id]:
            # Check length
            if len(word) != clue.length:
                continue
            # Check pattern if present
            if clue.pattern and not _matches_pattern(word, clue.pattern):
                continue
            filtered.append(word)

        candidates[clue_id] = filtered

    return candidates


def generate_candidates(
    clues: List[ClueInput],
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    on_batch_complete: Optional[callable] = None,
) -> Dict[str, List[str]]:
    """
    Generate candidate answers for all clues, batching API calls.

    Args:
        clues: List of ClueInput objects
        candidates_per_clue: Number of candidates to generate per clue
        batch_size: Number of clues per API call
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: OpenAI model to use
        on_batch_complete: Optional callback(batch_num, total_batches, results)

    Returns:
        Dict mapping clue_id to list of candidate words
    """
    if not clues:
        return {}

    all_candidates: Dict[str, List[str]] = {}
    total_batches = (len(clues) + batch_size - 1) // batch_size

    for i in range(0, len(clues), batch_size):
        batch = clues[i : i + batch_size]
        batch_num = i // batch_size + 1

        batch_results = generate_candidates_batch(
            batch,
            candidates_per_clue=candidates_per_clue,
            api_key=api_key,
            model=model,
        )

        all_candidates.update(batch_results)

        if on_batch_complete:
            on_batch_complete(batch_num, total_batches, batch_results)

    return all_candidates


def generate_synonyms_batch(
    clues: List[ClueInput],
    model: str = "gpt-4o",
) -> Dict[str, List[str]]:
    """
    Generate synonym-focused candidates for definition clues.

    Uses a targeted prompt that asks specifically for synonyms and related terms,
    catching simple answers that batch generation sometimes misses
    (e.g., DWINDLES for "gets smaller", CONVENT for "religious retreat").

    Args:
        clues: List of ClueInput objects (should be definition-type clues)
        model: OpenAI model to use

    Returns:
        Dict mapping clue_id to list of candidate words
    """
    if not clues:
        return {}

    client = OpenAI()
    all_candidates: Dict[str, List[str]] = {}

    # Process in batches of 15
    for i in range(0, len(clues), 15):
        batch = clues[i:i + 15]

        clue_lines = []
        for c in batch:
            pattern_info = f", pattern: {c.pattern}" if c.pattern and "_" in c.pattern else ""
            clue_lines.append(f"- {c.clue_id}: \"{c.text}\" ({c.length} letters{pattern_info})")

        prompt = f"""You are a crossword expert. For each clue, provide 10 candidate answers.

Think broadly about:
- Direct synonyms and related terms
- Less common synonyms (e.g., "dwindles" for "gets smaller")
- Domain-specific terms (e.g., "convent" for "religious retreat")
- Proper nouns if the clue suggests one (e.g., "Athens" for "capital on the Mediterranean")
- Compound words or phrases without spaces (e.g., "POLLENCOUNT" for "allergy season report")
- Abbreviations and acronyms common in crosswords

Rules:
- Each answer must be EXACTLY the specified number of letters
- UPPERCASE, NO SPACES, NO HYPHENS
- If a pattern is given, the answer MUST match it exactly

Clues:
{chr(10).join(clue_lines)}

Respond with ONLY a JSON object mapping clue_id to array of answers.
Example: {{"1-across": ["ANSWER1", "ANSWER2"]}}"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.8,  # Higher temp for diversity
            )
            response_text = response.choices[0].message.content
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
                    word = word.upper()
                    if len(word) != clue.length:
                        continue
                    if clue.pattern and not _matches_pattern(word, clue.pattern):
                        continue
                    filtered.append(word)
                batch_candidates[clue_id] = filtered

            all_candidates.update(batch_candidates)

        except Exception as e:
            logger.warning(f"Synonym generation error: {e}")

    return all_candidates
