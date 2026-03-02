"""Shared prompt building and response parsing for LLM candidate generation."""

import json
from typing import Dict, List

from loguru import logger

from crosswise.solver.candidates.models import ClueInput


def _build_prompt(clues: List[ClueInput], candidates_per_clue: int) -> str:
    """Build the prompt for the LLM to generate candidates."""
    clue_lines = []
    for c in clues:
        if c.pattern and "_" in c.pattern:
            # Show pattern constraint
            clue_lines.append(f"- {c.clue_id}: \"{c.text}\" ({c.length} letters, pattern: {c.pattern})")
        else:
            clue_lines.append(f"- {c.clue_id}: \"{c.text}\" ({c.length} letters)")
    clue_list = "\n".join(clue_lines)

    # Check if any clues have patterns
    has_patterns = any(c.pattern and "_" in c.pattern for c in clues)
    pattern_rules = """
- IMPORTANT: Some clues have letter patterns like "C_T" - answers MUST match the pattern exactly
  - Known letters (A-Z) must appear in those exact positions
  - Unknown positions (_) can be any letter""" if has_patterns else ""

    return f"""You are a crossword puzzle expert. For each clue below, provide exactly {candidates_per_clue} candidate answers.

Rules:
- Each answer must be EXACTLY the specified number of letters
- Answers should be UPPERCASE with NO SPACES (even for multi-word answers)
- Rank candidates from most likely to least likely
- Include common crossword answers and wordplay solutions{pattern_rules}

Clues:
{clue_list}

Respond with a JSON object mapping clue_id to an array of {candidates_per_clue} candidate answers.
Example format:
{{"1-across": ["PARIS", "LYONS", "TOURS", "BREST"], "2-down": ["ECHO", "ARIA", "SOLO", "DUET"]}}

JSON response:"""


def _parse_response(response_text: str, clue_ids: List[str]) -> Dict[str, List[str]]:
    """Parse LLM response into candidates dict."""
    # Find JSON in response (might have markdown code blocks)
    text = response_text.strip()
    if text.startswith("```"):
        # Remove markdown code block
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                # Return empty results instead of failing
                logger.warning("Could not parse LLM response, skipping batch")
                return {clue_id: [] for clue_id in clue_ids}
        else:
            logger.warning("No JSON in LLM response, skipping batch")
            return {clue_id: [] for clue_id in clue_ids}

    # Validate and normalize
    candidates = {}
    for clue_id in clue_ids:
        if clue_id in result:
            # Normalize: uppercase, no spaces
            words = [w.upper().replace(" ", "") for w in result[clue_id]]
            candidates[clue_id] = words
        else:
            candidates[clue_id] = []

    return candidates
