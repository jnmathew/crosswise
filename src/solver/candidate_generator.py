"""
LLM-based candidate generation for crossword clues using OpenAI API.
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Default batch size for API calls
DEFAULT_BATCH_SIZE = 30
DEFAULT_CANDIDATES_PER_CLUE = 4


@dataclass
class ClueInput:
    """Input for candidate generation."""

    clue_id: str  # e.g., "1-across"
    text: str  # e.g., "Capital of France"
    length: int  # Total answer length (sum of word lengths)


@dataclass
class CandidateResult:
    """Result of candidate generation for one clue."""

    clue_id: str
    candidates: List[str]
    error: Optional[str] = None


def _build_prompt(clues: List[ClueInput], candidates_per_clue: int) -> str:
    """Build the prompt for the LLM to generate candidates."""
    clue_list = "\n".join(
        f"- {c.clue_id}: \"{c.text}\" ({c.length} letters)" for c in clues
    )

    return f"""You are a crossword puzzle expert. For each clue below, provide exactly {candidates_per_clue} candidate answers.

Rules:
- Each answer must be EXACTLY the specified number of letters
- Answers should be UPPERCASE with NO SPACES (even for multi-word answers)
- Rank candidates from most likely to least likely
- Include common crossword answers and wordplay solutions

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
            result = json.loads(text[start:end])
        else:
            raise ValueError(f"Could not parse JSON from response: {e}")

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

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.7,
    )

    response_text = response.choices[0].message.content
    candidates = _parse_response(response_text, clue_ids)

    # Filter candidates to only those with correct length
    clue_lengths = {c.clue_id: c.length for c in clues}
    for clue_id in candidates:
        expected_len = clue_lengths.get(clue_id)
        if expected_len:
            candidates[clue_id] = [
                w for w in candidates[clue_id] if len(w) == expected_len
            ]

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


def clues_from_puzzle(puzzle: "Puzzle") -> List[ClueInput]:
    """
    Extract ClueInput list from a Puzzle object.

    Args:
        puzzle: Puzzle instance with clues

    Returns:
        List of ClueInput for candidate generation
    """
    from src.core.models import Direction

    clue_inputs = []

    for direction in [Direction.ACROSS, Direction.DOWN]:
        dir_name = direction.value
        for clue in puzzle.clues.get(direction, []):
            clue_id = f"{clue.number}-{dir_name}"
            clue_inputs.append(
                ClueInput(
                    clue_id=clue_id,
                    text=clue.text,
                    length=clue.total_length,
                )
            )

    return clue_inputs


def solve_with_llm_candidates(
    solver_input: "SolverInput",
    clue_inputs: List[ClueInput],
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    on_batch_complete: Optional[callable] = None,
) -> "SolveResult":
    """
    Generate candidates using LLM and solve the puzzle.

    This is a convenience function that combines candidate generation
    and solving into a single call.

    Args:
        solver_input: SolverInput with puzzle structure
        clue_inputs: List of ClueInput for candidate generation
        candidates_per_clue: Number of candidates per clue
        batch_size: Clues per API batch
        api_key: OpenAI API key
        model: OpenAI model to use
        on_batch_complete: Progress callback

    Returns:
        SolveResult from the CSP solver
    """
    from src.solver.csp import solve_csp

    # Generate candidates
    candidates = generate_candidates(
        clue_inputs,
        candidates_per_clue=candidates_per_clue,
        batch_size=batch_size,
        api_key=api_key,
        model=model,
        on_batch_complete=on_batch_complete,
    )

    # Solve with CSP
    return solve_csp(solver_input, candidates)
