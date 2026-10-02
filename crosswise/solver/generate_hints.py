"""
Generate hints and explanations for crossword puzzle clues using Claude.
"""

import json

import anthropic

from crosswise.solver.claude_client import SONNET_MODEL, create_message, response_text

# Fixed structured-output schema (compiled once by the API, then cached).
_HINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "hints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "hint": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["id", "hint", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hints"],
    "additionalProperties": False,
}


def load_solution(path: str) -> dict[str, str]:
    """Load solution JSON, handling both flat and nested formats."""
    with open(path) as f:
        data = json.load(f)

    # Nested format: {"assignment": {"1-across": "WAS", ...}}
    if "assignment" in data and isinstance(data["assignment"], dict):
        return data["assignment"]

    # Flat format: {"1-across": "WAS", ...}
    # Filter out non-clue keys (like metadata)
    return {k: v for k, v in data.items() if "-across" in k or "-down" in k}


def merge_answers(puzzle: dict, solution: dict[str, str]) -> dict:
    """Merge solution answers into puzzle clue objects."""
    for direction in ("across", "down"):
        for clue in puzzle["clues"][direction]:
            key = f"{clue['number']}-{direction}"
            answer = solution.get(key)
            clue["answer"] = answer
            clue["hint"] = None
            clue["explanation"] = None
    return puzzle


def generate_hints_batch(
    clues_with_answers: list[dict],
) -> list[dict[str, str]]:
    """Call Claude Sonnet to generate hints for all solved clues in one batch."""
    clue_lines = []
    for c in clues_with_answers:
        clue_lines.append(
            f'{c["number"]}-{c["direction"]}: "{c["text"]}" → {c["answer"]}'
        )

    prompt = f"""You are a crossword puzzle hint generator. For each clue+answer pair below, generate:
1. A **hint** — a brief nudge that helps the solver without giving the answer away. Should be a different angle or association than the original clue.
2. An **explanation** — a concise explanation of why the answer fits the clue (1-2 sentences).

Return one entry per clue with its id (e.g. "1-across"), hint, and explanation.

Clues:
{chr(10).join(clue_lines)}
"""

    from crosswise.solver.cost_tracker import get_tracker

    client = anthropic.Anthropic(timeout=120.0)
    # Sonnet 5.5 thinks by default; low effort suits this content-generation
    # task, and max_tokens leaves room for thinking on top of the JSON.
    response = create_message(
        client,
        model=SONNET_MODEL,
        max_tokens=16000,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": _HINTS_SCHEMA},
        },
        messages=[{"role": "user", "content": prompt}],
    )
    get_tracker().track(response, "hints")

    text = response_text(response, "hints")
    if not text:
        raise RuntimeError("Hint generation was declined")
    return json.loads(text)["hints"]
