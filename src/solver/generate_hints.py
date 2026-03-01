"""
Generate hints and explanations for crossword puzzle clues using Claude.
"""

import json

import anthropic


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
    """Call Claude Opus to generate hints for all solved clues in one batch."""
    clue_lines = []
    for c in clues_with_answers:
        clue_lines.append(
            f'{c["number"]}-{c["direction"]}: "{c["text"]}" → {c["answer"]}'
        )

    prompt = f"""You are a crossword puzzle hint generator. For each clue+answer pair below, generate:
1. A **hint** — a brief nudge that helps the solver without giving the answer away. Should be a different angle or association than the original clue.
2. An **explanation** — a concise explanation of why the answer fits the clue (1-2 sentences).

Return a JSON array with objects having keys: "id", "hint", "explanation".

Clues:
{chr(10).join(clue_lines)}

Respond with ONLY the JSON array, no other text."""

    from src.solver.cost_tracker import get_tracker

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    get_tracker().track(response, "hints")

    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    return json.loads(text)
