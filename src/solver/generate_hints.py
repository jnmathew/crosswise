"""
Generate hints and explanations for crossword puzzle clues using Claude.

Usage:
    .venv/bin/python3 -m src.solver.generate_hints \
        data/output/IMG_5526_puzzle.json \
        --solution data/output/IMG_5526_opus_solution.json \
        --output web/public/puzzles/IMG_5526.json
"""

import argparse
import json
import sys
from pathlib import Path

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

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Generate hints for crossword clues")
    parser.add_argument("puzzle", help="Path to puzzle JSON")
    parser.add_argument("--solution", required=True, help="Path to solution JSON")
    parser.add_argument("--output", required=True, help="Output path for enriched JSON")
    args = parser.parse_args()

    # Load puzzle and solution
    with open(args.puzzle) as f:
        puzzle = json.load(f)

    solution = load_solution(args.solution)
    puzzle = merge_answers(puzzle, solution)

    # Collect solved clues for hint generation
    solved_clues = []
    for direction in ("across", "down"):
        for clue in puzzle["clues"][direction]:
            if clue["answer"]:
                solved_clues.append(
                    {
                        "number": clue["number"],
                        "direction": direction,
                        "text": clue["text"],
                        "answer": clue["answer"],
                    }
                )

    total = len(puzzle["clues"]["across"]) + len(puzzle["clues"]["down"])
    print(f"Puzzle: {args.puzzle}")
    print(f"Solved: {len(solved_clues)}/{total} clues")

    if not solved_clues:
        print("No solved clues — skipping hint generation")
    else:
        # Batch in groups of 20 for faster responses
        batch_size = 20
        all_hints: list[dict[str, str]] = []
        for i in range(0, len(solved_clues), batch_size):
            batch = solved_clues[i : i + batch_size]
            print(
                f"Generating hints for batch {i // batch_size + 1} "
                f"({len(batch)} clues)..."
            )
            hints = generate_hints_batch(batch)
            all_hints.extend(hints)

        # Map hints back to clues
        hint_map = {h["id"]: h for h in all_hints}
        for direction in ("across", "down"):
            for clue in puzzle["clues"][direction]:
                key = f"{clue['number']}-{direction}"
                if key in hint_map:
                    clue["hint"] = hint_map[key]["hint"]
                    clue["explanation"] = hint_map[key]["explanation"]

    # Write enriched puzzle
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(puzzle, f, indent=2)

    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
