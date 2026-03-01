"""
LLM-based iterative crossword solver.

Instead of CSP search, uses Claude as the solver in multiple passes:
1. Confidence pass — commit answers the LLM is certain about
2. Pattern passes — use crossing letters to constrain remaining clues
3. CSP cleanup — tiny residual subproblem if needed

Guided by a crossword-solving skill prompt that teaches constraint-first strategy.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import anthropic
from loguru import logger

from src.solver.models import SolverInput


# Type aliases
ClueId = str
Word = str
CellPos = Tuple[int, int]

SKILL_PROMPT = """You are a championship-level crossword puzzle solver. You solve American-style crossword puzzles by committing high-confidence answers first, then propagating crossing letters to constrain harder clues.

## Hard Rules
- Every answer MUST be exactly the specified length.
- Every answer MUST match the crossing-letter pattern (known letters are fixed).
- If a candidate list is provided, STRONGLY prefer answers from that list.
- Wrong answers poison crossing clues. When unsure, skip.

## Solving Procedure (follow in order)

### Step 1: Fill-in-the-blank clues first
These have the highest certainty. The answer completes a known phrase.
- "— ex machina" → DEUS
- "Quentin Tarantino's ___ Fiction" → PULP
- "Much ___ About Nothing" → ADO

### Step 2: Proper nouns and one-answer clues
Names, places, and clues with only one plausible answer.
- "Boxing legend Muhammad" → ALI
- "Yale student" → ELI

IMPORTANT — count the dashes carefully for fill-in-the-blank names:
- Each dash represents one word. Match the word count to the letter count.
- "— Moines" (one dash = one word, answer is one word) vs "— — Gabor" (two dashes = two words)
- When unsure about a proper noun, use web search to verify before committing.

### Step 3: Use crossing letters to disambiguate
Even 1-2 known letters dramatically narrow candidates. A pattern like _A_E eliminates most 4-letter words. If only one candidate matches the pattern, commit it.

### Step 4: Check crossing consequences BEFORE committing
For each answer you're about to commit, mentally check: do the letters I'm placing create plausible patterns for the crossing clues? If your answer puts a Z, Q, X, or J in a crossing position where it would need to start a common English word, reconsider. This is the #1 cause of unsolvable grids.

Example: If your answer puts a Z in a crossing position where a common word needs to start, reconsider — almost no common words start with Z/Q/X/J. Always prefer answers with common letters (E, A, R, S, T) at crossing positions over answers with rare letters (Z, Q, X, J).

## Clue Decoding Rules

### Grammar matching (CRITICAL — use to constrain answer endings)
The clue and answer must be grammatically interchangeable:
- Plural clue → plural answer (ends in S): "Beverages" → ALES
- Past tense clue → past tense answer (ends in ED): "Consumed" → ATE
- "-ing" clue → "-ing" answer: "Running" → JOGGING
- Comparative clue → comparative answer (ends in ER)
- Adjective clue → adjective answer: "Furious" → IRATE

### Abbreviation signals
If the clue contains ANY of these, the answer is abbreviated:
- "(Abbr.)", "for short", "briefly", "in short"
- An abbreviated word in the clue itself (e.g., "TV" means answer is also short-form)
- "Org." or "Grp." → answer is an acronym

### Question mark = wordplay/pun
Your first-instinct answer is almost certainly WRONG. Think laterally:
- "Flower of London?" → THAMES (it flows, not a plant)
- "Bar members?" → LAWYERS (legal bar, not drinking bar)
- "Star trek?" → SPACEWALK
- "Current events?" → RAPIDS

### Quoted clues = spoken phrases
The answer is something someone would say:
- "Can't be done!" → NOWAY
- "Over here!" → PSST

### Foreign language cues
If the clue contains a foreign name or place, the answer is in that language:
- "Summer, in Nice" → ETE (French; Nice is in France)
- "Friend, to Pierre" → AMI (French)
- "Aunt, in Spain" → TIA (Spanish)
- "Water, to Juan" → AGUA (Spanish)

### "Kind of" / "Type of" clues
The answer forms a compound with a word in the clue:
- "Kind of plan" → MEAL (meal plan)
- "Type of code" → AREA (area code)

## Common Crossword Fill (crosswordese)
These short words appear in almost every crossword. When a clue is ambiguous and one of these fits the pattern, it's very likely correct:

3 letters: ERA, ETA, ALE, ORE, ODE, EEL, EMU, ERE, OLE, ADO, LEA, URN, IRE, AWE, AGO, APE, ARC, OAR, OAT, ORB, OWE
4 letters: ALOE, ARIA, AREA, OREO, OBOE, OLEO, EPEE, ETUI, ALEE, ERIE, EDEN, ANTE, EMIR, NAVE, IDEA, OGRE, AVID, EMIT
5 letters: AERIE, ELAND, EIDER, AIOLI, ARIEL, OPERA, ONSET, OATER

## Confidence Calibration
- COMMIT (>90%): Fill-in-the-blank with unique phrase, proper nouns, clues where only one word fits the length + pattern
- TENTATIVE (70-90%): Definition clues with one clear synonym, answers confirmed by 1+ crossing letter
- SKIP (<70%): Wordplay clues without crossing help, ambiguous short clues with multiple valid synonyms, any answer that would put unlikely letters in crossing positions

## Web Search
You have access to web search. Use it ONLY when you are unsure about:
- Proper nouns: "Actress — Gabor", "— Moines, Iowa", names you can't verify from memory
- Factual clues where you have multiple plausible candidates
- Clues where your answer would put rare letters (Z, Q, X, J) in crossing positions

Do NOT search for common vocabulary clues — only factual/proper noun verification.
Search BEFORE committing, not after. A quick verification prevents grid-poisoning mistakes."""


def build_grid_state(
    solver_input: SolverInput,
    assignment: Dict[ClueId, Word],
) -> str:
    """Build a text representation of the current grid state."""
    rows, cols = solver_input.grid_size
    grid = [["." for _ in range(cols)] for _ in range(rows)]

    # Mark black cells (cells not in any clue)
    all_clue_cells = set()
    for cells in solver_input.clue_cells.values():
        for cell in cells:
            all_clue_cells.add(cell)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in all_clue_cells:
                grid[r][c] = "#"

    # Fill in assigned letters
    for clue_id, word in assignment.items():
        cells = solver_input.clue_cells[clue_id]
        for i, (r, c) in enumerate(cells):
            grid[r][c] = word[i]

    # Fill unassigned clue cells with _
    for clue_id, cells in solver_input.clue_cells.items():
        if clue_id not in assignment:
            for r, c in cells:
                if grid[r][c] == ".":
                    grid[r][c] = "_"

    lines = []
    for row in grid:
        lines.append(" ".join(row))
    return "\n".join(lines)


def extract_patterns(
    solver_input: SolverInput,
    assignment: Dict[ClueId, Word],
) -> Dict[ClueId, str]:
    """Extract crossing-letter patterns for unsolved clues."""
    grid_letters: Dict[CellPos, str] = {}
    for clue_id, word in assignment.items():
        cells = solver_input.clue_cells[clue_id]
        for i, cell in enumerate(cells):
            grid_letters[cell] = word[i]

    patterns: Dict[ClueId, str] = {}
    for clue_id, cells in solver_input.clue_cells.items():
        if clue_id in assignment:
            continue
        pattern = ""
        for cell in cells:
            if cell in grid_letters:
                pattern += grid_letters[cell]
            else:
                pattern += "_"
        patterns[clue_id] = pattern

    return patterns


def validate_assignment(
    solver_input: SolverInput,
    assignment: Dict[ClueId, Word],
) -> Dict[ClueId, Word]:
    """Remove any assignments that conflict at crossing cells."""
    # Build letter grid
    grid_letters: Dict[CellPos, Dict[ClueId, str]] = {}
    for clue_id, word in assignment.items():
        cells = solver_input.clue_cells[clue_id]
        for i, cell in enumerate(cells):
            if cell not in grid_letters:
                grid_letters[cell] = {}
            grid_letters[cell][clue_id] = word[i]

    # Find conflicts
    conflicting: set = set()
    for cell, clue_letters in grid_letters.items():
        letters = set(clue_letters.values())
        if len(letters) > 1:
            # Conflict at this cell — mark all clues involved
            for cid in clue_letters:
                conflicting.add(cid)

    # Remove conflicting assignments
    return {cid: word for cid, word in assignment.items() if cid not in conflicting}


def solve_pass(
    solver_input: SolverInput,
    clue_text_lookup: Dict[str, str],
    candidates: Dict[str, List[str]],
    assignment: Dict[ClueId, Word],
    pass_num: int,
    model: str = "claude-opus-4-20250514",
) -> Dict[ClueId, Word]:
    """Run one LLM solve pass. Returns new assignments from this pass."""

    unsolved = [
        cid for cid in solver_input.clue_cells
        if cid not in assignment
    ]

    if not unsolved:
        return {}

    patterns = extract_patterns(solver_input, assignment)

    # Build clue list for the prompt
    clue_lines = []
    for cid in unsolved:
        text = clue_text_lookup.get(cid, "")
        length = solver_input.clue_length(cid)
        pattern = patterns.get(cid, "_" * length)
        cands = candidates.get(cid, [])

        cand_str = ""
        if cands:
            # Filter candidates to those matching the pattern
            matching = []
            for c in cands:
                c_upper = c.upper()
                if len(c_upper) != length:
                    continue
                match = True
                for j, ch in enumerate(pattern):
                    if ch != "_" and ch != c_upper[j]:
                        match = False
                        break
                if match:
                    matching.append(c_upper)
            if matching:
                cand_str = f" candidates=[{', '.join(matching[:12])}]"

        known_count = sum(1 for ch in pattern if ch != "_")
        clue_lines.append(
            f"{cid}: \"{text}\" ({length} letters) pattern={pattern}{cand_str}"
        )

    grid_str = build_grid_state(solver_input, assignment)

    if pass_num == 1:
        instruction = """This is pass 1. Commit answers you are CERTAIN about (>90% confidence).
These will be locked in and their letters will constrain crossing clues in the next pass.
Look for: proper nouns, fixed phrases ("— ex machina" = DEUS), and clues with only one plausible answer.
Aim for 15-25 confident answers."""
    elif pass_num <= 3:
        instruction = f"""This is pass {pass_num}. You now have crossing letters from previous passes.
Use the patterns to narrow down answers — if only one candidate matches the pattern, commit it.
For example, if pattern is A_B_R and the clue is "Ann —, Michigan", ARBOR is the only possibility.
Be more aggressive now — the crossing letters give you strong constraints."""
    else:
        instruction = f"""This is pass {pass_num}. Most of the grid should be filled.
For remaining clues, use ALL available crossing letters and your best judgment.
At this stage it's better to commit a reasonable answer than leave blanks.
Only skip clues where you genuinely have no idea."""

    prompt = f"""{instruction}

## Current Grid
{grid_str}

## Unsolved Clues ({len(unsolved)} remaining)
{chr(10).join(clue_lines)}

## Already Solved ({len(assignment)} clues)
{', '.join(f'{cid}={word}' for cid, word in sorted(assignment.items()))}

Return a JSON object mapping clue_id to answer for ONLY the clues you're confident about.
Example: {{"14-across": "DEUS", "8-across": "APSE"}}
Return ONLY the JSON object, no other text. If you're not confident about any, return {{}}."""

    from src.solver.cost_tracker import get_tracker

    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SKILL_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    get_tracker().track(response, f"solve_pass_{pass_num}")

    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if "```" in text:
            text = text[:text.rfind("```")]
        text = text.strip()

    # Try to extract JSON from the response (may have surrounding text)
    try:
        new_answers = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            try:
                new_answers = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response")
                return {}
        else:
            logger.warning("No JSON found in LLM response")
            return {}

    # Validate: correct length, matches pattern, and is in candidates (if we have them)
    validated: Dict[ClueId, Word] = {}
    for cid, word in new_answers.items():
        word_upper = word.upper()
        if cid not in solver_input.clue_cells:
            continue
        expected_len = solver_input.clue_length(cid)
        if len(word_upper) != expected_len:
            logger.warning(f"Rejected {cid}={word_upper} (length {len(word_upper)} != {expected_len})")
            continue

        # Check pattern match
        pattern = patterns.get(cid, "_" * expected_len)
        pattern_ok = True
        for j, ch in enumerate(pattern):
            if ch != "_" and ch != word_upper[j]:
                pattern_ok = False
                break
        if not pattern_ok:
            logger.warning(f"Rejected {cid}={word_upper} (doesn't match pattern {pattern})")
            continue

        validated[cid] = word_upper

    return validated


def find_conflict_clusters(
    solver_input: SolverInput,
    assignment: Dict[ClueId, Word],
    clue_text_lookup: Dict[str, str],
) -> List[Dict]:
    """Find clusters of unsolved clues blocked by wrong committed answers.

    For each unsolved clue whose crossing-letter pattern matches NO valid word,
    trace back to the committed crossing answers that created the impossible
    pattern. Group connected dead-ends into clusters for joint resolution.
    """
    from src.solver.word_index import WordIndex
    word_index = WordIndex()

    unsolved = [cid for cid in solver_input.clue_cells if cid not in assignment]
    if not unsolved:
        return []

    patterns = extract_patterns(solver_input, assignment)

    # Step 1: Find dead-end clues — pattern highly constrained, no word matches
    dead_clues: List[Tuple[ClueId, str, set]] = []

    for cid in unsolved:
        pattern = patterns.get(cid, "")
        length = solver_input.clue_length(cid)
        known = sum(1 for ch in pattern if ch != "_")

        # Only check clues with at least half their letters constrained
        if known < length * 0.5:
            continue

        # Check if any word in the word index matches this pattern
        matches = word_index.match_pattern(pattern, max_results=1)
        if matches:
            continue  # Pattern is viable, not a dead end

        # Dead end — trace which crossing clues provided the constraining letters
        blamed: set = set()
        cells = solver_input.clue_cells[cid]
        for i, cell in enumerate(cells):
            if pattern[i] != "_":
                for crossing_cid in solver_input.cell_to_clues.get(cell, []):
                    if crossing_cid != cid and crossing_cid in assignment:
                        blamed.add(crossing_cid)

        dead_clues.append((cid, pattern, blamed))

    if not dead_clues:
        return []

    # Step 2: Cluster dead-end clues that share blamed crossing answers
    clusters_map: Dict[ClueId, int] = {}  # blamed_cid -> cluster index
    clusters: List[Optional[Dict]] = []

    for cid, pattern, blamed in dead_clues:
        connected = set()
        for bc in blamed:
            if bc in clusters_map:
                connected.add(clusters_map[bc])

        if not connected:
            idx = len(clusters)
            clusters.append({
                "unsolved": [(cid, pattern)],
                "blamed": set(blamed),
            })
            for bc in blamed:
                clusters_map[bc] = idx
        elif len(connected) == 1:
            idx = connected.pop()
            clusters[idx]["unsolved"].append((cid, pattern))
            clusters[idx]["blamed"].update(blamed)
            for bc in blamed:
                clusters_map[bc] = idx
        else:
            # Merge multiple clusters
            target = min(connected)
            for other in connected:
                if other != target:
                    clusters[target]["unsolved"].extend(clusters[other]["unsolved"])
                    clusters[target]["blamed"].update(clusters[other]["blamed"])
                    for bc in clusters[other]["blamed"]:
                        clusters_map[bc] = target
                    clusters[other] = None
            clusters[target]["unsolved"].append((cid, pattern))
            clusters[target]["blamed"].update(blamed)
            for bc in blamed:
                clusters_map[bc] = target

    # Build result with full clue info
    result = []
    for cluster in clusters:
        if cluster is None:
            continue
        result.append({
            "unsolved": [
                {
                    "clue_id": cid,
                    "text": clue_text_lookup.get(cid, ""),
                    "length": solver_input.clue_length(cid),
                    "pattern": pat,
                }
                for cid, pat in cluster["unsolved"]
            ],
            "blamed": [
                {
                    "clue_id": bc,
                    "text": clue_text_lookup.get(bc, ""),
                    "current_answer": assignment[bc],
                    "length": solver_input.clue_length(bc),
                }
                for bc in cluster["blamed"]
            ],
        })

    return result


def resolve_conflict_cluster(
    solver_input: SolverInput,
    clue_text_lookup: Dict[str, str],
    candidates: Dict[str, List[str]],
    assignment: Dict[ClueId, Word],
    cluster: Dict,
    model: str = "claude-opus-4-20250514",
) -> Dict[ClueId, Word]:
    """Ask the LLM to re-solve a conflict cluster.

    Removes blamed answers from the grid, shows the LLM the conflict,
    and asks it to find consistent replacements for the whole cluster.
    """
    # Describe the dead-end patterns
    unsolved_lines = []
    for u in cluster["unsolved"]:
        cands = candidates.get(u["clue_id"], [])
        cand_str = ""
        if cands:
            valid = [c.upper() for c in cands if len(c) == u["length"]][:8]
            if valid:
                cand_str = f" candidates=[{', '.join(valid)}]"
        unsolved_lines.append(
            f'{u["clue_id"]}: "{u["text"]}" ({u["length"]} letters) '
            f'current pattern={u["pattern"]} — NO VALID WORD MATCHES THIS{cand_str}'
        )

    blamed_lines = []
    for b in cluster["blamed"]:
        blamed_lines.append(
            f'{b["clue_id"]}: "{b["text"]}" ({b["length"]} letters) '
            f'currently={b["current_answer"]}'
        )

    # All clue IDs to re-solve
    blamed_set = {b["clue_id"] for b in cluster["blamed"]}
    all_cids = [u["clue_id"] for u in cluster["unsolved"]] + list(blamed_set)

    # Temporary assignment with blamed answers removed
    temp_assignment = {cid: w for cid, w in assignment.items() if cid not in blamed_set}
    patterns_fresh = extract_patterns(solver_input, temp_assignment)

    # Build clue info for the cluster
    cluster_clue_lines = []
    for cid in all_cids:
        text = clue_text_lookup.get(cid, "")
        length = solver_input.clue_length(cid)
        pattern = patterns_fresh.get(cid, "_" * length)
        cands = candidates.get(cid, [])

        cand_str = ""
        if cands:
            matching = []
            for c in cands:
                c_upper = c.upper()
                if len(c_upper) != length:
                    continue
                ok = all(
                    pattern[j] == "_" or pattern[j] == c_upper[j]
                    for j in range(length)
                )
                if ok:
                    matching.append(c_upper)
            if matching:
                cand_str = f" candidates=[{', '.join(matching[:12])}]"

        prev = f" (was {assignment[cid]})" if cid in blamed_set else ""
        cluster_clue_lines.append(
            f'{cid}: "{text}" ({length} letters) pattern={pattern}{cand_str}{prev}'
        )

    grid_str = build_grid_state(solver_input, temp_assignment)

    prompt = f"""## Conflict Resolution

Your previous answers for some clues created impossible crossing patterns. I've removed the problematic answers so you can re-solve this cluster of interconnected clues.

### The Problem
These clues have crossing patterns that match NO valid English word:
{chr(10).join(unsolved_lines)}

The impossible letters came from these committed answers (now removed):
{chr(10).join(blamed_lines)}

### Clues to Re-solve (with problematic answers removed from grid)
{chr(10).join(cluster_clue_lines)}

### Current Grid (problematic answers removed)
{grid_str}

Think step by step:
1. For each "blamed" clue, what ALTERNATIVE answer would create viable crossing patterns?
2. With that alternative in place, what word fits each previously-dead crossing clue?
3. Verify: do ALL answers in the cluster agree at every shared cell?

Use web search to verify any proper nouns or obscure terms.

Return a JSON object with answers for ALL clues in this cluster (both the blamed clues and the previously-unsolvable ones).
Example: {{"9-down": "PEPO", "17-across": "OPEN"}}
Return ONLY the JSON object."""

    from src.solver.cost_tracker import get_tracker

    client = anthropic.Anthropic()

    tools = [{
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    }]

    messages = [{"role": "user", "content": prompt}]

    response = None
    tracker = get_tracker()
    for _turn in range(4):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SKILL_PROMPT,
            messages=messages,
            tools=tools,
        )
        tracker.track(response, "conflict_resolution")

        search_count = sum(
            1 for b in response.content
            if getattr(b, "type", None) == "server_tool_use"
        )
        if search_count:
            logger.info(f"Web searches used: {search_count}")

        if response.stop_reason != "pause_turn":
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": "Continue. Return the JSON answer."})

    # Extract text from mixed content blocks
    text_parts = [b.text for b in response.content if hasattr(b, "text")]
    text = "\n".join(text_parts).strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if "```" in text:
            text = text[:text.rfind("```")]
        text = text.strip()

    try:
        new_answers = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            try:
                new_answers = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse conflict resolution response")
                return {}
        else:
            logger.warning("No JSON found in conflict resolution response")
            return {}

    # Validate lengths
    validated: Dict[ClueId, Word] = {}
    for cid, word in new_answers.items():
        word_upper = word.upper()
        if cid not in solver_input.clue_cells:
            continue
        expected_len = solver_input.clue_length(cid)
        if len(word_upper) != expected_len:
            logger.warning(f"Rejected {cid}={word_upper} (length {len(word_upper)} != {expected_len})")
            continue
        validated[cid] = word_upper

    return validated


def prefill_from_db(
    solver_input: SolverInput,
    candidates: Dict[str, List[str]],
) -> Dict[ClueId, Word]:
    """Pre-fill answers where the DB has exactly one candidate of the right length.

    These are high-confidence — if 'Wildebeest' → GNU appears in hundreds of
    historical puzzles and it's the only 3-letter match, lock it in.
    """
    assignment: Dict[ClueId, Word] = {}

    for clue_id, cells in solver_input.clue_cells.items():
        expected_len = len(cells)
        cands = candidates.get(clue_id, [])

        # Filter to candidates of the correct length
        valid = [c.upper() for c in cands if len(c) == expected_len]

        # Only lock in if there's exactly one valid candidate
        if len(valid) == 1:
            assignment[clue_id] = valid[0]

    # Validate crossings — remove any that conflict
    validated = validate_assignment(solver_input, assignment)
    return validated


def _get_dictionary_definitions(word: str) -> Optional[str]:
    """Fetch definitions from Free Dictionary API. Returns combined definition text or None."""
    import requests

    try:
        resp = requests.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}",
            timeout=5,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        all_defs = []
        for entry in data:
            for meaning in entry.get("meanings", []):
                for defn in meaning.get("definitions", []):
                    all_defs.append(defn.get("definition", ""))
                for syn in meaning.get("synonyms", []):
                    all_defs.append(syn)
        return " | ".join(all_defs) if all_defs else None
    except Exception:
        return None


def _dictionary_and_haiku_confirm(word: str, clue_text: str) -> bool:
    """Verify a fully-constrained word fits the clue using dictionary + Haiku.

    1. Fetch dictionary definitions (free API)
    2. Quick word-overlap check — if clue word appears in definition, accept
    3. Otherwise, ask Haiku with the definition text: "does this word fit this clue?"

    Returns True on network failures (benefit of the doubt).
    """

    if not clue_text:
        return True

    # Step 1: Get dictionary definitions
    def_text = _get_dictionary_definitions(word)

    # Step 2: Quick word-overlap check
    if def_text:
        skip = {"a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "is",
                "it", "by", "at", "no", "not", "so", "as", "up", "do", "be", "he",
                "she", "we", "my", "me", "us", "its", "has", "had", "was", "are",
                "but", "if", "can", "may", "one", "two", "who", "how", "all", "any",
                "own", "too", "did", "get", "got", "let", "say", "new", "old", "big",
                "see", "way", "day", "man", "his", "her", "out", "now", "use"}
        clue_words = set()
        for w in clue_text.lower().split():
            cleaned = w.strip("\"'.,!?;:-\u2014()[]")
            if len(cleaned) > 2 and cleaned not in skip:
                clue_words.add(cleaned)

        if clue_words:
            def_lower = def_text.lower()
            for cw in clue_words:
                if cw in def_lower:
                    logger.debug(f"Dictionary match: '{cw}' found in definition of {word}")
                    return True

    # Step 3: Ask Haiku — is this word a valid answer for this clue?
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return True  # Can't verify, allow it

    try:
        import anthropic
        from src.solver.cost_tracker import get_tracker

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f'Crossword clue: "{clue_text}"\nProposed answer: {word}'
        if def_text:
            prompt += f'\nDictionary definition of {word}: {def_text}'
        prompt += '\n\nIs this answer correct for this crossword clue? Reply with ONLY "yes" or "no".'

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        get_tracker().track(response, "verify_word", model="claude-haiku-4-5-20251001")

        answer = response.content[0].text.strip().lower()
        confirmed = answer.startswith("yes")
        logger.debug(f"Haiku verification: {word} for \"{clue_text}\" → {answer}")
        return confirmed

    except Exception:
        return True  # Network error — allow it


def propagate_constraints(
    solver_input: SolverInput,
    candidates: Dict[str, List[str]],
    assignment: Dict[ClueId, Word],
    candidate_scores: Optional[Dict[str, Dict[str, float]]] = None,
    min_bouncer_score: float = 0.6,
    clue_text_lookup: Optional[Dict[str, str]] = None,
) -> Dict[ClueId, Word]:
    """Auto-commit clues where crossing-letter patterns eliminate all but one candidate.

    Pure logic, zero API cost, runs in milliseconds. Iterates until no more
    single-candidate clues remain (each commit may unlock others).

    For fully-constrained patterns (all letters known from crossings), verifies
    the word exists in the word index AND its dictionary definition matches the
    clue text before committing.

    Args:
        solver_input: Solver input with grid structure.
        candidates: Dict of clue_id -> list of candidate words.
        assignment: Current assignment (mutated in-place).
        candidate_scores: Optional dict of clue_id -> {word -> bouncer_score}.
            Used when 2+ candidates survive the pattern — auto-commits the top
            one if its score is >= min_bouncer_score and significantly higher.
        min_bouncer_score: Minimum bouncer score for auto-committing a candidate
            when it's the sole survivor or the clear top choice.
        clue_text_lookup: Optional dict of clue_id -> clue text for dictionary verification.

    Returns:
        Dict of newly committed clue_id -> word (subset added to assignment).
    """
    from src.solver.word_index import WordIndex
    word_index = WordIndex()

    new_commits: Dict[ClueId, Word] = {}
    changed = True

    while changed:
        changed = False
        patterns = extract_patterns(solver_input, assignment)

        for cid in list(patterns.keys()):
            if cid in assignment:
                continue

            pattern = patterns[cid]
            length = solver_input.clue_length(cid)
            known = sum(1 for ch in pattern if ch != "_")

            # Only consider clues with at least 1 crossing letter
            if known == 0:
                continue

            # Fully constrained pattern (all letters known from crossings)
            # — the word is forced. Verify it's valid before committing.
            if known == length:
                if not word_index.contains(pattern):
                    continue

                # If the word is already in the candidate list, trust it —
                # candidate generation already verified it fits the clue.
                in_candidates = pattern in [c.upper() for c in candidates.get(cid, [])]

                if in_candidates:
                    trial = {**assignment, cid: pattern}
                    validated = validate_assignment(solver_input, trial)
                    if cid in validated:
                        assignment[cid] = pattern
                        new_commits[cid] = pattern
                        changed = True
                        logger.debug(f"{cid} = {pattern} (fully constrained, in candidates)")
                else:
                    # Word NOT in candidates — use dictionary + Haiku to verify it fits the clue
                    clue_text = clue_text_lookup.get(cid, "") if clue_text_lookup else ""
                    confirmed = _dictionary_and_haiku_confirm(pattern, clue_text)
                    if confirmed:
                        trial = {**assignment, cid: pattern}
                        validated = validate_assignment(solver_input, trial)
                        if cid in validated:
                            assignment[cid] = pattern
                            new_commits[cid] = pattern
                            changed = True
                            logger.debug(f"{cid} = {pattern} (fully constrained, verified)")
                    else:
                        logger.debug(f"{cid}: pattern {pattern} is a word but verification rejected it for clue \"{clue_text}\"")
                continue

            cands = candidates.get(cid, [])
            if not cands:
                continue

            # Filter candidates to those matching the pattern
            matching = []
            for c in cands:
                c_upper = c.upper()
                if len(c_upper) != length:
                    continue
                ok = all(
                    pattern[j] == "_" or pattern[j] == c_upper[j]
                    for j in range(length)
                )
                if ok:
                    matching.append(c_upper)

            candidate_word = None
            if len(matching) == 1:
                word = matching[0]
                # Check bouncer score if available
                if candidate_scores:
                    score = candidate_scores.get(cid, {}).get(word, 0.0)
                    if score < min_bouncer_score:
                        continue
                candidate_word = word
            elif len(matching) >= 2 and candidate_scores:
                # 2+ survivors — check if one clearly dominates
                scored = [(w, candidate_scores.get(cid, {}).get(w, 0.0)) for w in matching]
                scored.sort(key=lambda x: x[1], reverse=True)
                top_word, top_score = scored[0]
                runner_up_score = scored[1][1]

                # Auto-commit if top candidate has strong score and clear lead
                if top_score >= min_bouncer_score and top_score - runner_up_score >= 0.15:
                    candidate_word = top_word

            if candidate_word:
                # Validate this single commit against current assignment
                trial = {**assignment, cid: candidate_word}
                validated = validate_assignment(solver_input, trial)
                if cid in validated:
                    assignment[cid] = candidate_word
                    new_commits[cid] = candidate_word
                    changed = True

    return new_commits


def solve_with_llm(
    solver_input: SolverInput,
    clue_text_lookup: Dict[str, str],
    candidates: Dict[str, List[str]],
    max_passes: int = 6,
    progress_callback=None,
    candidate_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[ClueId, Word]:
    """
    Iterative LLM-based crossword solver.

    Returns a full or partial assignment.
    """
    total = len(solver_input.clue_cells)

    # Phase 0: Pre-fill from DB (single-candidate clues)
    assignment = prefill_from_db(solver_input, candidates)
    if assignment:
        logger.info(f"DB pre-fill: {len(assignment)}/{total} locked in")

    t0 = time.time()

    for pass_num in range(1, max_passes + 1):
        remaining = total - len(assignment)
        if remaining == 0:
            break

        elapsed = time.time() - t0
        logger.info(f"[{elapsed:.1f}s] LLM solve pass {pass_num}: {len(assignment)}/{total} solved, {remaining} remaining")

        if progress_callback:
            progress_callback(pass_num, len(assignment), total)

        new_answers = solve_pass(
            solver_input, clue_text_lookup, candidates,
            assignment, pass_num,
        )

        if not new_answers:
            logger.info("No new answers — LLM is stuck")
            break

        # Merge and validate crossings
        trial = {**assignment, **new_answers}
        validated = validate_assignment(solver_input, trial)

        # Count how many new answers survived validation
        new_valid = {cid: w for cid, w in new_answers.items() if cid in validated}
        rejected = len(new_answers) - len(new_valid)

        assignment = validated
        elapsed = time.time() - t0
        logger.info(f"[{elapsed:.1f}s] +{len(new_valid)} answers ({rejected} rejected for conflicts), total: {len(assignment)}/{total}")

        if len(new_valid) == 0:
            logger.info("All new answers conflicted — stopping")
            break

    # Phase: Constraint propagation — auto-commit clues where patterns leave 1 candidate
    remaining = total - len(assignment)
    if remaining > 0:
        elapsed = time.time() - t0
        logger.info(f"[{elapsed:.1f}s] Constraint propagation ({remaining} unsolved)...")
        prop_commits = propagate_constraints(
            solver_input, candidates, assignment,
            candidate_scores=candidate_scores,
            clue_text_lookup=clue_text_lookup,
        )
        if prop_commits:
            elapsed = time.time() - t0
            logger.info(f"[{elapsed:.1f}s] +{len(prop_commits)} from constraint propagation, total: {len(assignment)}/{total}")
            for cid, word in sorted(prop_commits.items()):
                logger.debug(f"{cid} = {word}")
        else:
            logger.info("No auto-commits possible")

    # Phase: Conflict resolution — backtrack on wrong answers blocking unsolved clues
    remaining = total - len(assignment)
    if remaining > 0:
        elapsed = time.time() - t0
        logger.info(f"[{elapsed:.1f}s] Starting conflict resolution ({remaining} unsolved)...")

        clusters = find_conflict_clusters(solver_input, assignment, clue_text_lookup)

        if clusters:
            for i, cluster in enumerate(clusters):
                unsolved_ids = [u["clue_id"] for u in cluster["unsolved"]]
                blamed_strs = [
                    f'{b["clue_id"]}={b["current_answer"]}'
                    for b in cluster["blamed"]
                ]
                logger.info(f"Cluster {i+1}: {len(unsolved_ids)} dead-end clues, "
                            f"{len(cluster['blamed'])} blamed answers")
                logger.info(f"Dead: {', '.join(unsolved_ids)}")
                logger.info(f"Blamed: {', '.join(blamed_strs)}")

                new_answers = resolve_conflict_cluster(
                    solver_input, clue_text_lookup, candidates,
                    assignment, cluster,
                )

                if new_answers:
                    # Remove blamed answers, merge in new ones
                    blamed_set = {b["clue_id"] for b in cluster["blamed"]}
                    trial = {cid: w for cid, w in assignment.items()
                             if cid not in blamed_set}
                    trial.update(new_answers)
                    validated = validate_assignment(solver_input, trial)

                    old_count = len(assignment)
                    assignment = validated
                    delta = len(assignment) - old_count
                    elapsed = time.time() - t0
                    logger.info(f"[{elapsed:.1f}s] Resolved: {len(assignment)}/{total} "
                               f"({'+' if delta >= 0 else ''}{delta})")
                else:
                    logger.info("No resolution found")
            # After resolving conflicts, run follow-up passes to pick up
            # newly-unblocked clues (conflict resolution adds crossing letters)
            post_remaining = total - len(assignment)
            if post_remaining > 0 and post_remaining < remaining:
                elapsed = time.time() - t0
                logger.info(f"[{elapsed:.1f}s] Post-resolution pass ({post_remaining} remaining)...")
                new_answers = solve_pass(
                    solver_input, clue_text_lookup, candidates,
                    assignment, pass_num=max_passes + 1,
                )
                if new_answers:
                    trial = {**assignment, **new_answers}
                    validated = validate_assignment(solver_input, trial)
                    new_valid = {cid: w for cid, w in new_answers.items() if cid in validated}
                    assignment = validated
                    elapsed = time.time() - t0
                    logger.info(f"[{elapsed:.1f}s] +{len(new_valid)} from post-resolution, total: {len(assignment)}/{total}")
        else:
            logger.info("No conflict clusters detected")

    # Final constraint propagation after conflict resolution
    remaining = total - len(assignment)
    if remaining > 0:
        final_commits = propagate_constraints(
            solver_input, candidates, assignment,
            candidate_scores=candidate_scores,
            clue_text_lookup=clue_text_lookup,
        )
        if final_commits:
            elapsed = time.time() - t0
            logger.info(f"[{elapsed:.1f}s] +{len(final_commits)} from final propagation, total: {len(assignment)}/{total}")

    elapsed = time.time() - t0
    logger.info(f"[{elapsed:.1f}s] LLM solver done: {len(assignment)}/{total}")
    return assignment
