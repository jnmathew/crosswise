"""
LLM-based iterative crossword solver.

Instead of CSP search, uses Claude as the solver in multiple passes:
1. Confidence pass — commit answers the LLM is certain about
2. Pattern passes — use crossing letters to constrain remaining clues
3. CSP cleanup — tiny residual subproblem if needed

Guided by a crossword-solving skill prompt that teaches constraint-first strategy.
"""

import json
import time
from typing import Dict, List, Optional, Tuple

import anthropic

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
Use web search to verify any proper nouns you're unsure about.
Aim for 15-25 confident answers."""
    elif pass_num <= 3:
        instruction = f"""This is pass {pass_num}. You now have crossing letters from previous passes.
Use the patterns to narrow down answers — if only one candidate matches the pattern, commit it.
For example, if pattern is A_B_R and the clue is "Ann —, Michigan", ARBOR is the only possibility.
Search to verify ambiguous proper nouns or factual answers before committing.
Be more aggressive now — the crossing letters give you strong constraints."""
    else:
        instruction = f"""This is pass {pass_num}. Most of the grid should be filled.
For remaining clues, use ALL available crossing letters and your best judgment.
Use web search for any factual clues you're still unsure about.
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

    client = anthropic.Anthropic()

    tools = [{
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    }]

    messages = [{"role": "user", "content": prompt}]

    # Loop to handle pause_turn (server-side search may need continuation)
    response = None
    search_count = 0
    for _turn in range(4):  # up to 3 continuations + initial
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SKILL_PROMPT,
            messages=messages,
            tools=tools,
        )

        # Count web searches in this response
        for block in response.content:
            if getattr(block, "type", None) == "server_tool_use":
                search_count += 1

        if response.stop_reason != "pause_turn":
            break

        # Continue the conversation so Claude can finish after search results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": "Continue solving. Return your JSON answer."})

    if search_count > 0:
        print(f"    Web searches used: {search_count}")

    # Extract text from mixed content blocks (response may contain
    # server_tool_use and web_search_tool_result blocks alongside text)
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    text = "\n".join(text_parts).strip()
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
        import re
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            try:
                new_answers = json.loads(json_match.group())
            except json.JSONDecodeError:
                print(f"    Warning: could not parse LLM response")
                return {}
        else:
            print(f"    Warning: no JSON found in LLM response")
            return {}

    # Validate: correct length, matches pattern, and is in candidates (if we have them)
    validated: Dict[ClueId, Word] = {}
    for cid, word in new_answers.items():
        word_upper = word.upper()
        if cid not in solver_input.clue_cells:
            continue
        expected_len = solver_input.clue_length(cid)
        if len(word_upper) != expected_len:
            print(f"    Rejected {cid}={word_upper} (length {len(word_upper)} != {expected_len})")
            continue

        # Check pattern match
        pattern = patterns.get(cid, "_" * expected_len)
        pattern_ok = True
        for j, ch in enumerate(pattern):
            if ch != "_" and ch != word_upper[j]:
                pattern_ok = False
                break
        if not pattern_ok:
            print(f"    Rejected {cid}={word_upper} (doesn't match pattern {pattern})")
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

    client = anthropic.Anthropic()

    tools = [{
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
    }]

    messages = [{"role": "user", "content": prompt}]

    response = None
    for _turn in range(4):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SKILL_PROMPT,
            messages=messages,
            tools=tools,
        )

        search_count = sum(
            1 for b in response.content
            if getattr(b, "type", None) == "server_tool_use"
        )
        if search_count:
            print(f"      Web searches used: {search_count}")

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
        import re
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            try:
                new_answers = json.loads(json_match.group())
            except json.JSONDecodeError:
                print(f"      Warning: could not parse conflict resolution response")
                return {}
        else:
            print(f"      Warning: no JSON found in conflict resolution response")
            return {}

    # Validate lengths
    validated: Dict[ClueId, Word] = {}
    for cid, word in new_answers.items():
        word_upper = word.upper()
        if cid not in solver_input.clue_cells:
            continue
        expected_len = solver_input.clue_length(cid)
        if len(word_upper) != expected_len:
            print(f"      Rejected {cid}={word_upper} (length {len(word_upper)} != {expected_len})")
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


def solve_with_llm(
    solver_input: SolverInput,
    clue_text_lookup: Dict[str, str],
    candidates: Dict[str, List[str]],
    max_passes: int = 6,
    progress_callback=None,
) -> Dict[ClueId, Word]:
    """
    Iterative LLM-based crossword solver.

    Returns a full or partial assignment.
    """
    total = len(solver_input.clue_cells)

    # Phase 0: Pre-fill from DB (single-candidate clues)
    assignment = prefill_from_db(solver_input, candidates)
    if assignment:
        print(f"  DB pre-fill: {len(assignment)}/{total} locked in")

    t0 = time.time()

    for pass_num in range(1, max_passes + 1):
        remaining = total - len(assignment)
        if remaining == 0:
            break

        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s] LLM solve pass {pass_num}: {len(assignment)}/{total} solved, {remaining} remaining")

        if progress_callback:
            progress_callback(pass_num, len(assignment), total)

        new_answers = solve_pass(
            solver_input, clue_text_lookup, candidates,
            assignment, pass_num,
        )

        if not new_answers:
            print(f"    No new answers — LLM is stuck")
            break

        # Merge and validate crossings
        trial = {**assignment, **new_answers}
        validated = validate_assignment(solver_input, trial)

        # Count how many new answers survived validation
        new_valid = {cid: w for cid, w in new_answers.items() if cid in validated}
        rejected = len(new_answers) - len(new_valid)

        assignment = validated
        elapsed = time.time() - t0
        print(f"    [{elapsed:.1f}s] +{len(new_valid)} answers ({rejected} rejected for conflicts), total: {len(assignment)}/{total}")

        if len(new_valid) == 0:
            print(f"    All new answers conflicted — stopping")
            break

    # Phase: Conflict resolution — backtrack on wrong answers blocking unsolved clues
    remaining = total - len(assignment)
    if remaining > 0:
        elapsed = time.time() - t0
        print(f"  [{elapsed:.1f}s] Starting conflict resolution ({remaining} unsolved)...")

        clusters = find_conflict_clusters(solver_input, assignment, clue_text_lookup)

        if clusters:
            for i, cluster in enumerate(clusters):
                unsolved_ids = [u["clue_id"] for u in cluster["unsolved"]]
                blamed_strs = [
                    f'{b["clue_id"]}={b["current_answer"]}'
                    for b in cluster["blamed"]
                ]
                print(f"    Cluster {i+1}: {len(unsolved_ids)} dead-end clues, "
                      f"{len(cluster['blamed'])} blamed answers")
                print(f"      Dead: {', '.join(unsolved_ids)}")
                print(f"      Blamed: {', '.join(blamed_strs)}")

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
                    print(f"      [{elapsed:.1f}s] Resolved: {len(assignment)}/{total} "
                          f"({'+' if delta >= 0 else ''}{delta})")
                else:
                    print(f"      No resolution found")
            # After resolving conflicts, run follow-up passes to pick up
            # newly-unblocked clues (conflict resolution adds crossing letters)
            post_remaining = total - len(assignment)
            if post_remaining > 0 and post_remaining < remaining:
                elapsed = time.time() - t0
                print(f"  [{elapsed:.1f}s] Post-resolution pass ({post_remaining} remaining)...")
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
                    print(f"    [{elapsed:.1f}s] +{len(new_valid)} from post-resolution, total: {len(assignment)}/{total}")
        else:
            print(f"    No conflict clusters detected")

    elapsed = time.time() - t0
    print(f"  [{elapsed:.1f}s] LLM solver done: {len(assignment)}/{total}")
    return assignment
