"""
Candidate generation for crossword clues.

Primary source: TSV/SQLite database lookup (~9-11M historical pairs).
LLM fallback: Claude Opus for batch generation, Sonnet with extended thinking
for wordplay clues. Some OpenAI functions remain for the legacy solve_puzzle.py CLI.
"""

import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, TYPE_CHECKING, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

if TYPE_CHECKING:
    from crosswise.solver.clue_database import ClueDatabase

# Load environment variables from .env
load_dotenv()

# Default batch size for API calls
DEFAULT_BATCH_SIZE = 30
DEFAULT_CANDIDATES_PER_CLUE = 4
DEFAULT_MIN_CANDIDATES = 5


@dataclass
class ClueInput:
    """Input for candidate generation."""

    clue_id: str  # e.g., "1-across"
    text: str  # e.g., "Capital of France"
    length: int  # Total answer length (sum of word lengths)
    pattern: Optional[str] = None  # e.g., "C_T" for constrained generation
    category: Optional[str] = None  # "trivia", "definition", "wordplay", "fillin"
    num_crossings: int = 0  # Number of crossing clues (for domain sizing)


@dataclass
class ScoredCandidate:
    """A candidate answer with source and verification metadata."""

    word: str
    source: str  # "database", "llm", "word_index", "sniper"
    confidence: float  # 0.0-1.0 from source
    verified: bool  # True if found in DB or word index (Bouncer check)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScoredCandidate) and self.word == other.word

    def __hash__(self) -> int:
        return hash(self.word)


def to_plain_candidates(
    scored: Dict[str, List["ScoredCandidate"]],
) -> Dict[str, List[str]]:
    """Convert scored candidates to plain word lists (verified first)."""
    return {
        clue_id: [sc.word for sc in candidates]
        for clue_id, candidates in scored.items()
    }


def to_score_map(
    scored: Dict[str, List["ScoredCandidate"]],
) -> Dict[str, Dict[str, float]]:
    """Extract a score map: clue_id -> {word: composite_score}.

    Used by the CSP solver for value ordering (try high-score candidates first).
    """
    result: Dict[str, Dict[str, float]] = {}
    for clue_id, candidates in scored.items():
        result[clue_id] = {sc.word: sc.confidence for sc in candidates}
    return result


@dataclass
class CandidateResult:
    """Result of candidate generation for one clue."""

    clue_id: str
    candidates: List[str]
    error: Optional[str] = None


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


def _matches_pattern(word: str, pattern: str) -> bool:
    """Check if a word matches a pattern like 'C_T' where _ is wildcard."""
    if len(word) != len(pattern):
        return False
    for w_char, p_char in zip(word, pattern):
        if p_char != "_" and w_char != p_char:
            return False
    return True


def _is_pop_culture_clue(text: str) -> bool:
    """Heuristic: does this clue broadly reference pop culture or proper nouns?

    These are the clues where web search adds the most value — song titles,
    movie/TV names, celebrity references, book titles, brand names, etc.
    """
    t = text.strip()
    t_lower = t.lower()

    # Contains quoted text (e.g., '"— — Love Her"')
    if '"' in t or '\u201c' in t or '\u201d' in t:
        return True

    # Fill-in-the-blank with proper-noun context: has em-dash "—" and a capitalized word
    if '\u2014' in t or '—' in t or '---' in t:
        # Check for a capitalized word (not just the first word)
        words = t.split()
        for w in words:
            cleaned = w.strip('",.\u2014—\'')
            if cleaned and cleaned[0].isupper() and len(cleaned) > 1 and cleaned not in ('The', 'A', 'An', 'Or', 'And', 'In', 'On', 'Of', 'For', 'To', 'Is', 'It', 'At', 'By'):
                return True

    # References to media / entertainment
    media_signals = [
        'singer', 'actress', 'actor', 'director', 'author', 'novelist',
        'film', 'movie', 'show', 'series', 'song', 'album', 'band',
        'novel', 'book', 'play', 'musical', 'tv', 'cartoon', 'comic',
        'brand', 'company', 'network', 'channel', 'magazine',
        "beatles'", "beatles", 'grammy', 'oscar', 'emmy', 'tony',
        'broadway', 'hollywood', 'disney', 'marvel', 'netflix',
    ]
    for signal in media_signals:
        if signal in t_lower:
            return True

    # Contains a proper noun mid-clue (capitalized word that isn't the first word)
    words = t.split()
    if len(words) >= 3:
        for w in words[1:]:
            cleaned = w.strip('",.\u2014—\'?!;:')
            if cleaned and cleaned[0].isupper() and len(cleaned) > 1 and cleaned not in ('The', 'A', 'An', 'Or', 'And', 'In', 'On', 'Of', 'For', 'To', 'Is', 'It', 'At', 'By', 'No', 'Not', 'So'):
                return True

    return False


def _extract_answer(raw: str, expected_length: int) -> Optional[str]:
    """Extract a valid answer from Haiku's response text.

    Handles common response formats:
    - "SAM" (clean)
    - "The answer is SAM." (extra text)
    - "SAM (Sam Snead)" (parenthetical)
    - "**SAM**" (markdown bold)
    """
    # Clean up markdown bold
    cleaned = raw.replace("**", "").replace("*", "").strip()

    # Try 1: the whole response after basic cleanup
    simple = cleaned.strip('"\'.,!? ').replace(" ", "").upper()
    if len(simple) == expected_length and simple.isalpha():
        return simple

    # Try 2: find all-caps words of the right length (strongest signal)
    words = re.findall(r'[A-Za-z]+', cleaned)
    for w in words:
        if len(w) == expected_length and w.isupper():
            return w

    # Try 3: scan from the end — answers tend to be the last meaningful word
    for w in reversed(words):
        if len(w) == expected_length:
            return w.upper()

    return None


def web_search_prepass(
    clues: List[ClueInput],
) -> Dict[str, str]:
    """Run pop-culture clues through Haiku with web search to get verified candidates.

    Filters clues to those broadly referencing pop culture (proper nouns, media,
    fill-in-the-blank with names), then fires individual Haiku calls in parallel.
    Each call has web_search available (max_uses=1); Haiku decides whether to search.

    Args:
        clues: All clue inputs for the puzzle.

    Returns:
        Dict mapping clue_id to a single web-verified candidate answer (uppercase).
    """
    from crosswise.solver.cost_tracker import get_tracker

    pop_clues = [c for c in clues if _is_pop_culture_clue(c.text)]
    if not pop_clues:
        logger.info("Web pre-pass: no pop culture clues detected")
        return {}

    logger.info(f"Web pre-pass: {len(pop_clues)}/{len(clues)} clues identified as pop culture")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, skipping web pre-pass")
        return {}

    import anthropic

    results: Dict[str, str] = {}
    tracker = get_tracker()

    def _search_one(clue: ClueInput) -> Optional[tuple]:
        """Search for one clue. Returns (clue_id, answer) or None."""
        try:
            client = anthropic.Anthropic(api_key=api_key)

            prompt = (
                f'Crossword clue: "{clue.text}" ({clue.length} letters)\n\n'
                f'Use web search to find the answer, then reply with ONLY '
                f'the {clue.length}-letter answer in uppercase. Nothing else.'
            )

            tools = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1,
            }]

            messages = [{"role": "user", "content": prompt}]

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=messages,
                tools=tools,
            )
            tracker.track(response, "web_prepass", model="claude-haiku-4-5-20251001")

            # Handle pause_turn — follow up until we get a final answer (max 3 continuations)
            for _ in range(3):
                if response.stop_reason != "pause_turn":
                    break
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": f"Reply with ONLY the {clue.length}-letter answer in uppercase."})
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    messages=messages,
                    tools=tools,
                )
                tracker.track(response, "web_prepass_cont", model="claude-haiku-4-5-20251001")

            # Extract answer from response
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            raw = "".join(text_parts).strip()

            # Try to extract a word of the right length from the response
            answer = _extract_answer(raw, clue.length)
            if answer:
                return (clue.clue_id, answer)

            logger.debug(f"Web pre-pass {clue.clue_id}: no valid {clue.length}-letter answer from: {raw!r}")
            return None

        except Exception as e:
            logger.warning(f"Web pre-pass {clue.clue_id} failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_search_one, c): c for c in pop_clues}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results[result[0]] = result[1]

    found = len(results)
    logger.info(f"Web pre-pass: {found}/{len(pop_clues)} clues got web-verified candidates")
    return results


def categorize_clue(text: str) -> str:
    """
    Heuristic clue categorization.

    Returns one of: "trivia", "definition", "wordplay", "fillin".
    """
    text_stripped = text.strip()
    # Wordplay: ends with ? or has pun indicators
    if text_stripped.endswith("?") or "perhaps" in text_stripped.lower() or ", say" in text_stripped.lower():
        return "wordplay"
    # Fill-in-the-blank: contains blank markers
    if "___" in text_stripped or "…" in text_stripped or "..." in text_stripped:
        return "fillin"
    return "definition"


def compute_target_domain_size(
    category: str,
    db_hit_count: int,
    num_crossings: int,
) -> int:
    """
    Compute target domain size based on clue category and crossing density.

    Category-based targets:
    - High-confidence trivia (DB match >= 3): 3-5
    - Standard definition/fill-in-blank: 5-10
    - Wordplay/puns: 10-20

    Crossing-density floors:
    - 4-5 crossings: floor 2
    - 2-3 crossings: floor 5
    - 0-1 crossings: floor 10
    """
    # Category-based target
    if db_hit_count >= 3:
        target = 5
    elif category in ("definition", "fillin"):
        target = 10
    else:  # wordplay
        target = 20

    # Crossing-density floor
    if num_crossings >= 4:
        floor = 2
    elif num_crossings >= 2:
        floor = 5
    else:
        floor = 10

    return max(target, floor)


def bouncer_filter(
    candidates: Dict[str, List[str]],
    db: Optional["ClueDatabase"] = None,
    word_index: Optional[Any] = None,
    clue_text_lookup: Optional[Dict[str, str]] = None,
    candidate_sources: Optional[Dict[str, Dict[str, str]]] = None,
    web_candidates: Optional[Dict[str, str]] = None,
) -> Dict[str, List["ScoredCandidate"]]:
    """
    Cross-reference LLM candidates against DB and word index (Bouncer Filter).

    Computes a composite confidence score per candidate:
    - Source: DB-verified (0.8), word-index-verified (0.6), unverified (0.3)
    - Broda bonus: up to +0.15 from word index quality score
    - Clue-text match bonus: +0.1 if candidate was seen as answer to this clue in DB
    - Web confirmation bonus: +0.1 if candidate matches Haiku web pre-pass result

    Candidates sorted by composite score (highest first).

    Args:
        candidates: Dict mapping clue_id to list of candidate words
        db: ClueDatabase for answer verification
        word_index: WordIndex for membership testing and Broda scores
        clue_text_lookup: Optional dict of clue_id -> clue text (for DB clue match bonus)
        candidate_sources: Optional dict of clue_id -> {word -> source_label} for accurate source tracking
        web_candidates: Optional dict of clue_id -> web-verified answer from Haiku pre-pass

    Returns:
        Dict mapping clue_id to list of ScoredCandidate, sorted by score.
    """
    scored: Dict[str, List[ScoredCandidate]] = {}
    for clue_id, words in candidates.items():
        clue_scores: List[ScoredCandidate] = []
        clue_text = (clue_text_lookup or {}).get(clue_id, "")

        for word in words:
            word_upper = word.upper()
            is_in_db = False
            is_in_index = False
            is_clue_match = False

            if db is not None:
                try:
                    cursor = db._conn.execute(
                        "SELECT 1 FROM clues WHERE answer = ? LIMIT 1",
                        (word_upper,),
                    )
                    is_in_db = cursor.fetchone() is not None

                    # Check if this specific clue-answer pair exists in DB
                    if is_in_db and clue_text:
                        cursor2 = db._conn.execute(
                            "SELECT 1 FROM clues WHERE clue_normalized = ? AND answer = ? LIMIT 1",
                            (clue_text.lower(), word_upper),
                        )
                        is_clue_match = cursor2.fetchone() is not None
                except Exception:
                    pass

            if word_index is not None:
                is_in_index = word_index.contains(word_upper)

            # Compute composite score
            # Base: source reliability
            if is_in_db:
                base_score = 0.8
            elif is_in_index:
                base_score = 0.6
            else:
                base_score = 0.3

            # Broda quality bonus (0 to 0.15)
            broda_bonus = 0.0
            if word_index is not None:
                raw_score = word_index.score(word_upper)
                if raw_score > 0:
                    broda_bonus = min(raw_score / 100.0, 1.0) * 0.15

            # Clue-text match bonus
            clue_bonus = 0.1 if is_clue_match else 0.0

            # Web confirmation bonus: candidate matches Haiku web pre-pass result
            web_bonus = 0.0
            if web_candidates and clue_id in web_candidates:
                if word_upper == web_candidates[clue_id]:
                    web_bonus = 0.1

            confidence = min(base_score + broda_bonus + clue_bonus + web_bonus, 1.0)
            verified = is_in_db or is_in_index

            # Use tracked source if available, otherwise infer from verification
            if candidate_sources and clue_id in candidate_sources:
                source = candidate_sources[clue_id].get(word_upper, "llm")
            else:
                source = "llm"

            clue_scores.append(
                ScoredCandidate(word=word_upper, source=source, confidence=confidence, verified=verified)
            )

        # Sort by confidence descending (best candidates first)
        clue_scores.sort(key=lambda sc: sc.confidence, reverse=True)
        scored[clue_id] = clue_scores
    return scored


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


def generate_candidates_with_database(
    clues: List[ClueInput],
    db: "ClueDatabase",
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_llm_fallback: bool = True,
    on_progress: Optional[callable] = None,
) -> Dict[str, List[str]]:
    """
    Generate candidates using database lookup first, with optional LLM fallback.

    This is the preferred method when a clue database is available.
    Database lookups are fast and provide historically accurate answers.

    Args:
        clues: List of ClueInput objects
        db: ClueDatabase instance for lookups
        candidates_per_clue: Max candidates to return per clue
        batch_size: Clues per LLM batch (if fallback needed)
        api_key: OpenAI API key (for LLM fallback)
        model: OpenAI model to use (for LLM fallback)
        use_llm_fallback: Whether to use LLM for clues not in database
        on_progress: Optional callback(phase, clue_id, candidates_found)

    Returns:
        Dict mapping clue_id to list of candidate words
    """
    if not clues:
        return {}

    candidates: Dict[str, List[str]] = {}
    llm_needed: List[ClueInput] = []

    # Phase 1: Database lookup
    db_hits = 0
    for clue in clues:
        if clue.pattern and "_" in clue.pattern:
            # Have pattern constraint - use clue + pattern lookup
            answers = db.lookup_by_clue_and_pattern(
                clue.text,
                clue.pattern,
                max_results=candidates_per_clue,
            )
        else:
            # No pattern - use clue text lookup
            answers = db.lookup_by_clue(
                clue.text,
                clue.length,
                max_results=candidates_per_clue,
            )

        if answers:
            candidates[clue.clue_id] = answers
            db_hits += 1
            if on_progress:
                on_progress("database", clue.clue_id, len(answers))
        else:
            llm_needed.append(clue)

    logger.info(f"Database lookup: {db_hits}/{len(clues)} clues found")

    # Phase 2: LLM fallback for clues not in database
    if llm_needed and use_llm_fallback:
        # Use smaller batches (10 clues) for better LLM reliability
        llm_batch_size = min(batch_size, 10)
        logger.info(f"LLM fallback: generating for {len(llm_needed)} clues (batch size {llm_batch_size})...")

        def on_batch(batch_num: int, total: int, results: Dict[str, List[str]]) -> None:
            if on_progress:
                for clue_id, cands in results.items():
                    on_progress("llm", clue_id, len(cands))

        llm_candidates = generate_candidates(
            llm_needed,
            candidates_per_clue=candidates_per_clue,
            batch_size=llm_batch_size,
            api_key=api_key,
            model=model,
            on_batch_complete=on_batch,
        )

        # Merge LLM results
        candidates.update(llm_candidates)

        llm_hits = sum(1 for cid in llm_candidates if llm_candidates.get(cid))
        logger.info(f"LLM generated: {llm_hits}/{len(llm_needed)} clues")

    elif llm_needed and not use_llm_fallback:
        logger.info(f"{len(llm_needed)} clues have no database candidates (LLM fallback disabled)")

    return candidates


def regenerate_with_patterns(
    clues: List[ClueInput],
    db: "ClueDatabase",
    candidates_per_clue: int = DEFAULT_CANDIDATES_PER_CLUE,
) -> Dict[str, List[str]]:
    """
    Regenerate candidates for clues that have pattern constraints.

    Uses database pattern matching which is efficient for finding
    answers matching constraints like "C_T" (CAT, COT, CUT).

    Args:
        clues: List of ClueInput with pattern constraints
        db: ClueDatabase instance
        candidates_per_clue: Max candidates per clue

    Returns:
        Dict mapping clue_id to list of candidates matching pattern
    """
    candidates: Dict[str, List[str]] = {}

    for clue in clues:
        if not clue.pattern or "_" not in clue.pattern:
            # No pattern - skip
            continue

        # Try clue + pattern first
        answers = db.lookup_by_clue_and_pattern(
            clue.text,
            clue.pattern,
            max_results=candidates_per_clue,
        )

        # If no results from clue match, try pattern-only
        if not answers:
            answers = db.lookup_by_pattern(
                clue.pattern,
                max_results=candidates_per_clue,
            )

        if answers:
            candidates[clue.clue_id] = answers

    return candidates


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

    client = anthropic.Anthropic(api_key=api_key)
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


# ── Synonym Regeneration (for simple definition clues with missing candidates) ──


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


# ── Sniper Escalation Ladder ──────────────────────────────────────


def sniper_escalation(
    clue: ClueInput,
    db: Optional["ClueDatabase"] = None,
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


def generate_with_extended_thinking(
    clues: List[ClueInput],
    batch_size: int = 8,
    budget_tokens: int = 10000,
    word_index: Optional[Any] = None,
) -> Dict[str, List[ScoredCandidate]]:
    """
    Generate candidates using Claude extended thinking for hard clues.

    Uses Sonnet 4.5 with thinking enabled — the model reasons through
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

    client = anthropic.Anthropic(api_key=api_key)
    all_results: Dict[str, List[ScoredCandidate]] = {}

    for i in range(0, len(clues), batch_size):
        batch = clues[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(clues) + batch_size - 1) // batch_size

        # Build clue list — no patterns, just clue text + length
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


