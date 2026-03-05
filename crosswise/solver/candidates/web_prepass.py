"""Haiku web search pre-pass for pop culture clues.

Independent enrichment phase: detects pop culture clues (quotes, proper nouns,
media references) and fires parallel Haiku web searches to get verified answers.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from loguru import logger

from .models import ClueInput


def _is_pop_culture_clue(text: str) -> bool:
    """Heuristic: does this clue broadly reference pop culture or proper nouns?

    These are the clues where web search adds the most value -- song titles,
    movie/TV names, celebrity references, book titles, brand names, etc.
    """
    t = text.strip()
    t_lower = t.lower()

    # Contains quoted text (e.g., '"--- Love Her"')
    if '"' in t or '\u201c' in t or '\u201d' in t:
        return True

    # Fill-in-the-blank with proper-noun context: has em-dash and a capitalized word
    if '\u2014' in t or '\u2014' in t or '---' in t:
        # Check for a capitalized word (not just the first word)
        words = t.split()
        for w in words:
            cleaned = w.strip('",.\u2014\u2014\'')
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
            cleaned = w.strip('",.\u2014\u2014\'?!;:')
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

    # Try 3: scan from the end -- answers tend to be the last meaningful word
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
            client = anthropic.Anthropic(api_key=api_key, timeout=30.0)

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

            # Handle pause_turn -- follow up until we get a final answer (max 3 continuations)
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
