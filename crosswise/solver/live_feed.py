"""Live feed of the solver's answers, for the player's "watch the solve" view.

Two kinds of events go to the listener:

- ``tentative``: an answer the model has just written in its streaming reply,
  before the solver validates it. Parsed out of the text as it arrives.
- ``commit``: what actually changed in the solver's assignment after a phase
  (a pass, constraint propagation, conflict resolution, verification...):
  answers added or changed, and answers removed. A tentative answer that
  never shows up in a commit was rejected.

The feed only reports; nothing in the solve depends on it.
"""

import re
from typing import Callable, Dict, List, Optional

ClueId = str
Word = str
Event = dict

# A complete "12-across": "ARBOR" pair in streamed JSON text
_ANSWER_PAIR = re.compile(r'"(\d+-(?:across|down))"\s*:\s*"([A-Za-z]+)"')


class LiveFeed:
    def __init__(self, emit: Callable[[Event], None]):
        self._emit = emit
        self._shown: Dict[ClueId, Word] = {}  # last committed assignment
        self._tentative_pending = False  # tentatives sent since the last commit

    def snapshot(self) -> Dict[ClueId, Word]:
        """The last committed assignment (for a viewer that joins mid-solve)."""
        return dict(self._shown)

    def text_listener(self, phase: str, clue_lengths: Dict[ClueId, int]) -> Callable[[str], None]:
        """An ``on_text`` callback for one streaming request.

        Emits each well-formed answer once, as soon as its closing quote
        arrives. Answers of the wrong length (which the solver will reject
        anyway) are skipped.
        """
        chunks: List[str] = []
        seen = set()

        def on_text(chunk: str) -> None:
            chunks.append(chunk)
            for match in _ANSWER_PAIR.finditer("".join(chunks)):
                clue_id, word = match.group(1), match.group(2).upper()
                if (clue_id, word) in seen or clue_lengths.get(clue_id) != len(word):
                    continue
                seen.add((clue_id, word))
                self._tentative_pending = True
                self._emit({"type": "tentative", "phase": phase, "clue": clue_id, "word": word})

        return on_text

    def commit(self, assignment: Dict[ClueId, Word], phase: str) -> None:
        """Report what changed since the last commit.

        Also sent when nothing changed but tentative answers are outstanding,
        so the viewer learns they were all rejected.
        """
        added = {cid: w for cid, w in assignment.items() if self._shown.get(cid) != w}
        removed = [cid for cid in self._shown if cid not in assignment]
        if not added and not removed and not self._tentative_pending:
            return
        self._shown = dict(assignment)
        self._tentative_pending = False
        self._emit({
            "type": "commit", "phase": phase, "answers": added, "removed": removed,
            "total": len(assignment),
        })


def listener(live: Optional[LiveFeed], phase: str, clue_lengths: Dict[ClueId, int]):
    """``live.text_listener(...)``, or None when there's no feed."""
    return live.text_listener(phase, clue_lengths) if live else None
