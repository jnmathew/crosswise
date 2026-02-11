"""
Unified Crossword Word Index.

Loads multiple word list files into a single in-memory set for fast
membership testing and pattern matching. Sources:
- Broda List (classical crosswordese, high-frequency fill)
- Crossword Nexus List (broad contemporary coverage)
- Spread the Wordlist (additional depth)

All words are normalized to uppercase with spaces/hyphens stripped.
Supports semicolon-scored format (WORD;SCORE) and plain word-per-line format.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set


# Default paths relative to project root
DEFAULT_WORD_LIST_PATHS = [
    "data/wordlists/peter-broda-wordlist__gridtext__scored__july-25-2023.txt",
    "data/wordlists/crosswordnexuswordlist.txt",
    "data/wordlists/spreadthewordlist.txt",
]


class WordIndex:
    """In-memory set of valid crossword words with pattern matching."""

    def __init__(
        self,
        word_list_paths: Optional[List[str]] = None,
        project_root: Optional[str] = None,
    ):
        if project_root is None:
            project_root = str(Path(__file__).resolve().parents[2])

        root = Path(project_root)
        if word_list_paths is None:
            resolved_paths = [root / p for p in DEFAULT_WORD_LIST_PATHS]
        else:
            resolved_paths = [root / p if not Path(p).is_absolute() else Path(p) for p in word_list_paths]

        self._words: Set[str] = set()
        self._by_length: Dict[int, Set[str]] = {}
        self._scores: Dict[str, int] = {}  # word -> best score across sources
        self._load(resolved_paths)

    def _load(self, paths: List[Path]) -> None:
        """Load all word lists, normalizing to uppercase, stripping spaces/hyphens."""
        for path in paths:
            if not path.exists():
                print(f"  Warning: Word list not found: {path}")
                continue

            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Handle semicolon-scored format: WORD;SCORE
                    score = 50  # default
                    if ";" in line:
                        parts = line.split(";", 1)
                        raw_word = parts[0]
                        try:
                            score = int(parts[1])
                        except (ValueError, IndexError):
                            pass
                    else:
                        raw_word = line

                    word = raw_word.upper().replace(" ", "").replace("-", "")
                    if not word or not word.isalpha():
                        continue

                    self._words.add(word)
                    length = len(word)
                    if length not in self._by_length:
                        self._by_length[length] = set()
                    self._by_length[length].add(word)

                    # Keep the highest score across sources
                    if word not in self._scores or score > self._scores[word]:
                        self._scores[word] = score

                    count += 1

            print(f"  Loaded {count:,} words from {path.name}")

        print(f"  Word index total: {len(self._words):,} unique words")

    def contains(self, word: str) -> bool:
        """Check if word exists in the index. O(1) set lookup."""
        return word.upper().replace(" ", "").replace("-", "") in self._words

    def score(self, word: str) -> int:
        """Get the quality score for a word (0-100, higher is better). Returns 0 if not found."""
        return self._scores.get(word.upper().replace(" ", "").replace("-", ""), 0)

    def match_pattern(self, pattern: str, max_results: int = 50) -> List[str]:
        """
        Find words matching a pattern like 'C_T' where _ is any letter.
        Filters by length first, then applies regex.
        """
        length = len(pattern)
        candidates = self._by_length.get(length, set())
        if not candidates:
            return []

        # Convert pattern to regex: _ -> [A-Z]
        regex_str = "^" + pattern.upper().replace("_", "[A-Z]") + "$"
        compiled = re.compile(regex_str)

        results = []
        for word in candidates:
            if compiled.match(word):
                results.append(word)
                if len(results) >= max_results:
                    break
        return results

    def words_of_length(self, length: int) -> Set[str]:
        """Get all words of a given length."""
        return self._by_length.get(length, set())

    @property
    def size(self) -> int:
        return len(self._words)
