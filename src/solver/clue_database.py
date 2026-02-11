"""
SQLite-backed clue database for crossword puzzle solving.

Uses a TSV file with historical clue/answer pairs as the primary data source.
Converts TSV to SQLite on first load for efficient querying.
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Set
import unicodedata


# Default paths
DEFAULT_TSV_PATH = "data/xd 2/clues.tsv"
DEFAULT_DB_PATH = "data/clues.db"


def normalize_clue_text(text: str) -> str:
    """
    Normalize clue text for fuzzy matching.

    - Lowercase
    - Remove punctuation except apostrophes
    - Normalize unicode
    - Collapse whitespace
    """
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    # Lowercase
    text = text.lower()
    # Remove punctuation except apostrophes (keep contractions)
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


class ClueDatabase:
    """
    SQLite-backed database of crossword clue/answer pairs.

    Provides efficient lookup by:
    - Exact clue text match
    - Normalized (fuzzy) clue text match
    - Answer pattern matching (e.g., "C_T" matches "CAT", "COT", "CUT")
    - Answer length
    """

    def __init__(
        self,
        tsv_path: Optional[str] = None,
        db_path: Optional[str] = None,
        project_root: Optional[str] = None,
    ):
        """
        Initialize the clue database.

        Args:
            tsv_path: Path to TSV file (default: data/xd 2/clues.tsv)
            db_path: Path to SQLite database (default: data/clues.db)
            project_root: Project root directory for resolving relative paths
        """
        # Resolve project root
        if project_root is None:
            # Try to find project root by looking for CLAUDE.md
            current = Path(__file__).resolve()
            for parent in current.parents:
                if (parent / "CLAUDE.md").exists():
                    project_root = str(parent)
                    break
            else:
                project_root = str(Path.cwd())

        self.project_root = Path(project_root)

        # Resolve paths
        self.tsv_path = Path(tsv_path) if tsv_path else self.project_root / DEFAULT_TSV_PATH
        self.db_path = Path(db_path) if db_path else self.project_root / DEFAULT_DB_PATH

        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_database()

    def _ensure_database(self) -> None:
        """Ensure SQLite database exists and is populated."""
        if self.db_path.exists():
            # Database already exists, just connect
            self._connect()
            # Verify it has data
            cursor = self._conn.execute("SELECT COUNT(*) FROM clues")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"Loaded clue database: {count:,} entries")
                return
            # Empty database, need to rebuild
            self._conn.close()
            self.db_path.unlink()

        # Need to build database from TSV
        if not self.tsv_path.exists():
            raise FileNotFoundError(
                f"TSV file not found: {self.tsv_path}\n"
                f"Please add the clues.tsv file to the data directory."
            )

        print(f"Building SQLite database from {self.tsv_path}...")
        self._build_database()

    def _connect(self) -> None:
        """Open connection to SQLite database."""
        self._conn = sqlite3.connect(str(self.db_path))
        # Enable case-insensitive LIKE
        self._conn.execute("PRAGMA case_sensitive_like = OFF")

    def _build_database(self) -> None:
        """Build SQLite database from TSV file."""
        self._conn = sqlite3.connect(str(self.db_path))

        # Create table
        self._conn.execute("""
            CREATE TABLE clues (
                id INTEGER PRIMARY KEY,
                pubid TEXT,
                year INTEGER,
                answer TEXT NOT NULL,
                clue TEXT NOT NULL,
                clue_normalized TEXT NOT NULL,
                length INTEGER NOT NULL
            )
        """)

        # Bulk insert from TSV
        batch_size = 10000
        batch = []
        total_inserted = 0

        with open(self.tsv_path, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)

            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue

                pubid, year_str, answer, clue = parts[0], parts[1], parts[2], parts[3]

                # Skip empty answers
                if not answer or not answer.strip():
                    continue

                # Clean answer (uppercase, no spaces)
                answer = answer.upper().strip()

                # Skip if answer contains non-letter characters
                if not answer.isalpha():
                    continue

                # Parse year
                try:
                    year = int(year_str) if year_str else 0
                except ValueError:
                    year = 0

                # Normalize clue for fuzzy matching
                clue_normalized = normalize_clue_text(clue)

                batch.append((pubid, year, answer, clue, clue_normalized, len(answer)))

                if len(batch) >= batch_size:
                    self._conn.executemany(
                        "INSERT INTO clues (pubid, year, answer, clue, clue_normalized, length) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        batch
                    )
                    total_inserted += len(batch)
                    if total_inserted % 500000 == 0:
                        print(f"  Inserted {total_inserted:,} rows...")
                    batch = []

        # Insert remaining
        if batch:
            self._conn.executemany(
                "INSERT INTO clues (pubid, year, answer, clue, clue_normalized, length) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                batch
            )
            total_inserted += len(batch)

        print(f"  Total: {total_inserted:,} clue/answer pairs")

        # Create indexes
        print("  Creating indexes...")
        self._conn.execute("CREATE INDEX idx_clue_normalized ON clues(clue_normalized)")
        self._conn.execute("CREATE INDEX idx_length ON clues(length)")
        self._conn.execute("CREATE INDEX idx_answer ON clues(answer)")
        self._conn.execute("CREATE INDEX idx_clue_length ON clues(clue_normalized, length)")

        self._conn.commit()
        print(f"Database saved to {self.db_path}")

    def lookup_by_clue(
        self,
        clue_text: str,
        length: int,
        max_results: int = 20,
    ) -> List[str]:
        """
        Look up candidate answers by clue text.

        First tries exact match, then falls back to normalized match.

        Args:
            clue_text: The clue text to search for
            length: Required answer length
            max_results: Maximum number of unique answers to return

        Returns:
            List of candidate answers, ordered by frequency
        """
        answers: List[str] = []
        seen: Set[str] = set()

        # Normalize for fuzzy matching
        clue_normalized = normalize_clue_text(clue_text)

        # Query for normalized match (more flexible)
        cursor = self._conn.execute("""
            SELECT answer, COUNT(*) as freq
            FROM clues
            WHERE clue_normalized = ? AND length = ?
            GROUP BY answer
            ORDER BY freq DESC
            LIMIT ?
        """, (clue_normalized, length, max_results))

        for row in cursor:
            answer = row[0]
            if answer not in seen:
                answers.append(answer)
                seen.add(answer)

        return answers

    def lookup_by_pattern(
        self,
        pattern: str,
        max_results: int = 50,
    ) -> List[str]:
        """
        Look up answers matching a pattern like "C_T" (3 letters, C first, T last).

        Uses SQLite GLOB for pattern matching where _ is converted to ?.

        Args:
            pattern: Pattern string where _ represents unknown letters
            max_results: Maximum number of unique answers to return

        Returns:
            List of candidate answers matching the pattern
        """
        length = len(pattern)

        # Convert pattern: _ -> ? for GLOB
        glob_pattern = pattern.replace("_", "?")

        # Query using GLOB (case-sensitive by default, we store uppercase)
        cursor = self._conn.execute("""
            SELECT answer, COUNT(*) as freq
            FROM clues
            WHERE length = ? AND answer GLOB ?
            GROUP BY answer
            ORDER BY freq DESC
            LIMIT ?
        """, (length, glob_pattern, max_results))

        return [row[0] for row in cursor]

    def lookup_by_clue_and_pattern(
        self,
        clue_text: str,
        pattern: str,
        max_results: int = 20,
    ) -> List[str]:
        """
        Look up answers matching both clue text and pattern.

        First searches by clue, then filters by pattern.
        If no results, falls back to pattern-only search.

        Args:
            clue_text: The clue text to search for
            pattern: Pattern string where _ represents unknown letters
            max_results: Maximum number of answers to return

        Returns:
            List of candidate answers
        """
        length = len(pattern)
        clue_normalized = normalize_clue_text(clue_text)
        glob_pattern = pattern.replace("_", "?")

        # Try clue + pattern match
        cursor = self._conn.execute("""
            SELECT answer, COUNT(*) as freq
            FROM clues
            WHERE clue_normalized = ? AND length = ? AND answer GLOB ?
            GROUP BY answer
            ORDER BY freq DESC
            LIMIT ?
        """, (clue_normalized, length, glob_pattern, max_results))

        results = [row[0] for row in cursor]

        if results:
            return results

        # Fall back to pattern-only match
        return self.lookup_by_pattern(pattern, max_results)

    def lookup_by_length(
        self,
        length: int,
        max_results: int = 100,
    ) -> List[str]:
        """
        Get most common answers of a given length.

        This is a last resort when no clue or pattern matches.

        Args:
            length: Required answer length
            max_results: Maximum number of answers to return

        Returns:
            List of most common answers of that length
        """
        cursor = self._conn.execute("""
            SELECT answer, COUNT(*) as freq
            FROM clues
            WHERE length = ?
            GROUP BY answer
            ORDER BY freq DESC
            LIMIT ?
        """, (length, max_results))

        return [row[0] for row in cursor]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ClueDatabase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
