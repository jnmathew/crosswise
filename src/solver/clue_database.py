"""
SQLite-backed clue database for crossword puzzle solving.

Loads clue/answer pairs from two sources:
- xd TSV (data/xd 2/clues.tsv) — ~7.5M historical pairs
- CrosswordQA CSVs (data/crosswordqa/) — ~6.8M pairs from HuggingFace

Deduplicates CrosswordQA against xd on (answer, clue_normalized) pairs.
Converts to SQLite on first load for efficient querying.
"""

import csv
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Set, Tuple
import unicodedata


# Default paths
DEFAULT_XD_TSV_PATH = "data/xd 2/clues.tsv"
DEFAULT_CQA_DIR = "data/crosswordqa"
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
        db_path: Optional[str] = None,
        project_root: Optional[str] = None,
    ):
        """
        Initialize the clue database.

        Args:
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

        # Locate data sources
        xd_path = self.project_root / DEFAULT_XD_TSV_PATH
        cqa_dir = self.project_root / DEFAULT_CQA_DIR

        has_xd = xd_path.exists()
        has_cqa = (cqa_dir / "train.csv").exists()

        if not has_xd and not has_cqa:
            raise FileNotFoundError(
                "No clue data sources found.\n"
                f"  xd TSV:        {xd_path} (not found)\n"
                f"  CrosswordQA:   {cqa_dir}/ (not found)\n"
                "Run: bash scripts/download_crosswordqa.sh"
            )

        print("Building SQLite clue database...")
        self._build_database(
            xd_path if has_xd else None,
            cqa_dir if has_cqa else None,
        )

    def _connect(self) -> None:
        """Open connection to SQLite database."""
        self._conn = sqlite3.connect(str(self.db_path))
        # Enable case-insensitive LIKE
        self._conn.execute("PRAGMA case_sensitive_like = OFF")

    def _build_database(
        self,
        xd_path: Optional[Path],
        cqa_dir: Optional[Path],
    ) -> None:
        """Build SQLite database from available data sources."""
        self._conn = sqlite3.connect(str(self.db_path))

        # Create table (no pubid/year — never queried)
        self._conn.execute("""
            CREATE TABLE clues (
                id INTEGER PRIMARY KEY,
                answer TEXT NOT NULL,
                clue TEXT NOT NULL,
                clue_normalized TEXT NOT NULL,
                length INTEGER NOT NULL
            )
        """)

        xd_count = 0
        if xd_path:
            xd_count = self._load_xd_tsv(xd_path)

        cqa_count = 0
        if cqa_dir:
            # Build dedup set from xd data already loaded
            dedup: Set[Tuple[str, str]] = set()
            if xd_count > 0:
                print("  Building dedup index from xd data...")
                cursor = self._conn.execute(
                    "SELECT DISTINCT answer, clue_normalized FROM clues"
                )
                for row in cursor:
                    dedup.add((row[0], row[1]))
                print(f"  Dedup index: {len(dedup):,} unique (answer, clue) pairs")

            cqa_count = self._load_crosswordqa(cqa_dir, dedup)

        total = xd_count + cqa_count
        print(f"  Total: {total:,} clue/answer pairs")

        # Create indexes
        print("  Creating indexes...")
        self._conn.execute("CREATE INDEX idx_clue_normalized ON clues(clue_normalized)")
        self._conn.execute("CREATE INDEX idx_length ON clues(length)")
        self._conn.execute("CREATE INDEX idx_answer ON clues(answer)")
        self._conn.execute("CREATE INDEX idx_clue_length ON clues(clue_normalized, length)")

        self._conn.commit()
        print(f"Database saved to {self.db_path}")

    def _load_xd_tsv(self, path: Path) -> int:
        """Load clue/answer pairs from xd TSV file.

        Returns:
            Number of rows inserted.
        """
        print(f"  Loading xd TSV from {path}...")

        batch_size = 10000
        batch = []
        total_inserted = 0

        with open(path, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)

            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 4:
                    continue

                answer, clue = parts[2], parts[3]

                # Skip empty answers
                if not answer or not answer.strip():
                    continue

                # Clean answer (uppercase, no spaces)
                answer = answer.upper().strip()

                # Skip if answer contains non-letter characters
                if not answer.isalpha():
                    continue

                # Normalize clue for fuzzy matching
                clue_normalized = normalize_clue_text(clue)

                batch.append((answer, clue, clue_normalized, len(answer)))

                if len(batch) >= batch_size:
                    self._conn.executemany(
                        "INSERT INTO clues (answer, clue, clue_normalized, length) "
                        "VALUES (?, ?, ?, ?)",
                        batch
                    )
                    total_inserted += len(batch)
                    if total_inserted % 500000 == 0:
                        print(f"    {total_inserted:,} rows...")
                    batch = []

        # Insert remaining
        if batch:
            self._conn.executemany(
                "INSERT INTO clues (answer, clue, clue_normalized, length) "
                "VALUES (?, ?, ?, ?)",
                batch
            )
            total_inserted += len(batch)

        print(f"  xd: {total_inserted:,} rows loaded")
        return total_inserted

    def _load_crosswordqa(
        self,
        cqa_dir: Path,
        dedup: Set[Tuple[str, str]],
    ) -> int:
        """Load clue/answer pairs from CrosswordQA CSVs, deduplicating against existing data.

        Args:
            cqa_dir: Directory containing train.csv and valid.csv
            dedup: Set of (answer, clue_normalized) pairs already in the database

        Returns:
            Number of new rows inserted.
        """
        print(f"  Loading CrosswordQA from {cqa_dir}/...")

        batch_size = 10000
        batch = []
        total_inserted = 0
        skipped_dup = 0
        skipped_invalid = 0

        for csv_name in ("train.csv", "valid.csv"):
            csv_path = cqa_dir / csv_name
            if not csv_path.exists():
                print(f"    {csv_name} not found, skipping")
                continue

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip header
                next(reader, None)

                for row in reader:
                    # CSV columns: id, clue, answer
                    if len(row) < 3:
                        continue

                    clue, answer = row[1], row[2]

                    # Clean answer: strip spaces, uppercase
                    answer = answer.strip().replace(" ", "").upper()

                    # Skip empty or non-alpha answers
                    if not answer or not answer.isalpha():
                        skipped_invalid += 1
                        continue

                    clue_normalized = normalize_clue_text(clue)

                    # Deduplicate against xd
                    pair = (answer, clue_normalized)
                    if pair in dedup:
                        skipped_dup += 1
                        continue

                    # Add to dedup set so we don't insert CQA duplicates either
                    dedup.add(pair)

                    batch.append((answer, clue, clue_normalized, len(answer)))

                    if len(batch) >= batch_size:
                        self._conn.executemany(
                            "INSERT INTO clues (answer, clue, clue_normalized, length) "
                            "VALUES (?, ?, ?, ?)",
                            batch
                        )
                        total_inserted += len(batch)
                        if total_inserted % 500000 == 0:
                            print(f"    {total_inserted:,} rows...")
                        batch = []

        # Insert remaining
        if batch:
            self._conn.executemany(
                "INSERT INTO clues (answer, clue, clue_normalized, length) "
                "VALUES (?, ?, ?, ?)",
                batch
            )
            total_inserted += len(batch)

        print(f"  CrosswordQA: {total_inserted:,} new rows "
              f"({skipped_dup:,} duplicates, {skipped_invalid:,} invalid skipped)")
        return total_inserted

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
