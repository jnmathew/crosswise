"""Crash- and reader-safe JSON file writes."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any, **dump_kwargs) -> None:
    """Write JSON so readers see either the old file or the new one, never a partial.

    Writes to a temp file in the same directory, then renames it over the target
    (``os.replace`` is atomic on POSIX and Windows when both are on one filesystem).
    Puzzle and session files are read while solves rewrite them, e.g. by
    ``GET /api/puzzles`` and status polling.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, **dump_kwargs)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
