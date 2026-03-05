"""Tests for FastAPI API endpoints."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from crosswise.api import server
from crosswise.api.models import SessionStatus
from crosswise.api.session_manager import SessionManager


@pytest.fixture
def client(tmp_path):
    """TestClient with temp dirs for sessions and puzzles."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    puzzles_dir = tmp_path / "puzzles"
    puzzles_dir.mkdir()

    mgr = SessionManager(sessions_dir)

    original_sessions = server.SESSIONS_DIR
    original_puzzles = server.PUZZLES_DIR
    original_mgr = server.session_mgr

    server.SESSIONS_DIR = sessions_dir
    server.PUZZLES_DIR = puzzles_dir
    server.session_mgr = mgr

    yield TestClient(server.app)

    server.SESSIONS_DIR = original_sessions
    server.PUZZLES_DIR = original_puzzles
    server.session_mgr = original_mgr


def _create_session(tmp_path, status=SessionStatus.UPLOADED, **extra):
    """Helper: create a session dir with session.json."""
    sessions_dir = tmp_path / "sessions"
    mgr = SessionManager(sessions_dir)
    session_id = mgr.create_session()
    if status != SessionStatus.UPLOADED:
        mgr.update_status(session_id, status, **extra)
    elif extra:
        mgr.update_status(session_id, status, **extra)
    return session_id


def _write_puzzle(tmp_path, puzzle_id, data):
    """Helper: write a puzzle JSON file."""
    puzzles_dir = tmp_path / "puzzles"
    path = puzzles_dir / f"{puzzle_id}.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestGetConfig:
    """GET /api/config"""

    def test_returns_ocr_provider(self, client):
        """Should return the configured OCR provider."""
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "ocr_provider" in data
        assert data["ocr_provider"] == "gemini"


class TestListPuzzles:
    """GET /api/puzzles"""

    def test_empty_dir(self, client):
        """Should return empty list when no puzzles exist."""
        resp = client.get("/api/puzzles")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_with_puzzle_data(self, client, tmp_path):
        """Should return puzzle metadata."""
        _write_puzzle(tmp_path, "test-puzzle", {
            "metadata": {"name": "Sunday Special", "grid_size": [15, 15]},
            "clues": {
                "across": [
                    {"number": 1, "text": "Clue A", "answer": "PARIS"},
                    {"number": 5, "text": "Clue B"},
                ],
                "down": [
                    {"number": 2, "text": "Clue C", "answer": "ECHO"},
                ],
            },
        })

        resp = client.get("/api/puzzles")
        assert resp.status_code == 200
        puzzles = resp.json()
        assert len(puzzles) == 1

        p = puzzles[0]
        assert p["id"] == "test-puzzle"
        assert p["title"] == "Sunday Special"
        assert p["gridSize"] == [15, 15]
        assert p["totalClues"] == 3
        assert p["solved"] == 2


class TestUpdatePuzzle:
    """PATCH /api/puzzles/{puzzle_id}"""

    def test_update_name(self, client, tmp_path):
        """Should update puzzle name and persist to disk."""
        _write_puzzle(tmp_path, "p1", {
            "metadata": {"grid_size": [5, 5]},
            "clues": {"across": [], "down": []},
        })

        resp = client.patch("/api/puzzles/p1", json={"name": "My Puzzle"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Verify persisted
        with open(tmp_path / "puzzles" / "p1.json") as f:
            data = json.load(f)
        assert data["metadata"]["name"] == "My Puzzle"

    def test_not_found(self, client):
        """Should return 404 for missing puzzle."""
        resp = client.patch("/api/puzzles/nonexistent", json={"name": "X"})
        assert resp.status_code == 404


class TestSessionStatus:
    """GET /api/{session_id}/status"""

    def test_returns_status(self, client, tmp_path):
        """Should return session status with solve counts."""
        session_id = _create_session(
            tmp_path,
            status=SessionStatus.COMPLETE,
            solved_count=45,
            total_clues=78,
        )

        resp = client.get(f"/api/{session_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["solved_count"] == 45
        assert data["total_clues"] == 78


class TestDiagnostics:
    """GET /api/{session_id}/diagnostics"""

    def test_returns_diagnostics(self, client, tmp_path):
        """Should return diagnostics JSON when file exists."""
        session_id = _create_session(tmp_path)
        session_dir = tmp_path / "sessions" / session_id
        diag_data = [{"clue_id": "1-across", "answer": "PARIS", "candidates": ["PARIS", "LYONS"]}]
        with open(session_dir / "solve_diagnostics.json", "w") as f:
            json.dump(diag_data, f)

        resp = client.get(f"/api/{session_id}/diagnostics")
        assert resp.status_code == 200
        assert resp.json()[0]["clue_id"] == "1-across"

    def test_not_found(self, client, tmp_path):
        """Should return 404 when no diagnostics file."""
        session_id = _create_session(tmp_path)

        resp = client.get(f"/api/{session_id}/diagnostics")
        assert resp.status_code == 404


class TestUpload:
    """POST /api/upload"""

    def test_rejects_non_image(self, client):
        """Should return 400 for non-image content type."""
        resp = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400

    @patch("crosswise.api.server.pipeline")
    def test_success(self, mock_pipeline, client):
        """Should detect grid and return session info."""
        mock_pipeline.run_grid_detection.return_value = {
            "grid_size": [15, 15],
            "clue_slot_count": 78,
        }

        # 1x1 white PNG
        import struct
        import zlib
        def _make_png():
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
            raw = zlib.compress(b'\x00\xff\xff\xff')
            idat_crc = zlib.crc32(b'IDAT' + raw) & 0xffffffff
            idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
            iend_crc = zlib.crc32(b'IEND') & 0xffffffff
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
            return sig + ihdr + idat + iend

        resp = client.post(
            "/api/upload",
            files={"file": ("test.png", _make_png(), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "grid_detected"
        assert data["grid_size"] == [15, 15]
        assert data["clue_slot_count"] == 78
        mock_pipeline.run_grid_detection.assert_called_once()


class TestResizeGrid:
    """POST /api/{session_id}/resize-grid"""

    def test_rejects_out_of_range(self, client, tmp_path):
        """Should return 400 for rows/cols outside 3-30."""
        session_id = _create_session(tmp_path)

        resp = client.post(
            f"/api/{session_id}/resize-grid",
            json={"rows": 2, "cols": 15},
        )
        assert resp.status_code == 400

        resp = client.post(
            f"/api/{session_id}/resize-grid",
            json={"rows": 15, "cols": 31},
        )
        assert resp.status_code == 400


class TestCancel:
    """POST /api/{session_id}/cancel"""

    def test_no_active_solve(self, client, tmp_path):
        """Should return 404 when no cancel event exists."""
        session_id = _create_session(tmp_path)

        resp = client.post(f"/api/{session_id}/cancel")
        assert resp.status_code == 404
