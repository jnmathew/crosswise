"""Tests for FastAPI API endpoints."""

import json
import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from crosswise.api import server
from crosswise.api.models import SessionStatus
from crosswise.api.rate_limit import limiter
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
    limiter.reset()

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


class TestUnknownSession:
    """Unknown session IDs return 404, not 500."""

    def test_status(self, client):
        assert client.get("/api/nosuchsession/status").status_code == 404

    def test_solve(self, client):
        assert client.post("/api/nosuchsession/solve").status_code == 404

    def test_grid_edit(self, client):
        resp = client.post("/api/nosuchsession/grid-edit", json={"black_cells": [[False]]})
        assert resp.status_code == 404

    def test_diagnostics(self, client):
        assert client.get("/api/nosuchsession/diagnostics").status_code == 404


class TestConcurrentSolveGuard:
    """A second solve on a session with one already running is refused."""

    def test_solve_refused_while_active(self, client, tmp_path):
        session_id = _create_session(tmp_path)
        _write_puzzle(tmp_path, session_id, {"metadata": {}, "grid": {}, "clues": {}})
        server.cancel_events[session_id] = threading.Event()
        try:
            resp = client.post(f"/api/{session_id}/solve")
            assert resp.status_code == 409
        finally:
            server.cancel_events.pop(session_id, None)

    @patch("crosswise.api.server.pipeline")
    def test_mask_refused_before_ocr(self, mock_pipeline, client, tmp_path):
        """The guard runs before OCR, so a refused request costs no Gemini call."""
        session_id = _create_session(tmp_path)
        server.cancel_events[session_id] = threading.Event()
        try:
            resp = client.post(f"/api/{session_id}/mask", json={"rectangles": [], "separators": []})
            assert resp.status_code == 409
            mock_pipeline.run_ocr_and_verify.assert_not_called()
        finally:
            server.cancel_events.pop(session_id, None)


class TestListPuzzlesRobustness:
    def test_skips_unreadable_file(self, client, tmp_path):
        _write_puzzle(tmp_path, "good", {
            "metadata": {"name": "Good"}, "grid": {"cells": []},
            "clues": {"across": [], "down": []},
        })
        (tmp_path / "puzzles" / "bad.json").write_text('{"metadata": {"na')  # truncated write

        resp = client.get("/api/puzzles")
        assert resp.status_code == 200
        assert [p["id"] for p in resp.json()] == ["good"]


class TestEventLoopNotBlocked:
    @pytest.mark.asyncio
    @patch("crosswise.api.server.pipeline")
    async def test_ocr_does_not_block_other_requests(self, mock_pipeline, client, tmp_path):
        """Slow OCR runs in a worker thread, so other requests are served meanwhile."""
        import asyncio
        import time
        import httpx

        def slow_ocr(*args, **kwargs):
            time.sleep(1.0)
            return {"verification_passed": False, "ocr_clue_count": 0, "grid_slot_count": 0,
                    "matched_count": 0, "errors": []}

        mock_pipeline.run_ocr_and_verify.side_effect = slow_ocr
        session_id = _create_session(tmp_path)

        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            mask = asyncio.create_task(
                ac.post(f"/api/{session_id}/mask", json={"rectangles": [], "separators": []})
            )
            await asyncio.sleep(0.1)  # let the mask request reach the OCR call
            started = time.monotonic()
            config = await ac.get("/api/config")
            assert config.status_code == 200
            assert time.monotonic() - started < 0.5  # not stuck behind the 1s OCR
            assert not mask.done()
            assert (await mask).status_code == 200


class TestDeletePuzzle:
    def test_refused_while_solving(self, client, tmp_path):
        session_id = _create_session(tmp_path)
        _write_puzzle(tmp_path, session_id, {"metadata": {}, "grid": {}, "clues": {}})
        server.cancel_events[session_id] = threading.Event()
        try:
            assert client.delete(f"/api/puzzles/{session_id}").status_code == 409
            assert (tmp_path / "puzzles" / f"{session_id}.json").exists()
        finally:
            server.cancel_events.pop(session_id, None)

    def test_removes_session_dir(self, client, tmp_path):
        session_id = _create_session(tmp_path)
        _write_puzzle(tmp_path, session_id, {"metadata": {}, "grid": {}, "clues": {}})
        assert client.delete(f"/api/puzzles/{session_id}").status_code == 200
        assert not (tmp_path / "sessions" / session_id).exists()

    def test_puzzle_without_session(self, client, tmp_path):
        _write_puzzle(tmp_path, "sample", {"metadata": {}, "grid": {}, "clues": {}})
        assert client.delete("/api/puzzles/sample").status_code == 200


class TestInterruptedSessions:
    def test_running_sessions_marked_failed(self, tmp_path):
        mgr = SessionManager(tmp_path / "s")
        running = mgr.create_session(); mgr.update_status(running, SessionStatus.SOLVING)
        done = mgr.create_session(); mgr.update_status(done, SessionStatus.COMPLETE)
        assert mgr.mark_interrupted() == [running]
        assert mgr.get_status(running) == SessionStatus.FAILED
        assert mgr.get_status(done) == SessionStatus.COMPLETE


class TestRenameDuringSolve:
    def test_final_write_keeps_latest_name(self, tmp_path):
        """A rename made while the solve runs survives the solve's final save."""
        from crosswise.api import pipeline
        puzzles_dir, session_dir = tmp_path / "p", tmp_path / "s"
        puzzles_dir.mkdir(); session_dir.mkdir()
        cells = [[{"row": r, "col": c, "is_block": (r, c) == (1, 1)} for c in range(3)] for r in range(3)]
        puzzle = {"metadata": {"name": "Old name"}, "grid": {"rows": 3, "cols": 3, "cells": cells},
                  "clues": {"across": [{"number": 1, "text": "Feline pet", "start": [0, 0], "length": 3}],
                            "down": []}}
        path = puzzles_dir / "x.json"
        path.write_text(json.dumps(puzzle))

        def rename_mid_solve(puzzle_data, *args):
            data = json.loads(path.read_text()); data["metadata"]["name"] = "New name"
            path.write_text(json.dumps(data))
            return []

        class Sessions:
            def update_status(self, *a, **k): pass

        with patch.object(pipeline, "_generate_candidates",
                          return_value=({}, {}, {}, {}, {}, {}, None, {})), \
             patch.object(pipeline, "_run_solver", return_value=({"1-across": "CAT"}, 1)), \
             patch.object(pipeline, "_save_diagnostics"), \
             patch.object(pipeline, "_generate_and_apply_hints", side_effect=rename_mid_solve):
            pipeline._run_solve(session_dir, puzzles_dir, "x", lambda p: None, Sessions(), "x")
        assert json.loads(path.read_text())["metadata"]["name"] == "New name"


class TestRateLimit:
    def test_limiter_window(self):
        from crosswise.api.rate_limit import RateLimiter
        now = [0.0]
        lim = RateLimiter(clock=lambda: now[0])
        rule = [("k", 2, 60.0)]
        assert lim.acquire(rule) is None and lim.acquire(rule) is None
        assert lim.acquire(rule) == pytest.approx(60.0)
        now[0] = 61.0
        assert lim.acquire(rule) is None

    def test_rejected_request_uses_no_quota(self):
        from crosswise.api.rate_limit import RateLimiter
        lim = RateLimiter(clock=lambda: 0.0)
        assert lim.acquire([("a", 5, 60), ("b", 1, 60)]) is None
        assert lim.acquire([("a", 5, 60), ("b", 1, 60)]) is not None  # b is full
        assert len(lim._hits["a"]) == 1

    def test_solve_endpoint_returns_429(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(server.settings, "RATE_LIMIT_SOLVES_PER_HOUR", 1)
        session_id = _create_session(tmp_path)
        assert client.post(f"/api/{session_id}/solve").status_code == 404  # no puzzle; uses the slot
        resp = client.post(f"/api/{session_id}/solve")
        assert resp.status_code == 429 and "Retry-After" in resp.headers

    def test_disabled(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(server.settings, "RATE_LIMIT_ENABLED", False)
        monkeypatch.setattr(server.settings, "RATE_LIMIT_SOLVES_PER_HOUR", 0)
        session_id = _create_session(tmp_path)
        assert client.post(f"/api/{session_id}/solve").status_code == 404


class TestUploadLimits:
    def test_too_large(self, client, monkeypatch):
        monkeypatch.setattr(server.settings, "MAX_UPLOAD_MB", 1)
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (1024 * 1024 + 10)
        resp = client.post("/api/upload", files={"file": ("big.png", big, "image/png")})
        assert resp.status_code == 413

    def test_not_really_an_image(self, client):
        resp = client.post("/api/upload", files={"file": ("x.png", b"hello world", "image/png")})
        assert resp.status_code == 400
