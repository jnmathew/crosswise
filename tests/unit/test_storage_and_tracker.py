"""Tests for atomic JSON writes and per-solve cost tracking."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from crosswise.api.storage import write_json_atomic
from crosswise.solver.cost_tracker import get_tracker, reset_tracker, submit_in_context


def _fake_response(tokens: int):
    usage = SimpleNamespace(input_tokens=tokens, output_tokens=0,
                            cache_creation_input_tokens=0, cache_read_input_tokens=0,
                            server_tool_use=None)
    return SimpleNamespace(usage=usage, model="claude-opus-5-5")


class TestWriteJsonAtomic:
    def test_writes_and_leaves_no_temp_files(self, tmp_path):
        path = tmp_path / "p.json"
        write_json_atomic(path, {"a": 1})
        assert json.loads(path.read_text()) == {"a": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["p.json"]

    def test_failed_write_keeps_old_file(self, tmp_path):
        path = tmp_path / "p.json"
        write_json_atomic(path, {"old": True})
        with pytest.raises(TypeError):
            write_json_atomic(path, {"bad": object()})  # not JSON-serializable
        assert json.loads(path.read_text()) == {"old": True}
        assert [p.name for p in tmp_path.iterdir()] == ["p.json"]


class TestPerSolveTracker:
    def test_concurrent_solves_keep_separate_trackers(self):
        """Two solves in parallel threads each see only their own costs."""
        results = {}
        barrier = threading.Barrier(2)

        def solve(name, tokens):
            tracker = reset_tracker()
            barrier.wait()  # both trackers exist before either records
            get_tracker().track(_fake_response(tokens), name)
            barrier.wait()
            results[name] = [c.input_tokens for c in tracker._calls]

        threads = [threading.Thread(target=solve, args=(n, t)) for n, t in (("a", 100), ("b", 200))]
        for t in threads: t.start()
        for t in threads: t.join()
        assert results == {"a": [100], "b": [200]}

    def test_thread_pool_workers_record_on_solve_tracker(self):
        def solve():
            tracker = reset_tracker()
            with ThreadPoolExecutor(max_workers=2) as ex:
                for f in [submit_in_context(ex, lambda: get_tracker().track(_fake_response(5), "w"))
                          for _ in range(3)]:
                    f.result()
            return len(tracker._calls)

        with ThreadPoolExecutor(max_workers=1) as outer:
            assert outer.submit(solve).result() == 3
