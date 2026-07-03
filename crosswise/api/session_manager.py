import json
import uuid
from datetime import datetime
from pathlib import Path

from crosswise.api.models import SessionStatus


class SessionManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        session_dir = self.base_dir / session_id
        session_dir.mkdir()
        self._write_session(session_id, {
            "session_id": session_id,
            "status": SessionStatus.UPLOADED,
            "created_at": datetime.now().isoformat(),
        })
        return session_id

    def get_session_dir(self, session_id: str) -> Path:
        d = (self.base_dir / session_id).resolve()
        # Session IDs are server-generated hex, so a well-behaved client can
        # never trip this — it guards against crafted IDs escaping base_dir.
        if not d.is_relative_to(self.base_dir.resolve()) or d == self.base_dir.resolve():
            raise FileNotFoundError(f"Session {session_id} not found")
        if not d.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        return d

    def update_status(self, session_id: str, status: SessionStatus, **extra):
        data = self._read_session(session_id)
        data["status"] = status
        data["updated_at"] = datetime.now().isoformat()
        data.update(extra)
        self._write_session(session_id, data)

    def get_status(self, session_id: str) -> SessionStatus:
        data = self._read_session(session_id)
        return SessionStatus(data["status"])

    def get_session_data(self, session_id: str) -> dict:
        return self._read_session(session_id)

    def _session_file(self, session_id: str) -> Path:
        return self.base_dir / session_id / "session.json"

    def _read_session(self, session_id: str) -> dict:
        with open(self._session_file(session_id)) as f:
            return json.load(f)

    def _write_session(self, session_id: str, data: dict):
        with open(self._session_file(session_id), "w") as f:
            json.dump(data, f, indent=2, default=str)
