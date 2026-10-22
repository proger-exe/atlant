from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionRecord:
    session_id: str
    workspace: str
    model: str | None
    sandbox: str
    thread_id: str | None
    created_at: datetime
    updated_at: datetime
    generation_only: bool = False
    usage_thread_id: str | None = None
    cumulative_usage: dict[str, int] | None = None
    system_prompt_hash: str | None = None
    owner_run_id: str | None = None
    owner_attempt_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "model": self.model,
            "sandbox": self.sandbox,
            "thread_id": self.thread_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "generation_only": self.generation_only,
            "usage_thread_id": self.usage_thread_id,
            "cumulative_usage": self.cumulative_usage,
            "system_prompt_hash": self.system_prompt_hash,
            "owner_run_id": self.owner_run_id,
            "owner_attempt_id": self.owner_attempt_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | None]) -> "SessionRecord":
        return cls(
            session_id=str(data["session_id"]),
            workspace=str(data["workspace"]),
            model=data.get("model"),
            sandbox=str(data["sandbox"]),
            thread_id=data.get("thread_id"),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            updated_at=datetime.fromisoformat(str(data["updated_at"])),
            generation_only=bool(data.get("generation_only", False)),
            usage_thread_id=data.get("usage_thread_id"),
            cumulative_usage=data.get("cumulative_usage"),
            system_prompt_hash=data.get("system_prompt_hash"),
            owner_run_id=data.get("owner_run_id"),
            owner_attempt_id=data.get("owner_attempt_id"),
        )


class SessionStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.sessions_dir = self.base_dir / "sessions"
        self.transcripts_dir = self.base_dir / "transcripts"
        self._mutex = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        *,
        workspace: str,
        model: str | None,
        sandbox: str,
        generation_only: bool = False,
        owner_run_id: str | None = None,
        owner_attempt_id: str | None = None,
    ) -> SessionRecord:
        timestamp = utc_now()
        record = SessionRecord(
            session_id=str(uuid.uuid4()),
            workspace=workspace,
            model=model,
            sandbox=sandbox,
            thread_id=None,
            created_at=timestamp,
            updated_at=timestamp,
            generation_only=generation_only,
            owner_run_id=owner_run_id,
            owner_attempt_id=owner_attempt_id,
        )
        self.save(record)
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        return SessionRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, record: SessionRecord) -> None:
        record.updated_at = utc_now()
        path = self._session_path(record.session_id)
        payload = json.dumps(record.to_dict(), ensure_ascii=True, indent=2) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)

    def reset_thread(self, session_id: str) -> SessionRecord | None:
        record = self.get(session_id)
        if not record:
            return None
        record.thread_id = None
        record.usage_thread_id = None
        record.cumulative_usage = None
        record.system_prompt_hash = None
        self.save(record)
        return record

    def transcript_path(self, session_id: str) -> Path:
        return self.transcripts_dir / f"{session_id}.md"

    def append_transcript(self, session_id: str, role: str, content: str) -> None:
        safe = content.strip()
        if not safe:
            return
        path = self.transcript_path(session_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"## {role} [{utc_now().isoformat()}]\n\n")
            handle.write(safe)
            handle.write("\n\n")

    def acquire_session_lock(self, session_id: str) -> threading.Lock:
        with self._mutex:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

    def run_is_active(self, run_id: str) -> bool:
        for path in self.sessions_dir.glob("*.json"):
            record = self.get(path.stem)
            if (
                record
                and record.owner_run_id == run_id
                and self.acquire_session_lock(record.session_id).locked()
            ):
                return True
        return False

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"
