from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .session_store import utc_now


@dataclass
class JobRecord:
    job_id: str
    status: str
    session_id: str | None
    thread_id: str | None
    workspace: str | None
    model: str | None
    sandbox: str | None
    created_session: bool
    output_mode: str
    used_schema_id: str | None
    validation_retries: int
    duration_ms: int | None
    reply: str | dict[str, Any] | list[Any] | None
    error: dict[str, Any] | None
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "workspace": self.workspace,
            "model": self.model,
            "sandbox": self.sandbox,
            "created_session": self.created_session,
            "output_mode": self.output_mode,
            "used_schema_id": self.used_schema_id,
            "validation_retries": self.validation_retries,
            "usage": self.usage,
            "duration_ms": self.duration_ms,
            "reply": self.reply,
            "error": self.error,
            "submitted_at": self.submitted_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        started_at = data.get("started_at")
        finished_at = data.get("finished_at")
        return cls(
            job_id=str(data["job_id"]),
            status=str(data["status"]),
            session_id=data.get("session_id"),
            thread_id=data.get("thread_id"),
            workspace=data.get("workspace"),
            model=data.get("model"),
            sandbox=data.get("sandbox"),
            created_session=bool(data.get("created_session", False)),
            output_mode=str(data.get("output_mode", "raw")),
            used_schema_id=data.get("used_schema_id"),
            validation_retries=int(data.get("validation_retries", 0)),
            usage=data.get("usage"),
            duration_ms=data.get("duration_ms"),
            reply=data.get("reply"),
            error=data.get("error"),
            submitted_at=datetime.fromisoformat(str(data["submitted_at"])),
            started_at=datetime.fromisoformat(str(started_at)) if started_at else None,
            finished_at=(
                datetime.fromisoformat(str(finished_at)) if finished_at else None
            ),
            updated_at=datetime.fromisoformat(str(data["updated_at"])),
        )


class JobStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.jobs_dir = self.base_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.Lock()

    def create(
        self,
        *,
        session_id: str | None,
        workspace: str | None,
        model: str | None,
        sandbox: str | None,
        created_session: bool,
        output_mode: str,
        used_schema_id: str | None,
    ) -> JobRecord:
        now = utc_now()
        record = JobRecord(
            job_id=str(uuid.uuid4()),
            status="queued",
            session_id=session_id,
            thread_id=None,
            workspace=workspace,
            model=model,
            sandbox=sandbox,
            created_session=created_session,
            output_mode=output_mode,
            used_schema_id=used_schema_id,
            validation_retries=0,
            duration_ms=None,
            reply=None,
            error=None,
            submitted_at=now,
            started_at=None,
            finished_at=None,
            updated_at=now,
        )
        self.save(record)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        return JobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, record: JobRecord) -> None:
        with self._mutex:
            record.updated_at = utc_now()
            path = self.jobs_dir / f"{record.job_id}.json"
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
