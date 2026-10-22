from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .session_store import utc_now


@dataclass
class SchemaRecord:
    schema_id: str
    schema: dict[str, Any]
    name: str | None
    description: str | None
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema": self.schema,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaRecord":
        return cls(
            schema_id=str(data["schema_id"]),
            schema=dict(data["schema"]),
            name=data.get("name"),
            description=data.get("description"),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            updated_at=datetime.fromisoformat(str(data["updated_at"])),
        )


class SchemaStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.schemas_dir = self.base_dir / "schemas"
        self.schemas_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        schema: dict[str, Any],
        name: str | None,
        description: str | None,
    ) -> SchemaRecord:
        now = utc_now()
        record = SchemaRecord(
            schema_id=str(uuid.uuid4()),
            schema=schema,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )
        self.save(record)
        return record

    def get(self, schema_id: str) -> SchemaRecord | None:
        path = self.schemas_dir / f"{schema_id}.json"
        if not path.exists():
            return None
        return SchemaRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, record: SchemaRecord) -> None:
        record.updated_at = utc_now()
        path = self.schemas_dir / f"{record.schema_id}.json"
        path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
