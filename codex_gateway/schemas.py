from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


READ_ONLY_SANDBOX = "read-only"
WORKSPACE_WRITE_SANDBOX = "workspace-write"
SandboxMode = Literal["read-only", "workspace-write"]
ResponseFormat = Literal["raw", "json"]
JobStatus = Literal["queued", "running", "completed", "failed"]


class SessionCreateRequest(BaseModel):
    workspace: str | None = None
    model: str | None = None
    sandbox: SandboxMode | None = None
    generation_only: bool = False
    owner_run_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    owner_attempt_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class SessionResponse(BaseModel):
    session_id: str
    workspace: str
    model: str | None
    sandbox: SandboxMode
    thread_id: str | None
    created_at: datetime
    updated_at: datetime
    generation_only: bool = False


class ModelListResponse(BaseModel):
    models: list[str]
    default_model: str | None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    system_prompt: str | None = None
    session_id: str | None = None
    workspace: str | None = None
    model: str | None = None
    sandbox: SandboxMode | None = None
    response_format: ResponseFormat = "raw"
    schema_id: str | None = None
    chat_context: dict[str, Any] | list[Any] | str | None = None
    generation_only: bool = False

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("message must not be empty")
        return message

    @field_validator("system_prompt")
    @classmethod
    def strip_system_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        system_prompt = value.strip()
        if not system_prompt:
            raise ValueError("system_prompt must not be empty")
        return system_prompt


class ChatResponse(BaseModel):
    session_id: str
    thread_id: str | None
    reply: str | dict[str, Any] | list[Any]
    workspace: str
    model: str | None
    sandbox: SandboxMode
    duration_ms: int
    created_session: bool
    output_mode: ResponseFormat = "raw"
    used_schema_id: str | None = None
    validation_retries: int = 0
    usage: dict[str, int] | None = None


class JobCreateRequest(ChatRequest):
    pass


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    session_id: str | None
    thread_id: str | None
    reply: str | dict[str, Any] | list[Any] | None = None
    workspace: str | None
    model: str | None
    sandbox: SandboxMode | None = None
    duration_ms: int | None = None
    created_session: bool = False
    output_mode: ResponseFormat = "raw"
    used_schema_id: str | None = None
    validation_retries: int = 0
    usage: dict[str, int] | None = None
    error: ErrorResponse | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime


class SchemaCreateRequest(BaseModel):
    schema_body: dict[str, Any] = Field(alias="schema")
    name: str | None = None
    description: str | None = None

    model_config = {"populate_by_name": True}


class SchemaResponse(BaseModel):
    schema_id: str
    schema_body: dict[str, Any] = Field(alias="schema")
    name: str | None
    description: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"populate_by_name": True}


class ErrorResponse(BaseModel):
    code: str
    message: str
    retryable: bool = False
