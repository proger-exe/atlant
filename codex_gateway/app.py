from __future__ import annotations

import json
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .codex_runner import CodexRunnerError, CodexRunCancelledError, run_codex
from .config import (
    ALLOWED_SANDBOXES,
    READ_ONLY_PATH_CONFLICT_DETAIL,
    load_settings,
    read_only_path_conflicts,
)
from .job_store import JobRecord, JobStore
from .logging_utils import setup_logging
from .schema_store import SchemaStore
from .schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    JobCreateRequest,
    JobResponse,
    ModelListResponse,
    SchemaCreateRequest,
    SchemaResponse,
    SessionCreateRequest,
    SessionResponse,
)
from .security import require_bearer_token, validate_workspace
from .session_store import SessionRecord, SessionStore, utc_now
from .structured_output import (
    MAX_VALIDATION_RETRIES,
    StructuredOutputError,
    build_initial_prompt,
    build_retry_prompt,
    classify_failure,
    parse_and_validate_output,
    serialize_compact_json,
    validate_json_schema,
    native_output_schema,
)


def _ensure_read_only_workspace_allowed(
    workspace: str, sandbox: str, settings: Any
) -> None:
    if sandbox != "read-only":
        return
    if read_only_path_conflicts(
        Path(workspace),
        state_dir=settings.state_dir,
        log_file=settings.log_file,
    ):
        raise HTTPException(status_code=409, detail=READ_ONLY_PATH_CONFLICT_DETAIL)


def _build_session_response(record: SessionRecord) -> SessionResponse:
    return SessionResponse(
        session_id=record.session_id,
        workspace=record.workspace,
        model=record.model,
        sandbox=record.sandbox,
        thread_id=record.thread_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        generation_only=record.generation_only,
    )


def _build_schema_response(record: Any) -> SchemaResponse:
    return SchemaResponse(
        schema_id=record.schema_id,
        schema_body=record.schema,
        name=record.name,
        description=record.description,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _build_job_response(record: JobRecord) -> JobResponse:
    error = ErrorResponse(**record.error) if record.error else None
    return JobResponse(
        job_id=record.job_id,
        status=record.status,
        session_id=record.session_id,
        thread_id=record.thread_id,
        reply=record.reply,
        workspace=record.workspace,
        model=record.model,
        sandbox=record.sandbox,
        duration_ms=record.duration_ms,
        created_session=record.created_session,
        output_mode=record.output_mode,
        used_schema_id=record.used_schema_id,
        validation_retries=record.validation_retries,
        usage=record.usage,
        error=error,
        submitted_at=record.submitted_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        updated_at=record.updated_at,
    )


def _measure_json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=True))


def _is_context_overflow_error(message: str) -> bool:
    normalized = message.lower()
    markers = (
        "payload too large",
        "request too large",
        "maximum context length",
        "context length exceeded",
        "context window",
        "too many tokens",
        "prompt is too long",
        "input is too long",
        "413",
    )
    return any(marker in normalized for marker in markers)


def _build_context_overflow_detail(*, resumed_session: bool) -> str:
    if resumed_session:
        return (
            "session context too large; reduce message or chat_context, "
            "or reset the session thread via POST /sessions/{session_id}/reset"
        )
    return "request context too large; reduce message or chat_context"


def _run_codex_with_context_overflow_handling(
    *,
    session: SessionRecord,
    prompt: str,
    settings: Any,
    timeout_seconds: int,
    output_schema: dict | None = None,
    cancelled=None,
) -> Any:
    resumed_session = session.thread_id is not None
    try:
        return run_codex(
            codex_bin=settings.codex_bin,
            session=session,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            allow_network=settings.allow_network,
            state_dir=settings.state_dir,
            **({"output_schema": output_schema} if output_schema is not None else {}),
            **({"cancelled": cancelled} if cancelled is not None else {}),
        )
    except CodexRunCancelledError:
        raise
    except CodexRunnerError as exc:
        if _is_context_overflow_error(exc.message):
            raise HTTPException(
                status_code=413,
                detail=_build_context_overflow_detail(resumed_session=resumed_session),
            ) from exc
        # A follow-up prompt cannot bootstrap a replacement conversation.
        # Keep the old thread until the caller explicitly resets or starts a full task.
        if exc.retryable and session.thread_id is not None:
            raise StructuredOutputError(
                "session_context_lost",
                "Session context is unavailable; restart with the complete task. "
                + exc.message,
                status_code=409,
                retryable=False,
            ) from exc

        raise


def _record_turn_usage(
    record: SessionRecord, result: Any, store: SessionStore
) -> dict[str, int] | None:
    """Convert native thread totals to this invocation's usage under the session lock."""
    counts = result.usage
    previous = record.cumulative_usage
    delta = None
    if counts is not None and result.thread_id:
        if record.thread_id is None:
            delta = dict(counts)
        elif record.usage_thread_id == result.thread_id and previous is not None:
            if counts.keys() == previous.keys() and all(
                counts[key] >= previous[key] for key in counts
            ):
                delta = {key: value - previous[key] for key, value in counts.items()}
    # Unknown or decreasing totals must not invent usage; seed the next turn.
    record.thread_id = result.thread_id
    record.usage_thread_id = result.thread_id
    record.cumulative_usage = dict(counts) if counts is not None else None
    store.save(record)
    return delta


def _validate_chat_payload(payload: ChatRequest, settings: Any) -> None:
    if payload.response_format == "json" and not payload.schema_id:
        raise HTTPException(
            status_code=422, detail="schema_id is required for response_format=json"
        )
    if payload.response_format != "json" and payload.schema_id:
        raise HTTPException(
            status_code=422, detail="schema_id is only allowed for response_format=json"
        )


def _build_raw_prompt(payload: ChatRequest) -> str:
    parts: list[str] = []
    if payload.system_prompt is not None:
        parts.append(f"System prompt:\n{payload.system_prompt}")
    if payload.chat_context is not None:
        if isinstance(payload.chat_context, str):
            rendered_context = payload.chat_context
        else:
            rendered_context = json.dumps(
                payload.chat_context, ensure_ascii=True, indent=2
            )
        parts.append(f"Chat context:\n{rendered_context}")
    parts.append(f"User request:\n{payload.message}")
    if len(parts) == 1:
        return payload.message
    return "\n\n".join(parts)


def _resolve_model(model: str | None, settings: Any) -> str | None:
    resolved_model = model or settings.default_model
    if resolved_model and resolved_model not in settings.available_models:
        raise HTTPException(status_code=422, detail="unsupported model")
    return resolved_model


def _resolve_chat_session(
    payload: ChatRequest,
    *,
    store: SessionStore,
    settings: Any,
) -> tuple[SessionRecord, bool]:
    created_session = False
    if payload.session_id:
        if (
            payload.workspace
            or payload.model
            or payload.sandbox
            or payload.generation_only
        ):
            raise HTTPException(
                status_code=422,
                detail="workspace/model/sandbox are only allowed when creating a session",
            )
        record = store.get(payload.session_id)
        if not record:
            raise HTTPException(status_code=404, detail="session not found")
    else:
        workspace = validate_workspace(
            payload.workspace or str(settings.default_workspace),
            settings.allowed_workspace_root,
        )
        sandbox = (
            "read-only"
            if payload.generation_only
            else payload.sandbox or settings.default_sandbox
        )
        if sandbox not in ALLOWED_SANDBOXES:
            raise HTTPException(status_code=422, detail="unsupported sandbox")
        _ensure_read_only_workspace_allowed(workspace, sandbox, settings)
        record = store.create_session(
            workspace=workspace,
            model=_resolve_model(payload.model, settings),
            sandbox=sandbox,
            generation_only=payload.generation_only,
        )
        created_session = True
    return record, created_session


def _run_chat(
    *,
    payload: ChatRequest,
    store: SessionStore,
    schema_store: SchemaStore,
    settings: Any,
    timeout_seconds: int,
    run_cancelled=None,
) -> ChatResponse:
    _validate_chat_payload(payload, settings)
    record, created_session = _resolve_chat_session(
        payload,
        store=store,
        settings=settings,
    )
    _ensure_read_only_workspace_allowed(record.workspace, record.sandbox, settings)
    lock = store.acquire_session_lock(record.session_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="session is busy")

    try:
        # Resolve totals after acquiring the lock; another completed turn may
        # have saved a newer baseline between session lookup and acquisition.
        record = store.get(record.session_id) or record
        cancelled = None
        if record.owner_run_id:
            if run_cancelled is None:
                raise CodexRunCancelledError()
            cancelled = lambda: run_cancelled(
                record.owner_run_id, record.owner_attempt_id
            )
            if cancelled():
                raise CodexRunCancelledError()
        store.append_transcript(record.session_id, "User", payload.message)
        if payload.chat_context is not None:
            if isinstance(payload.chat_context, str):
                rendered_context = payload.chat_context
            else:
                rendered_context = json.dumps(
                    payload.chat_context, ensure_ascii=True, indent=2
                )
            store.append_transcript(record.session_id, "Context", rendered_context)
        started = time.monotonic()
        structured_reply: dict[str, Any] | list[Any] | None = None
        used_schema_id: str | None = None
        validation_retries = 0
        usage = None
        complete_usage = True
        output_mode = payload.response_format

        if payload.response_format == "json":
            schema_record = schema_store.get(payload.schema_id or "")
            if not schema_record:
                raise StructuredOutputError(
                    "schema_not_found", "schema not found", status_code=404
                )
            system_hash = hashlib.sha256(
                (payload.system_prompt or "").encode()
            ).hexdigest()
            send_system = not (
                record.generation_only
                and record.thread_id
                and record.system_prompt_hash == system_hash
            )
            prompt = build_initial_prompt(
                user_message=payload.message,
                schema_record=schema_record,
                chat_context=payload.chat_context,
                system_prompt=payload.system_prompt if send_system else None,
                native_only=record.generation_only,
            )
            attempts = 0
            while True:
                result = _run_codex_with_context_overflow_handling(
                    session=record,
                    prompt=prompt,
                    settings=settings,
                    timeout_seconds=timeout_seconds,
                    output_schema=native_output_schema(schema_record.schema),
                    cancelled=cancelled,
                )
                if record.generation_only:
                    record.system_prompt_hash = system_hash
                turn_usage = _record_turn_usage(record, result, store)
                if turn_usage is not None:
                    usage = {
                        key: (usage or {}).get(key, 0) + value
                        for key, value in turn_usage.items()
                    }
                else:
                    complete_usage = False
                record.thread_id = result.thread_id
                try:
                    parsed = parse_and_validate_output(result.reply, schema_record)
                    structured_reply = parsed.parsed
                    used_schema_id = schema_record.schema_id
                    break
                except StructuredOutputError as exc:
                    if attempts >= MAX_VALIDATION_RETRIES or not exc.retryable:
                        store.save(record)
                        exc.retryable = False
                        raise
                    failure = classify_failure(exc, result.reply)
                    prompt = build_retry_prompt(
                        original_prompt=build_initial_prompt(
                            user_message=payload.message,
                            schema_record=schema_record,
                            chat_context=payload.chat_context,
                            system_prompt=payload.system_prompt,
                            native_only=record.generation_only,
                        ),
                        schema_record=schema_record,
                        failure=failure,
                        native_only=record.generation_only,
                    )
                    attempts += 1
                    validation_retries = attempts
            reply_text = serialize_compact_json(structured_reply)
        else:
            prompt = _build_raw_prompt(payload)
            result = _run_codex_with_context_overflow_handling(
                session=record,
                prompt=prompt,
                settings=settings,
                timeout_seconds=timeout_seconds,
                cancelled=cancelled,
            )
            reply_text = result.reply
            usage = _record_turn_usage(record, result, store)

        duration_ms = int((time.monotonic() - started) * 1000)
        if cancelled and cancelled():
            raise CodexRunCancelledError()
        record.thread_id = result.thread_id
        store.save(record)
        store.append_transcript(record.session_id, "Codex", reply_text)
        return ChatResponse(
            session_id=record.session_id,
            thread_id=record.thread_id,
            reply=reply_text if output_mode == "raw" else (structured_reply or {}),
            workspace=record.workspace,
            model=record.model,
            sandbox=record.sandbox,
            duration_ms=duration_ms,
            created_session=created_session,
            output_mode=output_mode,
            used_schema_id=used_schema_id,
            validation_retries=validation_retries,
            usage=usage if complete_usage else None,
        )
    finally:
        lock.release()


def create_app(settings=None, *, run_cancelled=None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings)
    store = SessionStore(settings.state_dir)
    schema_store = SchemaStore(settings.state_dir)
    job_store = JobStore(settings.state_dir)
    app = FastAPI(
        title="Codex Local Gateway",
        version="0.1.0",
    )
    app.state.settings = settings
    app.state.store = store
    app.state.schema_store = schema_store
    app.state.job_store = job_store

    @app.exception_handler(CodexRunCancelledError)
    async def cancelled_error_handler(
        _: Request, exc: CodexRunCancelledError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=ErrorResponse(
                code="run_cancelled", message=exc.message, retryable=False
            ).model_dump(),
        )

    @app.exception_handler(CodexRunnerError)
    async def codex_runner_error_handler(
        _: Request, exc: CodexRunnerError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                code="codex_unavailable",
                message=exc.message,
                retryable=exc.retryable,
            ).model_dump(),
        )

    @app.exception_handler(StructuredOutputError)
    async def structured_error_handler(
        _: Request, exc: StructuredOutputError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                code=exc.code,
                message=exc.message,
                retryable=exc.retryable,
            ).model_dump(),
        )

    # NOTE: endpoints below are intentionally sync-def, not async-def.
    # FastAPI schedules sync handlers on its threadpool, so multiple /chat
    # requests actually run in parallel. Previous async-def wrapping sync
    # subprocess.Popen(codex-CLI).wait() blocked the entire event loop —
    # concurrent chats serialised, and /models/list check_status timed out
    # while a big chat was mid-flight. Do not change back to async without
    # first moving all blocking calls (subprocess, file I/O) into an
    # executor.

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "codex_bin": settings.codex_bin,
            "allowed_root": str(settings.allowed_workspace_root),
        }

    @app.get("/models/list", response_model=ModelListResponse)
    def list_models(
        authorization: str | None = Header(default=None),
    ) -> ModelListResponse:
        require_bearer_token(authorization, settings.api_token)
        return ModelListResponse(
            models=list(settings.available_models),
            default_model=settings.default_model,
        )

    @app.post("/schemas", response_model=SchemaResponse)
    def create_schema(
        payload: SchemaCreateRequest,
        authorization: str | None = Header(default=None),
    ) -> SchemaResponse:
        require_bearer_token(authorization, settings.api_token)
        if _measure_json_size(payload.schema_body) > settings.schema_json_limit:
            raise HTTPException(status_code=413, detail="schema too large")
        validate_json_schema(payload.schema_body)
        record = schema_store.create(
            schema=payload.schema_body,
            name=payload.name,
            description=payload.description,
        )
        return _build_schema_response(record)

    @app.get("/schemas/{schema_id}", response_model=SchemaResponse)
    def get_schema(
        schema_id: str,
        authorization: str | None = Header(default=None),
    ) -> SchemaResponse:
        require_bearer_token(authorization, settings.api_token)
        record = schema_store.get(schema_id)
        if not record:
            raise HTTPException(status_code=404, detail="schema not found")
        return _build_schema_response(record)

    @app.post("/sessions", response_model=SessionResponse)
    def create_session(
        payload: SessionCreateRequest,
        authorization: str | None = Header(default=None),
    ) -> SessionResponse:
        require_bearer_token(authorization, settings.api_token)
        workspace = validate_workspace(
            payload.workspace or str(settings.default_workspace),
            settings.allowed_workspace_root,
        )
        sandbox = (
            "read-only"
            if payload.generation_only
            else payload.sandbox or settings.default_sandbox
        )
        if sandbox not in ALLOWED_SANDBOXES:
            raise HTTPException(status_code=422, detail="unsupported sandbox")
        _ensure_read_only_workspace_allowed(workspace, sandbox, settings)

        if bool(payload.owner_run_id) != bool(payload.owner_attempt_id):
            raise HTTPException(422, "owner run and attempt must be supplied together")
        if payload.owner_run_id and (
            run_cancelled is None
            or run_cancelled(payload.owner_run_id, payload.owner_attempt_id)
        ):
            raise CodexRunCancelledError()
        record = store.create_session(
            workspace=workspace,
            model=_resolve_model(payload.model, settings),
            sandbox=sandbox,
            generation_only=payload.generation_only,
            owner_run_id=payload.owner_run_id,
            owner_attempt_id=payload.owner_attempt_id,
        )
        return _build_session_response(record)

    @app.get("/sessions/{session_id}", response_model=SessionResponse)
    def get_session(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> SessionResponse:
        require_bearer_token(authorization, settings.api_token)
        record = store.get(session_id)
        if not record:
            raise HTTPException(status_code=404, detail="session not found")
        return _build_session_response(record)

    @app.post(
        "/sessions/{session_id}/reset",
        response_model=SessionResponse,
    )
    def reset_session(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> SessionResponse:
        require_bearer_token(authorization, settings.api_token)
        current = store.get(session_id)
        if not current:
            raise HTTPException(status_code=404, detail="session not found")
        _ensure_read_only_workspace_allowed(
            current.workspace, current.sandbox, settings
        )
        lock = store.acquire_session_lock(session_id)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="session is busy")
        try:
            record = store.reset_thread(session_id)
        finally:
            lock.release()
        if not record:
            raise HTTPException(status_code=404, detail="session not found")
        return _build_session_response(record)

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        authorization: str | None = Header(default=None),
    ) -> ChatResponse:
        require_bearer_token(authorization, settings.api_token)
        return _run_chat(
            payload=payload,
            store=store,
            schema_store=schema_store,
            settings=settings,
            timeout_seconds=settings.codex_timeout_seconds,
            run_cancelled=run_cancelled,
        )

    @app.post("/jobs", response_model=JobResponse)
    def create_job(
        payload: JobCreateRequest,
        authorization: str | None = Header(default=None),
    ) -> JobResponse:
        require_bearer_token(authorization, settings.api_token)
        _validate_chat_payload(payload, settings)
        record, created_session = _resolve_chat_session(
            payload,
            store=store,
            settings=settings,
        )
        _ensure_read_only_workspace_allowed(record.workspace, record.sandbox, settings)
        job_record = job_store.create(
            session_id=record.session_id,
            workspace=record.workspace,
            model=record.model,
            sandbox=record.sandbox,
            created_session=created_session,
            output_mode=payload.response_format,
            used_schema_id=payload.schema_id,
        )
        job_payload = payload.model_copy(
            update={
                "session_id": record.session_id,
                "workspace": None,
                "model": None,
                "sandbox": None,
            }
        )

        def _run_job() -> None:
            current = job_store.get(job_record.job_id)
            if not current:
                return
            current.status = "running"
            current.started_at = utc_now()
            job_store.save(current)
            try:
                response = _run_chat(
                    payload=job_payload,
                    store=store,
                    schema_store=schema_store,
                    settings=settings,
                    timeout_seconds=settings.codex_job_timeout_seconds,
                    run_cancelled=run_cancelled,
                )
                current.status = "completed"
                current.session_id = response.session_id
                current.thread_id = response.thread_id
                current.reply = response.reply
                current.workspace = response.workspace
                current.model = response.model
                current.sandbox = response.sandbox
                current.duration_ms = response.duration_ms
                current.created_session = (
                    current.created_session or response.created_session
                )
                current.output_mode = response.output_mode
                current.used_schema_id = response.used_schema_id
                current.validation_retries = response.validation_retries
                current.usage = response.usage
                current.error = None
            except HTTPException as exc:
                current.status = "failed"
                error_code = (
                    "context_too_large" if exc.status_code == 413 else "http_error"
                )
                current.error = {
                    "code": error_code,
                    "message": str(exc.detail),
                    "retryable": exc.status_code >= 500,
                }
            except CodexRunCancelledError as exc:
                current.status = "failed"
                current.error = {
                    "code": "run_cancelled",
                    "message": exc.message,
                    "retryable": False,
                }
            except CodexRunnerError as exc:
                current.status = "failed"
                current.error = {
                    "code": "codex_unavailable",
                    "message": exc.message,
                    "retryable": exc.retryable,
                }
            except StructuredOutputError as exc:
                current.status = "failed"
                current.error = {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                }
            except Exception as exc:
                current.status = "failed"
                current.error = {
                    "code": "job_execution_failed",
                    "message": str(exc),
                    "retryable": False,
                }
            finally:
                current.finished_at = utc_now()
                job_store.save(current)

        threading.Thread(target=_run_job, daemon=True).start()
        return _build_job_response(job_record)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(
        job_id: str,
        authorization: str | None = Header(default=None),
    ) -> JobResponse:
        require_bearer_token(authorization, settings.api_token)
        record = job_store.get(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="job not found")
        return _build_job_response(record)

    return app
