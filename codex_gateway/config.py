from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .schemas import READ_ONLY_SANDBOX, WORKSPACE_WRITE_SANDBOX


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_SANDBOX = WORKSPACE_WRITE_SANDBOX
ALLOWED_SANDBOXES = {READ_ONLY_SANDBOX, WORKSPACE_WRITE_SANDBOX}
READ_ONLY_PATH_CONFLICT_DETAIL = (
    "read-only sandbox unavailable because CODEX_API_STATE_DIR or CODEX_API_LOG_FILE "
    "is inside the target workspace; move gateway state/logs outside the workspace"
)
DEFAULT_READ_ONLY_PATH_CONFLICT_MESSAGE = (
    "default read-only workspace conflicts with CODEX_API_STATE_DIR or CODEX_API_LOG_FILE; "
    "move gateway state/logs outside the workspace"
)


@dataclass(frozen=True)
class Settings:
    api_token: str
    host: str
    port: int
    codex_bin: str
    state_dir: Path
    log_file: Path
    log_level: str
    allowed_workspace_root: Path
    default_workspace: Path
    default_sandbox: str
    models_config_path: Path
    available_models: tuple[str, ...]
    default_model: str | None
    allow_network: bool
    schema_json_limit: int
    codex_timeout_seconds: int
    codex_job_timeout_seconds: int


def _path_is_inside(path: Path, parent: Path) -> bool:
    candidate = path.expanduser().resolve()
    root = parent.expanduser().resolve()
    return candidate == root or root in candidate.parents


def read_only_path_conflicts(
    workspace: Path,
    *,
    state_dir: Path,
    log_file: Path,
) -> tuple[str, ...]:
    conflicts: list[str] = []
    if _path_is_inside(state_dir, workspace):
        conflicts.append("CODEX_API_STATE_DIR")
    if _path_is_inside(log_file, workspace):
        conflicts.append("CODEX_API_LOG_FILE")
    return tuple(conflicts)


def _normalize_models(items: Any) -> tuple[str, ...]:
    if items is None:
        return ()
    if not isinstance(items, list):
        raise RuntimeError("models config field `models` must be an array")
    seen: set[str] = set()
    models: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise RuntimeError("models config entries must be strings")
        model = item.strip()
        if not model or model in seen:
            continue
        seen.add(model)
        models.append(model)
    return tuple(models)


def _load_models_config(path: Path) -> tuple[tuple[str, ...], str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"models config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"models config file is not valid JSON: {path}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError("models config root must be a JSON object")

    available_models = _normalize_models(raw.get("models"))
    default_model_raw = raw.get("default_model")
    if default_model_raw is None:
        default_model = None
    elif isinstance(default_model_raw, str):
        default_model = default_model_raw.strip() or None
    else:
        raise RuntimeError(
            "models config field `default_model` must be a string or null"
        )

    if default_model and default_model not in available_models:
        raise RuntimeError("models config default_model must be present in models")
    return available_models, default_model


def load_settings() -> Settings:
    load_dotenv()

    token = os.environ.get("CODEX_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("CODEX_API_TOKEN is required")

    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise RuntimeError("`codex` was not found in PATH")

    allowed_root = (
        Path(os.environ.get("CODEX_API_ALLOWED_ROOT", "/home/shelby/Projects"))
        .expanduser()
        .resolve()
    )
    default_workspace = (
        Path(os.environ.get("CODEX_API_DEFAULT_WORKSPACE", str(allowed_root)))
        .expanduser()
        .resolve()
    )

    sandbox = os.environ.get("CODEX_API_DEFAULT_SANDBOX", DEFAULT_SANDBOX).strip()
    if sandbox not in ALLOWED_SANDBOXES:
        raise RuntimeError(
            f"CODEX_API_DEFAULT_SANDBOX must be one of {sorted(ALLOWED_SANDBOXES)}"
        )

    state_dir = (
        Path(os.environ.get("CODEX_API_STATE_DIR", ".codex-api")).expanduser().resolve()
    )
    log_file = (
        Path(os.environ.get("CODEX_API_LOG_FILE", str(state_dir / "gateway.log")))
        .expanduser()
        .resolve()
    )
    allow_network_raw = os.environ.get("CODEX_API_ALLOW_NETWORK", "1").strip().lower()
    allow_network = allow_network_raw not in {"0", "false", "no", "off"}
    models_config_path = (
        Path(os.environ.get("CODEX_API_MODELS_FILE", "models.json"))
        .expanduser()
        .resolve()
    )
    available_models, default_model = _load_models_config(models_config_path)

    if sandbox == READ_ONLY_SANDBOX and read_only_path_conflicts(
        default_workspace,
        state_dir=state_dir,
        log_file=log_file,
    ):
        raise RuntimeError(DEFAULT_READ_ONLY_PATH_CONFLICT_MESSAGE)

    return Settings(
        api_token=token,
        host=os.environ.get("CODEX_API_HOST", DEFAULT_HOST),
        port=int(os.environ.get("CODEX_API_PORT", str(DEFAULT_PORT))),
        codex_bin=codex_bin,
        state_dir=state_dir,
        log_file=log_file,
        log_level=os.environ.get("CODEX_API_LOG_LEVEL", "INFO").strip().upper()
        or "INFO",
        allowed_workspace_root=allowed_root,
        default_workspace=default_workspace,
        default_sandbox=sandbox,
        models_config_path=models_config_path,
        available_models=available_models,
        default_model=default_model,
        allow_network=allow_network,
        schema_json_limit=int(os.environ.get("CODEX_API_SCHEMA_JSON_LIMIT", "24000")),
        codex_timeout_seconds=int(os.environ.get("CODEX_API_TIMEOUT_SECONDS", "600")),
        codex_job_timeout_seconds=int(
            os.environ.get(
                "CODEX_API_JOB_TIMEOUT_SECONDS",
                os.environ.get("CODEX_API_TIMEOUT_SECONDS", "600"),
            )
        ),
    )
