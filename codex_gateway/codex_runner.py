from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .session_store import SessionRecord
from .logging_utils import format_multiline_block
from .schemas import READ_ONLY_SANDBOX


logger = logging.getLogger(__name__)


@dataclass
class CodexRunResult:
    reply: str
    thread_id: str | None
    # Native turn.completed reports cumulative usage for the entire thread.
    usage: dict[str, int] | None = None


class CodexRunnerError(Exception):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class CodexRunCancelledError(CodexRunnerError):
    def __init__(self):
        super().__init__("Run stopped by user; model result discarded", retryable=False)


def _source_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser().resolve()


def _seed_read_only_auth(source_home: Path, managed_home: Path) -> None:
    source_auth = source_home / "auth.json"
    if not source_auth.exists():
        return

    managed_auth = managed_home / "auth.json"
    should_copy = not managed_auth.exists()
    if managed_auth.exists():
        should_copy = source_auth.stat().st_mtime_ns > managed_auth.stat().st_mtime_ns
    if should_copy:
        shutil.copy2(source_auth, managed_auth)


def _prepare_read_only_home(
    state_dir: Path, session: SessionRecord
) -> tuple[Path, Path]:
    managed_home = state_dir / "codex-read-only-home" / session.session_id
    sqlite_home = state_dir / "codex-read-only-sqlite" / session.session_id
    managed_home.mkdir(parents=True, exist_ok=True)
    sqlite_home.mkdir(parents=True, exist_ok=True)

    stale_config = managed_home / "config.toml"
    if stale_config.exists():
        stale_config.unlink()

    source_home = _source_codex_home()
    _seed_read_only_auth(source_home, managed_home)
    if session.generation_only:
        # CODEX_HOME isolation alone does not suppress ~/.agents skills. Disable
        # those paths explicitly, including system skills installed on startup.
        paths = set((Path.home() / ".agents" / "skills").glob("*/SKILL.md"))
        paths.update((managed_home / "skills").glob("**/SKILL.md"))
        for path in (source_home / "skills" / ".system").glob("*/SKILL.md"):
            paths.add(
                managed_home / "skills" / ".system" / path.parent.name / "SKILL.md"
            )
        entries = [
            "[[skills.config]]\npath = " + json.dumps(str(path)) + "\nenabled = false\n"
            for path in sorted(paths)
        ]
        stale_config.write_text("\n".join(entries), encoding="utf-8")
    return managed_home, sqlite_home


def build_subprocess_env(session: SessionRecord, *, state_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    if session.sandbox != READ_ONLY_SANDBOX:
        return environment

    managed_home, sqlite_home = _prepare_read_only_home(state_dir, session)
    environment["CODEX_HOME"] = str(managed_home)
    environment["CODEX_SQLITE_HOME"] = str(sqlite_home)
    return environment


def build_command(
    codex_bin: str,
    session: SessionRecord,
    prompt: str,
    output_path: str,
    *,
    allow_network: bool,
    output_schema_path: str | None = None,
) -> list[str]:
    command = [
        codex_bin,
        "-a",
        "never",
        "-c",
        f"sandbox.network_access={'true' if allow_network else 'false'}",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "-C",
        session.workspace,
        "-s",
        session.sandbox,
        "-o",
        output_path,
    ]
    if session.generation_only:
        # These sessions receive source only through coordinator tool results.
        options = [
            "-c",
            'web_search="disabled"',
            "-c",
            "mcp_servers={}",
            "-c",
            "plugins={}",
        ]
        options.extend(["--enable", "skip_host_skill_discovery"])
        for feature in (
            "shell_tool",
            "unified_exec",
            "multi_agent",
            "apps",
            "browser_use",
            "browser_use_external",
            "computer_use",
            "in_app_browser",
            "image_generation",
            "view_image",
            "code_mode",
            "code_mode_host",
            "artifact",
            "sleep_tool",
            "tool_suggest",
            "memories",
            "plugins",
            "hooks",
            "skill_search",
        ):
            options.extend(["--disable", feature])
        command[1:1] = options
    if session.model:
        command.extend(["-m", session.model])

    if session.thread_id:
        command.append("resume")
        if output_schema_path:
            command.extend(["--output-schema", output_schema_path])
        command.extend([session.thread_id, "-"])
    else:
        if output_schema_path:
            command.extend(["--output-schema", output_schema_path])
        command.append("-")
    return command


def run_codex(
    *,
    codex_bin: str,
    session: SessionRecord,
    prompt: str,
    timeout_seconds: int,
    allow_network: bool,
    state_dir: Path,
    output_schema: dict | None = None,
    cancelled=None,
) -> CodexRunResult:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        output_path = temp_file.name

    schema_path = None
    try:
        if session.generation_only:
            workspace = state_dir / "generation-workspaces" / session.session_id
            workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
            session = replace(
                session, workspace=str(workspace), sandbox=READ_ONLY_SANDBOX
            )
        if output_schema is not None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", encoding="utf-8", delete=False
            ) as schema_file:
                schema_path = schema_file.name
                json.dump(output_schema, schema_file)
        command = build_command(
            codex_bin,
            session,
            prompt,
            output_path,
            allow_network=allow_network,
            output_schema_path=schema_path,
        )
        environment = build_subprocess_env(session, state_dir=state_dir)
        logger.info(
            "starting codex run session_id=%s thread_id=%s workspace=%s model=%s sandbox=%s command=%s",
            session.session_id,
            session.thread_id,
            session.workspace,
            session.model,
            session.sandbox,
            (
                command[:-1] + ["<prompt>"]
                if not session.thread_id
                else command[:-1] + ["<prompt>"]
            ),
        )
        logger.debug(format_multiline_block("cli prompt payload", prompt))
        if cancelled and cancelled():
            raise CodexRunCancelledError()
        # A file gives the child the complete prompt and EOF independently of
        # communicate's timeout/retry state. Retrying a partially written stdin
        # pipe with input=None can strand a slow reader on Python 3.12.
        with tempfile.TemporaryFile(mode="w+b") as prompt_input:
            prompt_input.write(prompt.encode("utf-8"))
            prompt_input.seek(0)
            process = subprocess.Popen(
                command,
                stdin=prompt_input,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=True,
            )

        thread_id = session.thread_id
        usage = None
        non_json_lines: list[str] = []
        try:
            deadline = time.monotonic() + timeout_seconds
            while True:
                if cancelled and cancelled():
                    raise CodexRunCancelledError()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    output, _ = process.communicate(
                        timeout=min(0.1, remaining) if cancelled else remaining
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue
            if cancelled and cancelled():
                raise CodexRunCancelledError()
            for raw_line in output.splitlines():
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    non_json_lines.append(line)
                    continue

                if event.get("type") == "thread.started":
                    thread_id = event.get("thread_id") or thread_id
                elif event.get("type") == "turn.completed" and isinstance(
                    event.get("usage"), dict
                ):
                    counts = {
                        key: value
                        for key, value in event["usage"].items()
                        if isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                    }
                    if counts:
                        usage = counts
                elif event.get("type") == "error":
                    non_json_lines.append(str(event.get("message", "")))
            return_code = process.returncode
        except (subprocess.TimeoutExpired, CodexRunCancelledError) as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            if isinstance(exc, CodexRunCancelledError):
                raise
            logger.warning(
                "codex timed out session_id=%s thread_id=%s timeout_seconds=%s",
                session.session_id,
                session.thread_id,
                timeout_seconds,
            )
            raise CodexRunnerError(
                f"codex timed out after {timeout_seconds}s",
                retryable=True,
            ) from exc

        reply = ""
        try:
            with open(output_path, encoding="utf-8") as handle:
                reply = handle.read().strip()
        except OSError as exc:
            raise CodexRunnerError("failed to read codex output") from exc

        # Markers that indicate a transient / recoverable failure. Notably
        # "no such thread" / "session not found" mean the CLI-side thread
        # died out from under us (e.g. gateway restart mid-session, worker
        # was SIGINT'd) — the caller can retry with thread_id cleared.
        _RETRY_MARKERS = (
            "reconnect",
            "timed out",
            "session lost",
            "session not found",
            "no such thread",
            "thread not found",
            "unknown thread",
            "signal",  # covers "terminated by signal N" from a killed worker
        )

        if return_code != 0:
            details = "\n".join(non_json_lines[-5:]).strip()
            if not details:
                details = f"codex exited with status {return_code}"
            low = details.lower()
            # Negative return_code = killed by signal (SIGTERM/SIGKILL/SIGINT).
            # Always retryable — the worker died from outside, not a bad prompt.
            retryable = return_code < 0 or any(m in low for m in _RETRY_MARKERS)
            logger.error(
                format_multiline_block(
                    f"codex failed session_id={session.session_id} thread_id={thread_id} return_code={return_code}",
                    details,
                )
            )
            raise CodexRunnerError(details, retryable=retryable)

        if not reply:
            details = (
                "\n".join(non_json_lines[-5:]).strip()
                or "codex returned an empty response"
            )
            low = details.lower()
            retryable = any(m in low for m in _RETRY_MARKERS)
            logger.error(
                format_multiline_block(
                    f"codex returned empty reply session_id={session.session_id} thread_id={thread_id}",
                    details,
                )
            )
            raise CodexRunnerError(details, retryable=retryable)

        logger.info(
            "codex completed session_id=%s thread_id=%s reply_chars=%s",
            session.session_id,
            thread_id,
            len(reply),
        )
        logger.debug(format_multiline_block("cli response payload", reply))
        return CodexRunResult(reply=reply, thread_id=thread_id, usage=usage)
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass
        if schema_path:
            try:
                os.unlink(schema_path)
            except OSError:
                pass
