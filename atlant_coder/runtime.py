"""Local authenticated service and durable, detached run processes."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import requests
from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from codex_gateway.app import create_app
from codex_gateway.security import require_bearer_token
import traceback
import uvicorn
from codex_gateway.config import Settings

from atlant_coder.app import run as run_application
from atlant_coder.utils.checkpoint import clear

TERMINAL = {"completed", "failed", "stopped", "interrupted", "unverified"}
ACTIVE = {"starting", "running", "stopping", "waiting_approval", "waiting_input"}


def _error_reason(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args == ("attempt_id",):
        return "Старый сервис создал запуск без attempt_id; обновлённый worker не смог начать работу. Требуется перезапуск сервиса."
    if isinstance(exc, (InterruptedError, KeyboardInterrupt)):
        return "Остановка запрошена пользователем"
    text = str(exc).strip()
    if text == "project_tests_failed":
        return "Проверки проекта не прошли после разрешённых исправлений."
    return text[:1000] or type(exc).__name__


def _interruption_reason(directory: Path, exit_code=None) -> str:
    try:
        with (directory / "worker.log").open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 8192))
            tail = stream.read().decode("utf-8", errors="replace")
        if "ATLANT_WORKER_ATTEMPT " in tail:
            tail = tail.rsplit("ATLANT_WORKER_ATTEMPT ", 1)[1]
        if "KeyError: 'attempt_id'" in tail:
            return _error_reason(KeyError("attempt_id"))
        for line in reversed(tail.splitlines()):
            if line.startswith(
                (
                    "ModuleNotFoundError:",
                    "ImportError:",
                    "PermissionError:",
                    "RuntimeError:",
                    "ValueError:",
                )
            ):
                return "Worker завершился с ошибкой: " + line[:700]
    except OSError:
        pass
    if exit_code is not None:
        return f"Процесс worker завершился с кодом {exit_code}; результат последней команды может быть неизвестен."
    return "Процесс worker больше не работает; причина завершения не записана. Результат последней команды может быть неизвестен."


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            os.chmod(temp, 0o600)
            json.dump(data, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def directory(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("invalid run id")
        return self.root / run_id

    @contextlib.contextmanager
    def locked(self, run_id: str):
        directory = self.directory(run_id)
        if not directory.is_dir():
            raise KeyError(run_id)
        with (directory / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield directory

    def create(
        self, workspace: str, task: str, mode: str, policy_mode: str = "full"
    ) -> dict:
        run_id = uuid.uuid4().hex
        directory = self.directory(run_id)
        directory.mkdir(mode=0o700)
        state = dict(
            id=run_id,
            workspace=workspace,
            task=task,
            mode=mode,
            policy_mode=policy_mode,
            status="starting",
            stage="Подготовка",
            agents=[],
            plan=[],
            changes=[],
            checks=[],
            usage={},
            counters={},
            messages=[],
            last_seq=0,
            revision=0,
            stop_requested=False,
            attempt_id=uuid.uuid4().hex,
            pending_approval=None,
            created_at=time.time(),
            pid=None,
            can_resume=False,
            error=None,
            interruption_reason=None,
        )
        atomic_json(directory / "state.json", state)
        self.event(run_id, "task", task)
        return self.get(run_id)

    def get(self, run_id: str) -> dict:
        with self.locked(run_id) as directory:
            return json.loads((directory / "state.json").read_text())

    def update(self, run_id: str, **values) -> dict:
        with self.locked(run_id) as directory:
            state = json.loads((directory / "state.json").read_text())
            state.update(values)
            atomic_json(directory / "state.json", state)
            return state

    def event(self, run_id: str, kind: str, message: str, data=None) -> None:
        with self.locked(run_id) as directory:
            state = json.loads((directory / "state.json").read_text())
            state["last_seq"] += 1
            event = dict(
                seq=state["last_seq"],
                kind=kind,
                message=message,
                data=data or {},
                time=time.time(),
            )
            with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
            atomic_json(directory / "state.json", state)

    def events(self, run_id: str, after: int = 0) -> list[dict]:
        with self.locked(run_id) as directory:
            path = directory / "events.jsonl"
            if not path.exists():
                return []
            events = []
            for line in path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event["seq"] > after:
                    events.append(event)
                    if len(events) == 500:
                        break
            return events

    def steer(self, run_id: str, message: str) -> dict:
        with self.locked(run_id) as directory:
            state = json.loads((directory / "state.json").read_text())
            if state["status"] not in ACTIVE:
                raise ValueError("run is not active; start a new task or resume it")
            state["revision"] += 1
            state["messages"].append(
                {"revision": state["revision"], "message": message}
            )
            atomic_json(directory / "state.json", state)
        self.event(run_id, "user", message)
        return state

    def runs(self) -> list[dict]:
        return sorted(
            [
                self.get(p.name)
                for p in self.root.iterdir()
                if p.is_dir() and len(p.name) == 32
            ],
            key=lambda s: s["created_at"],
            reverse=True,
        )


class RunControl:
    def __init__(self, store: RunStore, run_id: str):
        self.store, self.run_id = store, run_id
        self.directory = store.directory(run_id)
        self.applied_revision = 0
        self.attempt_id = store.get(run_id).get("attempt_id")

    def stopped(self) -> bool:
        state = self.store.get(self.run_id)
        return state["stop_requested"] or state.get("attempt_id") != self.attempt_id

    def check_stopped(self):
        if self.stopped():
            raise InterruptedError("Остановка запрошена пользователем")

    @contextlib.contextmanager
    def publication(self):
        # Stop and the final short file write are ordered by the same lock.
        # Never hold it while waiting for a model or running project checks.
        with self.store.locked(self.run_id) as directory:
            state = json.loads((directory / "state.json").read_text())
            if state["stop_requested"] or state.get("attempt_id") != self.attempt_id:
                raise InterruptedError("Остановка запрошена пользователем")
            yield state["revision"] <= self.applied_revision

    def event(self, kind: str, message: str, data=None):
        self.store.event(self.run_id, kind, message, data)

    def pending(self) -> bool:
        state = self.store.get(self.run_id)
        return state["stop_requested"] or state["revision"] > self.applied_revision

    def messages(self) -> list[dict]:
        state = self.store.get(self.run_id)
        if state["stop_requested"]:
            raise InterruptedError("Остановка запрошена пользователем")
        return [m for m in state["messages"] if m["revision"] > self.applied_revision]

    def ask(self, question: str) -> str:
        self.store.update(self.run_id, status="waiting_input", stage=question)
        self.event("question", question)
        while True:
            state = self.store.get(self.run_id)
            if state["stop_requested"] or state.get("attempt_id") != self.attempt_id:
                raise InterruptedError("Остановка запрошена пользователем")
            answers = [
                item
                for item in state["messages"]
                if item["revision"] > self.applied_revision
            ]
            if answers:
                answer = answers[0]
                self.applied_revision = answer["revision"]
                self.store.update(self.run_id, status="running", stage="running")
                return answer["message"]
            time.sleep(0.2)

    def approval(self, description: str) -> bool:
        approval_id = uuid.uuid4().hex
        self.store.update(
            self.run_id,
            status="waiting_approval",
            pending_approval={"id": approval_id, "description": description},
        )
        self.event("approval", description)
        while True:
            state = self.store.get(self.run_id)
            if state["stop_requested"]:
                raise InterruptedError("Остановка запрошена пользователем")
            approval = state.get("pending_approval") or {}
            if approval.get("id") == approval_id and "allow" in approval:
                self.store.update(self.run_id, status="running", pending_approval=None)
                return approval["allow"]
            time.sleep(0.2)

    def publish(self, state: dict, reason: str):
        metrics = list(state.get("coordination_requests", []))
        for metric in state.get("generation_metrics", []):
            metrics.extend(metric.get("requests", []))
        usage = {}
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            vals = [(m.get("usage") or {}).get(key) for m in metrics]
            usage[key] = (
                sum(v for v in vals if isinstance(v, int))
                if vals and all(isinstance(v, int) for v in vals)
                else None
            )
        counters = {
            "plan_corrections": state.get("initial_plan_corrections", 0),
            "architecture_reworks": len(state.get("rework_requests", [])),
            "post_test_repairs": state.get("post_test_repairs", 0),
            "draft_reviews": sum(
                m.get("draft_reviews", 0) for m in state.get("generation_metrics", [])
            ),
        }
        completed = set(state.get("completed_files", []))
        active = {item.get("file") for item in state.get("active_agents", [])}
        plan = [
            {
                "path": path,
                "description": (
                    description
                    if isinstance(description, str)
                    else description.get("desc", "")
                ),
                "status": (
                    "completed"
                    if path in completed
                    else "running" if path in active else "pending"
                ),
            }
            for path, description in state.get("files", {}).items()
        ]
        extra = {}
        for field in ("result_summary", "final_summary"):
            if isinstance(state.get(field), str) and state[field].strip():
                extra["result_summary"] = state[field].strip()
                break
        self.store.update(
            self.run_id,
            stage=reason,
            agents=state.get("active_agents", []),
            changes=state.get("changes", []),
            checks=state.get("validation_results", []),
            usage=usage,
            counters=counters,
            plan=plan,
            **extra,
        )
        self.event("stage", reason)


class RuntimeClient:
    def __init__(self, url: str, token: str):
        self.url, self.token = url, token

    def _request(self, method: str, path: str, **kwargs):
        result = requests.request(
            method,
            self.url + path,
            headers={"Authorization": "Bearer " + self.token},
            timeout=10,
            **kwargs,
        )
        if not result.ok:
            raise RuntimeError(f"{result.status_code}: {result.text[:1000]}")
        return result.json()

    def runs(self):
        return self._request("GET", "/runs")

    def start(self, workspace, task, mode="agent", policy_mode=None):
        return self._request(
            "POST",
            "/runs",
            json=dict(
                workspace=workspace,
                task=task,
                mode=mode,
                policy_mode=policy_mode or os.environ.get("ATLANT_POLICY_MODE", "full"),
            ),
        )

    def snapshot(self, run_id):
        return self._request("GET", f"/runs/{run_id}")

    def events(self, run_id, after=0):
        return self._request("GET", f"/runs/{run_id}/events", params={"after": after})

    def steer(self, run_id, message):
        return self._request(
            "POST", f"/runs/{run_id}/messages", json={"message": message}
        )

    def stop(self, run_id):
        return self._request("POST", f"/runs/{run_id}/stop")

    def resume(self, run_id):
        return self._request("POST", f"/runs/{run_id}/resume")

    def approve(self, run_id, approval_id, allow):
        return self._request(
            "POST",
            f"/runs/{run_id}/approve",
            json=dict(approval_id=approval_id, allow=allow),
        )


def state_directory(workspace: str) -> Path:
    base = Path(
        os.environ.get(
            "ATLANT_STATE_DIR",
            str(
                Path(
                    os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))
                )
                / "atlant"
            ),
        )
    ).expanduser()
    key = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()[:16]
    directory = (base / key).resolve()
    if directory.is_relative_to(Path(workspace).resolve()):
        raise ValueError("ATLANT_STATE_DIR must be outside the target workspace")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    return directory


def _check_codex_capabilities(codex_bin: str) -> None:
    for command in (
        [codex_bin, "exec", "--help"],
        [codex_bin, "exec", "resume", "--help"],
    ):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Could not check Codex CLI structured-output support within 5 seconds"
            ) from exc
        if (
            result.returncode != 0
            or "--output-schema" not in result.stdout + result.stderr
        ):
            raise RuntimeError(
                f"Codex CLI must support --output-schema for {' '.join(command[1:-1])}; upgrade Codex before starting Atlant"
            )


def _worker_alive(state: dict, state_dir: Path) -> bool:
    pid = state.get("pid")
    if type(pid) is not int or pid <= 0:
        return False
    try:
        process = Path("/proc") / str(pid)
        args = process.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        args = [os.fsdecode(arg) for arg in args]
        process_state = (
            process.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0]
        )
        return (
            process_state not in {"Z", "X"}
            and len(args) == 8
            and args[1:5] == ["-m", "atlant_coder.runtime", "worker", "--state-dir"]
            and Path(args[5]).resolve() == state_dir.resolve()
            and args[6:] == ["--run-id", state["id"]]
        )
    except (OSError, ValueError, IndexError):
        return False


def ensure_service(workspace: str) -> RuntimeClient:
    workspace = str(Path(workspace).expanduser().resolve(strict=True))
    if not Path(workspace).is_dir():
        raise ValueError("workspace must be a directory")
    directory = state_directory(workspace)
    with (directory / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        descriptor = directory / "service.json"
        previous = None
        if descriptor.exists():
            previous = json.loads(descriptor.read_text())
            address = urlsplit(previous["url"])
            if (
                previous.get("workspace") != workspace
                or address.scheme != "http"
                or address.hostname != "127.0.0.1"
                or not address.port
                or address.username
                or address.password
                or address.path
                or address.query
                or address.fragment
                or not isinstance(previous.get("token"), str)
                or not previous["token"]
            ):
                raise RuntimeError(
                    "Invalid local service descriptor; refusing to replace its address or token"
                )
            client = RuntimeClient(previous["url"], previous["token"])
            try:
                health = client._request("GET", "/atlant/health")
            except requests.ConnectionError:
                pass
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "Existing local service could not be verified; refusing to replace its address or token"
                ) from exc
            else:
                if health.get("workspace") != workspace:
                    raise RuntimeError("local service workspace mismatch")
                client.service_info = health
                return client
        codex_bin = shutil.which("codex")
        if not codex_bin:
            raise RuntimeError(
                "Codex CLI is missing; install it and authenticate before running Atlant"
            )
        _check_codex_capabilities(codex_bin)
        token = previous["token"] if previous else secrets.token_urlsafe(32)
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", address.port if previous else 0))
            except OSError as exc:
                raise RuntimeError(
                    "Previous local service port is unavailable; refusing to change the endpoint used by detached workers"
                ) from exc
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        if not previous:
            atomic_json(descriptor, dict(url=url, token=token, workspace=workspace))
        env = dict(os.environ, ATLANT_SERVICE_TOKEN=token)
        with (directory / "service.log").open("a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "atlant_coder.runtime",
                    "serve",
                    "--state-dir",
                    str(directory),
                    "--port",
                    str(port),
                    "--workspace",
                    workspace,
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
                start_new_session=True,
            )
        client = RuntimeClient(url, token)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Gateway startup failed; see {directory / 'service.log'}"
                    )
                try:
                    health = client._request("GET", "/atlant/health")
                    if health.get("workspace") != workspace:
                        raise RuntimeError("local service workspace mismatch")
                    client.service_info = health
                    return client
                except requests.ConnectionError:
                    time.sleep(0.1)
            raise RuntimeError(
                f"Gateway startup timed out; see {directory / 'service.log'}"
            )
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            raise


def create_service(settings, state_dir: Path, *, launch=None):
    store = RunStore(state_dir / "runs")

    def run_cancelled(run_id, attempt_id):
        try:
            state = store.get(run_id)
            return (
                state.get("stop_requested", False)
                or state.get("attempt_id") != attempt_id
                or state["status"] not in ACTIVE
            )
        except (OSError, ValueError, KeyError):
            return True

    app = create_app(settings=settings, run_cancelled=run_cancelled)
    guard = threading.Lock()
    processes = {}

    def auth(authorization: str | None = Header(default=None)):
        require_bearer_token(authorization, settings.api_token)

    def get(run_id):
        try:
            state = store.get(run_id)
        except (ValueError, KeyError):
            raise HTTPException(404, "run not found")
        if state["status"] in ACTIVE:
            process = processes.get(run_id)
            alive = (
                (process is None or process.poll() is None)
                if run_id in processes
                else _worker_alive(state, state_dir)
            )
            if not alive:
                reason = _interruption_reason(
                    store.directory(run_id),
                    process.poll() if process is not None else None,
                )
                with store.locked(run_id) as directory:
                    current = json.loads((directory / "state.json").read_text())
                    if current["status"] in ACTIVE and current.get("pid") == state.get(
                        "pid"
                    ):
                        current.update(
                            status="interrupted",
                            can_resume=True,
                            pending_approval=None,
                            stage=reason,
                            interruption_reason=reason,
                            error=reason,
                        )
                        atomic_json(directory / "state.json", current)
                    state = current
        if state["status"] == "interrupted" and not state.get("interruption_reason"):
            reason = _interruption_reason(store.directory(run_id))
            state = store.update(run_id, interruption_reason=reason, error=reason)
        return state

    for state in store.runs():
        get(state["id"])

    def start_worker(state):
        if launch:
            launch(state)
            processes[state["id"]] = None
            return
        directory = store.directory(state["id"])
        env = dict(
            os.environ,
            CODEX_GATEWAY_URL=f"http://127.0.0.1:{settings.port}",
            CODEX_GATEWAY_TOKEN=settings.api_token,
            CODEX_GATEWAY_MODEL=settings.default_model or "gpt-5.6-sol",
            ATLANT_POLICY_MODE=state.get("policy_mode", "full"),
            ATLANT_RUN_ID=state["id"],
            ATLANT_RUN_ATTEMPT=state["attempt_id"],
        )
        with (directory / "worker.log").open("a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "atlant_coder.runtime",
                    "worker",
                    "--state-dir",
                    str(state_dir),
                    "--run-id",
                    state["id"],
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        processes[state["id"]] = process
        store.update(state["id"], pid=process.pid)

    class Start(BaseModel):
        model_config = ConfigDict(extra="forbid")
        workspace: str
        task: str = Field(min_length=1, max_length=200_000)
        mode: Literal["agent", "generate"] = "agent"
        policy_mode: Literal["full", "lite", "off"] = "full"

    class Message(BaseModel):
        model_config = ConfigDict(extra="forbid")
        message: str = Field(min_length=1, max_length=100_000)

    class Approval(BaseModel):
        model_config = ConfigDict(extra="forbid")
        approval_id: str
        allow: bool

    @app.get("/atlant/health", dependencies=[Depends(auth)])
    def health():
        return {
            "status": "ok",
            "workspace": str(settings.default_workspace),
            "runtime_protocol": 2,
            "capabilities": {"fast_cancel": True},
        }

    @app.get("/runs", dependencies=[Depends(auth)])
    def runs():
        return [get(s["id"]) for s in store.runs()]

    @app.post("/runs", dependencies=[Depends(auth)])
    def start(payload: Start):
        workspace = str(Path(payload.workspace).expanduser().resolve())
        if workspace != str(settings.default_workspace):
            raise HTTPException(403, "this service only owns its configured workspace")
        if not payload.task.strip():
            raise HTTPException(422, "task must not be blank")
        with guard:
            if any(get(s["id"])["status"] in ACTIVE for s in store.runs()):
                raise HTTPException(409, "workspace already has an active writer")
            state = store.create(
                workspace, payload.task, payload.mode, payload.policy_mode
            )
            try:
                start_worker(state)
            except Exception as exc:
                store.update(
                    state["id"],
                    status="failed",
                    stage=str(exc),
                    can_resume=True,
                    pending_approval=None,
                )
                raise HTTPException(503, "could not start worker") from exc
            return store.get(state["id"])

    @app.get("/runs/{run_id}", dependencies=[Depends(auth)])
    def snapshot(run_id: str):
        return get(run_id)

    @app.get("/runs/{run_id}/events", dependencies=[Depends(auth)])
    def events(run_id: str, after: int = 0):
        get(run_id)
        return store.events(run_id, max(0, after))

    @app.post("/runs/{run_id}/messages", dependencies=[Depends(auth)])
    def message(run_id: str, payload: Message):
        get(run_id)
        try:
            return store.steer(run_id, payload.message)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/runs/{run_id}/stop", dependencies=[Depends(auth)])
    def stop(run_id: str):
        get(run_id)
        with store.locked(run_id) as directory:
            state = json.loads((directory / "state.json").read_text())
            if state["status"] in ACTIVE:
                state.update(
                    stop_requested=True, status="stopping", pending_approval=None
                )
                atomic_json(directory / "state.json", state)
            return state

    @app.post("/runs/{run_id}/resume", dependencies=[Depends(auth)])
    def resume(run_id: str):
        with guard:
            state = get(run_id)
            if state["status"] not in {
                "failed",
                "stopped",
                "interrupted",
                "unverified",
            }:
                raise HTTPException(409, "run cannot be resumed")
            process = processes.get(run_id)
            if (process is not None and process.poll() is None) or _worker_alive(
                state, state_dir
            ):
                raise HTTPException(409, "previous worker has not exited yet")
            if app.state.store.run_is_active(run_id):
                raise HTTPException(409, "previous model request has not exited yet")
            if any(get(s["id"])["status"] in ACTIVE for s in store.runs()):
                raise HTTPException(409, "workspace already has an active writer")
            state = store.update(
                run_id,
                status="starting",
                stop_requested=False,
                attempt_id=uuid.uuid4().hex,
                pending_approval=None,
                can_resume=False,
                error=None,
                interruption_reason=None,
            )
            try:
                start_worker(state)
            except Exception as exc:
                store.update(
                    run_id,
                    status="failed",
                    stage=str(exc),
                    can_resume=True,
                    pending_approval=None,
                )
                raise HTTPException(503, "could not resume worker") from exc
            return store.get(run_id)

    @app.post("/runs/{run_id}/approve", dependencies=[Depends(auth)])
    def approve(run_id: str, payload: Approval):
        get(run_id)
        with store.locked(run_id) as directory:
            state = json.loads((directory / "state.json").read_text())
            pending = state.get("pending_approval")
            if (
                state["status"] != "waiting_approval"
                or state.get("stop_requested")
                or not pending
                or pending["id"] != payload.approval_id
                or "allow" in pending
            ):
                raise HTTPException(409, "approval is not pending")
            pending["allow"] = payload.allow
            atomic_json(directory / "state.json", state)
        return state

    app.state.atlant_store = store
    return app


def worker(state_dir: Path, run_id: str) -> int:
    store = RunStore(state_dir / "runs")
    control = None
    try:
        # A detached older daemon may launch current code with a legacy state.
        # Upgrade the record before constructing RunControl or reading its token.
        with store.locked(run_id) as directory:
            state = json.loads((directory / "state.json").read_text())
            if not state.get("attempt_id"):
                state["attempt_id"] = uuid.uuid4().hex
                atomic_json(directory / "state.json", state)
        control = RunControl(store, run_id)
        print("ATLANT_WORKER_ATTEMPT " + state["attempt_id"], flush=True)
        os.environ["ATLANT_CHECKPOINT_DIR"] = str(control.directory)
        os.environ["ATLANT_RUN_ID"] = run_id
        os.environ["ATLANT_RUN_ATTEMPT"] = state["attempt_id"]
        store.update(
            run_id,
            status="running",
            pid=os.getpid(),
            error=None,
            interruption_reason=None,
        )
        control.check_stopped()
        while True:
            result = run_application(
                state["workspace"], state["task"], mode=state["mode"], control=control
            )
            control.check_stopped()
            status = "completed" if result.get("project_verified") else "unverified"
            with store.locked(run_id) as directory:
                current = json.loads((directory / "state.json").read_text())
                if (
                    current["stop_requested"]
                    or current.get("attempt_id") != control.attempt_id
                ):
                    raise InterruptedError("Остановка запрошена пользователем")
                if current["revision"] > control.applied_revision:
                    current.update(status="running", stage="Обработка уточнения")
                    atomic_json(directory / "state.json", current)
                    stale = True
                else:
                    current.update(
                        status=status,
                        stage=status,
                        agents=[],
                        can_resume=status != "completed",
                        pending_approval=None,
                        error=None,
                        interruption_reason=None,
                    )
                    atomic_json(directory / "state.json", current)
                    stale = False
            if not stale:
                break
            control.event(
                "discarded", "Финальный результат отменён из-за изменения задачи"
            )
        if status == "completed":
            clear(state["workspace"])
        control.event("result", status)
        return 0 if status == "completed" else 2
    except BaseException as exc:
        status = (
            "stopped"
            if isinstance(exc, (InterruptedError, KeyboardInterrupt))
            or (control is not None and control.stopped())
            else "failed"
        )
        store.update(
            run_id,
            status=status,
            stage=str(exc),
            error=_error_reason(exc),
            interruption_reason=_error_reason(exc) if status == "stopped" else None,
            agents=[],
            can_resume=True,
            pending_approval=None,
        )
        store.event(run_id, "error", _error_reason(exc))

        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["serve", "worker"])
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--workspace")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.action == "worker":
        return worker(args.state_dir, args.run_id)

    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise RuntimeError(
            "Codex CLI is missing; install it and authenticate before running Atlant"
        )
    _check_codex_capabilities(codex_bin)
    model = os.environ.get("CODEX_GATEWAY_MODEL", "gpt-5.6-sol")
    settings = Settings(
        api_token=os.environ["ATLANT_SERVICE_TOKEN"],
        host="127.0.0.1",
        port=args.port,
        codex_bin=codex_bin,
        state_dir=args.state_dir / "gateway",
        log_file=args.state_dir / "gateway.log",
        log_level="INFO",
        allowed_workspace_root=Path(args.workspace),
        default_workspace=Path(args.workspace),
        default_sandbox="read-only",
        models_config_path=args.state_dir / "models.json",
        available_models=(model,),
        default_model=model,
        allow_network=True,
        schema_json_limit=250_000,
        codex_timeout_seconds=600,
        codex_job_timeout_seconds=600,
    )
    uvicorn.run(
        create_service(settings, args.state_dir), host=settings.host, port=settings.port
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
