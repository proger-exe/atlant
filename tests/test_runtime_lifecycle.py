import errno
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from atlant_coder import runtime
from tests.test_runtime import auth, service, settings


def test_restart_reuses_descriptor_address_and_token(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ATLANT_STATE_DIR", str(tmp_path / "state"))
    directory = runtime.state_directory(str(workspace))
    descriptor = directory / "service.json"
    previous = {
        "url": "http://127.0.0.1:54321",
        "token": "private-test-token",
        "workspace": str(workspace),
    }
    runtime.atomic_json(descriptor, previous)
    original = descriptor.read_bytes()
    bound, launched, probes = [], [], []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            bound.append(address)

        def getsockname(self):
            return bound[-1]

    def request(self, *args, **kwargs):
        probes.append(self.url)
        if len(probes) == 1:
            raise requests.ConnectionError("closed")
        return {"workspace": str(workspace)}

    monkeypatch.setattr(runtime.socket, "socket", Socket)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/fake/codex")
    monkeypatch.setattr(runtime, "_check_codex_capabilities", lambda _: None)
    monkeypatch.setattr(runtime.RuntimeClient, "_request", request)
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda args, **kwargs: (
            launched.append((args, kwargs)) or SimpleNamespace(poll=lambda: None)
        ),
    )
    client = runtime.ensure_service(str(workspace))
    assert client.url == previous["url"] and client.token == previous["token"]
    assert bound == [("127.0.0.1", 54321)]
    assert launched[0][1]["env"]["ATLANT_SERVICE_TOKEN"] == previous["token"]
    assert descriptor.read_bytes() == original


@pytest.mark.parametrize("failure", ["occupied", "wrong_workspace", "unauthorized"])
def test_restart_refuses_occupied_or_mismatched_endpoint(
    tmp_path, monkeypatch, failure
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ATLANT_STATE_DIR", str(tmp_path / "state"))
    descriptor = runtime.state_directory(str(workspace)) / "service.json"
    runtime.atomic_json(
        descriptor,
        {"url": "http://127.0.0.1:54321", "token": "test", "workspace": str(workspace)},
    )
    original = descriptor.read_bytes()

    def request(*args, **kwargs):
        if failure == "occupied":
            raise requests.ConnectionError("connection reset")
        if failure == "unauthorized":
            raise RuntimeError("401")
        return {"workspace": "/other/workspace"}

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def setsockopt(self, *args):
            pass

        def bind(self, address):
            raise OSError(errno.EADDRINUSE, "occupied")

    monkeypatch.setattr(runtime.RuntimeClient, "_request", request)
    monkeypatch.setattr(runtime.socket, "socket", Socket)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/fake/codex")
    monkeypatch.setattr(runtime, "_check_codex_capabilities", lambda _: None)
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("must not launch replacement"),
    )
    with pytest.raises(RuntimeError):
        runtime.ensure_service(str(workspace))
    assert descriptor.read_bytes() == original


def test_unrelated_live_pid_is_not_a_worker(tmp_path):
    assert not runtime._worker_alive({"id": "a" * 32, "pid": os.getpid()}, tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = runtime.RunStore(tmp_path / "state/runs")
    state = store.create(str(workspace), "task", "agent")
    store.update(state["id"], status="running", pid=os.getpid())
    runtime.create_service(settings(tmp_path, workspace), tmp_path / "state")
    assert store.get(state["id"])["status"] == "interrupted"


@pytest.mark.skipif(
    not Path("/proc/self/cmdline").exists(), reason="Linux worker identity"
)
def test_surviving_worker_is_preserved_then_death_detected_after_restart(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_dir = tmp_path / "state"
    store = runtime.RunStore(state_dir / "runs")
    state = store.create(str(workspace), "task", "agent")
    package = tmp_path / "fake/atlant_coder"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "runtime.py").write_text(
        "from pathlib import Path\nimport time\nPath('ready').touch()\ntime.sleep(30)\n"
    )
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
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(package.parent)),
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        store.update(state["id"], status="running", pid=process.pid)
        assert runtime._worker_alive(store.get(state["id"]), state_dir)
        assert not runtime._worker_alive(
            {**store.get(state["id"]), "id": "f" * 32}, state_dir
        )
        assert not runtime._worker_alive(
            store.get(state["id"]), tmp_path / "different-state"
        )
        app = runtime.create_service(settings(tmp_path, workspace), state_dir)
        client = TestClient(app)
        assert (
            client.get(f"/runs/{state['id']}", headers=auth()).json()["status"]
            == "running"
        )
        process.terminate()
        process.wait(timeout=5)
        assert (
            client.get(f"/runs/{state['id']}", headers=auth()).json()["status"]
            == "interrupted"
        )
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_recovery_does_not_overwrite_concurrent_terminal_result(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_dir = tmp_path / "state"
    store = runtime.RunStore(state_dir / "runs")
    state = store.create(str(workspace), "task", "agent")

    def just_finished(*args):
        store.update(state["id"], status="completed")
        return False

    monkeypatch.setattr(runtime, "_worker_alive", just_finished)
    runtime.create_service(settings(tmp_path, workspace), state_dir)
    assert store.get(state["id"])["status"] == "completed"


def test_resume_launch_error_is_failed_and_can_retry(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launches = []

    def launch(state):
        launches.append(state)
        if len(launches) == 2:
            raise OSError("cannot spawn")

    app = runtime.create_service(
        settings(tmp_path, workspace), tmp_path / "state", launch=launch
    )
    client = TestClient(app)
    store = app.state.atlant_store
    run_id = client.post(
        "/runs", headers=auth(), json={"workspace": str(workspace), "task": "task"}
    ).json()["id"]
    store.update(run_id, status="interrupted", can_resume=True)
    assert client.post(f"/runs/{run_id}/resume", headers=auth()).status_code == 503
    assert store.get(run_id)["status"] == "failed"
    assert store.get(run_id)["can_resume"] is True
    assert client.post(f"/runs/{run_id}/resume", headers=auth()).status_code == 200


def test_stop_invalidates_approval_and_stale_approval_cannot_replay(tmp_path):
    client, store, workspace, _ = service(tmp_path)
    run_id = client.post(
        "/runs", headers=auth(), json={"workspace": str(workspace), "task": "task"}
    ).json()["id"]
    pending = {"id": "approval-id", "description": "run command"}
    store.update(run_id, status="waiting_approval", pending_approval=pending)
    stopped = client.post(f"/runs/{run_id}/stop", headers=auth()).json()
    assert stopped["pending_approval"] is None
    for status in ("stopping", "failed", "stopped", "running"):
        store.update(run_id, status=status, pending_approval=pending)
        assert (
            client.post(
                f"/runs/{run_id}/approve",
                headers=auth(),
                json={"approval_id": pending["id"], "allow": True},
            ).status_code
            == 409
        )


@pytest.mark.parametrize("unsupported", ["exec", "resume", "timeout", None])
def test_capability_probe_requires_initial_and_resume_schema_support(
    monkeypatch, unsupported
):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if unsupported == "timeout":
            raise subprocess.TimeoutExpired(command, 5)
        which = "resume" if "resume" in command else "exec"
        return SimpleNamespace(
            returncode=0,
            stdout="--output-schema" if which != unsupported else "help",
            stderr="",
        )

    monkeypatch.setattr(runtime.subprocess, "run", run)
    if unsupported:
        with pytest.raises(RuntimeError, match="structured-output|output-schema"):
            runtime._check_codex_capabilities("/fake/codex")
    else:
        runtime._check_codex_capabilities("/fake/codex")
        assert len(calls) == 2
    assert all(
        command[-1] == "--help" and kwargs["timeout"] == 5 for command, kwargs in calls
    )


@pytest.mark.skipif(
    not Path("/proc/self/cmdline").exists(), reason="Linux worker identity"
)
def test_real_service_restart_keeps_worker_endpoint_and_liveness(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ATLANT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PYTHONPATH", str(Path(runtime.__file__).resolve().parents[1]))
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    codex = binary_dir / "codex"
    codex.write_text(
        f"#!{sys.executable}\nimport sys\nassert sys.argv[-1] == '--help'\nprint('--output-schema')\n"
    )
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary_dir) + os.pathsep + os.environ["PATH"])
    spawned = []
    real_popen = runtime.subprocess.Popen

    def record(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", record)
    worker = None
    try:
        first = runtime.ensure_service(str(workspace))
        directory = runtime.state_directory(str(workspace))
        descriptor = (directory / "service.json").read_bytes()
        store = runtime.RunStore(directory / "runs")
        state = store.create(str(workspace), "task", "agent")
        package = tmp_path / "fake/atlant_coder"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "runtime.py").write_text(
            "from pathlib import Path\nimport time\nPath('ready').touch()\ntime.sleep(30)\n"
        )
        worker = real_popen(
            [
                sys.executable,
                "-m",
                "atlant_coder.runtime",
                "worker",
                "--state-dir",
                str(directory),
                "--run-id",
                state["id"],
            ],
            cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH=str(package.parent)),
        )
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        store.update(state["id"], status="running", pid=worker.pid)
        assert first.snapshot(state["id"])["status"] == "running"
        service_process = next(
            process for process in spawned if "serve" in process.args
        )
        service_process.terminate()
        service_process.wait(timeout=5)
        assert worker.poll() is None
        second = runtime.ensure_service(str(workspace))
        assert (second.url, second.token) == (first.url, first.token)
        assert (directory / "service.json").read_bytes() == descriptor
        assert second.snapshot(state["id"])["status"] == "running"
        worker.terminate()
        worker.wait(timeout=5)
        assert second.snapshot(state["id"])["status"] == "interrupted"
    finally:
        for process in [worker, *spawned]:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
