"""Real blocking subprocesses and local HTTP, never a live model."""

import json
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from atlant_coder.runtime import RunControl, RunStore, create_service
from tests.test_runtime import auth, settings


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def process_running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[
            0
        ] not in {"Z", "X"}
    except FileNotFoundError:
        return False


def test_stop_kills_owned_native_call_and_rejects_late_attempt_after_resume(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cli = tmp_path / "fake-codex"
    marker = tmp_path / "model-pids.json"
    cli.write_text(
        "#!"
        + sys.executable
        + "\n"
        + f"""
import json,os,pathlib,subprocess,sys,time
prompt=sys.stdin.read()
if prompt == 'block':
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
 pathlib.Path({str(marker)!r}).write_text(json.dumps([os.getpid(),child.pid]))
 time.sleep(60)
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text('done')
print(json.dumps({{'type':'thread.started','thread_id':'fake-thread'}}))
"""
    )
    cli.chmod(0o700)
    config = settings(tmp_path, workspace)
    from dataclasses import replace

    config = replace(config, codex_bin=str(cli), codex_timeout_seconds=60)
    monkeypatch.setattr(
        "codex_gateway.codex_runner.build_subprocess_env",
        lambda *a, **kw: os.environ.copy(),
    )
    state_dir = tmp_path / "runtime"
    app = create_service(config, state_dir, launch=lambda state: None)
    client = TestClient(app)
    run = client.post(
        "/runs", headers=auth(), json={"workspace": str(workspace), "task": "task"}
    ).json()
    owner = {"owner_run_id": run["id"], "owner_attempt_id": run["attempt_id"]}
    sid = client.post("/sessions", headers=auth(), json=owner).json()["session_id"]
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            client.post,
            "/chat",
            headers=auth(),
            json={"session_id": sid, "message": "block"},
        )
        wait_for(marker.exists)
        started = time.monotonic()
        response = client.post(f"/runs/{run['id']}/stop", headers=auth())
        assert response.status_code == 200
        cancelled = future.result(timeout=3)
        assert time.monotonic() - started < 2
    assert cancelled.status_code == 409
    assert cancelled.json()["code"] == "run_cancelled"
    for pid in json.loads(marker.read_text()):
        wait_for(lambda: not process_running(pid))
    assert not app.state.atlant_store.get(run["id"])["changes"]
    # Persisted cancellation works in a reconstructed service, not only in RAM.
    app.state.atlant_store.update(run["id"], status="stopped", pid=None)
    restored = TestClient(create_service(config, state_dir, launch=lambda state: None))
    assert (
        restored.post(
            "/chat", headers=auth(), json={"session_id": sid, "message": "late"}
        ).json()["code"]
        == "run_cancelled"
    )
    resumed = restored.post(f"/runs/{run['id']}/resume", headers=auth()).json()
    assert resumed["attempt_id"] != run["attempt_id"]
    assert (
        restored.post(
            "/chat", headers=auth(), json={"session_id": sid, "message": "late"}
        ).json()["code"]
        == "run_cancelled"
    )
    assert (
        restored.post("/sessions", headers=auth(), json=owner).json()["code"]
        == "run_cancelled"
    )
    owner["owner_attempt_id"] = resumed["attempt_id"]
    fresh = restored.post("/sessions", headers=auth(), json=owner).json()["session_id"]
    assert (
        restored.post(
            "/chat", headers=auth(), json={"session_id": fresh, "message": "finish"}
        ).json()["reply"]
        == "done"
    )


def test_stop_kills_project_check_process_group(tmp_path, monkeypatch):
    from atlant_coder.handlers import coder

    store = RunStore(tmp_path / "runs")
    run = store.create(str(tmp_path), "task", "agent")
    control = RunControl(store, run["id"])
    monkeypatch.setattr(coder, "_control", control)
    marker = tmp_path / "check-pids.json"
    code = (
        "import json,os,pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps([os.getpid(),child.pid]));time.sleep(60)"
    )
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            coder._command, [sys.executable, "-c", code], str(tmp_path), 60
        )
        wait_for(marker.exists)
        started = time.monotonic()
        store.update(run["id"], stop_requested=True, status="stopping")
        with pytest.raises(InterruptedError):
            future.result(timeout=3)
        assert time.monotonic() - started < 2
    for pid in json.loads(marker.read_text()):
        wait_for(lambda: not process_running(pid))


def test_stopped_model_draft_never_publishes_and_checkpoint_resumes(
    tmp_path, monkeypatch
):
    from tests.test_app import run_repo, patch_reply
    from atlant_coder.utils.checkpoint import load

    monkeypatch.delenv("ATLANT_CHECKPOINT_DIR", raising=False)
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n")
    store = RunStore(tmp_path / "run-state")
    run = store.create(
        str(tmp_path), "Update VALUE and preserve unrelated work", "agent"
    )
    control = RunControl(store, run["id"])
    entered = threading.Event()

    def blocking_model(_):
        entered.set()
        wait_for(control.stopped)
        # A response arriving concurrently with stop must never publish.
        return patch_reply("VALUE = 1", "VALUE = 99")

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(run_repo, tmp_path, [blocking_model], control=control)
        assert entered.wait(3)
        started = time.monotonic()
        store.update(run["id"], stop_requested=True, status="stopping")
        with pytest.raises(InterruptedError):
            future.result(timeout=3)
        assert time.monotonic() - started < 2
    assert target.read_text() == "VALUE = 1\n"
    assert store.get(run["id"])["changes"] == []
    assert load(str(tmp_path)) is not None
    store.update(run["id"], stop_requested=False, status="running", attempt_id="a" * 32)
    run_repo(
        tmp_path,
        [patch_reply("VALUE = 1", "VALUE = 2")],
        control=RunControl(store, run["id"]),
    )
    assert target.read_text() == "VALUE = 2\n"


def test_stop_racing_final_write_is_rechecked_under_publication_lock(
    tmp_path, monkeypatch
):
    from atlant_coder.handlers import coder

    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n")
    store = RunStore(tmp_path / "runs")
    run = store.create(str(tmp_path), "task", "agent")
    control = RunControl(store, run["id"])
    monkeypatch.setattr(coder, "_control", control)
    assert not control.pending()
    read_bytes = Path.read_bytes

    def stop_between_initial_check_and_write(path):
        data = read_bytes(path)
        if path == target:
            store.update(run["id"], stop_requested=True, status="stopping")
        return data

    monkeypatch.setattr(Path, "read_bytes", stop_between_initial_check_and_write)
    with pytest.raises(InterruptedError):
        coder._accept(
            {"contract_revision": 1},
            str(tmp_path),
            {"path": "/app.py", "revision": 1, "status": "ok", "code": "VALUE = 99\n"},
        )
    assert target.read_text() == "VALUE = 1\n"
