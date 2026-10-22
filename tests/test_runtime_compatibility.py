import threading
import time

from fastapi.testclient import TestClient

from atlant_coder import runtime
from tests.test_runtime import auth, settings


def legacy_run(tmp_path, monkeypatch):
    store = runtime.RunStore(tmp_path / "runs")
    run = store.create(str(tmp_path), "original task", "agent")
    run.pop("attempt_id")
    runtime.atomic_json(store.directory(run["id"]) / "state.json", run)
    for key in ("ATLANT_RUN_ID", "ATLANT_RUN_ATTEMPT", "ATLANT_CHECKPOINT_DIR"):
        monkeypatch.setenv(key, "test-fixture")
    return store, run


def test_new_worker_starts_legacy_daemon_state_without_attempt_id(
    tmp_path, monkeypatch
):
    store, run = legacy_run(tmp_path, monkeypatch)
    calls = []

    def execute(home, task, **kwargs):
        current = store.get(run["id"])
        assert len(current["attempt_id"]) == 32
        assert kwargs["control"].attempt_id == current["attempt_id"]
        calls.append((home, task))
        return {"project_verified": False}

    monkeypatch.setattr(runtime, "run_application", execute)
    assert runtime.worker(tmp_path, run["id"]) == 2
    saved = store.get(run["id"])
    assert saved["status"] == "unverified"
    assert saved["error"] is None
    assert calls == [(str(tmp_path), "original task")]


def test_initialization_exception_is_persisted_instead_of_unexplained_interruption(
    tmp_path, monkeypatch
):
    store, run = legacy_run(tmp_path, monkeypatch)

    def fail_run(*args, **kwargs):
        raise ModuleNotFoundError("Required app dependency is missing")

    monkeypatch.setattr(runtime, "run_application", fail_run)
    assert runtime.worker(tmp_path, run["id"]) == 1
    state = store.get(run["id"])
    assert state["status"] == "failed"
    assert state["error"] == "Required app dependency is missing"
    assert state["can_resume"] is True
    assert "Traceback" not in state["error"]


def test_legacy_interruption_gets_specific_reason_from_worker_log(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state_dir = tmp_path / "runtime"
    store = runtime.RunStore(state_dir / "runs")
    run = store.create(str(workspace), "task", "agent")
    store.update(run["id"], status="interrupted")
    (store.directory(run["id"]) / "worker.log").write_text(
        "Traceback (most recent call last):\n  private stack details\nKeyError: 'attempt_id'\n"
    )
    app = runtime.create_service(
        settings(tmp_path, workspace), state_dir, launch=lambda state: None
    )
    snapshot = TestClient(app).get("/runs/" + run["id"], headers=auth()).json()
    assert "Старый сервис" in snapshot["interruption_reason"]
    assert "attempt_id" in snapshot["error"]
    assert "Traceback" not in snapshot["error"]
    assert "private stack" not in snapshot["error"]


def test_runtime_ask_waits_for_steer_and_consumes_answer(tmp_path):
    store = runtime.RunStore(tmp_path / "runs")
    run = store.create(str(tmp_path), "task", "agent")
    control = runtime.RunControl(store, run["id"])
    answers = []
    thread = threading.Thread(
        target=lambda: answers.append(control.ask("Какой формат?"))
    )
    thread.start()
    for _ in range(100):
        if store.get(run["id"])["status"] == "waiting_input":
            break
        time.sleep(0.01)
    assert thread.is_alive()
    store.steer(run["id"], "JSON")
    thread.join(timeout=2)
    assert answers == ["JSON"]
    assert control.applied_revision == 1
    assert control.messages() == []
    assert store.get(run["id"])["status"] == "running"
    assert [(event["kind"], event["message"]) for event in store.events(run["id"])] == [
        ("task", "task"),
        ("question", "Какой формат?"),
        ("user", "JSON"),
    ]


def test_runtime_ask_is_cancel_aware(tmp_path):
    store = runtime.RunStore(tmp_path / "runs")
    run = store.create(str(tmp_path), "task", "agent")
    control = runtime.RunControl(store, run["id"])
    errors = []

    def ask():
        try:
            control.ask("Продолжить?")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=ask)
    thread.start()
    for _ in range(100):
        if store.get(run["id"])["status"] == "waiting_input":
            break
        time.sleep(0.01)
    store.update(run["id"], stop_requested=True)
    thread.join(timeout=2)
    assert len(errors) == 1 and isinstance(errors[0], InterruptedError)


def test_worker_retries_when_steer_wins_final_commit(tmp_path, monkeypatch):
    store, run = legacy_run(tmp_path, monkeypatch)
    calls = []

    def run_app(home, task, *, control, **kwargs):
        calls.append((home, task))
        if len(calls) == 1:
            store.steer(run["id"], "Add the new requirement")
        else:
            control.applied_revision = store.get(run["id"])["revision"]
        return {"project_verified": True}

    monkeypatch.setattr(runtime, "run_application", run_app)
    assert runtime.worker(tmp_path, run["id"]) == 0
    assert len(calls) == 2
    assert store.get(run["id"])["status"] == "completed"
    assert any(event["kind"] == "discarded" for event in store.events(run["id"]))


def test_plan_snapshot_uses_actual_paths_and_preserves_real_summary_only(tmp_path):
    store = runtime.RunStore(tmp_path / "runs")
    run = store.create(str(tmp_path), "task", "agent")
    control = runtime.RunControl(store, run["id"])
    state = {
        "files": {"/a.py": "Core behavior", "/b.py": "Tests", "/c.md": "Documentation"},
        "completed_files": ["/a.py"],
        "active_agents": [{"file": "/b.py"}],
    }
    control.publish(state, "plan_ready")
    saved = store.get(run["id"])
    assert saved["plan"] == [
        {"path": "/a.py", "description": "Core behavior", "status": "completed"},
        {"path": "/b.py", "description": "Tests", "status": "running"},
        {"path": "/c.md", "description": "Documentation", "status": "pending"},
    ]
    assert "result_summary" not in saved
    state["result_summary"] = "Actual reviewed changes"
    control.publish(state, "unverified_no_project_checks")
    assert store.get(run["id"])["result_summary"] == "Actual reviewed changes"
    assert store.get(run["id"])["checks"] == []
