from pathlib import Path

from fastapi.testclient import TestClient

from atlant_coder.runtime import RunStore, create_service
from codex_gateway.config import Settings


def settings(tmp_path: Path, workspace: Path) -> Settings:
    return Settings(
        api_token="secret-token",
        host="127.0.0.1",
        port=8765,
        codex_bin="codex",
        state_dir=tmp_path / "gateway",
        log_file=tmp_path / "gateway.log",
        log_level="INFO",
        allowed_workspace_root=workspace,
        default_workspace=workspace,
        default_sandbox="read-only",
        models_config_path=tmp_path / "models.json",
        available_models=("test-model",),
        default_model="test-model",
        allow_network=False,
        schema_json_limit=10_000,
        codex_timeout_seconds=10,
        codex_job_timeout_seconds=10,
    )


def service(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launched = []
    app = create_service(
        settings(tmp_path, workspace), tmp_path / "runtime", launch=launched.append
    )
    return TestClient(app), app.state.atlant_store, workspace, launched


def auth():
    return {"Authorization": "Bearer secret-token"}


def test_auth_start_events_steer_and_single_writer(tmp_path: Path) -> None:
    client, store, workspace, launched = service(tmp_path)
    assert client.get("/runs").status_code == 401
    assert (
        client.get("/runs", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )

    response = client.post(
        "/runs",
        headers=auth(),
        json={"workspace": str(workspace), "task": "Собери проект", "mode": "agent"},
    )
    assert response.status_code == 200
    run_id = response.json()["id"]
    assert launched[0]["id"] == run_id
    assert (
        client.post(
            "/runs",
            headers=auth(),
            json={"workspace": str(workspace), "task": "Вторая задача"},
        ).status_code
        == 409
    )

    steered = client.post(
        f"/runs/{run_id}/messages", headers=auth(), json={"message": "Сначала тесты"}
    )
    assert steered.status_code == 200
    events = client.get(f"/runs/{run_id}/events", headers=auth()).json()
    assert [(event["seq"], event["kind"]) for event in events] == [
        (1, "task"),
        (2, "user"),
    ]
    assert (
        client.get(f"/runs/{run_id}/events?after=1", headers=auth()).json()[0][
            "message"
        ]
        == "Сначала тесты"
    )
    assert store.get(run_id)["messages"] == [
        {"revision": 1, "message": "Сначала тесты"}
    ]


def test_stop_resume_approval_and_persistence(tmp_path: Path) -> None:
    client, store, workspace, launched = service(tmp_path)
    run_id = client.post(
        "/runs", headers=auth(), json={"workspace": str(workspace), "task": "Задача"}
    ).json()["id"]
    stopped = client.post(f"/runs/{run_id}/stop", headers=auth()).json()
    assert stopped["status"] == "stopping" and stopped["stop_requested"] is True

    store.update(run_id, status="stopped", can_resume=True)
    resumed = client.post(f"/runs/{run_id}/resume", headers=auth())
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "starting"
    assert resumed.json()["stop_requested"] is False
    assert len(launched) == 2

    approval_id = "a" * 32
    store.update(
        run_id,
        status="waiting_approval",
        pending_approval={"id": approval_id, "description": "Запустить команду?"},
    )
    mismatch = client.post(
        f"/runs/{run_id}/approve",
        headers=auth(),
        json={"approval_id": "wrong", "allow": True},
    )
    assert mismatch.status_code == 409
    accepted = client.post(
        f"/runs/{run_id}/approve",
        headers=auth(),
        json={"approval_id": approval_id, "allow": False},
    )
    assert accepted.status_code == 200
    assert accepted.json()["pending_approval"]["allow"] is False
    assert (
        client.post(
            f"/runs/{run_id}/approve",
            headers=auth(),
            json={"approval_id": approval_id, "allow": True},
        ).status_code
        == 409
    )

    restored = RunStore(store.root).get(run_id)
    assert restored["id"] == run_id
    assert restored["pending_approval"]["allow"] is False
    assert [event["seq"] for event in RunStore(store.root).events(run_id)] == [1]


def test_rejects_wrong_workspace_and_invalid_run_id(tmp_path: Path) -> None:
    client, _, _, _ = service(tmp_path)
    outside = tmp_path / "other"
    outside.mkdir()
    assert (
        client.post(
            "/runs", headers=auth(), json={"workspace": str(outside), "task": "Задача"}
        ).status_code
        == 403
    )
    assert client.get("/runs/../../etc/passwd", headers=auth()).status_code == 404
    assert client.get("/runs/not-a-run", headers=auth()).status_code == 404
