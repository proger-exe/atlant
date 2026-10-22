from unittest.mock import patch

import pytest

from AurethosSDK.models.api_router import APIRouter
from AurethosSDK.models.base import BaseModel
from AurethosSDK.models.codex_gateway import (
    CodexGatewayClient,
    CodexGatewaySessionLostError,
    CodexGatewayRequestError,
)
from AurethosSDK.types.base_model import Model
from tests.handlers.test_core import review_project
from tests.handlers.test_parallel_batch import FakeAPI, entry, state_for
from atlant_coder.utils.project import file_context, make_prompt


def client():
    model = BaseModel(Model(name="CODER", system_instruction="test"))
    model.generation_read_only = True
    instance = CodexGatewayClient(model)
    instance.token = "test-only"
    instance._sessions_by_model[instance.use_model] = {
        "session_id": "old-session",
        "thread_id": "old-thread",
    }
    instance._sync_active_session(instance.use_model)
    return instance


def test_reset_reaches_gateway_and_next_prompt_contains_complete_context():
    gateway = client()
    gateway.history.add_entry("user", "old private conversation")
    api = APIRouter(BaseModel(Model(name="CODER", system_instruction="test")))
    api.initialized_classes = {"codex_gateway": gateway}
    api.model_history.add_entry("user", "old local task")
    prompt = make_prompt(file_context(state_for({"/a.py": entry()}), "/a.py"))
    with patch.object(
        gateway,
        "_request_json",
        side_effect=[
            (200, {}),
            (200, {"reply": '{"command":"OK"}', "thread_id": "new-thread"}),
        ],
    ) as request:
        api.reset_context()
        assert not api.model_history.history
        gateway.chat_completion(prompt)
    assert request.call_args_list[0].args[:2] == ("POST", "/sessions/old-session/reset")
    payload = request.call_args_list[1].args[2]
    assert payload["session_id"] == "old-session"
    assert "source_task" in payload["message"] and "small task" in payload["message"]
    assert "old private conversation" not in str(payload)
    assert gateway.thread_id == "new-thread"
    assert gateway.sandbox == "read-only"


def test_failed_remote_reset_is_not_treated_as_success():
    gateway = client()
    with patch.object(gateway, "_request_json", return_value=(503, {})):
        with pytest.raises(RuntimeError, match="reset failed"):
            gateway.reset_context()


def test_missing_session_does_not_replay_partial_continue():
    gateway = client()
    with patch.object(
        gateway, "_request_json", return_value=(404, {"detail": "session not found"})
    ) as request:
        with pytest.raises(CodexGatewaySessionLostError):
            gateway.chat_completion("CONTINUE")
    assert request.call_count == 1
    assert gateway.session_id is None


def test_lost_cli_thread_stops_without_partial_replay():
    gateway = client()
    with patch.object(
        gateway,
        "_request_json",
        return_value=(
            409,
            {
                "code": "session_context_lost",
                "message": "restart full task",
                "retryable": False,
            },
        ),
    ) as request:
        with pytest.raises(CodexGatewaySessionLostError):
            gateway.chat_completion("CONTINUE")
    assert request.call_count == 1
    assert gateway.session_id is None


def test_nonretryable_gateway_error_retains_cause_without_backoff():
    gateway = client()
    api = APIRouter(gateway.model)
    api.initialized_classes = {"codex_gateway": gateway}
    api.aviable_apis = ["codex_gateway"]
    with patch.object(
        gateway,
        "_request_json",
        return_value=(
            413,
            {
                "detail": "context too large",
                "retryable": False,
            },
        ),
    ) as request:
        with pytest.raises(CodexGatewayRequestError, match="413.*context too large"):
            api.send_api_request("MAKE full task")
    assert request.call_count == 1
    assert not any(
        entry["role"] == "user" for entry in gateway.history.get_all_entries()
    )


def test_provider_initialization_does_not_mutate_global_constructor():
    class Provider:
        def __init__(self, **kwargs):
            pass

    original = Provider.__init__
    api = APIRouter(BaseModel(Model(name="CODER", system_instruction="test")))
    api._init_api_class(Provider)
    api._init_api_class(Provider)
    assert Provider.__init__ is original


def test_core_receives_original_task_verified_facts_and_bounded_decision():
    state = state_for({"/a.py": entry()})
    state["source_task"] = "Original explicit constraint"
    state["completed_files"] = ["/a.py"]
    state["signatures_/a.py"] = ["VALUE = ..."]
    state["file_summaries"] = {"/a.py": {"behavior": "worker claim"}}
    api = FakeAPI(
        [
            {"command": "REPLAN", "files": ["/unplanned.py"], "reason": "bad"},
            {"command": "APPROVE", "reason": "consistent"},
        ]
    )
    assert review_project(api, state, "FINALIZE")["command"] == "APPROVE"
    assert len(api.calls) == 2
    assert "Original explicit constraint" in api.calls[0]
    assert "VALUE = ..." in api.calls[0]
    assert "worker claim" in api.calls[0]


def test_resume_does_not_shrink_full_plan_or_skip_final_gate(tmp_path, monkeypatch):
    from atlant_coder.app import run
    from atlant_coder.utils.checkpoint import CheckpointExitError, load, save
    from atlant_coder.utils.project import digest

    path = tmp_path / "report.md"
    path.write_text("Existing reviewed document")
    state = state_for({"/report.md": entry()})
    state.update(
        completed_files=["/report.md"],
        contract_approved=True,
        project_verified=False,
        file_hashes={"/report.md": digest(path.read_text())},
    )
    save(state, home_dir=str(tmp_path), reason="project_tests_failed")
    core = FakeAPI([{"command": "STOP", "reason": "Final review still required"}])
    monkeypatch.setattr(
        "atlant_coder.app.role_api",
        lambda role, home: core if role == "CORE" else FakeAPI(),
    )
    with pytest.raises(CheckpointExitError, match="Final review still required"):
        run(str(tmp_path), state["source_task"])
    assert len(core.calls) == 1 and core.calls[0].startswith("FINALIZE")
    saved = load(str(tmp_path))["memory"]
    assert saved["files"] == state["files"]
    assert saved["completed_files"] == ["/report.md"]
    assert saved["project_verified"] is False
