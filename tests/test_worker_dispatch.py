"""Worker commands must execute through the SDK, including corrections and tools."""

from unittest.mock import Mock

import pytest

from AurethosSDK.client.router import Router
from atlant_coder import config
from atlant_coder.handlers import parallel_batch
from atlant_coder.utils.project import file_context
from tests.handlers.test_parallel_batch import FakeAPI, entry, made, state_for


def record_dispatch(monkeypatch):
    seen = []
    original = Router.check_update

    def observe(self, update, *args, **kwargs):
        seen.append(update.command)
        return original(self, update, *args, **kwargs)

    monkeypatch.setattr(Router, "check_update", observe)
    return seen


def test_tools_and_draft_patch_use_sdk_handlers(monkeypatch):
    seen = record_dispatch(monkeypatch)
    state = state_for({"/audit.md": entry()})
    draft = made("/audit.md", "Total: 2")
    draft["summary"]["issues"] = ["Wrong count"]
    api = FakeAPI(
        [
            {"command": "LIST", "path": ""},
            draft,
            {
                "command": "PATCH",
                "file": "/audit.md",
                "edits": [{"old": "2", "new": "3"}],
                "summary": {"behavior": "Correct total", "exports": [], "issues": []},
            },
        ]
    )
    broker = Mock()
    broker.request.return_value = {"paths": ["a", "b", "c"]}
    monkeypatch.setattr(parallel_batch, "_clone_api_router", lambda _: api)
    context = {**file_context(state, "/audit.md"), "agent_mode": True, "broker": broker}
    result = parallel_batch._generate_one("/audit.md", context, api)
    assert result["status"] == "ok", result
    assert result["code"] == "Total: 3"
    assert result["draft_reviews"] == 1
    assert seen == [
        "MAKE",
        "REQUEST",
        "LIST",
        "REQUEST",
        "MADE",
        "REVIEW",
        "PATCH",
        "MADE",
        "VERIFY",
    ]
    broker.request.assert_called_once_with(
        "/audit.md", broker.tools.list, {"command": "LIST", "path": ""}
    )


def test_think_and_continuation_are_sdk_updates(monkeypatch):
    seen = record_dispatch(monkeypatch)
    monkeypatch.setattr(config, "min_plan_steps_to_skip_think", 0)
    monkeypatch.setattr(config, "max_think_iters_by_priority", {"high": 1})
    state = state_for({"/value.txt": entry(high=True)})
    api = FakeAPI(
        [
            {"command": "THINK", "content": "Which value?"},
            {"command": "NEEDRESUME", "file": "/value.txt", "code": "value="},
            made("/value.txt", "1"),
        ]
    )
    thinker = FakeAPI([{"final": "Use one"}])
    monkeypatch.setattr(parallel_batch, "_clone_api_router", lambda model: model)
    result = parallel_batch._generate_one(
        "/value.txt", file_context(state, "/value.txt"), api, thinker
    )
    assert result["status"] == "ok", result
    assert result["code"] == "value=1"
    assert seen == [
        "MAKE",
        "REQUEST",
        "THINK",
        "THINK",
        "ANSW",
        "REQUEST",
        "NEEDRESUME",
        "REQUEST",
        "MADE",
        "VERIFY",
    ]
    assert "THINK" not in api.options[1]["expects"]
    assert len(result["requests"]) == 4


@pytest.mark.parametrize(
    "command", ["DELETE", "VERIFY", "REVIEW", "MAKE", "READ", "OK", "THINK"]
)
def test_unknown_or_unavailable_command_cannot_invoke_handler(monkeypatch, command):
    seen = record_dispatch(monkeypatch)
    state = state_for({"/value.txt": entry()})
    api = FakeAPI([{"command": command}])
    monkeypatch.setattr(parallel_batch, "_clone_api_router", lambda _: api)
    result = parallel_batch._generate_one(
        "/value.txt", file_context(state, "/value.txt"), api
    )
    assert result["status"] == "error"
    assert "unexpected command" in result["reason"]
    assert seen == ["MAKE", "REQUEST"]
    assert "code" not in result


def test_tool_limit_and_cancellation_survive_dispatch(monkeypatch):
    state = state_for({"/value.txt": entry()})
    api = FakeAPI([{"command": "LIST", "path": ""}] * 21)
    broker = Mock()
    broker.request.return_value = {}
    monkeypatch.setattr(parallel_batch, "_clone_api_router", lambda _: api)
    context = {
        **file_context(state, "/value.txt"),
        "agent_mode": True,
        "broker": broker,
    }
    result = parallel_batch._generate_one("/value.txt", context, api)
    assert result["status"] == "error"
    assert "tool budget" in result["reason"]
    assert broker.request.call_count == 20
    api = FakeAPI([{"command": "LIST", "path": ""}])
    broker.request.side_effect = InterruptedError("Stopped by user")
    with pytest.raises(InterruptedError, match="Stopped by user"):
        parallel_batch._generate_one("/value.txt", context, api)
