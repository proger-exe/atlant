"""Planning uses SDK command dispatch with real project tools and fake models."""

import json

import pytest

from AurethosSDK.client.router import Router
from AurethosSDK.client.dispatcher import Dispatcher
from atlant_coder.handlers.arch import arch_router
from atlant_coder.handlers.planning import begin_request
from atlant_coder.macroses.workspace import WorkspaceTools
from atlant_coder.utils.checkpoint import CheckpointExitError


class API:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.request_metrics = []

    def send_api_request(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return next(self.responses)


def request_with_tools(api, prompt, commands, home, control=None):
    run = dict(home=home, state={"agent_mode": True}, control=control)
    dispatcher = Dispatcher("")
    router = Router(api, run=run)
    router.sync(arch_router.handlers)
    dispatcher.routers["ARCH"] = router
    pending = begin_request(run, "ARCH", prompt, commands, kind="revision")
    terminal = dispatcher.check_update(pending)
    return terminal.parameters["candidate"]


def test_planning_sdk_dispatch_reads_file_recovers_error_and_finishes(
    tmp_path, monkeypatch
):
    (tmp_path / "README.md").write_text("actual project evidence")
    final = {
        "command": "MADE",
        "conventions": "audit facts",
        "files": {
            "/audit.md": {
                "desc": "audit",
                "priority": "low",
                "depends_on": [],
                "parallel_safe": False,
            }
        },
    }
    api = API(
        [
            {"command": "READ", "path": "../outside"},
            {"command": "READ", "path": "README.md"},
            final,
        ]
    )
    dispatched = []
    original = Router.check_update

    def observe(self, update, *args, **kwargs):
        dispatched.append(update.command)
        return original(self, update, *args, **kwargs)

    monkeypatch.setattr(Router, "check_update", observe)
    assert request_with_tools(api, "Plan\n{}", ["MADE"], str(tmp_path)) == final
    assert dispatched == ["REQUEST", "READ", "REQUEST", "READ", "REQUEST", "MADE"]
    assert "error" in json.loads(api.calls[1][0].split("\n", 1)[1])
    assert "actual project evidence" in api.calls[2][0]
    assert all(
        call[1]["response_schema"] == api.calls[0][1]["response_schema"]
        for call in api.calls
    )


def test_planning_tool_budget_still_bounds_model_requests(tmp_path):
    api = API([{"command": "LIST", "path": ""}] * 21)
    with pytest.raises(CheckpointExitError, match="planning tool limit"):
        request_with_tools(api, "Plan", ["MADE"], str(tmp_path))
    assert len(api.calls) == 20


def test_planning_tool_interruption_is_not_reported_as_recoverable_error(
    tmp_path, monkeypatch
):
    api = API([{"command": "LIST", "path": ""}])

    def cancelled(*args):
        raise InterruptedError("stop requested")

    monkeypatch.setattr(WorkspaceTools, "list", cancelled)
    with pytest.raises(InterruptedError, match="stop requested"):
        request_with_tools(api, "Plan", ["MADE"], str(tmp_path))
    assert len(api.calls) == 1


def test_planning_stopped_model_reply_is_never_dispatched(tmp_path):
    class Control:
        checks = 0

        def check_stopped(self):
            self.checks += 1
            if self.checks == 2:
                raise InterruptedError("stop requested")

    api = API([{"command": "MADE", "files": {}}])
    with pytest.raises(InterruptedError):
        request_with_tools(api, "Plan", ["MADE"], str(tmp_path), Control())
    assert len(api.calls) == 1
