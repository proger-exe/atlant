"""Coordinator regressions, with real publication/checkpoints and fake LLMs."""

import json
import sys
import threading
import time
from unittest.mock import patch
from contextlib import contextmanager

import pytest

from atlant_coder.handlers.arch import _validate_made
from atlant_coder.handlers.coder import _accept, _run_project_unittest
from tests.handlers.support import run_generation
from tests.handlers.test_parallel_batch import FakeAPI, entry, state_for
from atlant_coder.utils import memory
from atlant_coder.utils.checkpoint import CheckpointExitError
from atlant_coder.utils.project import digest, target_path


def test_command_timeout_stops_descendants(tmp_path):
    from atlant_coder.handlers.coder import _command

    marker = tmp_path / "child-finished"
    child = f"import time,pathlib;time.sleep(.8);pathlib.Path({str(marker)!r}).touch()"
    parent = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child!r}]);time.sleep(10)"
    started = time.monotonic()
    assert "timed out" in _command(
        [sys.executable, "-c", parent], tmp_path, timeout=0.1
    )
    assert time.monotonic() - started < 1
    time.sleep(1)
    assert not marker.exists(), "timed-out verification left its child running"


def success(path, context, code="VALUE = 1\n"):
    return {
        "path": path,
        "revision": context["revision"],
        "status": "ok",
        "code": code,
        "signatures": ["VALUE = ..."],
        "summary": {"behavior": "value", "exports": ["VALUE"], "issues": []},
        "requests": [],
        "duration_s": 0,
    }


@contextmanager
def review_responses(reviewer):
    from atlant_coder.handlers.core import before_request as original

    def prepare(run):
        pending = original(run)
        if pending["kind"] == "review":
            state = run.get("review_state", run["state"])
            api = run["dispatcher"].routers["CORE"].model
            args = (api, state, pending["operation"])
            if pending["operation"] in {"RESOLVE", "STEER"}:
                args += (pending["payload"]["issues"],)
            result = dict(reviewer(*args))
            if pending["operation"] == "REVIEW_PLAN" and result["command"] != "STOP":
                result.setdefault(
                    "coverage",
                    [
                        dict(
                            section=section,
                            status="covered",
                            files=list(state["files"]),
                            evidence="Fixture checks the exact planned interfaces",
                        )
                        for section in pending["sections"]
                    ],
                )
            if (
                pending["operation"] in {"RESOLVE", "STEER"}
                and result["command"] == "APPROVE"
            ):
                result.setdefault("files", list(state["files"]))
            api.responses = iter([result])
        return pending

    with patch("atlant_coder.handlers.core.before_request", side_effect=prepare):
        yield


def _revise(state, home, core, architect, decision):
    from AurethosSDK.client.dispatcher import Dispatcher
    from AurethosSDK.client.router import Router
    from AurethosSDK.types.update import Update
    from atlant_coder.handlers.core import manager
    from atlant_coder.handlers.arch import arch_router
    from atlant_coder.handlers.coder import coder_router, close_generation

    run = dict(state=state, home=home, control=None, result=None)
    dispatcher = Dispatcher("")
    run["dispatcher"] = dispatcher
    for name, api, commands in (
        ("CORE", core, manager),
        ("ARCH", architect, arch_router),
        ("CODER", FakeAPI(), coder_router),
    ):
        router = Router(api, run=run)
        router.sync(commands.handlers)
        dispatcher.routers[name] = router
    try:
        dispatcher.check_update(
            Update(
                to_model="CODER",
                from_model="CORE",
                command="REPLAN_REQUIRED",
                parameters={
                    "decision": {"command": "REPLAN", **decision},
                    "review_context": {"phase": "initial"},
                },
            )
        )
    finally:
        close_generation(run)


def run(home, worker, *, workers=2, reviewer=None, architect=None):
    with (
        patch("atlant_coder.handlers.coder._generate_one", side_effect=worker),
        review_responses(
            reviewer or (lambda *a, **k: {"command": "APPROVE", "reason": "consistent"})
        ),
    ):
        return run_generation(
            FakeAPI(),
            str(home),
            workers=workers,
            core=FakeAPI(),
            architect=architect or FakeAPI(),
            thinker=FakeAPI(),
        )


def test_singleton_unlocks_parallel_children_without_waiting_for_slow_sibling(tmp_path):
    files = {
        "/base.py": entry(safe=False),
        "/fast.py": entry(["/base.py"]),
        "/slow.py": entry(["/base.py"]),
        "/child.py": entry(["/fast.py"]),
    }
    memory.restore(state_for(files))
    child_started = threading.Event()
    slow_started = threading.Event()
    order = []

    def worker(path, context, *args):
        order.append(path)
        if path == "/fast.py":
            assert slow_started.wait(3)
        if path == "/slow.py":
            slow_started.set()
            assert child_started.wait(3), "a layer barrier prevented child from running"
        if path == "/child.py":
            assert (tmp_path / "fast.py").is_file()
            child_started.set()
        return success(path, context)

    result = run(tmp_path, worker)
    assert order[0] == "/base.py"
    assert not result[
        "project_verified"
    ]  # No independent tests in this scheduler fixture.
    assert result["validation_results"][0]["status"] == "UNKNOWN"
    assert set(result["completed_files"]) == set(files)
    assert (tmp_path / ".atlant_checkpoint.json").exists()


def test_one_worker_uses_same_execution_path(tmp_path):
    memory.restore(state_for({"/a.py": entry(), "/b.py": entry(["/a.py"])}))
    calls = []

    def worker(path, context, *args):
        calls.append(path)
        return success(path, context)

    assert not run(tmp_path, worker, workers=1)[
        "project_verified"
    ]  # Fixture has no project tests.
    assert calls == ["/a.py", "/b.py"]


def test_incomplete_plan_audit_cannot_start_workers(tmp_path):
    memory.restore(state_for({"/a.py": entry()}))
    core = FakeAPI([{"command": "APPROVE", "reason": "looks good"}] * 2)
    with patch("atlant_coder.handlers.coder._generate_one") as worker:
        with pytest.raises(ValueError, match="complete coverage"):
            run_generation(
                FakeAPI(),
                str(tmp_path),
                core=core,
                architect=FakeAPI(),
                thinker=FakeAPI(),
            )
        worker.assert_not_called()
    checkpoint = json.loads((tmp_path / ".atlant_checkpoint.json").read_text())[
        "memory"
    ]
    assert checkpoint["completed_files"] == []
    assert checkpoint["project_verified"] is False


def test_arch_contract_rejects_missing_safety_and_cycles():
    state = state_for({"/a.py": entry(high=True)})
    assert not _validate_made(state["architecture"])
    del state["architecture"]["files"]["/a.py"]["parallel_safe"]
    assert any("parallel_safe" in e for e in _validate_made(state["architecture"]))
    state = state_for({"/a.py": entry(["/b.py"]), "/b.py": entry(["/a.py"])})
    assert any("cycle" in e for e in _validate_made(state["architecture"]))


def test_arch_rejects_test_only_but_accepts_web_project():
    assert any(
        "no production" in e
        for e in _validate_made(
            state_for({"/tests/test_a.py": entry()})["architecture"]
        )
    )
    assert not _validate_made(
        state_for({"/frontend/App.tsx": entry(), "/tests/app.test.tsx": entry()})[
            "architecture"
        ]
    )


def test_failed_dependency_never_unlocks_children_and_repair_limit_survives_resume(
    tmp_path,
):
    memory.restore(state_for({"/base.py": entry(), "/child.py": entry(["/base.py"])}))
    calls = []

    def worker(path, context, *args):
        calls.append(path)
        return {
            "path": path,
            "revision": context["revision"],
            "status": "error",
            "reason": "broken code",
        }

    with pytest.raises(CheckpointExitError, match="file_repair_limit"):
        run(tmp_path, worker)
    assert calls == ["/base.py"] * 3
    checkpoint = json.loads((tmp_path / ".atlant_checkpoint.json").read_text())
    assert checkpoint["memory"]["attempts_/base.py"] == 3
    assert not checkpoint["memory"]["project_verified"]
    with pytest.raises(CheckpointExitError, match="file_repair_limit"):
        run(tmp_path, worker)
    assert len(calls) == 3


def test_replan_is_reachable_after_second_failure(tmp_path):
    initial = state_for({"/base.py": entry()})
    memory.restore(initial)
    calls = []

    def worker(path, context, *args):
        calls.append(context["revision"])
        if len(calls) < 3:
            return {
                "path": path,
                "revision": context["revision"],
                "status": "error",
                "reason": "bad implementation",
            }
        return success(path, context)

    def reviewer(api, state, operation, issues=None):
        if operation == "RESOLVE":
            return {
                "command": "REPLAN",
                "files": ["/base.py"],
                "reason": "fix the plan",
            }
        return {"command": "APPROVE", "reason": "ok"}

    architect = FakeAPI([initial["architecture"]])
    result = run(tmp_path, worker, reviewer=reviewer, architect=architect)
    assert calls == [1, 1, 2]
    assert result["attempts_/base.py"] == 2
    assert len(architect.calls) == 1


def test_final_test_failure_preserves_all_written_checkpoint(tmp_path):
    memory.restore(
        state_for({"/app.py": entry(), "/tests/test_failure.py": entry(["/app.py"])})
    )

    def worker(path, context, *args):
        code = "VALUE = 1\n"
        if path.startswith("/tests/"):
            code = "import unittest\nclass Failure(unittest.TestCase):\n    def test_failure(self):\n        self.fail('gate reached')\n"
        return success(path, context, code)

    with pytest.raises(CheckpointExitError, match="project_tests_failed"):
        run(tmp_path, worker)
    state = json.loads((tmp_path / ".atlant_checkpoint.json").read_text())["memory"]
    assert len(state["completed_files"]) == 2
    assert "gate reached" in state["project_test_failure"]
    calls = []
    with pytest.raises(CheckpointExitError, match="project_tests_failed"):
        run(tmp_path, lambda *args: calls.append(args))
    assert calls == []


def test_modified_file_invalidates_dependents_on_resume(tmp_path):
    state = state_for(
        {"/base.py": entry(), "/child.py": entry(["/base.py"]), "/other.py": entry()}
    )
    state["completed_files"] = list(state["files"])
    state["file_hashes"] = {p: digest("VALUE = 1\n") for p in state["files"]}
    for path in state["files"]:
        target_path(str(tmp_path), path).write_text("VALUE = 1\n")
    (tmp_path / "base.py").write_text("VALUE = 2\n")
    memory.restore(state)
    calls = []

    def worker(path, context, *args):
        calls.append(path)
        return success(path, context)

    run(tmp_path, worker)
    assert calls == ["/base.py", "/child.py"]


def test_stale_revision_and_path_escape_are_rejected(tmp_path):
    state = state_for({"/a.py": entry()})
    with pytest.raises(ValueError, match="stale"):
        _accept(state, str(tmp_path), {"path": "/a.py", "revision": 0, "status": "ok"})
    for path in ("/../outside.py", "//outside.py", "/dir/../../outside.py", "/a\\b.py"):
        with pytest.raises(ValueError):
            target_path(str(tmp_path), path)
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        target_path(str(tmp_path), "/escape/outside.py")


def test_final_gate_uses_pytest_for_pytest_project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    (tmp_path / "tests/test_plain.py").write_text(
        "def test_plain():\n    assert True\n"
    )
    assert _run_project_unittest(str(tmp_path)) == ""


def test_revised_contract_regenerates_consumers_but_keeps_independent_result(tmp_path):
    import copy

    initial = state_for(
        {"/base.py": entry(), "/child.py": entry(["/base.py"]), "/other.py": entry()}
    )
    revised = copy.deepcopy(initial["architecture"])
    revised["files"]["/base.py"]["desc"] = "Define VALUE as 2"
    memory.restore(initial)
    calls = []

    def worker(path, context, *args):
        calls.append((path, context["revision"]))
        return success(path, context, f"VALUE = {context['revision']}\n")

    def reviewer(api, state, operation, issues=None):
        if operation == "FINALIZE" and state["contract_revision"] == 1:
            return {"command": "REPLAN", "files": ["/base.py"], "reason": "Use VALUE=2"}
        return {"command": "APPROVE", "reason": "ok"}

    out_of_scope = copy.deepcopy(revised)
    out_of_scope["files"]["/other.py"]["desc"] = "Unapproved change"
    architect = FakeAPI([out_of_scope, revised])
    result = run(tmp_path, worker, reviewer=reviewer, architect=architect)
    assert result["contract_revision"] == 2
    assert calls.count(("/other.py", 1)) == 1
    assert ("/other.py", 2) not in calls
    assert ("/base.py", 2) in calls and ("/child.py", 2) in calls
    assert (tmp_path / "base.py").read_text() == "VALUE = 2\n"
    assert (tmp_path / "other.py").read_text() == "VALUE = 1\n"
    assert result["rework_requests"] == [
        {"files": ["/base.py", "/child.py"], "reason": "Use VALUE=2"}
    ]


def test_failed_publication_restores_previous_file(tmp_path):
    state = state_for({"/a.py": entry()})
    state.update(file_summaries={}, file_hashes={})
    (tmp_path / "a.py").write_text("VALUE = 7\n")
    with patch(
        "atlant_coder.handlers.coder.smoke_check", return_value=["invalid import"]
    ):
        _accept(state, str(tmp_path), success("/a.py", {"revision": 1}))
    assert (tmp_path / "a.py").read_text() == "VALUE = 7\n"
    assert state["completed_files"] == []


def test_replan_correction_cannot_publish_unapproved_changes(tmp_path):
    import copy

    state = state_for({"/base.py": entry(), "/other.py": entry()})
    original = copy.deepcopy(state["architecture"])
    invalid = copy.deepcopy(original)
    invalid["files"]["/other.py"]["desc"] = "Unapproved change"
    architect = FakeAPI([invalid, invalid])
    with pytest.raises(CheckpointExitError, match="changed unrelated files"):
        _revise(
            state,
            str(tmp_path),
            FakeAPI(),
            architect,
            {"files": ["/base.py"], "reason": "Repair base only"},
        )
    assert state["architecture"] == original
    assert state["contract_revision"] == 1
    assert len(architect.calls) == 2
    assert (tmp_path / ".atlant_checkpoint.json").is_file()


def test_revised_plan_does_not_reuse_failed_checks_from_previous_source(tmp_path):
    import copy

    state = state_for({"/base.py": entry(), "/check.py": entry(["/base.py"])})
    state.update(
        contract_approved=True,
        completed_files=["/base.py", "/check.py"],
        validation_results=[{"check": "contract", "status": "FAIL"}],
        project_test_failure="old case-insensitive check failed",
    )
    state["code_/check.py"] = "old incorrect test"
    revised = copy.deepcopy(state["architecture"])
    revised["files"]["/check.py"]["desc"] = "Use case-sensitive check"

    def approve(api, trial, operation):
        assert operation == "REVIEW_PLAN"
        assert not trial.get("validation_results")
        assert not trial.get("project_test_failure")
        assert "code_/check.py" not in trial
        return {"command": "APPROVE", "reason": "correct repair plan"}

    with review_responses(approve):
        _revise(
            state,
            str(tmp_path),
            FakeAPI(),
            FakeAPI([revised]),
            {"files": ["/check.py"], "reason": "Use a case-sensitive check"},
        )
    assert state["contract_revision"] == 2
    assert state["completed_files"] == ["/base.py"]
    assert state["project_verified"] is False


@pytest.mark.parametrize("final_command", ["APPROVE", "STOP", "REPLAN"])
def test_plan_review_can_request_another_file_without_publishing_rejected_plan(
    tmp_path,
    final_command,
):
    import copy

    state = state_for(
        {"/base.py": entry(), "/server.py": entry(), "/other.py": entry()}
    )
    state["contract_approved"] = True
    original = copy.deepcopy(state["architecture"])
    first = copy.deepcopy(original)
    first["files"]["/base.py"]["desc"] = "Fix click movement"
    second = copy.deepcopy(first)
    second["files"]["/server.py"]["desc"] = "Start timers for new games"
    architect = FakeAPI([first, second])
    decisions = iter(
        [
            {
                "command": "REPLAN",
                "files": ["/server.py"],
                "reason": "Fix timer lifecycle",
            },
            {
                "command": final_command,
                "files": ["/base.py"],
                "reason": "review result",
            },
        ]
    )

    def review(api, trial, operation):
        assert operation == "REVIEW_PLAN"
        assert state["architecture"] == original
        assert state["contract_revision"] == 1
        return next(decisions)

    with review_responses(review):
        if final_command == "APPROVE":
            _revise(
                state,
                str(tmp_path),
                FakeAPI(),
                architect,
                {"files": ["/base.py"], "reason": "Fix movement"},
            )
            assert state["architecture"] == second
            assert state["contract_revision"] == 2
            assert state["contract_approved"] is True
        else:
            with pytest.raises(
                CheckpointExitError,
                match=(
                    "review result"
                    if final_command == "STOP"
                    else "replan limit reached"
                ),
            ):
                _revise(
                    state,
                    str(tmp_path),
                    FakeAPI(),
                    architect,
                    {"files": ["/base.py"], "reason": "Fix movement"},
                )
            checkpoint = json.loads((tmp_path / ".atlant_checkpoint.json").read_text())[
                "memory"
            ]
            assert checkpoint["architecture"] == original
            assert not checkpoint["project_verified"]
    assert len(architect.calls) == 2
    request, _ = json.JSONDecoder().raw_decode(architect.calls[1].split("\n", 1)[1])
    assert request["architecture"] == first
    assert request["decision"]["reason"] == "Fix timer lifecycle"
    assert request["editable_files"] == ["/base.py", "/server.py"]
    assert state["replans_/base.py"] == state["replans_/server.py"] == 1


def test_initial_plan_can_correct_same_file_twice_before_workers_start(tmp_path):
    state = state_for({"/base.py": entry()})
    memory.restore(state)
    decision = {"command": "REPLAN", "files": ["/base.py"], "reason": "Close contract"}
    reviews = []

    def review(api, trial, operation):
        reviews.append(operation)
        if operation == "REVIEW_PLAN" and len(reviews) <= 2:
            return decision
        return {"command": "APPROVE", "reason": "complete"}

    def worker(path, context, *args):
        assert reviews == ["REVIEW_PLAN"] * 3
        assert memory.get("contract_approved") is True
        return success(path, context)

    result = run(
        tmp_path,
        worker,
        reviewer=review,
        architect=FakeAPI([state["architecture"], state["architecture"]]),
    )
    assert (
        result["project_verified"] is False
    )  # Plan approval alone is not verification.
    assert result["initial_plan_corrections"] == 2
    assert "replans_/base.py" not in result
    assert not result.get("rework_requests")


def test_initial_plan_correction_limit_persists_on_resume(tmp_path):
    from atlant_coder import config

    state = state_for({"/base.py": entry()})
    decision = {
        "command": "REPLAN",
        "files": ["/base.py"],
        "reason": "Still incomplete",
    }
    architect = FakeAPI([state["architecture"]] * config.max_initial_plan_corrections)
    with review_responses(lambda *args: decision):
        with pytest.raises(CheckpointExitError, match="initial plan correction limit"):
            _revise(state, str(tmp_path), FakeAPI(), architect, decision)
        checkpoint = json.loads((tmp_path / ".atlant_checkpoint.json").read_text())[
            "memory"
        ]
        assert (
            checkpoint["initial_plan_corrections"]
            == config.max_initial_plan_corrections
        )
        assert not checkpoint.get("contract_approved")
        assert "replans_/base.py" not in checkpoint
        memory.restore(checkpoint)
        with patch("atlant_coder.handlers.coder._generate_one") as worker:
            with pytest.raises(
                CheckpointExitError, match="initial plan correction limit"
            ):
                run_generation(
                    FakeAPI(),
                    str(tmp_path),
                    core=FakeAPI(),
                    architect=architect,
                    thinker=FakeAPI(),
                )
        worker.assert_not_called()
    assert len(architect.calls) == config.max_initial_plan_corrections


def test_python_interfaces_preserve_argument_kinds_and_field_defaults():
    from atlant_coder.utils.signatures import extract_signatures

    source = 'def f(a: int, /, b=1, *, flag: bool=False, **kw: str): pass\nclass C:\n    x: int = 2\n    STATUS = "ready"\n'
    extracted = "\n".join(extract_signatures(source))
    assert "a: int, /, b=1, *, flag: bool=False, **kw: str" in extracted
    assert "x: int = 2" in extracted and "STATUS = 'ready'" in extracted


def test_unittest_zero_discovery_is_not_a_pass(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_empty.py").write_text("# no tests\n")
    assert "zero tests" in _run_project_unittest(str(tmp_path))


def test_dispatcher_runs_original_task_through_core_arch_workers_and_final_review(
    tmp_path,
):
    from AurethosSDK.models.api_router import APIRouter
    from tests.handlers.test_parallel_batch import made

    (tmp_path / "task.atl").write_text("Keep VALUE=1, expose no extra features.")
    memory.restore({})
    architecture = state_for(
        {"/base.py": entry(high=True), "/tests/test_base.py": entry(["/base.py"])}
    )["architecture"]
    commands = []

    def provider(self, message, **kwargs):
        name = self.model.settings.name
        commands.append((name, message.splitlines()[0]))
        if name == "CORE":
            if message.startswith("REVIEW_PLAN\n"):
                payload = json.loads(message.split("\n", 1)[1])
                return {
                    "command": "APPROVE",
                    "reason": "all sections checked",
                    "coverage": [
                        {
                            "section": section,
                            "status": "covered",
                            "files": ["/base.py"],
                            "evidence": "VALUE=1 is owned by base.py and tested by test_base.py",
                        }
                        for section in payload["requirement_sections"]
                    ],
                }
            if message.startswith("TASK "):
                return {"command": "OK"}
            if message.startswith("MAKE"):
                return {
                    "command": "MADE",
                    "data": "Keep VALUE=1, expose no extra features.",
                }
            return {"command": "APPROVE", "reason": "matches original task"}
        if name == "ARCH":
            return {"command": "OK"} if message.startswith("TASK ") else architecture
        if message.startswith("REVIEW"):
            return {"command": "OK"}
        path = message.splitlines()[0].removeprefix("MAKE ")
        if path == "/base.py":
            return made(path)
        return {
            "command": "MADE",
            "file": path,
            "code": "import unittest\nfrom base import VALUE\nclass Contract(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(VALUE, 1)\n",
            "summary": {
                "behavior": "Tests the public constant.",
                "exports": ["Contract"],
                "issues": [],
            },
        }

    from atlant_coder.app import run as run_app

    with (
        patch.object(APIRouter, "send_api_request", provider),
        patch("atlant_coder.handlers.core.ask_loops_count", 0),
    ):
        run_app(str(tmp_path), (tmp_path / "task.atl").read_text())
    assert memory.get("source_task") == "Keep VALUE=1, expose no extra features."
    assert memory.get("project_verified") is True
    assert ("CORE", "REVIEW_PLAN") in commands and ("CORE", "FINALIZE") in commands
    assert not any(
        name == "CODER" and prompt.startswith(("TASK ", "STRUCTURE "))
        for name, prompt in commands
    )
