import copy

import pytest

from AurethosSDK.client.dispatcher import Dispatcher
from atlant_coder.handlers.coder import coder_router
from tests.handlers.support import run_generation
from atlant_coder.utils import memory
from atlant_coder.utils.checkpoint import CheckpointExitError
from tests.handlers.test_parallel_batch import FakeAPI, entry, state_for
from tests.handlers.test_parallel_scheduler import run, success


def observe_sdk(monkeypatch):
    received = []
    dispatch = Dispatcher.check_update

    def observe(self, update):
        if update.to_model == "CORE" and update.command in {
            "APPROVE",
            "REPLAN",
            "STOP",
        }:
            received.append(update.command)
            assert set(update.parameters) == {"response"}
        return dispatch(self, update)

    monkeypatch.setattr(Dispatcher, "check_update", observe)
    return received


def test_coordinator_continuations_are_module_level_sdk_handlers():
    assert len(coder_router.handlers) >= 8
    assert all(
        "<locals>" not in handler.function.__qualname__
        for handler in coder_router.handlers
    )
    assert all(
        handler.function.__module__ == "atlant_coder.handlers.coder"
        for handler in coder_router.handlers
    )


def test_unknown_core_command_does_not_start_workers(tmp_path, monkeypatch):
    memory.restore(state_for({"/a.py": entry()}))

    def forbidden(*args):
        pytest.fail("invalid CORE command launched worker")

    monkeypatch.setattr("atlant_coder.handlers.coder._generate_one", forbidden)
    with pytest.raises(ValueError, match="unexpected planning command"):
        run_generation(
            FakeAPI(),
            str(tmp_path),
            core=FakeAPI([{"command": "MADE"}]),
            architect=FakeAPI(),
            thinker=FakeAPI(),
        )
    assert not (tmp_path / "a.py").exists()


def test_stop_reaches_sdk_and_prevents_worker_launch(tmp_path, monkeypatch):
    received = observe_sdk(monkeypatch)
    memory.restore(state_for({"/a.py": entry()}))

    def worker(*args):
        pytest.fail("STOP launched a worker")

    with pytest.raises(CheckpointExitError, match="missing requirement"):
        run(
            tmp_path,
            worker,
            reviewer=lambda *args: {"command": "STOP", "reason": "missing requirement"},
        )
    assert received == ["STOP"]
    assert not memory.get("contract_approved")
    assert memory.get("completed_files") == []


@pytest.mark.parametrize("verdict", ["APPROVE", "STOP"])
def test_replan_candidate_stays_private_until_sdk_approval(
    tmp_path, monkeypatch, verdict
):
    received = observe_sdk(monkeypatch)
    state = state_for({"/a.py": entry()})
    original = copy.deepcopy(state["architecture"])
    proposal = copy.deepcopy(original)
    proposal["files"]["/a.py"]["desc"] = "Correct exact requirement"
    memory.restore(state)
    reviews = iter(
        [
            {"command": "REPLAN", "files": ["/a.py"], "reason": "correct requirement"},
            {"command": verdict, "reason": "candidate verdict"},
            {"command": "APPROVE", "reason": "complete"},
        ]
    )

    def worker(path, context, *args):
        assert verdict == "APPROVE"
        assert memory.get("contract_approved") is True
        assert memory.get("architecture") == proposal
        return success(path, context)

    if verdict == "STOP":
        with pytest.raises(CheckpointExitError, match="candidate verdict"):
            run(
                tmp_path,
                worker,
                reviewer=lambda *args: next(reviews),
                architect=FakeAPI([proposal]),
            )
        assert memory.get("architecture") == original
        assert not (tmp_path / "a.py").exists()
        assert received == ["REPLAN", "STOP"]
    else:
        result = run(
            tmp_path,
            worker,
            reviewer=lambda *args: next(reviews),
            architect=FakeAPI([proposal]),
        )
        assert result["initial_plan_corrections"] == 1
        assert result["contract_revision"] == 2
        assert not result["project_verified"]
        assert received == ["REPLAN", "APPROVE", "APPROVE"]


def test_role_event_schemas_match_registered_commands():
    import ast
    import inspect

    from atlant_coder.handlers import arch, coder, core, parallel_batch, planning
    from atlant_coder.schemas.arch import ArchAcceptedCommands
    from atlant_coder.schemas.coder import CoderAcceptedCommands
    from atlant_coder.schemas.core import CoreAcceptedCommands

    registered = {"CORE": set(), "ARCH": set(), "CODER": set()}
    router_roles = {
        "manager": "CORE",
        "core_router": "CORE",
        "arch_router": "ARCH",
        "coder_router": "CODER",
        "worker_router": "CODER",
    }
    for module in (arch, coder, core, parallel_batch, planning):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "handler"
                    and isinstance(decorator.func.value, ast.Name)
                ):
                    continue
                role = router_roles.get(decorator.func.value.id)
                if role:
                    for part in ast.walk(decorator):
                        if (
                            isinstance(part, ast.Compare)
                            and isinstance(part.left, ast.Attribute)
                            and part.left.attr == "command"
                        ):
                            registered[role].update(
                                value.value
                                for value in part.comparators
                                if isinstance(value, ast.Constant)
                                and isinstance(value.value, str)
                            )
    for role, enum in (
        ("CORE", CoreAcceptedCommands),
        ("ARCH", ArchAcceptedCommands),
        ("CODER", CoderAcceptedCommands),
    ):
        assert {member.value for member in enum} == registered[role]


def test_internal_event_schemas_reject_unregistered_commands_and_wrong_roles():
    from pydantic import ValidationError
    from atlant_coder.schemas import ArchUpdate, CoderUpdate, CoreUpdate

    for schema, role, command in (
        (ArchUpdate, "ARCH", "PLAN"),
        (CoreUpdate, "CORE", "START"),
        (CoderUpdate, "CODER", "GENERATE"),
    ):
        update = schema(command=command, parameters={})
        assert update.to_model == update.from_model == role
        with pytest.raises(ValidationError):
            schema(command="setup_finish", parameters={})
        with pytest.raises(ValidationError):
            schema(to_model="UNKNOWN", command=command, parameters={})
