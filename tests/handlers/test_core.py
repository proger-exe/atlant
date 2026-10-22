from unittest.mock import patch

from AurethosSDK.client.dispatcher import Dispatcher
from AurethosSDK.client.router import Router
from atlant_coder.handlers.core import manager
from atlant_coder.handlers.planning import transition
from tests.handlers.test_parallel_batch import FakeAPI


def core_dispatch(api, run, command, **parameters):
    dispatcher = Dispatcher("")
    router = Router(api, run=run)
    router.sync(manager.handlers)
    dispatcher.routers["CORE"] = router
    return dispatcher.check_update(transition("CORE", command, **parameters))


def review_project(api, state, operation, issues=None):
    result = core_dispatch(
        api,
        dict(state=state, home=state.get("home_path", "."), control=None),
        operation,
        issues=issues,
    )
    return result.parameters["decision"]


def test_core_forwards_made_instead_of_user_confirmation():
    state = dict(source_task="original task", tech_task="original task")
    api = FakeAPI(
        [
            {"command": "ASK", "data": "confirm"},
            {"command": "MADE", "data": "complete specification"},
        ]
    )
    with (
        patch("atlant_coder.handlers.core.ask_loops_count", 1),
        patch("atlant_coder.handlers.core.wait_user_enter", return_value="yes"),
    ):
        update = core_dispatch(api, dict(state=state, home=".", control=None), "START")
    assert update.to_model == "ARCH" and update.command == "PLAN"
    assert state["tech_task"] == "complete specification"
    assert state["source_task"] == "original task"


def test_noninteractive_task_is_preserved_without_a_model_paraphrase():
    state = dict(
        source_task="Exact supplied specification",
        tech_task="Exact supplied specification",
    )
    api = FakeAPI([])
    with patch("atlant_coder.handlers.core.ask_loops_count", 0):
        update = core_dispatch(api, dict(state=state, home=".", control=None), "START")
    assert update.to_model == "ARCH" and update.command == "PLAN"
    assert state["source_task"] == state["tech_task"]
    assert not api.calls


def test_background_ask_uses_control_and_never_reads_stdin():
    class Control:
        def check_stopped(self):
            pass

        def ask(self, message):
            assert message == "Which scope?"
            return "Only documentation"

    state = dict(source_task="Audit", tech_task="Audit")
    api = FakeAPI(
        [
            {"command": "ASK", "data": "Which scope?"},
            {"command": "MADE", "data": "Audit only documentation"},
        ]
    )
    with (
        patch("atlant_coder.handlers.core.ask_loops_count", 1),
        patch(
            "atlant_coder.handlers.core.wait_user_enter",
            side_effect=AssertionError("stdin forbidden"),
        ),
    ):
        update = core_dispatch(
            api, dict(state=state, home=".", control=Control()), "START"
        )
    assert update.to_model == "ARCH"
    assert "Only documentation" in api.calls[1]


def test_core_resolve_excludes_stale_unpublished_draft(tmp_path):
    import json

    from atlant_coder.utils.project import digest
    from tests.handlers.test_parallel_batch import FakeAPI, entry, state_for

    (tmp_path / "base.py").write_text("BASE = 2\n")
    state = state_for({"/base.py": entry(), "/consumer.py": entry(["/base.py"])})
    state.update(
        home_path=str(tmp_path),
        completed_files=["/base.py"],
        file_hashes={"/base.py": digest("BASE = 2\n")},
        file_drafts={
            "/consumer.py": {
                "code": "VALUE = 1\n",
                "summary": {"behavior": "draft", "exports": [], "issues": []},
                "revision": 1,
                "base_hash": None,
                "dependency_hashes": {"/base.py": digest("BASE = 1\n")},
            }
        },
    )
    api = FakeAPI([{"command": "STOP", "reason": "cannot repair safely"}])
    review_project(api, state, "RESOLVE", {"/consumer.py": "failed"})
    payload = json.loads(api.calls[0].split("\n", 1)[1])
    assert payload["unpublished_drafts"] == {}


def test_plan_approval_requires_complete_coverage_and_cannot_hide_gaps():
    import json
    import pytest
    from tests.handlers.test_parallel_batch import FakeAPI, entry, state_for

    state = state_for({"/a.py": entry(), "/b.py": entry()})
    state["source_task"] = "Exact events.\n\nNo extra dependencies."
    rows = [
        {
            "section": "1",
            "status": "covered",
            "files": ["/a.py"],
            "evidence": "event payload fixed",
        },
        {
            "section": "2",
            "status": "covered",
            "files": ["/b.py"],
            "evidence": "stdlib only",
        },
    ]
    accepted = {
        "command": "APPROVE",
        "reason": "checked both sections",
        "coverage": rows,
    }
    for invalid in [
        {"command": "APPROVE", "reason": "looks good"},
        {**accepted, "coverage": rows[:1]},
        {**accepted, "coverage": [rows[0], rows[0]]},
        {**accepted, "coverage": [rows[0], {**rows[1], "status": "gap"}]},
        {**accepted, "coverage": [rows[0], {**rows[1], "files": ["/unknown.py"]}]},
    ]:
        api = FakeAPI([invalid, accepted])
        assert review_project(api, state, "REVIEW_PLAN") == accepted
        assert len(api.calls) == 2
    with pytest.raises(ValueError, match="complete coverage"):
        review_project(FakeAPI([invalid, invalid]), state, "REVIEW_PLAN")
    payload = json.loads(api.calls[0].split("\n", 1)[1])
    assert payload["requirement_sections"] == {
        "1": "Exact events.",
        "2": "No extra dependencies.",
    }
    gaps = [{**row, "status": "gap"} for row in rows]
    partial = {"command": "REPLAN", "reason": "fix both", "files": [], "coverage": gaps}
    complete = {**partial, "files": ["/a.py", "/b.py"]}
    assert (
        review_project(FakeAPI([partial, complete]), state, "REVIEW_PLAN") == complete
    )
    stop = {"command": "STOP", "reason": "incompatible user requirements"}
    assert review_project(FakeAPI([stop]), state, "REVIEW_PLAN") == stop
