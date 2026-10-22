"""Real coordinator/publication/checks, with deterministic model responses."""

import json
import subprocess
from unittest.mock import patch

import pytest

from atlant_coder.app import run
from atlant_coder.handlers.parallel_batch import _generate_one
from tests.handlers.test_parallel_batch import FakeAPI, entry, state_for
from atlant_coder.utils.project import digest, file_context


def patch_reply(old, new, path="/app.py"):
    return {
        "command": "PATCH",
        "file": path,
        "edits": [{"old": old, "new": new}],
        "summary": {"behavior": "Updates VALUE", "exports": ["VALUE"], "issues": []},
    }


class Script(FakeAPI):
    def send_api_request(self, message, **kwargs):
        response = super().send_api_request(message, **kwargs)
        return response(message) if callable(response) else response


class Reviewer(FakeAPI):
    def send_api_request(self, message, **kwargs):
        self.calls.append(message)
        self.request_metrics.append({"duration_s": 0, "input_chars": len(message)})
        operation, raw = message.split("\n", 1)
        payload = json.loads(raw)
        response = {
            "command": "APPROVE",
            "reason": "Existing VALUE contract is unchanged",
        }
        if operation == "REVIEW_PLAN":
            response["coverage"] = [
                {
                    "section": key,
                    "status": "covered",
                    "files": ["/app.py"],
                    "evidence": "app.py owns the integer VALUE export",
                }
                for key in payload["requirement_sections"]
            ]
        if operation in {"RESOLVE", "STEER"}:
            response["files"] = ["/app.py"]
        return response


def run_repo(home, responses, *, control=None):
    plan = {
        "command": "MADE",
        "conventions": "VALUE remains an integer",
        "files": {"/app.py": entry()},
    }
    architect = Script([{"command": "READ", "path": "app.py"}, plan])
    coder = Script(responses)
    core = Reviewer()
    roles = {"ARCH": architect, "CODER": coder, "CORE": core, "CODER_THINK": FakeAPI()}
    with (
        patch(
            "atlant_coder.app.role_api",
            side_effect=lambda name, home: roles.get(name, FakeAPI()),
        ),
        patch(
            "atlant_coder.handlers.parallel_batch._clone_api_router",
            side_effect=lambda api: api,
        ),
    ):
        result = run(
            home,
            "Update VALUE and preserve unrelated work",
            mode="agent",
            workers=1,
            control=control,
        )
    return result, architect, coder, core


@pytest.fixture(autouse=True)
def isolated_checkpoints(monkeypatch):
    monkeypatch.delenv("ATLANT_CHECKPOINT_DIR", raising=False)


def test_patch_preserves_unrelated_git_dirty_work_and_no_checks_is_unverified(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    unrelated = tmp_path / "user.txt"
    unrelated.write_text("committed\n")
    for args in (
        ["init", "-q"],
        ["add", "user.txt"],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )
    unrelated.write_text("uncommitted user edit\n")
    result, architect, coder, _ = run_repo(
        tmp_path, [patch_reply("VALUE = 1", "VALUE = 2")]
    )
    assert (tmp_path / "app.py").read_text() == "VALUE = 2\n"
    assert unrelated.read_text() == "uncommitted user edit\n"
    assert not result["project_verified"]
    assert result["validation_results"][0]["status"] == "UNKNOWN"
    assert len(architect.calls) == 2 and len(coder.calls) == 1


def test_post_test_fix_reuses_contract_without_architecture_rework(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_value.py").write_text(
        "import unittest\nfrom app import VALUE\nclass ValueTest(unittest.TestCase):\n"
        " def test_value(self): self.assertEqual(VALUE, 3)\n"
    )
    result, architect, _, core = run_repo(
        tmp_path,
        [patch_reply("VALUE = 1", "VALUE = 2"), patch_reply("VALUE = 2", "VALUE = 3")],
    )
    assert result["project_verified"]
    assert result["post_test_repairs"] == 1
    assert not result.get("rework_requests")
    assert len(architect.calls) == 2
    assert [prompt.split("\n")[0] for prompt in core.calls] == [
        "REVIEW_PLAN",
        "RESOLVE",
        "FINALIZE",
    ]
    assert result["validation_results"] == [{"check": "python tests", "status": "PASS"}]


def test_concurrent_edit_is_preserved_and_fresh_source_reaches_retry(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n")

    def changed_while_model_was_busy(_):
        target.write_text("VALUE = 10\n# user note\n")
        return patch_reply("VALUE = 1", "VALUE = 2")

    result, _, coder, _ = run_repo(
        tmp_path, [changed_while_model_was_busy, patch_reply("VALUE = 10", "VALUE = 2")]
    )
    assert result["concurrent_edits"] == 1
    assert "VALUE = 10" in coder.calls[1] and "user note" in coder.calls[1]
    assert target.read_text() == "VALUE = 2\n# user note\n"


@pytest.mark.parametrize("waiting_on_tool", [False, True])
def test_mid_worker_steering_discards_pending_tool_and_stale_patch(
    tmp_path, waiting_on_tool
):
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n")

    class Control:
        applied_revision = 0
        queued = False
        events = []
        published = []

        def check_stopped(self):
            pass

        def pending(self):
            return self.queued and self.applied_revision == 0

        def messages(self):
            return [{"revision": 1, "message": "Use VALUE 2"}] if self.pending() else []

        def event(self, kind, message, data=None):
            self.events.append((kind, message))

        def publish(self, state, reason):
            self.published.append(target.read_text())

    control = Control()

    def queue_clarification(_):
        control.queued = True
        return (
            {"command": "READ", "path": "app.py"}
            if waiting_on_tool
            else patch_reply("VALUE = 1", "VALUE = 9")
        )

    result, architect, coder, core = run_repo(
        tmp_path,
        [queue_clarification, patch_reply("VALUE = 1", "VALUE = 2")],
        control=control,
    )
    assert target.read_text() == "VALUE = 2\n"
    assert all("VALUE = 9" not in source for source in control.published)
    assert "User clarification" in coder.calls[1]
    assert not coder.calls[1].startswith("TOOL RESULT")
    assert any(kind == "discarded" for kind, _ in control.events)
    assert result["user_revision"] == 1 and control.applied_revision == 1
    assert len(architect.calls) == 2
    assert any(prompt.startswith("STEER\n") for prompt in core.calls)


def test_patch_cannot_write_another_workers_file(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    state = state_for({"/app.py": entry()})
    state.update(home_path=str(tmp_path), agent_mode=True)
    coder = Script([patch_reply("VALUE = 1", "VALUE = 9", path="/unrelated.py")])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=coder
    ):
        result = _generate_one("/app.py", file_context(state, "/app.py"), coder)
    assert result["status"] == "error" and "another worker" in result["reason"]
    assert (tmp_path / "app.py").read_text() == "VALUE = 1\n"
    assert not (tmp_path / "unrelated.py").exists()


@pytest.mark.parametrize("delete", [False, True])
def test_user_edit_during_validation_never_gets_rolled_back(tmp_path, delete):
    from atlant_coder.handlers.coder import _accept
    from tests.handlers.test_parallel_scheduler import success

    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n")
    state = state_for({"/app.py": entry()})
    state.update(file_summaries={}, file_hashes={})

    def user_changes(*args):
        if delete:
            target.unlink()
        else:
            target.write_text("VALUE = 99  # user edit\n")
        return ["validation failed"]

    with patch("atlant_coder.handlers.coder.smoke_check", side_effect=user_changes):
        action, _ = _accept(
            state, str(tmp_path), success("/app.py", {"revision": 1}, "VALUE = 2\n")
        )
    assert action == "fatal"
    if delete:
        assert not target.exists()
    else:
        assert target.read_text() == "VALUE = 99  # user edit\n"


def test_atomic_patch_preserves_executable_mode(tmp_path):
    from atlant_coder.macroses import write_file_after_generic

    target = tmp_path / "tool.py"
    target.write_text("old\n")
    target.chmod(0o755)
    write_file_after_generic(target, "new\n")
    assert target.stat().st_mode & 0o777 == 0o755
    assert target.read_text() == "new\n"


def test_changed_dependency_prevents_publishing_consumer(tmp_path):
    from atlant_coder.handlers.coder import _accept
    from tests.handlers.test_parallel_scheduler import success
    from atlant_coder.utils.project import digest

    (tmp_path / "base.py").write_text("VALUE = 99\n")
    state = state_for({"/base.py": entry(), "/consumer.py": entry(["/base.py"])})
    state.update(
        completed_files=["/base.py"],
        file_summaries={},
        file_hashes={"/base.py": digest("VALUE = 1\n")},
    )
    result = success("/consumer.py", {"revision": 1})
    result["dependency_hashes"] = {"/base.py": digest("VALUE = 1\n")}
    action, reason = _accept(state, str(tmp_path), result)
    assert action == "fatal" and "dependencies changed" in reason
    assert not (tmp_path / "consumer.py").exists()
    assert "/base.py" not in state["completed_files"]
    assert (tmp_path / "base.py").read_text() == "VALUE = 99\n"


def test_publication_guard_discards_result_when_steer_wins_lock(tmp_path, monkeypatch):
    from atlant_coder import runtime
    from atlant_coder.handlers import coder as coordinator
    from atlant_coder.handlers.coder import _accept
    from tests.handlers.test_parallel_scheduler import success

    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n")
    store = runtime.RunStore(tmp_path / "runs")
    run = store.create(str(tmp_path), "task", "agent")
    store.update(run["id"], status="running")
    control = runtime.RunControl(store, run["id"])
    state = state_for({"/app.py": entry()})
    state.update(file_summaries={}, file_hashes={})
    store.steer(run["id"], "Use VALUE 3")
    monkeypatch.setattr(coordinator, "_control", control)

    action, _ = _accept(
        state, str(tmp_path), success("/app.py", {"revision": 1}, "VALUE = 2\n")
    )
    assert action == "discarded"
    assert target.read_text() == "VALUE = 1\n"
    assert state["generation_metrics"] == []
    assert state.get("file_drafts", {}) == {}


@pytest.mark.parametrize("stale", ["baseline", "revision", "dependency"])
def test_stale_draft_is_not_reused(tmp_path, stale):
    target = tmp_path / "consumer.py"
    target.write_text("VALUE = 1\n")
    dependency = tmp_path / "base.py"
    dependency.write_text("BASE = 1\n")
    state = state_for({"/base.py": entry(), "/consumer.py": entry(["/base.py"])})
    state.update(
        home_path=str(tmp_path),
        completed_files=["/base.py"],
        file_hashes={"/base.py": digest("BASE = 1\n")},
        file_drafts={
            "/consumer.py": {
                "code": "VALUE = 2\n",
                "summary": {"behavior": "draft", "exports": [], "issues": []},
                "revision": 1,
                "base_hash": digest("VALUE = 1\n"),
                "dependency_hashes": {"/base.py": digest("BASE = 1\n")},
            }
        },
    )
    if stale == "baseline":
        target.write_text("VALUE = 9\n")
    elif stale == "revision":
        state["contract_revision"] = 2
    else:
        state["file_hashes"]["/base.py"] = digest("BASE = 9\n")

    assert "previous_draft" not in file_context(state, "/consumer.py")


def test_accept_records_draft_dependency_snapshot(tmp_path):
    from atlant_coder.handlers.coder import _accept

    state = state_for({"/base.py": entry(), "/consumer.py": entry(["/base.py"])})
    dependency_hashes = {"/base.py": digest("BASE = 1\n")}
    result = {
        "path": "/consumer.py",
        "revision": 1,
        "status": "conflict",
        "reason": "repair dependency use",
        "base_hash": None,
        "dependency_hashes": dependency_hashes,
        "draft": {
            "code": "VALUE = 1\n",
            "summary": {"behavior": "draft", "exports": [], "issues": []},
        },
    }
    _accept(state, str(tmp_path), result)
    assert (
        state["file_drafts"]["/consumer.py"]["dependency_hashes"] == dependency_hashes
    )


def test_document_only_checks_reject_empty_and_ignore_nested_projects(tmp_path):
    from atlant_coder.handlers.coder import _project_checks

    (tmp_path / "report.md").write_text("")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "package.json").write_text('{"scripts":{"test":"exit 1"}}')
    state = state_for({"/report.md": entry()})
    with patch("atlant_coder.handlers.coder._command") as command:
        failures, checks = _project_checks(state, str(tmp_path))
    assert failures == {"/report.md": "document is empty"}
    assert checks == []
    command.assert_not_called()

    (tmp_path / "report.md").write_text("Evidence\n")
    with patch("atlant_coder.handlers.coder._command") as command:
        failures, checks = _project_checks(state, str(tmp_path))
    assert failures == {}
    assert checks[0]["status"] == "PASS"
    command.assert_not_called()


def test_ponytail_avoids_redundant_model_review_but_keeps_draft_issue_review(tmp_path):
    from tests.handlers.test_parallel_batch import made

    state = state_for({"/app.py": entry(high=True)})
    state.update(home_path=str(tmp_path), policy_mode="full")
    clean = made("/app.py")
    coder = Script([clean])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=coder
    ):
        result = _generate_one("/app.py", file_context(state, "/app.py"), coder)
    assert result["status"] == "ok" and result["draft_reviews"] == 0
    assert len(coder.calls) == 1
    broken = {
        **clean,
        "summary": {**clean["summary"], "issues": ["Needs export correction"]},
    }
    coder = Script([broken, clean])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=coder
    ):
        result = _generate_one("/app.py", file_context(state, "/app.py"), coder)
    assert result["status"] == "ok" and result["draft_reviews"] == 1
    assert coder.calls[1].startswith("REVIEW")


def test_patch_does_not_reformat_or_remove_unrelated_user_code(tmp_path):
    target = tmp_path / "app.py"
    before = "import os  # preexisting unused import\nVALUE = 1\n\ndef user_note():\n    return 'keep this spelling'\n"
    target.write_text(before)
    run_repo(tmp_path, [patch_reply("VALUE = 1", "VALUE = 2")])
    assert target.read_text() == before.replace("VALUE = 1", "VALUE = 2")
