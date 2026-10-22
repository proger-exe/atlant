"""Worker regressions: exact targets, isolated context and one-shot generation."""

from unittest.mock import patch

import pytest

from atlant_coder.handlers.parallel_batch import _generate_one
from atlant_coder.utils.memory import memory
from atlant_coder.utils.project import file_context, install_plan


class FakeAPI:
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.calls = []
        self.options = []
        self.request_metrics = []

    def send_api_request(self, message, **kwargs):
        self.calls.append(message)
        self.options.append(kwargs)
        self.request_metrics.append({"duration_s": 0, "input_chars": len(message)})
        return next(self.responses)

    def reset_context(self):
        pass


def entry(deps=(), safe=True, high=False):
    item = {
        "desc": "implement VALUE",
        "priority": "high" if high else "low",
        "depends_on": list(deps),
        "parallel_safe": safe,
    }
    if high:
        item.update(
            plan=["define VALUE", "use agreed type", "export VALUE"],
            public_names=["VALUE"],
        )
    return item


def state_for(files):
    state = {
        "tech_task": "small task",
        "source_task": "small task",
        "completed_files": [],
        "contract_revision": 1,
    }
    install_plan(
        state,
        {
            "command": "MADE",
            "conventions": "VALUE is an integer. Reuse shared contracts.",
            "files": files,
        },
    )
    return state


def made(path, code="VALUE = 1\n"):
    return {
        "command": "MADE",
        "file": path,
        "code": code,
        "summary": {"behavior": "Defines VALUE.", "exports": ["VALUE"], "issues": []},
    }


def test_wrong_target_is_rejected_without_writing_or_marking_complete(tmp_path):
    state = state_for({"/target.js": entry()})
    memory.restore(state)
    before = memory.snapshot()
    api = FakeAPI([made("/different.js", "export const VALUE = 1;")])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/target.js", file_context(state, "/target.js"), api)
    assert result["status"] == "error"
    assert "wrong file" in result["reason"]
    assert memory.snapshot() == before
    assert list(tmp_path.iterdir()) == []
    assert len(api.calls) == 1


def test_one_make_contains_full_context_and_only_dependencies():
    state = state_for(
        {"/dep.py": entry(), "/noise.py": entry(), "/target.py": entry(["/dep.py"])}
    )
    state["completed_files"] = ["/dep.py", "/noise.py"]
    state["signatures_/dep.py"] = ["DEP = 1"]
    state["signatures_/noise.py"] = ["UNRELATED_SECRET = 2"]
    api = FakeAPI([made("/target.py")])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/target.py", file_context(state, "/target.py"), api)
    assert result["status"] == "ok", result
    assert len(api.calls) == 1
    assert api.calls[0].startswith("MAKE /target.py")
    assert "small task" in api.calls[0] and "DEP = 1" in api.calls[0]
    assert "UNRELATED_SECRET" not in api.calls[0]
    assert result["summary"]["verified_exports"] == ["VALUE"]


def test_chunks_are_concatenated_in_original_order():
    state = state_for({"/target.py": entry()})
    api = FakeAPI(
        [
            {"command": "NEEDRESUME", "file": "/target.py", "code": 'VALUE = "'},
            made("/target.py", 'right"\n'),
        ]
    )
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/target.py", file_context(state, "/target.py"), api)
    assert result["status"] == "ok", result
    assert '"right"' in result["code"]


def test_transitive_shared_styles_reach_screen_without_unrelated_styles():
    state = state_for(
        {
            "/tokens.css": entry(),
            "/style.css": entry(["/tokens.css"]),
            "/Button.tsx": entry(["/style.css"]),
            "/App.tsx": entry(["/Button.tsx"]),
            "/other.css": entry(),
        }
    )
    state["completed_files"] = [p for p in state["files"] if p != "/App.tsx"]
    state["code_/style.css"] = ".app { padding: var(--space); }"
    state["code_/tokens.css"] = ":root { --space: 8px; }"
    state["code_/other.css"] = ".unrelated { color: red; }"
    context = file_context(state, "/App.tsx")
    assert context["dependencies"]["/style.css"]["source"] == state["code_/style.css"]
    assert context["dependencies"]["/tokens.css"]["source"] == state["code_/tokens.css"]
    assert "/other.css" not in context["dependencies"]


def test_review_cannot_replace_another_file():
    state = state_for({"/target.py": entry(high=True)})
    api = FakeAPI([made("/target.py"), made("/wrong.py")])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/target.py", file_context(state, "/target.py"), api)
    assert result["status"] == "error"
    assert "wrong file" in result["reason"]
    assert len(api.calls) == 2


def test_declared_exports_are_checked_against_source():
    state = state_for({"/target.py": entry(high=True)})
    api = FakeAPI([made("/target.py", "OTHER = 1\n"), {"command": "OK"}])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/target.py", file_context(state, "/target.py"), api)
    assert result["status"] == "error"
    assert "VALUE" in result["reason"]


def test_worker_router_does_not_share_model_or_history():
    from AurethosSDK.models.api_router import APIRouter
    from AurethosSDK.models.base import BaseModel
    from AurethosSDK.types.base_model import Model
    from atlant_coder.handlers.parallel_batch import _clone_api_router

    base = APIRouter(BaseModel(Model(name="CODER", system_instruction="test")))
    left, right = _clone_api_router(base), _clone_api_router(base)
    left.model.settings.system_instruction = "changed"
    left.model_history.add_entry("user", "private")
    assert right.model.settings.system_instruction == "test"
    assert right.model_history.history == []
    assert base.model.settings.system_instruction == "test"


def test_draft_issues_are_corrected_before_publication_without_replanning():
    state = state_for({"/package.json": entry()})
    draft = made("/package.json", '{"dependencies":{"forbidden-plugin":"latest"}}')
    draft["summary"]["issues"] = [
        "The contract forbids this plugin; remove it from my draft."
    ]
    fixed = made("/package.json", '{"dependencies":{}}')
    api = FakeAPI([draft, fixed])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one(
            "/package.json", file_context(state, "/package.json"), api
        )
    assert result["status"] == "ok"
    assert result["code"] == fixed["code"]
    assert len(api.calls) == 2
    assert "Known draft issues" in api.calls[1]


def test_review_cannot_silently_clear_issues_but_can_report_real_contract_conflict():
    state = state_for({"/package.json": entry()})
    draft = made("/package.json", "{}")
    draft["summary"]["issues"] = ["Unresolved defect"]
    for review, status in [
        ({"command": "OK"}, "error"),
        (
            {
                "command": "CONFLICT",
                "issues": ["Two required dependency versions disagree"],
            },
            "conflict",
        ),
    ]:
        api = FakeAPI([draft, review])
        with patch(
            "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
        ):
            result = _generate_one(
                "/package.json", file_context(state, "/package.json"), api
            )
        assert result["status"] == status
        assert "code" not in result


def test_agent_review_patch_fixes_new_unpublished_draft():
    state = state_for({"/audit.md": entry()})
    context = {**file_context(state, "/audit.md"), "agent_mode": True}
    draft = made("/audit.md", "Found 59 active projects.\nKeep this evidence.\n")
    draft["summary"]["issues"] = ["Active total must be 68 according to tool evidence."]
    fixed = {
        "command": "PATCH",
        "file": "/audit.md",
        "edits": [{"old": "59 active", "new": "68 active"}],
        "summary": {
            "behavior": "Correct totals",
            "exports": [],
            "issues": [],
            "limitations": ["Audited projects were not executed."],
        },
    }
    api = FakeAPI([draft, fixed])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/audit.md", context, api)
    assert result["status"] == "ok", result
    assert result["code"] == "Found 68 active projects.\nKeep this evidence.\n"
    assert result["summary"]["limitations"] == fixed["summary"]["limitations"]
    assert api.options[1]["expects"] == ["OK", "MADE", "CONFLICT", "PATCH"]


@pytest.mark.parametrize(
    "wrong_file,old", [(True, "59"), (False, "active"), (False, "missing")]
)
def test_review_patch_rejects_wrong_target_or_nonunique_edit(wrong_file, old):
    state = state_for({"/audit.md": entry()})
    draft = made("/audit.md", "59 active; active")
    draft["summary"]["issues"] = ["Incorrect total"]
    fixed = {
        "command": "PATCH",
        "file": "/other.md" if wrong_file else "/audit.md",
        "edits": [{"old": old, "new": "68"}],
        "summary": made("/audit.md")["summary"],
    }
    api = FakeAPI([draft, fixed])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one(
            "/audit.md", {**file_context(state, "/audit.md"), "agent_mode": True}, api
        )
    assert result["status"] == "error"
    assert result["draft"]["code"] == draft["code"]
    assert "code" not in result


def test_limitations_do_not_trigger_review_but_issues_still_block():
    state = state_for({"/audit.md": entry()})
    context = {**file_context(state, "/audit.md"), "agent_mode": True}
    draft = made("/audit.md", "Evidence and explicit verification limits.")
    draft["summary"]["limitations"] = [
        "Runtime not verified; not required by the task."
    ]
    api = FakeAPI([draft])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/audit.md", context, api)
    assert result["status"] == "ok"
    assert len(api.calls) == 1
    draft["summary"]["issues"] = ["A required project is missing from the audit."]
    api = FakeAPI([draft, {"command": "OK"}])
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/audit.md", context, api)
    assert result["status"] == "error"
    assert "unresolved implementation issues" in result["reason"]
    assert result["draft"]["summary"]["issues"] == draft["summary"]["issues"]


def test_conflicting_draft_can_be_patched_next_attempt_without_research():
    state = state_for({"/audit.md": entry()})
    context = {**file_context(state, "/audit.md"), "agent_mode": True}
    draft = made("/audit.md", "Total: 93\nEvidence remains here.")
    draft["summary"]["issues"] = ["README and actual count disagree."]
    api = FakeAPI(
        [draft, {"command": "CONFLICT", "issues": ["Which count is authoritative?"]}]
    )
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        failed = _generate_one("/audit.md", context, api)
    assert failed["status"] == "conflict"
    assert failed["draft"]["code"] == draft["code"]
    context.update(
        previous_draft=failed["draft"],
        current_source="Unrelated disk baseline",
        feedback="Use actual tool evidence: 92 projects.",
    )
    api = FakeAPI(
        [
            {
                "command": "PATCH",
                "file": "/audit.md",
                "edits": [{"old": "Total: 93", "new": "Total: 92"}],
                "summary": {
                    "behavior": "Correct factual count",
                    "exports": [],
                    "issues": [],
                },
            }
        ]
    )
    with patch(
        "atlant_coder.handlers.parallel_batch._clone_api_router", return_value=api
    ):
        result = _generate_one("/audit.md", context, api)
    assert result["status"] == "ok", result
    assert result["code"] == "Total: 92\nEvidence remains here."
    assert len(api.calls) == 1
    assert "PATCH" in api.options[0]["expects"]
    assert "UNPUBLISHED" in api.calls[0]
