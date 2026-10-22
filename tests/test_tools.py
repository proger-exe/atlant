import threading
import time
from pathlib import Path

import pytest

from atlant_coder.macroses.workspace import (
    ToolBroker,
    WorkspaceTools,
    apply_edits,
    inventory,
    read_text,
)
from atlant_coder.utils.project import digest


class Control:
    def __init__(self, allow: bool):
        self.allow = allow
        self.approvals = []
        self.events = []

    def approval(self, description):
        self.approvals.append(description)
        return self.allow

    def event(self, kind, message, data):
        self.events.append((kind, message, data))


def test_paths_do_not_escape_or_follow_symlinks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "safe.txt").write_text("safe")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    (workspace / "link.txt").symlink_to(secret)
    private = workspace / ".git"
    private.mkdir()
    (private / "config").write_text("credential = secret")
    (workspace / "git-config.txt").symlink_to(private / "config")

    assert inventory(str(workspace)) == ["safe.txt"]
    assert read_text(str(workspace), "safe.txt") == "safe"
    with pytest.raises(ValueError, match="invalid project path|escapes project"):
        read_text(str(workspace), "../secret.txt")
    with pytest.raises(ValueError, match="escapes project"):
        read_text(str(workspace), "link.txt")
    with pytest.raises(ValueError, match="reserved|private"):
        read_text(str(workspace), "git-config.txt")
    with pytest.raises(ValueError, match="private"):
        read_text(str(workspace), ".env")


def test_patch_requires_unique_current_text_and_hashes_result() -> None:
    source = "alpha\nbeta\n"
    changed = apply_edits(source, [{"old": "beta", "new": "gamma"}])
    assert changed == "alpha\ngamma\n"
    assert digest(changed) != digest(source)
    assert len(digest(changed)) == 64
    with pytest.raises(ValueError, match="exactly once"):
        apply_edits("same same", [{"old": "same", "new": "new"}])
    with pytest.raises(ValueError, match="at least one"):
        apply_edits(source, [])


def test_broker_executes_on_coordinator_and_records_owner(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello")
    control = Control(True)
    broker = ToolBroker(WorkspaceTools(str(tmp_path), control))
    result = []
    requester = threading.Thread(
        target=lambda: result.append(
            broker.request(
                "CODER-2", broker.tools.read, {"command": "READ", "path": "note.txt"}
            )
        )
    )
    requester.start()
    for _ in range(100):
        if not broker.queue.empty():
            break
        time.sleep(0.001)
    assert requester.is_alive(), "worker requests must wait for the coordinator"
    broker.drain()
    requester.join(timeout=1)
    assert result == [{"text": "hello", "sha256": digest("hello")}]
    assert control.events == [
        ("tool", "CODER-2: READ", {"command": "READ", "path": "note.txt"})
    ]


def test_commands_require_approval_except_fixed_checks(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        "atlant_coder.macroses.workspace.run_command",
        lambda argv, home, **kwargs: calls.append((argv, home, kwargs)) or "",
    )

    denied = Control(False)
    assert (
        WorkspaceTools(str(tmp_path), denied).run(
            {"command": "RUN", "argv": ["git", "status"]}
        )["status"]
        == "denied"
    )
    assert calls == []
    assert "git" in denied.approvals[0]

    allowed = Control(True)
    result = WorkspaceTools(str(tmp_path), allowed).run(
        {"command": "RUN", "argv": ["git", "status"]}
    )
    assert result["status"] == "PASS"
    assert len(allowed.approvals) == 1
    assert calls[-1][2]["control"] is allowed

    automatic = Control(False)
    result = WorkspaceTools(str(tmp_path), automatic).run(
        {"command": "RUN", "argv": ["python3", "-m", "pytest", "-q"]}
    )
    assert result["status"] == "PASS"
    assert automatic.approvals == []
    assert calls[-1][2]["control"] is automatic

    extra = Control(False)
    result = WorkspaceTools(str(tmp_path), extra).run(
        {
            "command": "RUN",
            "argv": ["python3", "-m", "pytest", "-q", "tests/test_one.py"],
        }
    )
    assert result["status"] == "denied"
    assert len(extra.approvals) == 1
