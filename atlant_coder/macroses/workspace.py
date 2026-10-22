"""Workspace tools executed by the coordinator, never by model processes."""

from __future__ import annotations

import json
import os
import queue
import threading
import subprocess
from concurrent.futures import Future
from pathlib import Path

from atlant_coder.macroses.command import run_command
from atlant_coder.utils.project import digest, target_path

EXCLUDED = {
    ".git",
    ".codex",
    ".agents",
    ".aws",
    ".ssh",
    ".gnupg",
    ".venv",
    "node_modules",
    "__pycache__",
    ".atlant",
    ".atlant-bench",
    "dist",
    "build",
}


def visible(path: str) -> bool:
    parts = Path(path).parts
    return not any(
        p in EXCLUDED
        or p in {"auth.json", "credentials.json", "id_rsa", "id_ed25519"}
        or (p.startswith(".env") and p != ".env.example")
        or p.startswith(".atlant")
        or p.endswith((".pem", ".key"))
        for p in parts
    )


def inventory(home: str) -> list[str]:
    """Use git's ignore rules where available, including untracked user files."""
    result = subprocess.run(
        ["git", "-C", home, "ls-files", "-co", "-z", "--exclude-standard"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        paths = [p for p in result.stdout.split("\0") if p]
    else:
        paths = []
        for root, dirs, files in os.walk(home, followlinks=False):
            dirs[:] = [d for d in dirs if visible(d) and not Path(root, d).is_symlink()]
            paths.extend(str(Path(root, f).relative_to(home)) for f in files)
    return sorted(
        {
            p
            for p in paths
            if visible(p) and Path(home, p).is_file() and not Path(home, p).is_symlink()
        }
    )


def read_text(home: str, path: str, *, limit: int = 100_000) -> str:
    if not visible(path):
        raise ValueError("private or generated path is not available to tools")
    target = target_path(home, "/" + path.lstrip("/"))
    if target.stat().st_size > limit:
        raise ValueError(
            f"file exceeds {limit} bytes; use SEARCH to narrow the context"
        )
    return target.read_text(encoding="utf-8")


def apply_edits(source: str, edits: list[dict]) -> str:
    if not edits:
        raise ValueError("PATCH requires at least one edit")
    result = source
    for edit in edits:
        old, new = edit["old"], edit["new"]
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise ValueError("PATCH requires nonempty old text and string new text")
        if result.count(old) != 1:
            raise ValueError(
                "PATCH old text must match exactly once; READ the file again"
            )
        result = result.replace(old, new, 1)
    return result


def ordinary_check(argv: list[str]) -> bool:
    """Only fixed test/build invocations auto-run; other commands need approval."""
    if not argv:
        return False
    binary = Path(argv[0]).name
    if binary in {"python", "python3", "python3.12"}:
        return argv[1:] in (
            ["-m", "pytest", "-q"],
            ["-m", "unittest", "discover", "-s", "tests", "-v"],
        )
    return argv in (
        ["npm", "run", "test"],
        ["npm", "run", "build"],
        ["npm", "run", "typecheck"],
    )


class WorkspaceTools:
    def __init__(self, home: str, control=None):
        self.home = str(Path(home).resolve())
        self.control = control

    def _response(self, response: dict) -> dict:
        response = dict(response)
        if isinstance(response.get("path"), str) and response["path"].startswith(
            self.home + "/"
        ):
            response["path"] = response["path"][len(self.home) :]
        return response

    def read(self, response: dict) -> dict:
        response = self._response(response)
        text = read_text(self.home, response["path"])
        return {"text": text, "sha256": digest(text)}

    def list(self, response: dict) -> dict:
        response = self._response(response)
        prefix = response["path"].strip("/")
        if prefix and (".." in Path(prefix).parts or not visible(prefix)):
            raise ValueError("invalid listing path")
        paths = [
            p
            for p in inventory(self.home)
            if not prefix or p == prefix or p.startswith(prefix + "/")
        ]
        return {"paths": paths[:1000], "truncated": len(paths) > 1000}

    def search(self, response: dict) -> dict:
        response = self._response(response)
        pattern = response["pattern"]
        if not pattern or len(pattern) > 1000:
            raise ValueError(
                "SEARCH requires a nonempty literal pattern up to 1000 characters"
            )
        prefix = response["path"].strip("/")
        matches = []
        for path in inventory(self.home):
            if prefix and not (path == prefix or path.startswith(prefix + "/")):
                continue
            try:
                lines = read_text(self.home, path).splitlines()
            except (ValueError, OSError, UnicodeError):
                continue
            for line, text in enumerate(lines, 1):
                if pattern in text:
                    matches.append({"path": path, "line": line, "text": text[:1000]})
                    if len(matches) == 100:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def run(self, response: dict) -> dict:
        response = self._response(response)
        argv = response["argv"]
        if not argv or not all(
            isinstance(a, str) and a and "\0" not in a for a in argv
        ):
            raise ValueError("RUN requires a nonempty argv array")
        if not ordinary_check(argv):
            if self.control is None or not self.control.approval(
                "Запустить команду: " + json.dumps(argv, ensure_ascii=False)
            ):
                return {"status": "denied", "message": "Command not approved"}
        output = []
        error = run_command(argv, self.home, output=output, control=self.control)
        return {
            "status": "FAIL" if error else "PASS",
            "output": error or "".join(output),
        }


class ToolBroker:
    """Workers queue requests; only the coordinator executes them."""

    def __init__(self, tools: WorkspaceTools):
        self.tools = tools
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False

    def request(self, owner: str, operation, response: dict) -> dict:
        future = Future()
        with self.lock:
            if self.closed:
                raise InterruptedError("Coordinator stopped")
            self.queue.put((owner, operation, response, future))
        return future.result()

    def close(self):
        with self.lock:
            self.closed = True
            self.drain(cancelled=True)

    def drain(self, *, cancelled: bool = False) -> None:
        while True:
            try:
                owner, operation, response, future = self.queue.get_nowait()
            except queue.Empty:
                return
            if cancelled:
                future.set_exception(
                    InterruptedError("Task revision changed; discard this result")
                )
                continue
            try:
                if self.tools.control:
                    self.tools.control.event(
                        "tool", f"{owner}: {response['command']}", response
                    )
                future.set_result(operation(response))
            except Exception as exc:
                future.set_result({"error": str(exc)})
