"""Project contracts and per-file snapshots shared by every execution path."""

from atlant_coder.utils.prompt_rules import build_rules

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

from atlant_coder.utils.skip_think import normalize_files, transitive_deps


def target_path(home: str, path: str) -> Path:
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or path.startswith("//")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path[1:].split("/"))
        or PurePosixPath(path).as_posix() != path
    ):
        raise ValueError(f"invalid project path: {path!r}")
    parts = PurePosixPath(path).parts[1:]
    if any(
        part
        in {
            ".aws",
            ".ssh",
            ".gnupg",
            "auth.json",
            "credentials.json",
            "id_rsa",
            "id_ed25519",
        }
        or (part.startswith(".env") and part != ".env.example")
        or part.endswith((".pem", ".key"))
        for part in parts
    ):
        raise ValueError(f"private project path: {path}")
    if (
        parts[0] in {".git", ".codex", ".agents", ".atlant"}
        or parts[0].startswith(".atlant_")
        or path
        in {
            "/task.atl",
            "/.atlant_checkpoint.json",
            "/.atlant_metrics.json",
            "/.atlant.log",
        }
    ):
        raise ValueError(f"reserved project path: {path}")
    root = Path(home).resolve()
    target = (root / path[1:]).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes project: {path}")
    resolved = "/" + target.relative_to(root).as_posix()
    if resolved != path:
        target_path(home, resolved)
    return target


def digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def current_plan(state: dict) -> dict:
    if state.get("architecture"):
        return copy.deepcopy(state["architecture"])
    # Upgrade the original map-based checkpoint without discarding its files.
    files = dict(state.get("files") or {})
    for path in state.get("completed_files") or []:
        files.setdefault(path, f"Existing implementation: {path}")
    entries = {}
    for path, desc in files.items():
        priority = (state.get("file_priorities") or {}).get(path, "high")
        entry = {
            "desc": desc,
            "priority": priority,
            "depends_on": (state.get("file_deps") or {}).get(path, []),
            "parallel_safe": path in (state.get("parallel_safe_files") or []),
        }
        if priority == "high":
            entry["plan"] = (state.get(f"task_{path}") or desc).splitlines()
            entry["public_names"] = (state.get("file_public_names") or {}).get(path, [])
        entries[path] = entry
    return {
        "command": "MADE",
        "conventions": state.get("project_conventions")
        or "Preserve the original task and existing interfaces.",
        "files": entries,
    }


def install_plan(state: dict, plan: dict) -> None:
    descs, priorities, deps, plans, publics, safe = normalize_files(plan["files"])
    state.update(
        architecture=copy.deepcopy(plan),
        files=descs,
        file_priorities=priorities,
        file_deps=deps,
        file_public_names=publics,
        parallel_safe_files=[p for p, value in safe.items() if value],
        project_conventions=plan["conventions"],
    )
    for path in descs:
        state[f"task_{path}"] = plans.get(path, "")


def affected_files(paths, deps: dict) -> set:
    affected = set(paths)
    while True:
        added = {
            p for p, needs in deps.items() if affected.intersection(needs)
        } - affected
        if not added:
            return affected
        affected.update(added)


def file_context(state: dict, path: str) -> dict:
    deps = state.get("file_deps") or {}
    needed = transitive_deps(path, deps)
    completed = set(state.get("completed_files") or [])
    if not needed <= completed:
        raise ValueError(
            f"unpublished dependencies for {path}: {sorted(needed - completed)}"
        )
    dependencies = {}
    for dep in sorted(needed):
        item = {
            "public_names": (state.get("file_public_names") or {}).get(dep, []),
            "signatures": state.get(f"signatures_{dep}") or [],
            "summary": (state.get("file_summaries") or {}).get(dep, {}),
        }
        # Shared styles are contracts even through a component dependency: a
        # summary of Button must not hide the layout classes from its consumers.
        if (dep in deps.get(path, []) and Path(dep).suffix != ".py") or Path(
            dep
        ).suffix in {".css", ".scss"}:
            item["source"] = state.get(f"code_{dep}", "")
        dependencies[dep] = item
    context = copy.deepcopy(
        {
            "source_task": state.get("source_task") or state.get("tech_task", ""),
            "tech_task": state.get("tech_task", ""),
            "revision": state.get("contract_revision", 0),
            "conventions": state.get("project_conventions", ""),
            "structure": list(state.get("files") or {}),
            "file": path,
            "entry": current_plan(state)["files"][path],
            "dependencies": dependencies,
            "dependency_hashes": {
                dep: state.get("file_hashes", {}).get(dep)
                for dep in needed
                if state.get("file_hashes", {}).get(dep)
            },
            "feedback": state.get(f"last_error_{path}", ""),
        }
    )

    if context["tech_task"] == context["source_task"]:
        context.pop("tech_task")

    context["rules"] = build_rules(
        context["source_task"], "CODER", path, mode=state.get("policy_mode", "full")
    )
    if state.get("home_path"):
        target = target_path(state["home_path"], path)
        context["base_hash"] = (
            digest(target.read_text(encoding="utf-8")) if target.exists() else None
        )
        if target.exists():
            context["current_source"] = target.read_text(encoding="utf-8")
    context["agent_mode"] = state.get("agent_mode", False)
    context["policy_mode"] = state.get("policy_mode", "off")
    draft = state.get("file_drafts", {}).get(path)
    if (
        draft
        and draft.get("revision") == context["revision"]
        and draft.get("base_hash") == context.get("base_hash")
        and draft.get("dependency_hashes", {}) == context["dependency_hashes"]
    ):
        context["previous_draft"] = {"code": draft["code"], "summary": draft["summary"]}
    return context


def make_prompt(context: dict) -> str:
    style_rule = (
        "Reuse class names from shared stylesheet sources exactly; report CONFLICT "
        "if a required style is missing instead of inventing an unstyled class.\n"
        if any(Path(p).suffix in {".css", ".scss"} for p in context["dependencies"])
        else ""
    )
    return (
        "MAKE " + context["file"] + "\n"
        "The following JSON is the complete task context, not a TASK/STRUCTURE handshake. "
        "Implement only this file. Preserve the original task, project conventions, "
        "public interfaces and shared UI components/tokens. Return MADE with code "
        "and summary {behavior: string, exports: string[], issues: string[], limitations?: string[]}. "
        "issues contains only unresolved defects or unmet requirements; limitations records evidence boundaries "
        "such as tests not run when the task only requested a document audit. "
        "This is an unpublished file draft: the coordinator publishes it, then runs required project tests/builds "
        "and CORE FINALIZE before declaring the task complete. Pending post-publication checks are limitations, "
        "not draft defects; do not RUN tests against the old disk file to verify your unpublished patch. "
        "A known failing behavior in your draft remains an issue. Never claim pending checks passed. "
        "If the contract is contradictory, return CONFLICT with issues: string[]. "
        "Never silently change a contract. NEEDRESUME is only for real output limits.\n"
        + style_rule
        + (
            "Use PATCH {file,edits:[{old,new}],summary} for an existing file; exact old text must match once. "
            "READ(path), SEARCH(path,pattern), LIST(path), RUN(argv) request coordinator tools. "
            "Do not execute tools yourself or rewrite unrelated code.\n"
            if context.get("agent_mode")
            else ""
        )
        + json.dumps(
            {k: v for k, v in context.items() if k != "broker"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
