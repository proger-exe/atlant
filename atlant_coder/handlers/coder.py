"""Single-owner coordinator with a continuously replenished dependency queue."""

import copy
import difflib
import json
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, nullcontext
from pathlib import Path

from AurethosSDK.client.router import ABSRouter
from AurethosSDK.filters.filter import F
from atlant_coder.schemas import ArchUpdate, CoreUpdate
from atlant_coder import config
from atlant_coder.handlers.arch import _validate_made
from atlant_coder.handlers.parallel_batch import _generate_one
from atlant_coder.macroses import write_file_after_generic
from atlant_coder.utils import memory, smoke_check, extract_signatures
from atlant_coder.utils.checkpoint import CheckpointExitError, save
from atlant_coder.macroses.command import run_command
from atlant_coder.utils.project import (
    affected_files,
    current_plan,
    digest,
    file_context,
    install_plan,
    target_path,
)
from atlant_coder.macroses.workspace import ToolBroker, WorkspaceTools, inventory

logger = logging.getLogger(__name__)
coder_router = ABSRouter("CODER")
_control = None


def _save(state, home, reason):
    state["home_path"] = home
    memory.restore(state)
    save(memory.snapshot(), home_dir=home, reason=reason)
    if _control:
        _control.publish(state, reason)


def _abort(state, home, reason):
    state["project_verified"] = False
    state["failure_reason"] = reason
    _save(state, home, reason)
    raise CheckpointExitError(reason)


def _command(command, home, timeout=300, require_tests=False, output=None):
    return run_command(
        command,
        home,
        timeout=timeout,
        require_tests=require_tests,
        output=output,
        control=_control,
    )


def _run_project_unittest(home_path: str) -> str:
    root = Path(home_path)
    if not (root / "tests").is_dir() or not any((root / "tests").rglob("test*.py")):
        return ""
    pyproject = root / "pyproject.toml"
    uses_pytest = pyproject.is_file() and "pytest" in pyproject.read_text(
        encoding="utf-8"
    )
    command = [sys.executable, "-B", "-m"] + (
        ["pytest", "-q", "-p", "no:cacheprovider"]
        if uses_pytest
        else ["unittest", "discover", "-s", "tests", "-v"]
    )
    return _command(command, root, require_tests=True)


def _project_checks(state, home):
    results = []
    for path in state["files"]:
        target = target_path(home, path)
        if not target.is_file():
            return {path: "planned file is missing"}, results
        if path.endswith(".py"):
            errors = smoke_check(str(target), home)
            if errors:
                return {path: "\n".join(errors)}, results
    # A document-only task must not install dependencies or build every audited repository.
    if state["files"] and all(
        Path(p).suffix.lower() in {".md", ".rst", ".txt"} for p in state["files"]
    ):
        for path in state["files"]:
            if not target_path(home, path).read_text(encoding="utf-8").strip():
                return {path: "document is empty"}, results
        return {}, [
            {
                "check": "Документы: наличие и непустое содержимое",
                "status": "PASS",
                "detail": "Проверены только документы; сборка, запуск и тесты описанных проектов не выполнялись.",
            }
        ]
    project_files = state["files"]
    if state.get("agent_mode"):
        project_files = ["/" + p for p in inventory(home)]
    python_tests = any(
        p.startswith("/tests/")
        and Path(p).name.startswith("test")
        and p.endswith(".py")
        for p in project_files
    )
    if python_tests:
        error = _run_project_unittest(home)
        results.append({"check": "python tests", "status": "FAIL" if error else "PASS"})
        if error:
            return {"project": error}, results
    manifests = [p for p in project_files if Path(p).name == "package.json"]
    web_sources = [
        p for p in project_files if Path(p).suffix in {".tsx", ".jsx", ".ts", ".vue"}
    ]
    if web_sources and not manifests:
        return {"project": "typed/component web project has no package.json"}, results
    for manifest in manifests:
        package = target_path(home, manifest)
        data = json.loads(package.read_text())
        scripts = data.get("scripts", {})
        if not (package.parent / "node_modules").is_dir() and (
            data.get("dependencies") or data.get("devDependencies")
        ):
            error = _command(
                ["npm", "install", "--ignore-scripts", "--no-audit", "--no-fund"],
                package.parent,
            )
            results.append(
                {
                    "check": f"{manifest}:dependencies",
                    "status": "UNKNOWN" if error else "PASS",
                }
            )
            if error:
                return {"environment": error}, results
        required = ["build", "test"]
        if any(p.endswith((".ts", ".tsx")) for p in web_sources):
            required.insert(0, "typecheck")
        for script in required:
            if script not in scripts:
                return {manifest: f"missing required package script: {script}"}, results
            error = _command(["npm", "run", script], package.parent)
            results.append(
                {"check": f"{manifest}:{script}", "status": "FAIL" if error else "PASS"}
            )
            if error:
                return {manifest: error}, results
    return {}, results


def _local_repair(state, paths, reason):
    """Implementation repairs do not invalidate unchanged consumer contracts."""
    selected = set(paths)
    if not selected or not selected <= set(state["files"]):
        raise ValueError("local repair requires existing planned paths")
    state["completed_files"] = [
        p for p in state.get("completed_files", []) if p not in selected
    ]
    state["project_verified"] = False
    state.pop("validation_results", None)
    for path in selected:
        state["last_error_" + path] = reason


def _review_event(run, operation, phase, issues=None, trial=None):
    run["review_context"] = {"phase": phase}
    if trial is None:
        run.pop("review_state", None)
    else:
        run["review_state"] = trial
    return CoreUpdate(
        to_model="CORE",
        from_model="CODER",
        command=operation,
        parameters={"issues": issues or {}, "review_context": run["review_context"]},
    )


def _commit_steer(run):
    state = run["state"]
    state["contract_revision"] += 1
    state["user_revision"] = run.pop("steer_messages")[-1]["revision"]
    run["control"].applied_revision = state["user_revision"]
    _save(state, run["home"], "user_clarification_applied")


def _invalidate(state, paths):
    affected = affected_files(paths, state["file_deps"])
    if affected:
        # Checks describe the previous source revision, not the proposed repair.
        state.pop("validation_results", None)
        state.pop("project_test_failure", None)
    state["completed_files"] = [
        p for p in state.get("completed_files", []) if p not in affected
    ]
    for path in affected:
        for prefix in ("code_", "signatures_"):
            state.pop(prefix + path, None)
        for key in ("file_hashes", "file_summaries"):
            state.setdefault(key, {}).pop(path, None)
    state["project_verified"] = False
    return affected


def _restore_completed(state, home):
    invalid = set()
    state.setdefault("file_hashes", {})
    state.setdefault("file_summaries", {})
    for path in state.get("completed_files", []):
        if path not in state["files"]:
            invalid.add(path)
            continue
        target = target_path(home, path)
        if not target.is_file():
            invalid.add(path)
            continue
        code = target.read_text(encoding="utf-8")
        previous = state["file_hashes"].get(path)
        if previous and previous != digest(code):
            invalid.add(path)
            continue
        if smoke_check(str(target), home):
            invalid.add(path)
            continue
        state["file_hashes"][path] = digest(code)
        state["code_" + path] = code
        state["signatures_" + path] = (
            extract_signatures(code) if path.endswith(".py") else []
        )
        state["file_summaries"].setdefault(
            path,
            {
                "behavior": "Recovered from an older checkpoint; semantic review required.",
                "exports": state["file_public_names"].get(path, []),
                "issues": [],
            },
        )
    _invalidate(state, invalid)


@coder_router.handler(F.command == "REPLAN_REQUIRED")
def replan_required(run, decision, review_context=None):
    state, home = run["state"], run["home"]
    phase = (review_context or run.get("review_context", {})).get("phase")
    revision = run.get("revision")
    if revision is None:
        old = current_plan(state)
        revision = run["revision"] = {
            "old": old,
            "candidate": old,
            "paths": set(),
            "steer": phase == "steer",
        }
    if revision["steer"] and phase == "revision":
        _abort(state, home, "updated user plan was not approved: " + decision["reason"])
    revision["decision"] = decision
    revision["scope_attempts"] = 0
    if revision["steer"]:
        prompt = "MAKE updated complete plan for this user clarification. Preserve unrelated entries exactly. New paths are allowed; removing a path from the plan does not delete it from disk.\n"
        payload = {
            "source_task": state["source_task"],
            "architecture": revision["old"],
            "decision": decision,
        }
    else:
        state.setdefault("plan_decisions", []).append(copy.deepcopy(decision))
        rework = affected_files(decision["files"], state["file_deps"]) & set(
            state.get("completed_files", [])
        )
        if rework:
            state.setdefault("rework_requests", []).append(
                {"files": sorted(rework), "reason": decision["reason"]}
            )
        if not state.get("contract_approved"):
            used = state.get("initial_plan_corrections", 0)
            if used >= config.max_initial_plan_corrections:
                _abort(state, home, "initial plan correction limit reached")
            state["initial_plan_corrections"] = used + 1
        else:
            for path in set(decision["files"]):
                used = state.get("replans_" + path, 0)
                if used >= config.max_replans:
                    _abort(state, home, f"replan limit reached: {path}")
                state["replans_" + path] = used + 1
        _save(state, home, "replan_requested")
        revision["paths"].update(decision["files"])
        revision["allowed"] = affected_files(revision["paths"], state["file_deps"])
        payload = {
            "source_task": state.get("source_task") or state.get("tech_task"),
            "architecture": revision["candidate"],
            "decision": decision,
            "summaries": state.get("file_summaries", {}),
            "editable_files": sorted(revision["allowed"]),
        }
        prompt = (
            "REPLAN: implement the CORE decision, not revise it. Return the COMPLETE MADE. "
            "Preserve every path and copy entries outside editable_files exactly unchanged. "
            "Within editable entries, preserve unrelated requirements and steps verbatim; "
            "change only clauses necessary for the CORE decision. Do not summarize or rewrite away existing constraints.\n"
        )
    return ArchUpdate(
        to_model="ARCH",
        from_model="CODER",
        command="REVISE",
        parameters={
            "prompt": prompt + json.dumps(payload, ensure_ascii=False),
            "reset": True,
        },
    )


@coder_router.handler(F.command == "REPLAN_CANDIDATE")
def replan_candidate(run, candidate):
    state, home = run["state"], run["home"]
    revision = run["revision"]
    old = revision["old"]
    state["pending_architecture"] = copy.deepcopy(candidate)
    _save(state, home, "candidate_plan_ready")
    if not revision["steer"] and set(candidate["files"]) != set(old["files"]):
        _abort(state, home, "REPLAN must preserve the approved set of project paths")
    changed = {
        p
        for p in set(old["files"]) | set(candidate["files"])
        if old["files"].get(p) != candidate["files"].get(p)
    }
    if not revision["steer"] and changed - revision["allowed"]:
        unrelated = sorted(changed - revision["allowed"])
        if revision["scope_attempts"]:
            _abort(state, home, f"REPLAN changed unrelated files: {unrelated}")
        revision["scope_attempts"] += 1
        return ArchUpdate(
            to_model="ARCH",
            from_model="CODER",
            command="REVISE",
            parameters={
                "reset": False,
                "prompt": "REPLAN scope violation: restore these entries EXACTLY from the original architecture: "
                + str(unrelated)
                + ". Return the complete MADE, changing only editable_files as instructed by CORE.",
            },
        )
    revision["candidate"] = candidate
    if candidate["conventions"] != old["conventions"]:
        changed.update(candidate["files"])
    trial = copy.deepcopy(state)
    install_plan(trial, candidate)
    invalid = _invalidate(trial, changed | revision["paths"])
    if revision["steer"]:
        trial["completed_files"] = [
            p for p in trial["completed_files"] if p in trial["files"]
        ]
        trial.setdefault("user_plan_changes", []).append(
            {"files": sorted(changed), "reason": revision["decision"]["reason"]}
        )
    else:
        trial["contract_revision"] = state["contract_revision"] + 1
    revision.update(trial=trial, invalid=invalid)
    return _review_event(run, "REVIEW_PLAN", "revision", trial=trial)


@coder_router.handler(F.command == "PLAN_APPROVED")
def plan_approved(run, decision, review_context=None):
    state, home = run["state"], run["home"]
    revision = run.pop("revision", None)
    if revision:
        state.clear()
        state.update(revision["trial"])
        state["contract_approved"] = True
        state.pop("pending_architecture", None)
        if revision["steer"]:
            _commit_steer(run)
        else:
            for path in revision["invalid"]:
                state["last_error_" + path] = revision["decision"]["reason"]
            _save(state, home, "contract_revised")
    state["contract_approved"] = True
    run.pop("review_state", None)
    _save(state, home, "contract_approved")
    return _advance_generation(run)


@coder_router.handler(F.command == "STEER_APPROVED")
def steer_approved(run, decision, review_context=None):
    _local_repair(run["state"], decision.get("files", []), decision["reason"])
    _commit_steer(run)
    return _advance_generation(run)


@coder_router.handler(F.command == "RESOLVE_APPROVED")
def resolve_approved(run, decision, review_context=None):
    state, home = run["state"], run["home"]
    if (review_context or run["review_context"])["phase"] == "checks":
        if (
            not decision.get("files")
            or state.get("post_test_repairs", 0) >= config.max_post_test_repairs
        ):
            _abort(state, home, "project_tests_failed")
        state["post_test_repairs"] = state.get("post_test_repairs", 0) + 1
        _local_repair(
            state,
            decision["files"],
            state["project_test_failure"] + "\nCORE: " + decision["reason"],
        )
        _save(state, home, "post_test_repair")
    else:
        for path, reason in run["conflicts"].items():
            state["last_error_" + path] = reason + "\nCORE: " + decision["reason"]
    return _advance_generation(run)


@coder_router.handler(F.command == "FINAL_APPROVED")
def final_approved(run, decision, review_context=None):
    run["final_approved"] = True
    if isinstance(decision.get("reason"), str) and decision["reason"].strip():
        run["state"]["result_summary"] = decision["reason"].strip()
    return _advance_generation(run)


@coder_router.handler(F.command == "STOP_REQUESTED")
def stop_requested(run, decision, review_context=None):
    phase = (review_context or run.get("review_context", {})).get("phase")
    reason = decision["reason"]
    if phase == "checks":
        reason = "project_tests_failed"
    elif phase == "revision":
        reason = (
            "updated user plan was not approved: "
            if run["revision"]["steer"]
            else "revised contract rejected: "
        ) + reason
    _abort(run["state"], run["home"], reason)


def _accept(state, home, result):
    path = result["path"]
    previous_draft = copy.deepcopy(state.get("file_drafts", {}).get(path))
    metric = {
        k: result[k]
        for k in (
            "path",
            "status",
            "revision",
            "duration_s",
            "queue_wait_s",
            "requests",
            "draft_reviews",
        )
        if k in result
    }
    state.setdefault("generation_metrics", []).append(metric)
    if result["revision"] != state["contract_revision"]:
        raise ValueError("refusing a result from a stale contract revision")
    if result.get("draft"):
        state.setdefault("file_drafts", {})[path] = {
            **copy.deepcopy(result["draft"]),
            "revision": result["revision"],
            "base_hash": result.get("base_hash"),
            "dependency_hashes": copy.deepcopy(result.get("dependency_hashes", {})),
        }
    if result["status"] == "transport_error":
        state["last_error_" + path] = result["reason"]
        return "fatal", f"transport failed for {path}: {result['reason']}"
    if result["status"] == "ok":
        changed_dependencies = []
        for dep, expected in result.get("dependency_hashes", {}).items():
            source = target_path(home, dep)
            if (
                not source.is_file()
                or digest(source.read_text(encoding="utf-8")) != expected
            ):
                changed_dependencies.append(dep)
        if changed_dependencies:
            _invalidate(state, changed_dependencies)
            return (
                "fatal",
                "dependencies changed during generation; resume to reread: "
                + ", ".join(changed_dependencies),
            )
        target = target_path(home, path)
        old = target.read_bytes() if target.exists() else None
        actual_hash = digest(old.decode("utf-8")) if old is not None else None
        if "base_hash" in result and result["base_hash"] != actual_hash:
            state["concurrent_edits"] = state.get("concurrent_edits", 0) + 1
            state["last_error_" + path] = (
                "User changed the file during generation. Read the current source and preserve those edits."
            )
            _save(state, home, "concurrent_edit:" + path)
            if state["concurrent_edits"] >= 3:
                return "fatal", "workspace repeatedly changed during generation"
            return "", ""
        published = False
        try:
            guard = getattr(_control, "publication", nullcontext)
            with guard() as permitted:
                if permitted is False:
                    state["generation_metrics"].pop()
                    if previous_draft is None:
                        state.get("file_drafts", {}).pop(path, None)
                    else:
                        state.setdefault("file_drafts", {})[path] = previous_draft
                    return "discarded", "task revision changed before publication"
                write_file_after_generic(target, result["code"])
                published = True
            errors = smoke_check(str(target), home)
            if (
                not target.exists()
                or target.read_text(encoding="utf-8") != result["code"]
            ):
                return (
                    "fatal",
                    f"user changed {path} during validation; preserving their file",
                )
            if errors:
                raise ValueError("\n".join(errors))
        except InterruptedError:
            raise
        except Exception as exc:
            if published and (
                not target.exists()
                or target.read_text(encoding="utf-8") != result["code"]
            ):
                return (
                    "fatal",
                    f"user edited {path} during validation; preserving their file",
                )
            if published and old is None:
                target.unlink(missing_ok=True)
            elif published:
                write_file_after_generic(target, old.decode("utf-8"))
            result = {**result, "status": "error", "reason": str(exc)}
            metric["status"] = "error"
        else:
            previous = (old or b"").decode("utf-8")
            change = {
                "path": path,
                "diff": "".join(
                    difflib.unified_diff(
                        previous.splitlines(True),
                        result["code"].splitlines(True),
                        fromfile=path,
                        tofile=path,
                    )
                ),
            }
            changes = state.setdefault("changes", [])
            changes.append(change)
            state["code_" + path] = result["code"]
            state["signatures_" + path] = result["signatures"]
            state["file_summaries"][path] = {
                **result["summary"],
                "revision": result["revision"],
            }
            state["file_hashes"][path] = digest(result["code"])
            if path not in state["completed_files"]:
                state["completed_files"].append(path)
            state["last_error_" + path] = ""
            state.get("file_drafts", {}).pop(path, None)
            state["project_verified"] = False
            _save(state, home, f"file_completed:{path}")
            return "", ""
    attempts = state.get("attempts_" + path, 0) + 1
    state["attempts_" + path] = attempts
    state["last_error_" + path] = result.get("reason", "generation failed")
    _save(state, home, f"file_failed:{path}")
    if attempts >= (config.max_make_attempts or 3):
        state.update(repair_failure_file=path, repair_failure_attempts=attempts)
        return "fatal", f"file_repair_limit:{path}"
    if result["status"] == "conflict" or attempts >= config.replan_after_attempts:
        return "resolve", state["last_error_" + path]
    return "", ""


def _generation_steps(run, base_api):
    global _control
    control = run.get("control")
    _control = control
    home, state = run["home"], run["state"]
    install_plan(state, current_plan(state))
    errors = _validate_made(state["architecture"])
    if errors:
        _abort(state, home, "; ".join(errors))
    state.setdefault("contract_revision", 1)
    state.setdefault("completed_files", [])
    _restore_completed(state, home)
    core = run["dispatcher"].routers["CORE"].model
    architect = run["dispatcher"].routers["ARCH"].model
    thinker = run["dispatcher"].routers["CODER_THINK"].model
    started = time.monotonic()
    limit = max(
        1,
        (
            run["workers"]
            if run.get("workers") is not None
            else config.max_parallel_workers
        ),
    )
    ready_since = {}
    broker = ToolBroker(WorkspaceTools(home, control))
    _save(state, home, "coordinator_started")
    try:
        if not state.get("contract_approved"):
            yield _review_event(run, "REVIEW_PLAN", "initial")
        conflicts = run["conflicts"] = {}
        fatal = ""
        running = {}
        with ThreadPoolExecutor(max_workers=limit) as pool, closing(broker):
            while True:
                pending_control = control is not None and control.pending()
                broker.drain(cancelled=pending_control)
                if not running and pending_control:
                    messages = control.messages()
                    if messages:
                        for item in messages:
                            state["source_task"] += (
                                "\n\nUser clarification:\n" + item["message"]
                            )
                        state["tech_task"] = state["source_task"]
                        run["steer_messages"] = messages
                        yield _review_event(
                            run,
                            "STEER",
                            "steer",
                            {"user": "\n".join(m["message"] for m in messages)},
                        )
                    conflicts.clear()
                    fatal = ""
                    continue
                completed = set(state["completed_files"])
                active = set(running.values())
                pending = [
                    p for p in state["files"] if p not in completed and p not in active
                ]
                exhausted = [
                    p
                    for p in pending
                    if state.get("attempts_" + p, 0) >= (config.max_make_attempts or 3)
                ]
                if exhausted:
                    fatal = f"file_repair_limit:{exhausted[0]}"
                if not running and fatal:
                    _abort(state, home, fatal)
                if not running and conflicts:
                    yield _review_event(run, "RESOLVE", "conflict", conflicts)
                    conflicts.clear()
                    _save(state, home, "conflict_resolved")
                    continue
                if not running and not pending:
                    failures, checks = _project_checks(state, home)
                    state["validation_results"] = checks
                    if control and control.pending():
                        continue
                    if failures:
                        state["project_test_failure"] = "\n".join(failures.values())
                        if "environment" in failures:
                            _abort(
                                state,
                                home,
                                "validation environment unavailable: "
                                + failures["environment"],
                            )
                        yield _review_event(run, "RESOLVE", "checks", failures)
                        continue
                    run["final_approved"] = False
                    yield _review_event(run, "FINALIZE", "final")
                    if not run["final_approved"] or (control and control.pending()):
                        continue
                    if not checks:
                        state["project_verified"] = False
                        state["validation_results"] = [
                            {
                                "check": "independent project checks",
                                "status": "UNKNOWN",
                                "detail": "No runnable tests/build checks discovered",
                            }
                        ]
                        _save(state, home, "unverified_no_project_checks")
                        return state
                    state["project_verified"] = True
                    state.pop("failure_reason", None)
                    state.pop("project_test_failure", None)
                    _save(state, home, "project_verified")
                    return state
                if not conflicts and not fatal and not pending_control:
                    ready = [
                        p
                        for p in pending
                        if set(state["file_deps"].get(p, [])) <= completed
                    ]
                    for path in ready:
                        ready_since.setdefault(path, time.monotonic())
                    safe = set(state["parallel_safe_files"])
                    ready.sort(
                        key=lambda p: (
                            p in safe,
                            -sum(p in deps for deps in state["file_deps"].values()),
                        )
                    )
                    # Contract owners run exclusively. Ordinary ready work is
                    # replenished after each completed future, without layer barriers.
                    exclusive_running = any(p not in safe for p in active)
                    candidates = []
                    if not running and ready and ready[0] not in safe:
                        candidates = ready[:1]
                    elif not exclusive_running:
                        candidates = [p for p in ready if p in safe][
                            : limit - len(running)
                        ]
                    for path in candidates:
                        context = file_context(state, path)
                        context["queue_wait_s"] = time.monotonic() - ready_since.pop(
                            path
                        )
                        if state.get("agent_mode"):
                            context["broker"] = broker
                        future = pool.submit(
                            _generate_one, path, context, base_api, thinker
                        )
                        running[future] = path
                    if candidates:
                        state["active_agents"] = [
                            {"role": "CODER", "file": p, "status": "running"}
                            for p in running.values()
                        ]
                        _save(state, home, "generating")
                if not running:
                    _abort(
                        state, home, "dependency queue blocked: " + ", ".join(pending)
                    )
                done, _ = wait(running, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    path = running.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "path": path,
                            "revision": state["contract_revision"],
                            "status": "error",
                            "reason": str(exc),
                        }
                    if control and control.pending():
                        state.setdefault("generation_metrics", []).append(
                            {
                                "path": path,
                                "status": "discarded",
                                "revision": result.get("revision"),
                                "requests": result.get("requests", []),
                                "duration_s": result.get("duration_s"),
                            }
                        )
                        control.event(
                            "discarded",
                            f"Результат {path} отменён из-за изменения задачи",
                        )
                        continue
                    action, reason = _accept(state, home, result)
                    if action == "discarded":
                        state.setdefault("generation_metrics", []).append(
                            {
                                "path": path,
                                "status": "discarded",
                                "revision": result.get("revision"),
                                "requests": result.get("requests", []),
                                "duration_s": result.get("duration_s"),
                            }
                        )
                        if control:
                            control.event(
                                "discarded",
                                f"Результат {path} отменён из-за изменения задачи",
                            )
                    elif action == "fatal":
                        fatal = reason
                    elif action == "resolve":
                        conflicts[path] = reason
    except BaseException:
        state["project_verified"] = False
        _save(state, home, state.get("failure_reason") or "coordinator_interrupted")
        raise
    finally:
        broker.drain(cancelled=True)
        state["run_duration_s"] = time.monotonic() - started
        state["coordination_requests"] = (
            state.get("planning_requests", [])
            + list(getattr(core, "request_metrics", []))
            + list(getattr(architect, "request_metrics", []))
        )
        memory.restore(state)
        report = {
            k: state.get(k)
            for k in (
                "project_verified",
                "failure_reason",
                "run_duration_s",
                "generation_metrics",
                "initial_plan_corrections",
                "rework_requests",
                "coordination_requests",
                "validation_results",
                "post_test_repairs",
                "plan_decisions",
            )
        }
        write_file_after_generic(
            Path(os.environ.get("ATLANT_CHECKPOINT_DIR", home))
            / ".atlant_metrics.json",
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        if control:
            control.publish(state, "finished")
        _control = None


def _advance_generation(run):
    scheduler = run.get("scheduler")
    if scheduler is None:
        run["result"] = run["state"]
        return None
    try:
        return next(scheduler)
    except StopIteration as finished:
        run["result"] = finished.value
        run["scheduler"] = None
        return None


def close_generation(run):
    global _control
    scheduler = run.get("scheduler")
    try:
        if scheduler is not None:
            scheduler.close()
    finally:
        run["scheduler"] = None
        _control = None


@coder_router.handler(F.command == "GENERATE")
def generate_handler(model, run):
    if run.get("scheduler") is not None:
        raise RuntimeError("generation is already active")
    run["scheduler"] = _generation_steps(run, model)
    return _advance_generation(run)
