"""ARCH plan commands and corrections dispatched through the SDK."""

import json
import logging

from AurethosSDK.filters.filter import F
from atlant_coder import config as _cfg
from atlant_coder.handlers.planning import (
    arch_router,
    begin_request,
    before_request,
    response_update,
    transition,
)
from atlant_coder.macroses.workspace import inventory, read_text
from atlant_coder.schemas.protocol import decode_response
from atlant_coder.utils.checkpoint import save
from atlant_coder.utils.memory import memory
from atlant_coder.utils.project import target_path, install_plan
from atlant_coder.utils.prompt_rules import build_rules

logger = logging.getLogger(__name__)


def _validate_made(res: dict) -> list[str]:
    """Reject ambiguous ARCH contracts before six coders amplify them."""
    if not isinstance(res, dict):
        return ["MADE response must be an object"]
    errors: list[str] = []
    extra = set(res) - {"command", "conventions", "files"}
    if res.get("command") != "MADE" or extra:
        errors.append(f"invalid MADE envelope; extra keys: {sorted(extra)}")
    files = res.get("files")
    if not isinstance(res.get("conventions"), str) or not res["conventions"].strip():
        errors.append("conventions must be a non-empty project contract")
    if not isinstance(files, dict) or not files:
        return errors + ["files must be a non-empty object"]
    known = set(files)
    if any(not isinstance(path, str) for path in known):
        return errors + ["file paths must be strings"]
    source_extensions = (".py", ".js", ".jsx", ".ts", ".tsx", ".vue", ".html")
    has_tests = any(path.startswith("/tests/") for path in known)
    has_production = any(
        path.endswith(source_extensions)
        and not path.startswith(("/tests/", "/alembic/", "/migrations/"))
        for path in known
    )
    if has_tests and not has_production and not memory.get("agent_mode"):
        errors.append("plan includes tests but no production Python or web files")
    for path, entry in files.items():
        try:
            target_path(memory.get("home_path") or ".", path)
        except ValueError as exc:
            errors.append(str(exc))
        prefixes = getattr(_cfg, "allowed_path_prefixes", None)
        if prefixes and not path.startswith(tuple(prefixes)):
            errors.append(f"forbidden project path: {path}")
        if not isinstance(entry, dict):
            errors.append(f"{path}: entry must be an object")
            continue
        if not str(entry.get("desc") or "").strip():
            errors.append(f"{path}: missing desc")
        priority = entry.get("priority")
        if priority not in {"high", "low"}:
            errors.append(f"{path}: priority must be high or low")
        deps = entry.get("depends_on")
        if not isinstance(deps, list):
            errors.append(f"{path}: depends_on must be a list")
        else:
            unknown = [
                dep
                for dep in deps
                if not isinstance(dep, str) or dep not in known or dep == path
            ]
            if unknown:
                errors.append(f"{path}: invalid dependencies {unknown}")
        if not isinstance(entry.get("parallel_safe"), bool):
            errors.append(f"{path}: parallel_safe must be boolean")
        if priority == "high":
            if not isinstance(entry.get("plan"), list) or not entry["plan"]:
                errors.append(f"{path}: high priority requires a non-empty plan")
            if not isinstance(entry.get("public_names"), list):
                errors.append(f"{path}: high priority requires public_names")
        elif entry.get("plan") or entry.get("public_names"):
            errors.append(f"{path}: low priority cannot define plan/public_names")
        for field in ("plan", "public_names"):
            values = entry.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(v, str) or not v.strip() for v in values
            ):
                errors.append(f"{path}: {field} must contain nonempty strings")
    unresolved = {
        path: {dep for dep in entry.get("depends_on", []) if isinstance(dep, str)}
        for path, entry in files.items()
        if isinstance(entry, dict) and isinstance(entry.get("depends_on"), list)
    }
    while unresolved:
        ready = {
            path for path, deps in unresolved.items() if not (deps & unresolved.keys())
        }
        if not ready:
            errors.append(f"dependency cycle: {sorted(unresolved)}")
            break
        for path in ready:
            unresolved.pop(path)
    return errors


def repository_context(home):
    paths = inventory(home)
    evidence = {}
    for path in paths:
        if path in {
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "Cargo.toml",
            "go.mod",
        } or path.endswith("/AGENTS.md"):
            try:
                evidence[path] = read_text(home, path, limit=30_000)
            except (ValueError, UnicodeError, OSError) as exc:
                evidence[path] = str(exc)
    return dict(paths=paths[:1500], truncated=len(paths) > 1500, documents=evidence)


@arch_router.handler(F.command == "PLAN")
def plan_handler(run):
    state, control = run["state"], run.get("control")
    if control:
        state["active_agents"] = [
            {
                "role": "ARCH",
                "status": "planning",
                "task": "Изучение проекта и планирование",
            }
        ]
        control.publish(state, "planning")
        control.event("stage", "ARCH: изучение проекта и планирование")
    prompt = (
        "MAKE\nCreate the smallest complete file plan for the exact user task. "
        "For an existing repository, list only files needing edits/new files. Read relevant source and AGENTS.md "
        "before planning; preserve unrelated user code and interfaces. depends_on references only other planned files; "
        "existing unmodified dependencies are available through READ. Do not reimplement the whole repository.\n"
        + json.dumps(
            {
                "source_task": state["source_task"],
                "tech_task": state["tech_task"],
                "repository": (
                    repository_context(run["home"]) if state.get("agent_mode") else {}
                ),
                "rules": build_rules(
                    state["source_task"], "ARCH", mode=state.get("policy_mode", "full")
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return begin_request(run, "ARCH", prompt, ["MADE"], kind="plan")


@arch_router.handler(F.command == "REVISE")
def revise_handler(model, run, prompt, reset=True):
    if reset:
        model.reset_context()
    return begin_request(run, "ARCH", prompt, ["MADE"], kind="revision")


@arch_router.handler(F.command == "REQUEST")
def request_handler(model, run):
    pending = before_request(run)
    prompt = pending["prompt"]
    if pending["requests"] == 1:
        prompt += (
            "\nWire format: MADE.files is an array of objects with path, desc, priority, depends_on, "
            "parallel_safe, plan and public_names. Low priority uses empty plan/public_names arrays."
        )
    response = model.send_api_request(
        prompt, expects=pending["commands"], response_schema=pending["schema"]
    )
    return response_update(run, response)


@arch_router.handler(F.command == "MADE")
def made_handler(model, run, response):
    pending = run["planning"]
    # Native validation happens in the SDK; the application owns its wire adapter.
    candidate = (
        decode_response(response, pending["schema"])
        if isinstance(response.get("files"), list)
        else response
    )
    errors = _validate_made(candidate)
    if errors:
        if pending["corrections"] >= 1:
            raise ValueError("ARCH returned invalid MADE twice: " + "; ".join(errors))
        pending["corrections"] += 1
        pending["prompt"] = (
            "ANSW Correct the MADE protocol errors and return the complete plan: "
            + "; ".join(errors)
        )
        return transition("ARCH", "REQUEST")
    if pending["kind"] == "revision":
        return transition(
            "CODER", "REPLAN_CANDIDATE", sender="ARCH", candidate=candidate
        )
    state = run["state"]
    install_plan(state, candidate)
    state["planning_requests"] = list(model.request_metrics)
    state.update(contract_approved=False, contract_revision=1, completed_files=[])
    memory.restore(state)
    save(state, home_dir=run["home"], reason="arch_made")
    if run.get("control"):
        run["control"].publish(state, "plan_ready")
    return transition("CODER", "GENERATE", sender="ARCH")
