"""Shared planning state and SDK tool handlers for the real CORE/ARCH roles."""

import json

from AurethosSDK.client.router import ABSRouter
from AurethosSDK.filters.filter import F
from atlant_coder.schemas import ArchUpdate, CoreUpdate, CoderUpdate
from atlant_coder.macroses.workspace import WorkspaceTools
from atlant_coder.schemas.protocol import response_schema
from atlant_coder.utils.checkpoint import CheckpointExitError

core_router = ABSRouter("CORE")
arch_router = ABSRouter("ARCH")


def transition(role, command, *, sender=None, **parameters):
    update_type = {"CORE": CoreUpdate, "ARCH": ArchUpdate, "CODER": CoderUpdate}[role]
    return update_type(
        to_model=role, from_model=sender or role, command=command, parameters=parameters
    )


def begin_request(
    run, role, prompt, commands, *, kind, operation="", sections=None, payload=None
):
    state = run.get("review_state", run["state"])
    allowed = list(commands)
    if state.get("agent_mode") and kind != "clarify":
        allowed += ["READ", "LIST", "SEARCH"]
        contract = (
            f"Use READ(path), LIST(path), SEARCH(path,pattern) relative to USER PROJECT ROOT {run['home']!r}. "
            "LIST('') lists project files. Native working-directory paths are not project paths."
        )
        if payload is not None:
            payload["tool_contract"] = contract
            prompt = (
                operation
                + "\n"
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )
        else:
            prompt += "\n" + contract
    run["planning"] = dict(
        role=role,
        prompt=prompt,
        commands=allowed,
        kind=kind,
        operation=operation,
        sections=sections or {},
        payload=payload,
        corrections=0,
        requests=0,
        schema=response_schema(
            role,
            allowed,
            coverage=operation == "REVIEW_PLAN",
            repair=operation in {"STEER", "RESOLVE"},
        ),
    )
    return transition(role, "REQUEST")


def before_request(run):
    pending = run["planning"]
    if pending["requests"] >= 20:
        raise CheckpointExitError("planning tool limit reached")
    if run.get("control"):
        run["control"].check_stopped()
    pending["requests"] += 1
    return pending


def response_update(run, response):
    pending = run["planning"]
    if run.get("control"):
        run["control"].check_stopped()
    if (
        not isinstance(response, dict)
        or response.get("command") not in pending["commands"]
    ):
        raise ValueError("unexpected planning command")
    return transition(pending["role"], response["command"], response=response)


def tool_feedback(run, response, result):
    if run.get("control"):
        run["control"].event("tool", response["command"], response)
    run["planning"]["prompt"] = "TOOL RESULT\n" + json.dumps(
        result, ensure_ascii=False, separators=(",", ":")
    )
    return transition(run["planning"]["role"], "REQUEST")


def tool_error(exc):
    return {"error": str(exc), "hint": "Use project-relative paths from LIST('')."}


@core_router.handler(F.command == "READ")
@arch_router.handler(F.command == "READ")
def read_handler(run, response):
    try:
        result = WorkspaceTools(run["home"], run.get("control")).read(response)
    except InterruptedError:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        result = tool_error(exc)
    return tool_feedback(run, response, result)


@core_router.handler(F.command == "LIST")
@arch_router.handler(F.command == "LIST")
def list_handler(run, response):
    try:
        result = WorkspaceTools(run["home"], run.get("control")).list(response)
    except InterruptedError:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        result = tool_error(exc)
    return tool_feedback(run, response, result)


@core_router.handler(F.command == "SEARCH")
@arch_router.handler(F.command == "SEARCH")
def search_handler(run, response):
    try:
        result = WorkspaceTools(run["home"], run.get("control")).search(response)
    except InterruptedError:
        raise
    except (OSError, ValueError, UnicodeError) as exc:
        result = tool_error(exc)
    return tool_feedback(run, response, result)
