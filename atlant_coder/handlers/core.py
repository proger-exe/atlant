"""CORE decisions and clarification commands routed by the SDK dispatcher."""

import json
import re
from pathlib import Path

from AurethosSDK.filters.filter import F
from AurethosSDK.models.api_router import APIRouter
from AurethosSDK.models.base import BaseModel
from AurethosSDK.utils.model_parser import model_parser
from atlant_coder import MODELS_PATH
from atlant_coder.config import ask_loops_count, max_ask_rounds
from atlant_coder.handlers.planning import (
    core_router as manager,
    transition,
    begin_request,
    before_request,
    response_update,
)
from atlant_coder.schemas.protocol import response_schema
from atlant_coder.utils.memory import memory
from atlant_coder.utils.project import current_plan, file_context
from atlant_coder.utils.prompt_rules import build_rules
from atlant_coder.utils.tui import wait_user_enter


def role_api(name, home):
    return APIRouter(
        BaseModel(model_parser(str(MODELS_PATH))[name]),
        json_mode=True,
        generation_read_only=True,
        workspace=str(Path(home).resolve()),
    )


@manager.handler(F.command == "START")
def start_handler(run):
    if run["state"].get("files"):
        return transition("CODER", "GENERATE", sender="CORE")
    if ask_loops_count <= 0:
        return transition("ARCH", "PLAN", sender="CORE")
    run["clarification_rounds"] = 0
    return begin_request(
        run,
        "CORE",
        "TASK " + run["state"]["source_task"],
        ["OK", "ASK", "MADE"],
        kind="clarify",
    )


@manager.handler(F.command == "REQUEST")
def request_handler(model, run):
    pending = before_request(run)
    response = model.send_api_request(
        pending["prompt"],
        expects=pending["commands"],
        response_schema=pending["schema"],
    )
    return response_update(run, response)


@manager.handler(F.command == "ASK")
def ask_handler(run, response):
    run["clarification_rounds"] += 1
    if run.get("control"):
        ask = getattr(run["control"], "ask", None)
        if ask is None:
            raise RuntimeError(
                "interactive clarification unavailable in this background run"
            )
        answer = ask(response["data"])
    else:
        answer = wait_user_enter(response["data"])
    pending = run["planning"]
    pending["prompt"] = str(answer)
    if run["clarification_rounds"] >= max_ask_rounds:
        pending[
            "prompt"
        ] += "\nNo more questions. Return the complete MADE specification."
        pending["commands"] = ["MADE"]
        pending["schema"] = response_schema("CORE", ["MADE"])
    return transition("CORE", "REQUEST")


@manager.handler(F.command == "OK")
def ok_handler(run):
    run["planning"].update(
        prompt="MAKE the complete specification.",
        commands=["MADE"],
        schema=response_schema("CORE", ["MADE"]),
    )
    return transition("CORE", "REQUEST")


@manager.handler(F.command == "MADE")
def specification_handler(run, response):
    specification = response.get("data")
    if not isinstance(specification, str) or not specification.strip():
        raise ValueError("CORE must produce MADE with a nonempty specification")
    run["state"]["tech_task"] = specification
    memory.restore(run["state"])
    return transition("ARCH", "PLAN", sender="CORE")


def review_payload(state, operation, issues=None):
    """Source facts and unchanged coverage requirements for the CORE review."""
    if operation not in {"REVIEW_PLAN", "RESOLVE", "FINALIZE", "STEER"}:
        raise ValueError(f"invalid CORE operation: {operation}")
    evidence = {}
    for path in state.get("completed_files", []):
        item = {
            "signatures": state.get(f"signatures_{path}", []),
            "sha256": state.get("file_hashes", {}).get(path),
        }
        if (
            path in (issues or {})
            or not state["architecture"]["files"][path]["parallel_safe"]
        ):
            code = state.get(f"code_{path}", "")
            item["source_excerpt"] = code[:6000]
            item["excerpt_truncated"] = len(code) > 6000
        evidence[path] = item
    payload = {
        "source_task": state.get("source_task") or state.get("tech_task", ""),
        "tech_task": state.get("tech_task", ""),
        "revision": state.get("contract_revision", 0),
        "architecture": current_plan(state),
        "summaries": state.get("file_summaries", {}),
        "evidence": evidence,
        "issues": issues or {},
        "checks": state.get("validation_results", []),
    }
    if operation == "RESOLVE":
        payload["unpublished_drafts"] = {}
        for path in issues or {}:
            try:
                draft = file_context(state, path).get("previous_draft")
            except (KeyError, ValueError):
                continue
            if draft:
                payload["unpublished_drafts"][path] = {
                    "summary": draft["summary"],
                    "source_excerpt": draft["code"][:12000],
                    "excerpt_truncated": len(draft["code"]) > 12000,
                }

    payload["rules"] = build_rules(
        payload["source_task"], "CORE", mode=state.get("policy_mode", "full")
    )
    if payload["source_task"] == payload["tech_task"]:
        payload.pop("tech_task")
    if operation == "RESOLVE":
        payload["repair_contract"] = (
            "If interfaces and architecture are correct, return APPROVE with nonempty files naming the smallest "
            "set of implementation files to repair and precise reason. Do not weaken tests to hide a failure. "
            "Use REPLAN only for a real contradiction of contracts; STOP if unable to identify a repair."
        )
    if operation == "STEER":
        payload["steering_contract"] = (
            "User clarified the task. APPROVE with nonempty files only if current contracts remain valid and local "
            "implementation edits suffice. Otherwise REPLAN with affected files and precise contract changes. "
            "New files require REPLAN. Never ignore new requirements."
        )
    sections = {}
    if operation == "REVIEW_PLAN":
        sections = {
            str(i): text.strip()
            for i, text in enumerate(
                re.split(r"\n\s*\n", payload["source_task"].strip()), 1
            )
            if text.strip()
        }
        payload["requirement_sections"] = sections
        payload["review_contract"] = (
            "Audit EVERY clause of EVERY requirement section and EVERY producer/consumer edge, "
            "including negative constraints, dependency versions, exact event names/payloads, "
            "startup/new-instance/reconnect/cleanup lifecycle and validation hooks. "
            "Do not approve alternatives such as 'event or method'; both sides must agree on one interface. "
            "Return coverage: [{section: string, status: 'covered'|'gap', files: string[], evidence: string}] "
            "with exactly one row per requirement_sections key. evidence must give concrete contract facts "
            "or precise missing clauses, not a generic claim. Check ALL clauses within each section. "
            "Collect ALL gaps in one REPLAN, naming all affected files; do not stop at the first issue. "
            "APPROVE requires every row covered. Never invent new requirements."
        )
    return payload, sections


@manager.handler(F.command == "REVIEW_PLAN")
@manager.handler(F.command == "RESOLVE")
@manager.handler(F.command == "FINALIZE")
@manager.handler(F.command == "STEER")
def review_handler(model, run, update, issues=None, review_context=None):
    operation = update.command
    state = run.get("review_state", run["state"])
    if review_context is not None:
        run["review_context"] = review_context
    model.reset_context()
    payload, sections = review_payload(state, operation, issues)
    return begin_request(
        run,
        "CORE",
        operation + "\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        ["APPROVE", "REPLAN", "STOP"],
        kind="review",
        operation=operation,
        payload=payload,
        sections=sections,
    )


def validate_decision(response, state, operation, sections):
    if not isinstance(response.get("reason"), str) or not response["reason"].strip():
        return "CORE response requires a nonempty reason"
    if response["command"] == "STOP":
        return ""
    if operation == "REVIEW_PLAN":
        coverage = response.get("coverage")
        if not isinstance(coverage, list) or len(coverage) != len(sections):
            return "REVIEW_PLAN requires complete coverage with evidence; gaps require REPLAN"
        seen, gaps = set(), set()
        for row in coverage:
            if not isinstance(row, dict):
                return "REVIEW_PLAN requires complete coverage with evidence"
            section, paths = row.get("section"), row.get("files")
            if (
                not isinstance(section, str)
                or section not in sections
                or section in seen
                or row.get("status") not in ("covered", "gap")
                or not isinstance(paths, list)
                or not paths
                or any(not isinstance(p, str) or p not in state["files"] for p in paths)
                or not isinstance(row.get("evidence"), str)
                or not row["evidence"].strip()
            ):
                return "REVIEW_PLAN requires complete coverage with evidence"
            seen.add(section)
            if row["status"] == "gap":
                gaps.update(paths)
        if response["command"] == "APPROVE" and gaps:
            return "REVIEW_PLAN requires complete coverage with evidence; gaps require REPLAN"
        if response["command"] == "REPLAN" and not gaps.issubset(
            set(response.get("files") or [])
        ):
            return "REPLAN must name every file with a requirement coverage gap"
    if response["command"] == "REPLAN" or operation in {"STEER", "RESOLVE"}:
        paths = response.get("files")
        if (
            not isinstance(paths, list)
            or not paths
            or any(not isinstance(p, str) or p not in state["files"] for p in paths)
        ):
            return "CORE repair/replan requires nonempty files containing only existing planned paths"
    return ""


def checked_decision(run, response, continuation):
    pending = run["planning"]
    error = validate_decision(
        response,
        run.get("review_state", run["state"]),
        pending["operation"],
        pending["sections"],
    )
    if error:
        if pending["corrections"] >= 1:
            raise ValueError(error)
        pending["corrections"] += 1
        pending["prompt"] = (
            error
            + ". Correct your response.\n"
            + pending["operation"]
            + "\n"
            + json.dumps(pending["payload"], ensure_ascii=False)
        )
        return transition("CORE", "REQUEST")
    return transition(
        "CODER",
        continuation,
        sender="CORE",
        decision=response,
        review_context=run.get("review_context", {}),
    )


@manager.handler(F.command == "APPROVE")
def approve_handler(run, response):
    continuation = {
        "REVIEW_PLAN": "PLAN_APPROVED",
        "RESOLVE": "RESOLVE_APPROVED",
        "FINALIZE": "FINAL_APPROVED",
        "STEER": "STEER_APPROVED",
    }[run["planning"]["operation"]]
    return checked_decision(run, response, continuation)


@manager.handler(F.command == "REPLAN")
def replan_handler(run, response):
    return checked_decision(run, response, "REPLAN_REQUIRED")


@manager.handler(F.command == "STOP")
def stop_handler(run, response):
    return checked_decision(run, response, "STOP_REQUESTED")
