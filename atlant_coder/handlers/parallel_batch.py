"""One isolated file task. Workers never publish files or mutate project state."""

import copy
import json
from collections import Counter
from dataclasses import dataclass, field
import time
from pathlib import Path, PurePosixPath

from AurethosSDK.client.dispatcher import Dispatcher
from AurethosSDK.client.router import ABSRouter
from AurethosSDK.filters.filter import F
from AurethosSDK.types.update import Update
from AurethosSDK.models.api_router import APIRouter
from AurethosSDK.models.codex_gateway import CodexGatewaySessionLostError
from atlant_coder import MODELS_PATH, config
from atlant_coder.macroses import analyze_code, make
from atlant_coder.utils.analyze_report import format_error_feedback
from atlant_coder.utils.project import make_prompt
from atlant_coder.macroses.workspace import apply_edits
from atlant_coder.schemas.protocol import response_schema
from atlant_coder.schemas import CoderUpdate
from atlant_coder.utils.signatures import extract_signatures
from atlant_coder.utils.smoke_check import _module_publics_from_code


def _clone_api_router(base: APIRouter) -> APIRouter:
    return APIRouter(
        copy.deepcopy(base.model), json_mode=True, generation_read_only=True
    )


def _same_target(returned, expected) -> bool:
    if not isinstance(returned, str) or not returned.strip() or not expected:
        return False
    return (
        PurePosixPath(returned.strip().lstrip("/")).as_posix()
        == PurePosixPath(expected.lstrip("/")).as_posix()
    )


def _validate_response(response: dict, path: str) -> str:
    if not _same_target(response.get("file"), path):
        return f"wrong file: expected {path!r}, got {response.get('file')!r}"
    if not isinstance(response.get("code"), str):
        return "code must be a string"
    return ""


def _summary(response: dict) -> dict:
    summary = response.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("behavior"), str):
        raise ValueError(
            "MADE requires summary.behavior, summary.exports and summary.issues"
        )
    if len(summary["behavior"]) > 4000:
        raise ValueError("summary.behavior exceeds 4000 characters")
    summary = {"limitations": [], **summary}
    for field in ("exports", "issues", "limitations"):
        if not isinstance(summary.get(field), list) or any(
            not isinstance(v, str) for v in summary[field]
        ):
            raise ValueError(f"summary.{field} must be a string array")
    return summary


worker_router = ABSRouter("CODER")
worker_think_router = ABSRouter("CODER_THINK")


@dataclass
class FileTask:
    path: str
    context: dict
    result: dict
    cheap: bool
    cap: int
    patch_source: str | None = None
    chunks: list[str] = field(default_factory=list)
    thinks: int = 0
    continues: int = 0
    tool_calls: int = 0
    reviewing: bool = False
    think_api: APIRouter | None = None


def _update(command, *, sender="CODER", **parameters):
    return CoderUpdate(
        to_model="CODER", from_model=sender, command=command, parameters=parameters
    )


def _reply(response, allowed):
    if not isinstance(response, dict):
        raise ValueError("expected a JSON object from CODER")
    if response.get("command") not in allowed:
        raise ValueError(f"unexpected command: {response.get('command')!r}")
    return _update(response["command"], response=response)


@worker_router.handler(F.command == "MAKE")
def start(prompt):
    return _update("REQUEST", prompt=prompt)


@worker_router.handler(F.command == "REQUEST")
def request(model, task: FileTask, prompt):
    allowed = ["MADE", "NEEDRESUME", "CONFLICT"]
    if task.context.get("agent_mode"):
        if task.patch_source is not None:
            allowed = ["PATCH", "CONFLICT"]
        allowed += ["READ", "SEARCH", "LIST", "RUN"]
    if task.thinks < task.cap:
        allowed.append("THINK")
    return _reply(
        model.send_api_request(
            prompt,
            expects=allowed,
            cheap=task.cheap,
            response_schema=response_schema("CODER", allowed),
        ),
        allowed,
    )


@worker_router.handler(F.command == "PATCH")
def patch(task: FileTask, response):
    if not _same_target(response.get("file"), task.path):
        raise ValueError("PATCH cannot modify another worker's file")
    source = task.result["draft"]["code"] if task.reviewing else task.patch_source
    if source is None:
        raise ValueError("PATCH requires an existing file or previous draft")
    return _update(
        "MADE",
        response={
            **response,
            "command": "MADE",
            "code": apply_edits(source, response["edits"]),
        },
    )


@worker_router.handler(F.command == "CONFLICT")
def conflict(task: FileTask, response):
    issues = response.get("issues")
    if (
        not isinstance(issues, list)
        or not issues
        or any((not isinstance(i, str) or not i.strip() for i in issues))
    ):
        raise ValueError("CONFLICT requires nonempty issues: string[]")
    task.result.update(status="conflict", reason="\n".join(issues))


@worker_router.handler(F.command == "NEEDRESUME")
def resume(task: FileTask, response):
    error = _validate_response(response, task.path)
    if error:
        raise ValueError(error)
    task.chunks.append(response["code"])
    task.continues += 1
    if task.continues > 3:
        raise ValueError("continuation limit reached")
    return _update(
        "REQUEST",
        prompt="CONTINUE exactly where the previous chunk ended; final MADE must include summary.",
    )


@worker_router.handler(F.command == "MADE")
def made(task: FileTask, response):
    error = _validate_response(response, task.path)
    if error:
        raise ValueError(error)
    code = (
        response["code"] if task.reviewing else "".join(task.chunks) + response["code"]
    )
    summary = _summary(response)
    task.result["draft"] = {"code": code, "summary": summary}
    if not task.reviewing and (
        summary["issues"]
        or (
            config.enable_self_review
            and task.context.get("policy_mode", "off") == "off"
            and (not task.cheap)
            and (Path(task.path).suffix == ".py")
        )
    ):
        return _update("REVIEW")
    return _update("VERIFY")


@worker_router.handler(F.command == "REVIEW")
def review(model, task: FileTask):
    task.reviewing = True
    task.result["draft_reviews"] += 1
    review_commands = ["OK", "MADE", "CONFLICT"]
    patch_instruction = ""
    if task.context.get("agent_mode"):
        review_commands.append("PATCH")
        patch_instruction = " Prefer PATCH for local corrections: its exact unique old/new edits apply to the COMPLETE draft below, including a new file, not to the original disk file. Return the same file and updated factual summary."
    response = model.send_api_request(
        "REVIEW: verify this COMPLETE file against the original task, contract, dependency interfaces, shared UI tokens/components and current feedback. Return OK or a complete MADE replacement with the same file and a factual summary. Fix defects in your draft without changing the agreed contract. Return CONFLICT only if the requirements themselves contradict each other, with the incompatible requirements in issues."
        + patch_instruction
        + " summary.issues contains unresolved defects or unmet task requirements and blocks publication. summary.limitations contains evidence/scope limits, not defects; do not move an unmet requirement there or claim unperformed checks passed. Known draft issues: "
        + repr(task.result["draft"]["summary"]["issues"])
        + "\n"
        + task.result["draft"]["code"],
        expects=review_commands,
        response_schema=response_schema("CODER", review_commands),
    )
    return _reply(response, review_commands)


@worker_router.handler((F.command == "OK") | (F.command == "VERIFY"))
def verify(task: FileTask):
    code, summary = (task.result["draft"]["code"], task.result["draft"]["summary"])
    if summary["issues"]:
        raise ValueError(
            "unresolved implementation issues: " + "\n".join(summary["issues"])
        )
    signatures = []
    if task.path.endswith(".py"):
        if task.context.get("agent_mode") and "current_source" in task.context:
            analyzed = analyze_code(code, fix=False)
            baseline = analyze_code(task.context["current_source"], fix=False)

            def problems(report):
                return Counter(
                    (
                        (item.get("code"), item.get("message"))
                        for item in report["stdout_report"]
                    )
                )

            if analyzed["stderr_error"] or problems(analyzed) - problems(baseline):
                raise ValueError(
                    str(analyzed["stdout_report"] or analyzed["stderr_report"])
                )
        else:
            analyzed = analyze_code(code)
            formatted = make(analyzed["analyzed_code"])
            if analyzed["error"] or formatted["error"]:
                raise ValueError(format_error_feedback(analyzed, formatted))
            code = formatted["formatted_code"]
        compile(code, task.path, "exec")
        actual = _module_publics_from_code(code)
        expected = set(task.context["entry"].get("public_names", []))
        missing = expected - actual
        claimed_missing = set(summary["exports"]) - actual
        if missing or claimed_missing:
            raise ValueError(
                f"public exports missing from source: {sorted(missing | claimed_missing)}"
            )
        signatures = extract_signatures(code)
        summary["verified_exports"] = sorted(actual)
    elif task.path.endswith(".json"):
        json.loads(code)
    task.result.update(
        status="ok", code=code, summary=summary, signatures=signatures, reason=""
    )


def _tool(task, response, operation):
    task.tool_calls += 1
    if task.tool_calls > 20:
        raise ValueError("tool budget exhausted or tools unavailable")
    tool_result = task.context["broker"].request(task.path, operation, response)
    return _update(
        "REQUEST",
        prompt="TOOL RESULT\n"
        + json.dumps(tool_result, ensure_ascii=False, separators=(",", ":")),
    )


def _tools(task):
    broker = task.context.get("broker")
    if broker is None:
        raise ValueError("tool budget exhausted or tools unavailable")
    return broker.tools


@worker_router.handler(F.command == "READ")
def read(task, response):
    return _tool(task, response, _tools(task).read)


@worker_router.handler(F.command == "LIST")
def list_files(task, response):
    return _tool(task, response, _tools(task).list)


@worker_router.handler(F.command == "SEARCH")
def search(task, response):
    return _tool(task, response, _tools(task).search)


@worker_router.handler(F.command == "RUN")
def run_command(task, response):
    return _tool(task, response, _tools(task).run)


@worker_router.handler(F.command == "THINK")
def think(task, response):
    task.thinks += 1
    return Update(
        to_model="CODER_THINK",
        from_model="CODER",
        command="THINK",
        parameters={"question": str(response.get("content", ""))},
    )


@worker_think_router.handler(F.command == "THINK")
def resolve_question(task, model, question):
    if model is None:
        raise ValueError("THINK requested but no thinker configured")
    if task.think_api is None:
        task.think_api = _clone_api_router(model)
    thought = task.think_api.send_api_request(
        make_prompt(task.context)
        + "\nResolve this specific implementation question:\n"
        + question,
        cheap=True,
        response_schema=response_schema("CODER_THINK"),
    )
    if not isinstance(thought, dict) or not thought.get("final"):
        raise ValueError("thinker returned no final answer")
    return _update("ANSW", sender="CODER_THINK", answer=str(thought["final"]))


@worker_router.handler(F.command == "ANSW")
def answer(answer):
    return _update(
        "REQUEST",
        prompt="ANSWER "
        + answer
        + "\nProduce the requested file; do not change the contract.",
    )


def _generate_one(path, context, base_apir, thinker=None):
    """Run one isolated SDK chain; only the coordinator publishes its result."""
    started = time.monotonic()
    api = _clone_api_router(base_apir)
    entry = context["entry"]
    cheap = entry["priority"] == "low"
    cap = config.max_think_iters_by_priority.get(
        entry["priority"], config.max_think_iters
    )
    if cheap or len(entry.get("plan", [])) >= config.min_plan_steps_to_skip_think > 0:
        cap = 0
    result = {
        "path": path,
        "revision": context["revision"],
        "status": "error",
        "queue_wait_s": context.get("queue_wait_s", 0),
        "draft_reviews": 0,
        "dependency_hashes": context.get("dependency_hashes", {}),
    }
    if "base_hash" in context:
        result["base_hash"] = context["base_hash"]
    task = FileTask(path, context, result, cheap, cap, context.get("current_source"))
    dispatcher = Dispatcher(str(MODELS_PATH))
    dispatcher.include_routers([worker_router, worker_think_router])
    dispatcher.init_routers(task=task)
    dispatcher.routers["CODER"].set_custom_model(api)
    dispatcher.routers["CODER_THINK"].set_custom_model(thinker)
    try:
        prompt = make_prompt(context)
        if context.get("previous_draft") is not None:
            previous = context["previous_draft"]
            if not isinstance(previous, dict) or not isinstance(
                previous.get("code"), str
            ):
                raise ValueError("previous_draft requires complete code and summary")
            task.patch_source = previous["code"]
            result["draft"] = {"code": task.patch_source, "summary": _summary(previous)}
            prompt += (
                "\nprevious_draft is the complete UNPUBLISHED draft from the previous attempt. "
                "Correct the reported feedback/last_error locally; PATCH edits must target previous_draft.code, "
                "not current_source on disk. Reuse the supplied evidence when sufficient; only gather "
                "additional facts needed to resolve remaining uncertainty. Do not repeat the whole investigation."
            )
        dispatcher.check_update(_update("MAKE", prompt=prompt))
    except (ValueError, SyntaxError) as exc:
        result["reason"] = str(exc)
    except (RuntimeError, CodexGatewaySessionLostError) as exc:
        result.update(status="transport_error", reason=str(exc))
    finally:
        result["duration_s"] = time.monotonic() - started
        result["requests"] = list(api.request_metrics)
        if task.think_api is not None:
            result["requests"].extend(task.think_api.request_metrics)
    return result
