import os
from typing import Optional

from pydantic import BaseModel


politoken = os.environ.get("POLLINATIONS_TOKEN", "")
ask_loops_count = int(os.environ.get("ATLANT_ASK_LOOPS", "0"))
# Ceiling on how many ASK rounds CORE can do before it force-sends MAKE.
# Prevents infinite ASK-loop when the user's answers don't satisfy the model.
max_ask_rounds = 2
max_think_iters = 3
# Stop a run after three failed repairs of the same file. The checkpoint keeps
# the failure and lets an operator resume after fixing the cause.
max_make_attempts = 3
# Trigger ARCH REPLAN after this many consecutive analyze/format fails on a
# single file — the plan is probably wrong, not just the code. 0 = disabled.
replan_after_attempts = 2
# Initial CORE/ARCH agreement has its own global budget before workers start.
max_initial_plan_corrections = 3
# After approval, stop the run when a file exhausts its replan budget.
max_replans = 1
# Per-priority THINK-iteration caps. Falls back to `max_think_iters` when
# a file's priority is unknown. Was {high:5, low:1} — dropped high to 2:
# codex-family models do internal reasoning already, external THINK-loop
# above 2 rounds was pure round-trip overhead.
max_think_iters_by_priority = {"high": 2, "low": 0}
# When ARCH gave a plan with at least this many concrete steps, skip THINK
# entirely — the algorithm is already spelled out, thinker adds nothing.
# 0 = never skip on plan basis.
min_plan_steps_to_skip_think = 3
# Ready tasks run concurrently; shared contract owners run exclusively.
# Set ATLANT_WORKERS=1 to use the same pipeline sequentially.
max_parallel_workers = max(1, int(os.environ.get("ATLANT_WORKERS", "6")))

# Optional caller-supplied scope restriction. Rejected paths cause replanning;
# they are never silently removed. CORE validates the plan against the task.
allowed_path_prefixes: Optional[list[str]] = None
enable_self_review = True


# === codex-gateway ===
codex_gateway_url = os.environ.get("CODEX_GATEWAY_URL", "http://127.0.0.1:8765")
codex_gateway_token = os.environ.get("CODEX_GATEWAY_TOKEN") or os.environ.get(
    "CODEX_API_TOKEN"
)
codex_gateway_model = os.environ.get("CODEX_GATEWAY_MODEL", "gpt-5.6-sol")
codex_gateway_workspace = os.environ.get("CODEX_GATEWAY_WORKSPACE", os.getcwd())
codex_gateway_sandbox = os.environ.get("CODEX_GATEWAY_SANDBOX", "workspace-write")
try:
    codex_gateway_timeout_seconds = int(
        os.environ.get("CODEX_GATEWAY_TIMEOUT_SECONDS", "600")
    )
except ValueError:
    codex_gateway_timeout_seconds = 600

# == GEMINI CONFIGURATION ==
gemini_token = os.environ.get("GEMINI_API_KEY", "")


# === llm7 ===
llm7_token = os.environ.get("LLM7_TOKEN", "")


class GeminiAnswers(BaseModel):
    command: str
    content: Optional[str] = ""
    file: Optional[str] = ""
    code: Optional[str] = ""
    task: Optional[str] = ""

    def __str__(self):
        return f"{self.command} {self.content}"


class ThinkGeminiAnswers(BaseModel):
    contemplator: str
    final: dict

    def __str__(self):
        return f"{self.final}"


# Contract-preserving repairs after project checks have a separate bounded budget.
max_post_test_repairs = 2
