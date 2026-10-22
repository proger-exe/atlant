"""Workspace lifecycle and SDK dispatcher bootstrap for Atlant runs."""

import fcntl
import os
from pathlib import Path

from AurethosSDK.client.dispatcher import Dispatcher
from atlant_coder import MODELS_PATH
from atlant_coder.handlers.arch import arch_router
from atlant_coder.handlers.core import manager, role_api
from atlant_coder.handlers.coder import coder_router, close_generation
from atlant_coder.handlers.planning import transition
from atlant_coder.utils.checkpoint import CheckpointExitError, clear, load, save
from atlant_coder.utils.memory import memory


def run(workspace, task, *, mode="generate", control=None, workers=None):
    home = str(Path(workspace).resolve())
    if mode not in {"agent", "generate"} or not task.strip():
        raise ValueError("mode must be agent/generate and task must not be empty")
    lock_path = Path(home) / ".atlant_writer.lock"
    if lock_path.is_symlink():
        raise ValueError("workspace lock must not be a symlink")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CheckpointExitError("workspace already has an active writer") from exc
        memory.restore({})
        checkpoint = load(home)
        if checkpoint:
            state = checkpoint["memory"]
            if state.get("initial_task", state.get("source_task")) != task:
                raise CheckpointExitError(
                    "task changed since checkpoint; start a new run"
                )
            state["home_path"] = home
            if control:
                control.applied_revision = state.get("user_revision", 0)
        else:
            state = dict(
                source_task=task,
                initial_task=task,
                tech_task=task,
                home_path=home,
                agent_mode=mode == "agent",
                policy_mode=os.environ.get("ATLANT_POLICY_MODE", "full"),
            )
            if control:
                for item in control.messages():
                    state["source_task"] += (
                        "\n\nUser clarification:\n" + item["message"]
                    )
                    control.applied_revision = item["revision"]
                state["user_revision"] = control.applied_revision
                state["tech_task"] = state["source_task"]
        memory.restore(state)
        context = dict(
            state=state, home=home, control=control, workers=workers, result=None
        )
        dispatcher = Dispatcher(str(MODELS_PATH))
        context["dispatcher"] = dispatcher
        dispatcher.include_routers([manager, arch_router, coder_router])
        dispatcher.init_routers(run=context)
        for role, router in dispatcher.routers.items():
            router.set_custom_model(role_api(role, home))
        try:
            dispatcher.check_update(transition("CORE", "START"))
            if context["result"] is None:
                raise CheckpointExitError("dispatcher ended before completing the run")
            if control is None and context["result"].get("project_verified"):
                clear(home)
            return context["result"]
        except BaseException:
            latest = memory.snapshot()
            if latest.get("files") and latest.get("home_path") == home:
                save(latest, home_dir=home, reason="run_interrupted")
            raise
        finally:
            close_generation(context)
