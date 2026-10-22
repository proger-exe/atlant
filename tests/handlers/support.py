"""Test bootstrap for exercising the registered coordinator with fake role APIs."""

from pathlib import Path

from AurethosSDK.client.dispatcher import Dispatcher
from AurethosSDK.client.router import Router
from AurethosSDK.types.update import Update
from atlant_coder.handlers.arch import arch_router
from atlant_coder.handlers.coder import close_generation, coder_router
from atlant_coder.handlers.core import manager, role_api
from atlant_coder.handlers.parallel_batch import worker_think_router
from atlant_coder.utils import memory


def run_generation(
    base_api,
    home,
    *,
    core=None,
    architect=None,
    thinker=None,
    workers=None,
    control=None,
):
    """Run the registered role workflow for an already prepared plan."""
    home = str(Path(home).resolve())
    run = {
        "state": memory.snapshot(),
        "home": home,
        "control": control,
        "workers": workers,
        "result": None,
    }
    dispatcher = Dispatcher("")
    run["dispatcher"] = dispatcher
    for name, api, commands in (
        ("CORE", core or role_api("CORE", home), manager),
        ("ARCH", architect or role_api("ARCH", home), arch_router),
        ("CODER", base_api, coder_router),
        ("CODER_THINK", thinker or role_api("CODER_THINK", home), worker_think_router),
    ):
        router = Router(api, run=run)
        router.sync(commands.handlers)
        dispatcher.routers[name] = router
    try:
        dispatcher.check_update(
            Update(
                to_model="CODER", from_model="CORE", command="GENERATE", parameters={}
            )
        )
        return run["result"]
    finally:
        close_generation(run)
