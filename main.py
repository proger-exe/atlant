import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from atlant_coder import config
from atlant_coder.app import run
from atlant_coder.runtime import ensure_service
from atlant_coder.utils.checkpoint import CheckpointExitError, save
from atlant_coder.utils.memory import memory


HOME_PATH = "./task/"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and verify a project with Atlant"
    )
    parser.add_argument(
        "--home",
        default=HOME_PATH,
        help="Directory containing task.atl and generated output",
    )
    parser.add_argument("--workers", type=int, default=config.max_parallel_workers)
    args = parser.parse_args(argv)
    home = str(Path(args.home).resolve()) + "/"
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not Path(home, "task.atl").is_file():
        parser.error(f"no task.atl in {home}")
    config.max_parallel_workers = args.workers
    log = RotatingFileHandler(
        Path(home) / ".atlant.log", maxBytes=10_000_000, backupCount=2, encoding="utf-8"
    )
    log.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.basicConfig(
        level=logging.INFO, handlers=[log, RichHandler(show_path=False)]
    )
    memory.restore({})
    try:
        if not os.environ.get("CODEX_GATEWAY_URL"):
            client = ensure_service(home)
            os.environ["CODEX_GATEWAY_URL"] = client.url
            os.environ["CODEX_GATEWAY_TOKEN"] = client.token
        run(
            home,
            Path(home, "task.atl").read_text(encoding="utf-8"),
            workers=args.workers,
        )
        if memory.get("project_verified") is not True:
            raise CheckpointExitError("pipeline ended without project verification")
    except Exception as exc:
        if memory.get("files"):
            save(memory.snapshot(), home_dir=home, reason=str(exc))
        logging.getLogger("atlant").error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
