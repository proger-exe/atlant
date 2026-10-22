from __future__ import annotations

import logging
from pathlib import Path

from .config import Settings


def setup_logging(settings: Settings) -> None:
    root_logger = logging.getLogger()
    target_path = settings.log_file
    target_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in root_logger.handlers:
        if getattr(handler, "_codex_gateway_handler", False):
            return

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(target_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._codex_gateway_handler = True  # type: ignore[attr-defined]

    root_logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    root_logger.addHandler(file_handler)


def format_multiline_block(title: str, content: str) -> str:
    return f"{title}\n---\n{content}\n---"
