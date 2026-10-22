"""Checkpoint save/load — resume a run after a fatal gateway/network/OS event.

The checkpoint is a single JSON file that snapshots `MemoryStorage`. Callers
autosave after every ARCH MADE and every completed file, and force-save on
fatal errors (e.g. `APIRouter` giving up after backoff exhaustion). On next
start, `main.py` loads the checkpoint if present and skips CORE + ARCH,
jumping straight to CODER with the remaining files.

Files already written to disk in `home_path` are not touched — only the
in-memory state around them.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Checkpoint path is fixed relative to the workspace root of the run.
# main.py writes / reads it under home_path (task/) so a run always ships
# alongside the files it produced.
_FILENAME = ".atlant_checkpoint.json"


class CheckpointExitError(Exception):
    """Raised after a checkpoint is saved on a fatal condition to unwind the
    dispatcher stack cleanly and let main.py exit with a non-zero status.
    Message is included in the exit log line — keep it short and factual.
    """


def _path(home_dir: str) -> Path:
    return Path(os.environ.get("ATLANT_CHECKPOINT_DIR", home_dir)) / _FILENAME


def save(memory_snapshot: dict[str, Any], *, home_dir: str, reason: str) -> None:
    """Atomically write the checkpoint to disk. Tmp + rename so a crash
    mid-write can never corrupt an existing valid checkpoint.
    """
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "memory": memory_snapshot,
    }
    target = _path(home_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=".atlant_ckpt_",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2, default=str)
        tmp_path = tmp.name
    os.replace(tmp_path, target)
    completed = payload["memory"].get("completed_files") or []
    logger.info(
        "checkpoint saved: reason=%s completed=%d path=%s",
        reason,
        len(completed),
        target,
    )


def load(home_dir: str) -> dict[str, Any] | None:
    """Return the checkpoint payload dict or None if none/corrupt.

    A corrupt file is renamed (`.corrupt`) rather than deleted, so an
    operator can inspect what went wrong.
    """
    target = _path(home_dir)
    if not target.exists():
        return None
    try:
        with target.open(encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or "memory" not in payload:
            raise ValueError("checkpoint payload missing 'memory' key")
        return payload
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        corrupt = target.with_suffix(target.suffix + ".corrupt")
        try:
            os.replace(target, corrupt)
        except OSError:
            pass
        logger.warning(
            "checkpoint at %s corrupt (%s) — moved to %s, starting fresh",
            target,
            exc,
            corrupt,
        )
        return None


def clear(home_dir: str) -> None:
    """Remove a completed-run checkpoint. Safe if it doesn't exist."""
    target = _path(home_dir)
    try:
        target.unlink()
        logger.info("checkpoint cleared: %s", target)
    except FileNotFoundError:
        pass
