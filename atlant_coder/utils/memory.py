import copy
import inspect
import logging
import threading
from pathlib import Path


class MemoryStorage:
    def __init__(self):
        self.memory = {}
        # Guards read-modify-write on list values (completed_files/failed_files
        # etc.) when parallel workers hit the same key. Path-scoped keys like
        # signatures_/task_/attempts_ are inherently race-free (unique keys)
        # so the lock only matters for shared lists.
        self._lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_caller_info(self):
        """Returns formatted caller info: '{function}{class} from {file}:{line}'"""
        frame = (
            inspect.currentframe().f_back.f_back
        )  # Skip current method and direct caller

        # Get calling function name
        func_name = frame.f_code.co_name

        if func_name == "<module>":
            func_name = "none"

        # Get class name if exists
        class_name = ""
        if "self" in frame.f_locals:
            class_name = f".{frame.f_locals['self'].__class__.__name__}"

        # Get filename without path
        filename = Path(frame.f_code.co_filename).name

        # Get line number
        line_no = frame.f_lineno

        return f"called by [rgb(140,31,235)]{func_name}{class_name}[/] from [rgb(255,200,0)]{filename}[/]:[blue]{line_no}[/]"

    def set(self, key: str, value):
        self.memory[key] = value
        self.logger.debug(
            "key=%r, setted to new value: %r, %s", key, value, self._get_caller_info()
        )

    def get(self, key: str):
        value = self.memory.get(key)
        self.logger.debug(
            "key=%r, requested value: %r, %s", key, value, self._get_caller_info()
        )
        return value

    def snapshot(self) -> dict:
        """Deep-copy of the storage for checkpoint serialization.

        Values must be JSON-serializable (str/int/float/bool/dict/list/None).
        This is fine for all keys the pipeline sets — no ORM instances or
        clients live here. Callers that stash non-JSON values must convert
        first.
        """

        with self._lock:
            return copy.deepcopy(self.memory)

    def restore(self, data: dict) -> None:
        """Replace all in-memory state from a previously saved snapshot."""
        with self._lock:
            self.memory = dict(data or {})

    def append_unique(self, key: str, item):
        """Atomic append-if-absent for a list value. Race-safe under threads."""
        with self._lock:
            lst = list(self.memory.get(key) or [])
            if item not in lst:
                lst.append(item)
                self.memory[key] = lst

    def get_all_with_prefix(self, prefix: str):
        res = []
        for sector in self.memory:
            if prefix in sector:
                res.append((sector, self.memory[sector]))
        self.logger.debug(
            "requested prefix: %r, finded records: %r, %s",
            prefix,
            res,
            self._get_caller_info(),
        )

        return res


memory = MemoryStorage()
