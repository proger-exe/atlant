"""Decide whether the coder can skip the THINK round-trip for a given file."""

import logging
import pathlib
from typing import Any, Dict, List, Tuple

from atlant_coder.utils.memory import memory

logger = logging.getLogger(__name__)

# Files whose content is trivial or purely declarative — THINK adds no value.
SKIP_THINK_EXTENSIONS = {
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".env",
    ".gitignore",
    ".toml",
    ".cfg",
    ".ini",
    ".lock",
    ".conf",
}
SKIP_THINK_FILENAMES = {
    "__init__.py",
    "LICENSE",
    "MANIFEST.in",
    "py.typed",
    ".dockerignore",
}


def normalize_files(
    files_raw: Any,
) -> Tuple[
    Dict[str, str],
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, bool],
]:
    """Normalize ARCH file entries into path-keyed maps.

    Returns descriptions, priorities, dependencies, plans, public names and
    the explicit parallel-safety contract. `plans` is the joined step string;
    `public_names` is the ARCH-declared exported-symbol contract per file
    (used by parallel workers to avoid sibling name-drift).
    """
    descs: Dict[str, str] = {}
    prios: Dict[str, str] = {}
    deps: Dict[str, List[str]] = {}
    plans: Dict[str, str] = {}
    publics: Dict[str, List[str]] = {}
    parallel_safe: Dict[str, bool] = {}

    if not isinstance(files_raw, dict):
        return descs, prios, deps, plans, publics, parallel_safe

    for path, entry in files_raw.items():
        if isinstance(entry, dict):
            descs[path] = str(entry.get("desc") or entry.get("description") or "")
            raw_prio = entry.get("priority")
            if raw_prio:
                prios[path] = str(raw_prio).lower()
            raw_deps = entry.get("depends_on") or entry.get("deps") or []
            if isinstance(raw_deps, list):
                deps[path] = [str(d) for d in raw_deps if isinstance(d, str)]
            raw_plan = entry.get("plan") or entry.get("steps") or []
            if isinstance(raw_plan, list):
                steps = [str(s).strip() for s in raw_plan if s]
                if steps:
                    plans[path] = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
            elif isinstance(raw_plan, str) and raw_plan.strip():
                plans[path] = raw_plan.strip()
            raw_pubs = (
                entry.get("public_names")
                or entry.get("exports")
                or entry.get("public")
                or []
            )
            if isinstance(raw_pubs, list):
                names = [str(n).strip() for n in raw_pubs if n]
                if names:
                    publics[path] = names
            if isinstance(entry.get("parallel_safe"), bool):
                parallel_safe[path] = entry["parallel_safe"]
        else:
            descs[path] = str(entry) if entry is not None else ""

    return descs, prios, deps, plans, publics, parallel_safe


def topo_sort_files(descs: Dict[str, str], deps: Dict[str, List[str]]) -> List[str]:
    """Kahn's topological sort — leaves-first ordering (deps before dependents).

    Robust to:
      - Missing dependency targets (silently dropped).
      - Cycles (breaks arbitrarily, appends remaining, warns).
      - No dependency info at all (returns original insertion order).
    """
    all_paths = list(descs.keys())
    if not all_paths:
        return []

    known = set(all_paths)
    filtered: Dict[str, List[str]] = {
        p: [d for d in deps.get(p, []) if d in known and d != p] for p in all_paths
    }

    in_degree = {p: 0 for p in all_paths}
    reverse: Dict[str, List[str]] = {p: [] for p in all_paths}
    for path, path_deps in filtered.items():
        for d in path_deps:
            in_degree[path] += 1
            reverse[d].append(path)

    ready: List[str] = sorted([p for p in all_paths if in_degree[p] == 0])
    order: List[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        newly_ready: List[str] = []
        for dependent in reverse[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                newly_ready.append(dependent)
        ready.extend(sorted(newly_ready))

    if len(order) < len(all_paths):
        remaining = [p for p in all_paths if p not in order]
        logger.warning(
            "Dependency cycle in ARCH output — %d files fall back to input order: %s",
            len(remaining),
            remaining,
        )
        order.extend(remaining)

    return order


def store_priorities(prios: Dict[str, str]) -> None:
    """Persist per-file priority hints for later lookup by the coder."""
    memory.set("file_priorities", prios)


def topo_sort_layered(
    descs: Dict[str, str], deps: Dict[str, List[str]]
) -> List[List[str]]:
    """Same order as topo_sort_files, packed into layers by BFS depth.

    Layer 0 = files with in_degree=0 (no deps). Layer N = files whose deps
    all sit in layers < N. Files inside one layer are mutually independent —
    safe to generate in parallel. Cycle-safe: any cycle leftovers become a
    trailing layer, matching topo_sort_files' behavior.
    """
    all_paths = list(descs.keys())
    if not all_paths:
        return []

    known = set(all_paths)
    filtered: Dict[str, List[str]] = {
        p: [d for d in deps.get(p, []) if d in known and d != p] for p in all_paths
    }
    in_degree = {p: len(filtered[p]) for p in all_paths}
    reverse: Dict[str, List[str]] = {p: [] for p in all_paths}
    for path, path_deps in filtered.items():
        for d in path_deps:
            reverse[d].append(path)

    layers: List[List[str]] = []
    ready = sorted([p for p in all_paths if in_degree[p] == 0])
    placed = set()
    while ready:
        layers.append(ready)
        placed.update(ready)
        next_ready: List[str] = []
        for current in ready:
            for dep in reverse[current]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    next_ready.append(dep)
        ready = sorted(next_ready)

    remaining = [p for p in all_paths if p not in placed]
    if remaining:
        logger.warning(
            "Dependency cycle in ARCH output — %d files fall into a trailing layer: %s",
            len(remaining),
            remaining,
        )
        layers.append(remaining)
    return layers


def transitive_deps(path: str, deps_map: Dict[str, List[str]]) -> set:
    """Every file reachable from `path` via depends_on edges (excluding path).

    Used to trim sibling-signature context to only what this file could
    actually reference. Cycle-safe. Missing keys → empty edges.
    """
    seen: set = set()
    stack = list(deps_map.get(path, []))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(deps_map.get(p, []))
    return seen


def should_skip_think(filepath: str) -> bool:
    """True if THINK should be bypassed for this file.

    Decision order:
      1. If ARCH gave an explicit priority — use it. "low" → skip, else THINK.
      2. Otherwise fall through to extension/filename heuristic.
    """
    prios = memory.get("file_priorities") or {}
    explicit = prios.get(filepath)
    if explicit is not None:
        return explicit.lower() == "low"

    p = pathlib.Path(filepath)
    if p.suffix.lower() in SKIP_THINK_EXTENSIONS:
        return True
    if p.name in SKIP_THINK_FILENAMES:
        return True

    return False
