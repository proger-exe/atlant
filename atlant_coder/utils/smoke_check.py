"""Post-write smoke check: catch import/compile errors that AST+format missed.

analyze_code checks syntax; format_code checks style. Neither notices a
missing dependency, a typo'd module name, or a broken CPython bytecode
pass. This check does both — cheaply, without executing the module.
"""

import ast
import importlib.util
import logging
import pathlib
import subprocess
import sys
from typing import List, Set

logger = logging.getLogger(__name__)


def _resolve(top: str, project_root: pathlib.Path) -> bool:
    """True if `top` (first path segment of an import) resolves.

    Order: installed/stdlib via find_spec → sibling file/package under
    project_root. Anything else is a probable typo or missing dep.
    """
    try:
        if importlib.util.find_spec(top) is not None:
            return True
    except (ImportError, ValueError):
        # find_spec raises ImportError on some malformed names; treat as unresolved.
        pass
    if (project_root / f"{top}.py").exists():
        return True
    if (project_root / top).is_dir():
        return True
    return False


def _module_publics(target_file: pathlib.Path) -> Set[str]:
    """Top-level names defined in a .py file (AST scan, no code execution).

    Catches classes, functions, and simple/annotated assignments. Used by
    the from-import name checker below — safer than importing the module,
    which would run any side effects at file-scope.
    """
    try:
        tree = ast.parse(target_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    return _publics(tree)


def _package_has_submodule(target: pathlib.Path, name: str) -> bool:
    """Check package children statically; importing would execute user code."""
    package = target.parent if target.name == "__init__.py" else target
    if not package.is_dir():
        return False
    return (package / f"{name}.py").is_file() or (package / name).is_dir()


def _module_publics_from_code(code: str) -> Set[str]:
    return _publics(ast.parse(code))


def _publics(tree) -> Set[str]:
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            # `from x import Y` also re-exports Y at module top-level.
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _resolve_local_module_file(
    module: str,
    level: int,
    current_file: pathlib.Path,
    project_root: pathlib.Path,
) -> pathlib.Path | None:
    """Locate the .py file backing a `from X import ...` import, if local.

    Handles both absolute (`from pkg.sub import x`) and relative
    (`from .sib import y`, `from ..other import z`) forms. Returns the
    file (or package __init__.py) if it lives inside project_root,
    otherwise None.
    """
    if level and level > 0:
        # Relative — walk up from current file's directory.
        base = current_file.parent
        for _ in range(level - 1):
            base = base.parent
        if not module:
            candidate = base / "__init__.py"
            return candidate if candidate.exists() else None
        segments = module.split(".")
    else:
        base = project_root
        segments = module.split(".")

    target_dir = base
    for seg in segments[:-1]:
        target_dir = target_dir / seg
    leaf = segments[-1]
    for candidate in (target_dir / f"{leaf}.py", target_dir / leaf / "__init__.py"):
        if candidate.exists():
            return candidate
    namespace = target_dir / leaf
    if namespace.is_dir():
        return namespace
    return None


def smoke_check(full_path: str, project_root: str, timeout_s: int = 15) -> List[str]:
    """Return a list of human-readable error strings; empty = clean.

    Two passes: (1) `python -m py_compile` for CPython-strict syntax, and
    (2) static import-resolution over top-level import names. Relative
    imports (`from . import x`) are trusted — package layout is on the
    generator, not us.
    """
    p = pathlib.Path(full_path)
    if p.suffix != ".py" or not p.exists():
        return []

    errors: List[str] = []

    # (1) py_compile — safer than importing (no module-level code runs).
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(p)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            errors.append(f"py_compile FAILED:\n{proc.stderr.strip()}")
            return errors  # no point resolving imports if it doesn't parse
    except subprocess.TimeoutExpired:
        errors.append(f"py_compile TIMEOUT after {timeout_s}s")
        return errors
    except OSError as exc:
        errors.append(f"cannot run py_compile: {exc}")
        return errors

    # (2) static import resolution.
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return errors  # py_compile would have caught real syntax issues

    root = pathlib.Path(project_root)
    missing_modules: Set[str] = set()
    missing_names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if not _resolve(top, root):
                    missing_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Resolve the target module (relative + absolute) to a local
            # file when possible. If it's local, verify each imported name
            # actually exists there — catches sibling-parallel drift like
            # `from .models import AuthorStatsRow` when models.py never
            # defined that name.
            target_file = _resolve_local_module_file(
                node.module or "", node.level or 0, p, root
            )
            if target_file is None:
                # Non-local (relative import outside root, or absolute
                # module that isn't a project file): keep the module-level
                # existence check only.
                if node.module and (not node.level or node.level == 0):
                    top = node.module.split(".")[0]
                    if not _resolve(top, root):
                        missing_modules.add(node.module)
                continue
            available = _module_publics(target_file)
            for alias in node.names:
                name = alias.name
                if name == "*":
                    continue
                if name not in available and not _package_has_submodule(
                    target_file, name
                ):
                    missing_names.append(
                        f"'{name}' from '{node.module}' (defined in {target_file.name})"
                    )

    if missing_modules:
        errors.append(
            "Импорты не резолвятся (модуль не установлен и не найден в проекте): "
            + ", ".join(sorted(missing_modules))
        )
    if missing_names:
        errors.append(
            "Имена не найдены в целевых модулях (sibling-drift, "
            "модель придумала имя которого нет): " + "; ".join(missing_names)
        )

    return errors
