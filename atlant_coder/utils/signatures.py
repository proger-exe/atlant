"""AST-based signature extraction — compact context for cross-file continuity."""

import ast
from typing import List


def _format_args(node: ast.arguments) -> str:
    parts: List[str] = []
    positional = [*node.posonlyargs, *node.args]
    total = len(positional)
    defaults_offset = total - len(node.defaults)
    for i, arg in enumerate(positional):
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if i >= defaults_offset:
            piece += f"={ast.unparse(node.defaults[i - defaults_offset])}"
        parts.append(piece)
        if node.posonlyargs and i + 1 == len(node.posonlyargs):
            parts.append("/")
    if node.vararg is not None:
        annotation = (
            f": {ast.unparse(node.vararg.annotation)}" if node.vararg.annotation else ""
        )
        parts.append(f"*{node.vararg.arg}{annotation}")
    elif node.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(node.kwonlyargs):
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        default = node.kw_defaults[i] if i < len(node.kw_defaults) else None
        if default is not None:
            piece += f"={ast.unparse(default)}"
        parts.append(piece)
    if node.kwarg is not None:
        annotation = (
            f": {ast.unparse(node.kwarg.annotation)}" if node.kwarg.annotation else ""
        )
        parts.append(f"**{node.kwarg.arg}{annotation}")
    return ", ".join(parts)


def _format_returns(node) -> str:
    ret = getattr(node, "returns", None)
    return f" -> {ast.unparse(ret)}" if ret is not None else ""


def extract_signatures(code: str) -> List[str]:
    """Return a flat list of top-level class/function signatures from Python source.

    Silently returns an empty list on parse errors — callers treat missing
    signatures as "unknown" rather than as an error.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return []

    lines: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            lines.append(
                f"{prefix} {node.name}({_format_args(node.args)}){_format_returns(node)}"
            )
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            head = f"class {node.name}({bases})" if bases else f"class {node.name}"
            methods: List[str] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    p = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    methods.append(
                        f"    {p} {item.name}({_format_args(item.args)}){_format_returns(item)}"
                    )
                elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    ann = ast.unparse(item.annotation) if item.annotation else ""
                    default = f" = {ast.unparse(item.value)}" if item.value else ""
                    methods.append(f"    {item.target.id}: {ann}{default}")
                elif isinstance(item, ast.Assign):
                    methods.append("    " + ast.unparse(item))
            lines.append(head + ":")
            if methods:
                lines.extend(methods)
            else:
                lines.append("    ...")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            ann = ast.unparse(node.annotation) if node.annotation else ""
            lines.append(f"{node.target.id}: {ann}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    lines.append(f"{target.id} = ...")

    return lines


def collect_context_block(signatures_by_file: dict) -> str:
    """Render {path: [signature lines]} into a compact prompt fragment."""
    if not signatures_by_file:
        return ""
    chunks = ["УЖЕ СГЕНЕРИРОВАННЫЕ ПУБЛИЧНЫЕ ИМЕНА (используй их, не переизобретай):"]
    for path, sigs in signatures_by_file.items():
        if not sigs:
            continue
        chunks.append(f"# {path}")
        for line in sigs:
            chunks.append(line)
    return "\n".join(chunks) if len(chunks) > 1 else ""
