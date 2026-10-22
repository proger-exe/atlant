"""Guard application boundaries that previously bypassed the SDK."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_application_uses_module_imports_and_registered_handlers():
    for path in (ROOT / "atlant_coder").rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports = {
            id(node)
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                assert id(node) in imports, (path, node.lineno)
            if isinstance(node, ast.ClassDef):
                assert not any(
                    isinstance(base, ast.Name) and base.id == "APIRouter"
                    for base in node.bases
                ), path
            if isinstance(node, ast.FunctionDef):
                if any(
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute)
                    and d.func.attr == "handler"
                    for d in node.decorator_list
                ):
                    assert node in tree.body, (path, node.name)


def test_sdk_does_not_depend_on_application_and_tests_live_outside_package():
    for path in (ROOT / "AurethosSDK").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("atlant_coder"), path
            elif isinstance(node, ast.Import):
                assert not any(
                    n.name.startswith("atlant_coder") for n in node.names
                ), path
    assert not list((ROOT / "atlant_coder").rglob("test_*.py"))
