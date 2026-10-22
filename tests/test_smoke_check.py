from pathlib import Path

from atlant_coder.utils.smoke_check import smoke_check


def check(source: str, root: Path) -> list[str]:
    target = root / "consumer.py"
    target.write_text(source, encoding="utf-8")
    return smoke_check(str(target), str(root))


def test_from_package_accepts_existing_submodules_without_reexports(
    tmp_path: Path,
) -> None:
    package = tmp_path / "app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "rules.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert check("from app import engine, rules\n", tmp_path) == []


def test_namespace_package_accepts_only_existing_submodule(tmp_path: Path) -> None:
    namespace = tmp_path / "plugins"
    namespace.mkdir()
    (namespace / "local.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert check("from plugins import local\n", tmp_path) == []
    errors = check("from plugins import missing\n", tmp_path)
    assert errors and "missing" in errors[-1]


def test_missing_package_name_still_rejected(tmp_path: Path) -> None:
    package = tmp_path / "app"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

    errors = check("from app import Missing\n", tmp_path)
    assert errors and "Missing" in errors[-1]


def test_sibling_file_does_not_supply_name_for_nonpackage_module(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "engine.py").write_text("VALUE = 2\n", encoding="utf-8")

    errors = check("from app import engine\n", tmp_path)
    assert errors and "engine" in errors[-1]
