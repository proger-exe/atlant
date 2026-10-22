from pathlib import Path


MODELS = Path(__file__).parents[1] / "atlant_coder" / "models"


def prompt(name: str) -> str:
    return (MODELS / name).read_text(encoding="utf-8")


def test_role_prompts_are_compact_and_follow_current_wire_contract() -> None:
    prompts = {
        name: prompt(name)
        for name in ("br_manager.md", "br_structure.md", "br_coder.md", "br_think.md")
    }
    assert sum(map(len, prompts.values())) <= 14_000
    assert all(len(text) <= 6_000 for text in prompts.values())
    assert "files — список записей с path" in prompts["br_structure.md"]
    assert "нативной схем" in prompts["br_manager.md"]
    assert "нативной схем" in prompts["br_coder.md"]
    assert "поле final" in prompts["br_think.md"]


def test_prompts_keep_cross_file_and_safety_invariants() -> None:
    manager = prompt("br_manager.md")
    architect = prompt("br_structure.md")
    coder = prompt("br_coder.md")
    assert all(
        term in manager
        for term in ("coverage", "producer/consumer", "FINALIZE", "REPLAN")
    )
    assert all(
        term in architect
        for term in (
            "depends_on",
            "parallel_safe",
            "public_names",
            "rollback",
            "cleanup",
        )
    )
    assert all(
        term in coder for term in ("PATCH", "READ", "RUN", "CONFLICT", "browser PASS")
    )
    assert "нативный cwd" in architect and "нативный cwd" in coder
