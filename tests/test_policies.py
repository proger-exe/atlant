import json

import pytest

from atlant_coder.utils.prompt_rules import (
    CATALOG,
    PONYTAIL,
    build_rules,
    select_rules,
)


def ids(task, role="CODER", path=""):
    return {entry["id"] for entry in select_rules(task, role, path)}


def test_catalog_contains_ten_attributed_original_adaptations():
    assert len(CATALOG) == len({entry["id"] for entry in CATALOG}) == 10
    for entry in CATALOG:
        assert entry["source"].startswith("https://raw.githubusercontent.com/")
        assert entry["popularity_source"].startswith("https://skills.sh/")
        assert entry["license"] in {"MIT", "Apache-2.0"}
        assert entry["license_source"] and entry["version"] and entry["observed_at"]
        assert entry["popularity"] and len(entry["rules"]) == 2
    assert PONYTAIL["id"] not in {entry["id"] for entry in CATALOG}
    json.dumps(CATALOG)


def test_role_and_file_scope_select_relevant_guidance():
    assert {"improve-codebase-architecture", "grill-me"} <= ids(
        "Python service", "ARCH"
    )
    assert {"code-review", "verification-before-completion"} <= ids(
        "Python service", "CORE"
    )
    assert "grill-me" not in ids("Python service", "CODER")
    assert "vercel-react-best-practices" in ids(
        "React frontend and Python backend", path="web/App.tsx"
    )
    backend = ids("React frontend and Python backend", path="app/server.py")
    assert backend == {"verification-before-completion"}
    assert "vercel-react-best-practices" not in ids("Vue component", path="src/App.vue")


def test_debugging_tests_and_browser_are_selected_for_their_jobs():
    assert "systematic-debugging" in ids(
        "Исправь ошибку сериализации", path="app/storage.py"
    )
    assert "tdd" in ids("Python service", path="tests/test_service.py")
    assert "agent-browser" in ids(
        "Verify browser interactions", path="tests/browser.test.mjs"
    )
    assert "frontend-design" in ids("Build a product page", path="src/page.html")
    assert "tdd" not in ids("Change the heading color", path="src/style.css")


@pytest.mark.parametrize(
    "role", ["CORE", "ARCH", "ARCH_THINK", "CODER", "CODER_THINK", "AGENT"]
)
def test_guidance_is_bounded_deduplicated_and_preserves_safeguards(role):
    task = "Fix React browser UI regression with TDD and code review. " * 1000
    selected = select_rules(task, role, "src/App.tsx")
    assert 0 < len(selected) <= 4
    assert len({entry["id"] for entry in selected}) == len(selected)
    full = build_rules(task, role, "src/App.tsx")
    lite = build_rules(task, role, "src/App.tsx", "lite")
    assert len(lite) < len(full) < 2400
    for prompt in (full, lite):
        assert "[ponytail]" in prompt
        assert "Never weaken validation, security" in prompt
        assert "approved contracts" in prompt
    assert build_rules(task, role, mode="off") == ""


def test_invalid_mode_fails_instead_of_silently_disabling_rules():
    with pytest.raises(ValueError, match="full, lite or off"):
        build_rules("small task", "CODER", mode="typo")
