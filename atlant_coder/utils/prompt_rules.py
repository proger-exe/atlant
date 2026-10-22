"""Select bounded generation rules from the bundled prompt asset."""

import json
import re
from pathlib import Path

_DATA = json.loads(
    (Path(__file__).parents[1] / "models" / "prompt_rules.json").read_text(
        encoding="utf-8"
    )
)
CATALOG = tuple(_DATA["catalog"])
PONYTAIL = _DATA["ponytail"]


def select_rules(task: str, role: str, path: str = "") -> list[dict]:
    """Return at most four relevant catalog entries, without invoking skills."""
    text, path = task.casefold(), path.replace("\\", "/").casefold()
    role = role.upper().split("_", 1)[0]
    ui = (
        bool(re.search(r"\.(tsx|jsx|vue|svelte|html|css|scss)$", path))
        if path
        else bool(
            re.search(
                r"\b(react|next\.?js|vue|svelte|frontend|html|css|ui|website)\b|интерфейс|фронтенд|сайт",
                text,
            )
        )
    )
    react = ui and (
        path.endswith((".tsx", ".jsx"))
        or bool(re.search(r"\b(react|next\.?js)\b", text))
    )
    debugging = bool(
        re.search(
            r"\b(bug|debug|fix|failure|regression|broken)\b|ошиб|исправ|сбой|падает",
            text,
        )
    )
    testing = bool(
        re.search(r"(^|/)(test[^/]*|tests)/|(^|/)test_|\.(test|spec)\.", path)
    ) or bool(
        re.search(
            r"\b(tdd|test-driven|regression test|integration test)\b|регрессионн|интеграционн.*тест",
            text,
        )
    )
    browser = (not path or testing or ui) and bool(
        re.search(r"\b(browser|playwright|e2e|webgl)\b|браузер", text + " " + path)
    )
    selected = []
    if role == "ARCH":
        selected += ["improve-codebase-architecture", "grill-me"]
    elif role == "CORE":
        selected += ["code-review", "verification-before-completion"]
    else:
        selected.append("verification-before-completion")
    if debugging:
        selected.append("systematic-debugging")
    if testing:
        selected.append("tdd")
    if browser:
        selected.append("agent-browser")
    if ui:
        if react:
            selected.append("vercel-react-best-practices")
        selected += ["web-design-guidelines", "frontend-design"]
    if re.search(r"\b(review|audit)\b|ревью|аудит", text):
        selected.append("code-review")
    by_id = {entry["id"]: entry for entry in CATALOG}
    return [by_id[identifier] for identifier in dict.fromkeys(selected)][:4]


def build_rules(task: str, role: str, path: str = "", mode: str = "full") -> str:
    """Render bounded context; off disables guidance, never coordinator safeguards."""
    if mode not in {"full", "lite", "off"}:
        raise ValueError("policy mode must be full, lite or off")
    if mode == "off":
        return ""
    lines = [
        "BUILT-IN GENERATION RULES (original compact adaptations)",
        "The task, approved contracts and response protocol remain authoritative; work only within the assigned scope.",
        "Never weaken validation, security, data integrity, accessibility or required acceptance checks to save code or tokens.",
        "Use only available capabilities; these rules do not authorize downloads, external actions or changes to the approved plan.",
    ]
    for entry in (PONYTAIL, *select_rules(task, role, path)):
        rules = entry["rules"][:1] if mode == "lite" else entry["rules"]
        lines.append(f"[{entry['id']}] " + " ".join(rules))
    return "\n".join(lines)
