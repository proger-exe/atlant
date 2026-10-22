"""Wire contracts for model replies; coordinator schemas remain separate."""

from copy import deepcopy


def _object(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def response_schema(role, commands=None, *, coverage=False, repair=False):
    role = role.upper()
    text = {"type": "string"}
    nonempty = {"type": "string", "minLength": 1}
    strings = {"type": "array", "items": text}
    if "THINK" in role:
        return _object({"final": nonempty})
    if not commands:
        if role.startswith("CORE"):
            commands = ["OK", "ASK", "MADE"]
        elif role.startswith("ARCH"):
            commands = ["THINK", "MADE"]
        elif role.startswith("CODER"):
            commands = ["OK", "THINK", "MADE", "NEEDRESUME", "CONFLICT"]
        else:
            raise ValueError(f"No strict response schema for role {role!r}")
    summary = _object(
        {
            "behavior": nonempty,
            "exports": strings,
            "issues": strings,
            "limitations": strings,
        },
        required=["behavior", "exports", "issues"],
    )
    file_entry = _object(
        {
            "path": nonempty,
            "desc": nonempty,
            "priority": {"enum": ["high", "low"], "type": "string"},
            "depends_on": strings,
            "parallel_safe": {"type": "boolean"},
            "plan": strings,
            "public_names": strings,
        }
    )
    rows = {
        "type": "array",
        "items": _object(
            {
                "section": nonempty,
                "status": {"type": "string", "enum": ["covered", "gap"]},
                "files": {**strings, "minItems": 1},
                "evidence": nonempty,
            }
        ),
    }
    variants = []
    for command in commands:
        fields = {"command": {"type": "string", "const": str(command)}}
        required = None
        if command == "THINK":
            fields["data" if role.startswith("ARCH") else "content"] = nonempty
        elif command == "MADE" and role.startswith("ARCH"):
            fields.update(
                conventions=nonempty,
                files={"type": "array", "items": file_entry, "minItems": 1},
            )
        elif command in {"MADE", "NEEDRESUME"} and role.startswith("CODER"):
            fields.update(file=nonempty, code=text)
            if command == "MADE":
                fields["summary"] = summary
        elif command == "PATCH":
            fields.update(
                file=nonempty,
                edits={
                    "type": "array",
                    "minItems": 1,
                    "items": _object({"old": nonempty, "new": text}),
                },
                summary=summary,
            )
        elif command in {"READ", "LIST"}:
            fields["path"] = text
        elif command == "SEARCH":
            fields.update(path=text, pattern=nonempty)
        elif command == "RUN":
            fields["argv"] = {**strings, "minItems": 1}
        elif command == "CONFLICT":
            fields["issues"] = {"type": "array", "items": nonempty, "minItems": 1}
        elif command in {"APPROVE", "REPLAN", "STOP"}:
            fields["reason"] = nonempty
            if command != "STOP":
                fields["files"] = {"type": "array", "items": nonempty, "minItems": 1}
                if coverage:
                    fields["coverage"] = rows
                if command == "APPROVE" and not repair:
                    required = [key for key in fields if key != "files"]
        elif command == "MADE":
            fields["data"] = nonempty
        elif command == "ASK":
            fields["data"] = nonempty
        elif command != "OK":
            raise ValueError(f"No response schema for {role}/{command}")
        variants.append(_object(fields, required))
    result = {"anyOf": variants}
    if role.startswith("ARCH") and "MADE" in commands:
        result["x-atlant-files-map"] = True
    return deepcopy(result)


def decode_response(value, schema):
    if schema.get("x-atlant-files-map") and value.get("command") == "MADE":
        entries = value["files"]
        paths = [entry["path"] for entry in entries]
        if len(paths) != len(set(paths)):
            raise ValueError("ARCH returned duplicate file paths")
        value = {
            **value,
            "files": {
                entry["path"]: {
                    key: item for key, item in entry.items() if key != "path"
                }
                for entry in entries
            },
        }
    return value
