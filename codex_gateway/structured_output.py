from __future__ import annotations

import json
import copy
from dataclasses import dataclass
from typing import Any

import jsonschema

from .schema_store import SchemaRecord


MAX_VALIDATION_RETRIES = 1


def native_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Codex requires a closed root object, even for a command union or array."""

    def strict(node):
        if isinstance(node, list):
            return [strict(value) for value in node]
        if not isinstance(node, dict):
            return node
        result = {key: strict(value) for key, value in node.items()}
        if node.get("type") == "object" or "properties" in node:
            properties = result.setdefault("properties", {})
            required = node.get("required", [])
            for key in properties:
                if key not in required:
                    properties[key] = {"anyOf": [properties[key], {"type": "null"}]}
            result["required"] = list(properties)
            result.setdefault("additionalProperties", False)
        return result

    body = strict(copy.deepcopy(schema))
    definitions = body.pop("$defs", None)
    result = {
        "type": "object",
        "properties": {"response": body},
        "required": ["response"],
        "additionalProperties": False,
    }
    if definitions is not None:
        result["$defs"] = definitions
    return result


def _omit_optional_nulls(value: Any, schema: dict[str, Any], root=None) -> Any:
    root = schema if root is None else root
    if schema.get("$ref", "").startswith("#/"):
        resolved = root
        for component in schema["$ref"][2:].split("/"):
            resolved = resolved[component.replace("~1", "/").replace("~0", "~")]
        return _omit_optional_nulls(value, resolved, root)
    if isinstance(value, dict):
        for branch in schema.get("anyOf", schema.get("oneOf", [])):
            candidate = _omit_optional_nulls(value, branch, root)
            try:
                jsonschema.validate(
                    candidate, {"$defs": root.get("$defs", {}), **branch}
                )
                return candidate
            except (jsonschema.ValidationError, jsonschema.SchemaError):
                pass
        properties = schema.get("properties", {})

        def synthetic_null(key, item):
            if (
                item is not None
                or key not in properties
                or key in schema.get("required", [])
            ):
                return False
            # Native strict mode adds null to represent omitted optional fields.
            # Preserve null when it was already meaningful in the source schema.
            try:
                jsonschema.validate(
                    None, {"$defs": root.get("$defs", {}), **properties[key]}
                )
            except jsonschema.ValidationError:
                return True
            return False

        return {
            key: _omit_optional_nulls(item, properties.get(key, {}), root)
            for key, item in value.items()
            if not synthetic_null(key, item)
        }
    if isinstance(value, list):
        return [
            _omit_optional_nulls(item, schema.get("items", {}), root) for item in value
        ]
    return value


class StructuredOutputError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 422,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class StructuredValidationFailure:
    kind: str
    message: str
    details: list[str]
    raw_output: str


@dataclass
class StructuredParseResult:
    parsed: dict[str, Any] | list[Any]
    raw_output: str


def validate_json_schema(schema: dict[str, Any]) -> None:
    try:
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise StructuredOutputError(
            "invalid_schema_payload",
            f"invalid JSON Schema: {exc.message}",
            status_code=422,
        ) from exc


def serialize_compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def build_initial_prompt(
    *,
    user_message: str,
    schema_record: SchemaRecord,
    chat_context: dict[str, Any] | list[Any] | str | None,
    system_prompt: str | None = None,
    native_only: bool = False,
) -> str:
    parts = [
        "You must return only valid JSON that matches the provided schema.",
        "Do not include markdown fences, explanations, commentary, or extra text.",
        'The native output envelope is {"response": <your JSON response>}; put the requested command inside response. Use null for optional fields required by the native schema.',
        f"Schema ID: {schema_record.schema_id}",
    ]
    if not native_only:
        parts.append(
            f"JSON Schema: {json.dumps(schema_record.schema, ensure_ascii=True, indent=2)}"
        )
    if system_prompt:
        parts.insert(0, f"System prompt:\n{system_prompt}")
    if chat_context is not None:
        if isinstance(chat_context, str):
            rendered_context = chat_context
        else:
            rendered_context = json.dumps(chat_context, ensure_ascii=True, indent=2)
        parts.append(f"Chat context:\n{rendered_context}")
    parts.append(f"User request:\n{user_message}")
    parts.append("Return only the final JSON response.")
    return "\n\n".join(parts)


def parse_and_validate_output(
    raw_output: str,
    schema_record: SchemaRecord,
) -> StructuredParseResult:
    text = raw_output.strip()
    if not text:
        raise StructuredOutputError(
            "invalid_model_output",
            "model returned empty output",
            status_code=502,
            retryable=True,
        )

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        failure = StructuredValidationFailure(
            kind="json_parse_error",
            message="model output was not valid JSON",
            details=[
                f"JSON parse error at line {exc.lineno} column {exc.colno}: {exc.msg}"
            ],
            raw_output=text,
        )
        raise StructuredOutputError(
            "invalid_model_output",
            failure.message,
            status_code=502,
            retryable=True,
        ) from exc

    if isinstance(parsed, dict) and set(parsed) == {"response"}:
        parsed = _omit_optional_nulls(parsed["response"], schema_record.schema)
    try:
        jsonschema.validate(parsed, schema_record.schema)
    except jsonschema.ValidationError as exc:
        details = [exc.message]
        if list(exc.path):
            details.append(f"path: {'/'.join(str(part) for part in exc.path)}")
        if exc.validator:
            details.append(f"validator: {exc.validator}")
        raise StructuredOutputError(
            "schema_validation_failed",
            "; ".join(details),
            status_code=502,
            retryable=True,
        ) from exc

    if not isinstance(parsed, (dict, list)):
        raise StructuredOutputError(
            "schema_validation_failed",
            "model output must be a JSON object or array",
            status_code=502,
            retryable=True,
        )

    return StructuredParseResult(parsed=parsed, raw_output=text)


def classify_failure(
    error: StructuredOutputError,
    raw_output: str,
) -> StructuredValidationFailure:
    if error.code == "schema_validation_failed":
        return StructuredValidationFailure(
            kind="schema_validation_error",
            message="model output failed schema validation",
            details=[part.strip() for part in error.message.split(";") if part.strip()],
            raw_output=raw_output.strip(),
        )
    if raw_output.strip():
        return StructuredValidationFailure(
            kind="json_parse_error",
            message="model output was not valid JSON",
            details=[error.message],
            raw_output=raw_output.strip(),
        )
    return StructuredValidationFailure(
        kind="empty_or_garbage_output",
        message="model output was empty",
        details=[error.message],
        raw_output="",
    )


def build_retry_prompt(
    *,
    original_prompt: str,
    schema_record: SchemaRecord,
    failure: StructuredValidationFailure,
    native_only: bool = False,
) -> str:
    if failure.kind == "json_parse_error":
        corrective = (
            "Your previous answer was not valid JSON. Return only a valid JSON object or array. "
            "Do not use markdown fences or any explanatory text."
        )
    elif failure.kind == "schema_validation_error":
        corrective = "Your previous answer did not match the required schema. Fix the JSON so it matches exactly."
    else:
        corrective = "Your previous answer was empty or unusable. Generate the full JSON response from scratch."

    issues = (
        "\n".join(f"- {item}" for item in failure.details)
        if failure.details
        else "- unspecified validation failure"
    )
    if native_only:
        return "\n\n".join(
            [
                corrective,
                f"Validation issues:\n{issues}",
                'Return only the corrected native {"response": ...} envelope using the current output schema.',
            ]
        )
    return "\n\n".join(
        [
            corrective,
            f"Schema ID: {schema_record.schema_id}",
            f"JSON Schema: {json.dumps(schema_record.schema, ensure_ascii=True, indent=2)}",
            "Validation issues:",
            issues,
            "Return only the corrected final JSON. No prose. No code fences.",
            f"Original task context:\n{original_prompt}",
        ]
    )
