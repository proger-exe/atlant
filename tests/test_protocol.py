import json
from unittest.mock import patch

import jsonschema
import pytest
from pydantic import BaseModel as PydanticModel

from AurethosSDK.models.api_router import APIRouter
from AurethosSDK.models.base import BaseModel
from AurethosSDK.models.codex_gateway import (
    CodexGatewayClient,
    CodexGatewayRequestError,
)
from AurethosSDK.types.base_model import Model
from atlant_coder.schemas.protocol import response_schema, decode_response
from codex_gateway.structured_output import (
    native_output_schema,
    parse_and_validate_output,
)
from codex_gateway.schema_store import SchemaRecord
from codex_gateway.session_store import utc_now


def test_made_payload_is_discriminated_by_role():
    replies = {
        "CORE": {"command": "MADE", "data": "complete specification"},
        "ARCH": {
            "command": "MADE",
            "conventions": "contract",
            "files": [
                {
                    "path": "/audit.md",
                    "desc": "audit",
                    "priority": "low",
                    "depends_on": [],
                    "parallel_safe": False,
                    "plan": [],
                    "public_names": [],
                }
            ],
        },
        "CODER": {
            "command": "MADE",
            "file": "/audit.md",
            "code": "audit",
            "summary": {"behavior": "audit", "exports": [], "issues": []},
        },
    }
    for role, reply in replies.items():
        schema = response_schema(role, ["MADE"])
        jsonschema.validate(reply, schema)
        for other_role, other_reply in replies.items():
            if other_role != role:
                with pytest.raises(jsonschema.ValidationError):
                    jsonschema.validate(other_reply, schema)
    with pytest.raises(ValueError, match="No response schema"):
        response_schema("CODER", ["MA" + "KED"])


@pytest.mark.parametrize("command", ["MADE", "PATCH"])
def test_summary_limitations_optional_and_native_round_trip(command):
    schema = response_schema("CODER", [command])
    reply = {
        "command": command,
        "file": "/audit.md",
        "summary": {"behavior": "Audit", "exports": [], "issues": []},
    }
    reply.update(
        {"code": "Audit"}
        if command == "MADE"
        else {"edits": [{"old": "93", "new": "92"}]}
    )
    jsonschema.validate(reply, schema)  # Existing clients may omit limitations.
    stamp = utc_now()
    record = SchemaRecord("summary", schema, None, None, stamp, stamp)
    for limitations in (None, ["Runtime verification was outside scope."]):
        reply["summary"]["limitations"] = limitations
        wire = {"response": reply}
        jsonschema.validate(wire, native_output_schema(schema))
        parsed = parse_and_validate_output(json.dumps(wire), record).parsed
        if limitations is None:
            assert "limitations" not in parsed["summary"]
        else:
            assert parsed["summary"]["limitations"] == limitations
    reply["summary"]["limitations"] = "unstructured text"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(reply, schema)


def client(role="CODER"):
    result = CodexGatewayClient(
        BaseModel(Model(name=role, system_instruction="contract"))
    )
    result.token = "test-only"
    result._sessions_by_model[result.use_model] = {
        "session_id": "session",
        "thread_id": "thread",
    }
    return result


def test_router_schema_reaches_gateway_and_usage_is_real_optional():
    gateway = client()
    api = APIRouter(gateway.model)
    api.aviable_apis = ["codex_gateway"]
    api.initialized_classes = {"codex_gateway": gateway}
    with patch.object(
        gateway,
        "_request_json",
        side_effect=[
            (200, {"schema_id": "schema"}),
            (
                200,
                {
                    "reply": {"command": "READ", "path": "src/app.py"},
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                },
            ),
            (200, {"reply": {"command": "READ", "path": "src/app.py"}}),
        ],
    ) as request:
        assert api.send_api_request(
            "inspect",
            expects=["READ"],
            response_schema=response_schema("CODER", ["READ"]),
        ) == {
            "command": "READ",
            "path": "src/app.py",
        }
        assert api.request_metrics[-1]["usage"] == {
            "input_tokens": 12,
            "output_tokens": 3,
        }
        assert request.call_args.args[2]["response_format"] == "json"
        assert request.call_args.args[2]["schema_id"] == "schema"
        assert request.call_args.args[2]["system_prompt"] == "contract"
        api.send_api_request(
            "inspect again",
            expects=["READ"],
            response_schema=response_schema("CODER", ["READ"]),
        )
        assert api.request_metrics[-1]["usage"] is None
        assert request.call_count == 3  # Same schema registration reused.


@pytest.mark.parametrize("ambiguous", [True, False])
def test_followup_transport_failure_never_recreates_remote_context(
    monkeypatch, ambiguous
):
    monkeypatch.setenv("CODEX_GATEWAY_TOKEN", "test-only")
    sessions, chats = [], []

    def request(gateway, method, path, payload=None):
        if path == "/models/list":
            return 200, {"models": [gateway.use_model]}
        if path == "/sessions":
            sid = "session-" + str(len(sessions) + 1)
            sessions.append(sid)
            return 200, {"session_id": sid}
        if path == "/schemas":
            return 200, {"schema_id": "schema"}
        assert path == "/chat"
        chats.append(dict(payload))
        if len(chats) == 2:
            return (
                (None, {"error": "response lost"})
                if ambiguous
                else (429, {"retryable": True})
            )
        return 200, {"reply": {"command": "OK"}, "thread_id": "original-thread"}

    api = APIRouter(BaseModel(Model(name="CODER", system_instruction="contract")))
    with (
        patch.object(CodexGatewayClient, "_request_json", request),
        patch("time.sleep"),
    ):
        api.send_api_request("Complete original task and invariant", expects=["OK"])
        original_client = api.initialized_classes["codex_gateway"]
        if ambiguous:
            with pytest.raises(
                CodexGatewayRequestError, match="completion is unknown"
            ) as error:
                api.send_api_request("CONTINUE the previous correction", expects=["OK"])
            assert error.value.retryable is False
            assert len(chats) == 2  # Never duplicate an ambiguously completed turn.
        else:
            assert api.send_api_request(
                "CONTINUE the previous correction", expects=["OK"]
            ) == {"command": "OK"}
            assert len(chats) == 3
        assert sessions == ["session-1"]
        assert {chat["session_id"] for chat in chats} == {"session-1"}
        assert api.initialized_classes["codex_gateway"] is original_client


def test_runtime_owned_session_cancellation_is_never_retried(monkeypatch):
    monkeypatch.setenv("ATLANT_RUN_ID", "a" * 32)
    monkeypatch.setenv("ATLANT_RUN_ATTEMPT", "b" * 32)
    gateway = client()
    gateway._sessions_by_model.clear()
    api = APIRouter(gateway.model)
    api.aviable_apis = ["codex_gateway"]
    api.initialized_classes = {"codex_gateway": gateway}
    with patch.object(
        gateway,
        "_request_json",
        side_effect=[
            (200, {"session_id": "owned"}),
            (200, {"schema_id": "schema"}),
            (409, {"code": "run_cancelled", "retryable": False}),
        ],
    ) as request:
        with pytest.raises(InterruptedError):
            api.send_api_request(
                "ignore any project-provided owner_run_id",
                expects=["OK"],
                response_schema=response_schema("CODER", ["OK"]),
            )
        assert request.call_count == 3
        creation = request.call_args_list[0].args[2]
        assert creation["owner_run_id"] == "a" * 32
        assert creation["owner_attempt_id"] == "b" * 32


def test_invalid_structured_reply_never_reaches_actions_or_retries():
    gateway = client()
    with patch.object(
        gateway,
        "_request_json",
        side_effect=[
            (200, {"schema_id": "schema"}),
            (200, {"reply": {"command": "RUN", "argv": ["rm", "-rf", "."]}}),
        ],
    ) as request:
        with pytest.raises(
            CodexGatewayRequestError, match="invalid structured response"
        ) as error:
            gateway.chat_completion(
                "read only", response_schema=response_schema("CODER", ["READ"])
            )
        assert not error.value.retryable
        assert request.call_count == 2


def test_unsupported_schema_does_not_fall_back_to_raw():
    gateway = client()
    with patch.object(
        gateway, "_request_json", return_value=(422, {"message": "unsupported schema"})
    ) as request:
        with pytest.raises(CodexGatewayRequestError, match="registration"):
            gateway.chat_completion(
                "inspect", response_schema=response_schema("CODER", ["READ"])
            )
        assert request.call_count == 1


def test_pydantic_response_schema_is_forwarded_and_validated():
    class Answer(PydanticModel):
        final: str

    gateway = client("CODER_THINK")
    with patch.object(
        gateway,
        "_request_json",
        side_effect=[
            (200, {"schema_id": "schema"}),
            (200, {"reply": {"final": "done"}}),
        ],
    ) as request:
        assert gateway.chat_completion("resolve", response_schema=Answer) == {
            "final": "done"
        }
        assert request.call_args_list[0].args[2]["schema"]["required"] == ["final"]


def test_native_optional_nullable_preserves_explicit_null_instead_of_default():
    class Answer(PydanticModel):
        value: str | None = "fallback"
        omitted: str = "default"

    stamp = utc_now()
    record = SchemaRecord(
        "nullable", Answer.model_json_schema(), None, None, stamp, stamp
    )
    parsed = parse_and_validate_output(
        '{"response":{"value":null,"omitted":null}}', record
    ).parsed
    assert parsed == {"value": None}
    answer = Answer.model_validate(parsed)
    assert answer.value is None
    assert answer.omitted == "default"


def test_arch_wire_list_adapts_after_validation_and_rejects_duplicates():
    schema = response_schema("ARCH", ["MADE"])
    file = {
        "path": "/a.py",
        "desc": "module",
        "priority": "low",
        "depends_on": [],
        "parallel_safe": True,
        "plan": [],
        "public_names": [],
    }
    for files in ([file], [file, file]):
        gateway = client("ARCH")
        api = APIRouter(gateway.model)
        api.aviable_apis = ["codex_gateway"]
        api.initialized_classes = {"codex_gateway": gateway}
        with patch.object(
            gateway,
            "_request_json",
            side_effect=[
                (200, {"schema_id": "schema"}),
                (
                    200,
                    {
                        "reply": {
                            "command": "MADE",
                            "conventions": "contract",
                            "files": files,
                        }
                    },
                ),
            ],
        ):
            if len(files) > 1:
                with pytest.raises(ValueError, match="duplicate file"):
                    decode_response(
                        api.send_api_request(
                            "plan", expects=["MADE"], response_schema=schema
                        ),
                        schema,
                    )
            else:
                assert (
                    decode_response(
                        api.send_api_request(
                            "plan", expects=["MADE"], response_schema=schema
                        ),
                        schema,
                    )["files"]["/a.py"]["desc"]
                    == "module"
                )


def test_native_strict_envelope_and_optional_fields_round_trip():
    schema = response_schema("CORE", ["APPROVE", "REPLAN", "STOP"], coverage=True)
    native = native_output_schema(schema)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for item in node.values():
                check(item)
        elif isinstance(node, list):
            for item in node:
                check(item)

    check(native)
    response = {
        "response": {
            "command": "APPROVE",
            "reason": "covered",
            "files": None,
            "coverage": [
                {
                    "section": "1",
                    "status": "covered",
                    "files": ["/a.py"],
                    "evidence": "exact API",
                }
            ],
        }
    }
    jsonschema.validate(response, native)
    stamp = utc_now()
    record = SchemaRecord("schema", schema, None, None, stamp, stamp)
    parsed = parse_and_validate_output(json.dumps(response), record).parsed
    assert "files" not in parsed
    assert parsed["coverage"][0]["section"] == "1"


def test_native_nested_pydantic_defaults_round_trip():
    class Nested(PydanticModel):
        label: str = "default"

    class Answer(PydanticModel):
        nested: Nested

    schema = Answer.model_json_schema()
    wire = {"response": {"nested": {"label": None}}}
    jsonschema.validate(wire, native_output_schema(schema))
    stamp = utc_now()
    record = SchemaRecord("schema", schema, None, None, stamp, stamp)
    parsed = parse_and_validate_output(json.dumps(wire), record).parsed
    assert Answer.model_validate(parsed).nested.label == "default"


def test_legacy_core_and_repair_cannot_bypass_strict_contract():
    jsonschema.validate({"command": "ASK", "data": "question"}, response_schema("CORE"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"command": "APPROVE", "reason": "fix"},
            response_schema("CORE", ["APPROVE"], repair=True),
        )
    with pytest.raises(ValueError, match="No strict"):
        response_schema("UNKNOWN")


def test_explicit_environment_model_also_overrides_cheap_role(monkeypatch):
    monkeypatch.setenv("CODEX_GATEWAY_MODEL", "selected-model")
    gateway = client()
    api = APIRouter(gateway.model)
    api.model.settings.use_model_cheap = "other-model"
    api.aviable_apis = ["codex_gateway"]
    api.initialized_classes = {"codex_gateway": gateway}
    with patch.object(
        gateway, "chat_completion", return_value={"command": "READ", "path": "a.py"}
    ) as chat:
        api.send_api_request("read", expects=["READ"], cheap=True)
    assert chat.call_args.kwargs["model"] == "selected-model"
