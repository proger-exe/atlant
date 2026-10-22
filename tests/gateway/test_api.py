from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
import httpx
from pathlib import Path
from unittest.mock import patch

from codex_gateway.app import create_app
from codex_gateway.config import (
    DEFAULT_READ_ONLY_PATH_CONFLICT_MESSAGE,
    READ_ONLY_PATH_CONFLICT_DETAIL,
    load_settings,
)
from codex_gateway.codex_runner import (
    CodexRunResult,
    CodexRunnerError,
    build_command,
    build_subprocess_env,
)
from codex_gateway.session_store import SessionRecord
from codex_gateway.structured_output import validate_json_schema


class GatewayApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_only_persists_and_forces_read_only(self):
        response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={"generation_only": True, "sandbox": "workspace-write"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["generation_only"])
        self.assertEqual(response.json()["sandbox"], "read-only")
        loaded = await self.client.get(
            "/sessions/" + response.json()["session_id"], headers=self.auth_headers()
        )
        self.assertTrue(loaded.json()["generation_only"])

    def setUp(self) -> None:
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.allowed_root = Path(self.temp_dir.name) / "projects"
        self.allowed_root.mkdir(parents=True)

        os.environ["CODEX_API_TOKEN"] = "test-token"
        os.environ["CODEX_API_ALLOWED_ROOT"] = str(self.allowed_root)
        os.environ["CODEX_API_DEFAULT_WORKSPACE"] = str(self.allowed_root)
        os.environ["CODEX_API_DEFAULT_SANDBOX"] = "workspace-write"
        os.environ["CODEX_API_STATE_DIR"] = str(Path(self.temp_dir.name) / "state")
        os.environ.pop("CODEX_API_LOG_FILE", None)
        self.models_config_path = Path(self.temp_dir.name) / "models.json"
        self.write_models_config()
        os.environ["CODEX_API_MODELS_FILE"] = str(self.models_config_path)

        self.which_patcher = patch(
            "codex_gateway.config.shutil.which", return_value="/usr/bin/codex"
        )
        self.which_patcher.start()
        self.addCleanup(self.which_patcher.stop)

    async def asyncSetUp(self) -> None:
        self.app = create_app()
        self.transport = httpx.ASGITransport(app=self.app)
        self.client = httpx.AsyncClient(
            transport=self.transport,
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer test-token"}

    async def recreate_app(self) -> None:
        await self.client.aclose()
        self.app = create_app()
        self.transport = httpx.ASGITransport(app=self.app)
        self.client = httpx.AsyncClient(
            transport=self.transport,
            base_url="http://testserver",
        )

    def write_models_config(
        self,
        models: list[str] | None = None,
        default_model: str | None = None,
    ) -> None:
        self.models_config_path.write_text(
            json.dumps(
                {
                    "models": models or [],
                    "default_model": default_model,
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    async def test_health_requires_no_auth(self) -> None:
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    async def test_lost_context_is_explicit_http_error_without_fresh_replay(
        self,
    ) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="original-thread"),
        ):
            created = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"message": "full original task"},
            )
        self.assertEqual(created.status_code, 200)
        session_id = created.json()["session_id"]
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=CodexRunnerError("session lost", retryable=True),
        ) as run:
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"session_id": session_id, "message": "CONTINUE"},
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "session_context_lost")
        self.assertFalse(response.json()["retryable"])
        self.assertEqual(run.call_count, 1)

    async def test_create_schema_and_read_it_back(self) -> None:
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        }
        create_response = await self.client.post(
            "/schemas",
            headers=self.auth_headers(),
            json={"name": "status-only", "schema": schema},
        )
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        get_response = await self.client.get(
            f"/schemas/{payload['schema_id']}",
            headers=self.auth_headers(),
        )
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json()["schema"], schema)

    async def test_create_schema_rejects_invalid_json_schema(self) -> None:
        response = await self.client.post(
            "/schemas",
            headers=self.auth_headers(),
            json={"schema": {"type": "wat"}},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "invalid_schema_payload")

    async def test_create_session_requires_auth(self) -> None:
        response = await self.client.post("/sessions", json={})
        self.assertEqual(response.status_code, 401)

    async def test_create_session_rejects_workspace_outside_root(self) -> None:
        outside = Path(self.temp_dir.name) / "outside"
        response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={"workspace": str(outside)},
        )
        self.assertEqual(response.status_code, 403)

    async def test_list_models_returns_configured_allowlist(self) -> None:
        self.write_models_config(["gpt-5-codex", "custom-a", "custom-b"], "custom-a")
        await self.recreate_app()

        response = await self.client.get(
            "/models/list",
            headers=self.auth_headers(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "models": ["gpt-5-codex", "custom-a", "custom-b"],
                "default_model": "custom-a",
            },
        )

    async def test_create_session_rejects_model_outside_allowlist(self) -> None:
        self.write_models_config(["custom-a", "custom-b"])
        await self.recreate_app()

        response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={"model": "custom-c"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "unsupported model")

    async def test_create_session_uses_default_model_from_allowlist(self) -> None:
        self.write_models_config(["custom-a", "custom-b"], "custom-b")
        await self.recreate_app()

        response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "custom-b")

    async def test_chat_creates_session_and_returns_reply(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="thread-1"),
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"message": "hello"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["reply"], "ok")
        self.assertTrue(payload["created_session"])
        session_response = await self.client.get(
            f"/sessions/{payload['session_id']}",
            headers=self.auth_headers(),
        )
        self.assertEqual(session_response.status_code, 200)
        self.assertEqual(session_response.json()["thread_id"], "thread-1")

    async def test_chat_accepts_message_above_old_local_limit(self) -> None:
        long_message = "x" * 15000
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="thread-1"),
        ) as run_mock:
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"message": long_message},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "ok")
        self.assertEqual(run_mock.call_args.kwargs["prompt"], long_message)

    async def test_chat_raw_system_prompt_wraps_prompt(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="thread-1"),
        ) as run_mock:
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={
                    "message": "hello",
                    "system_prompt": "Be concise.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            run_mock.call_args.kwargs["prompt"],
            "System prompt:\nBe concise.\n\nUser request:\nhello",
        )

    async def test_chat_raw_system_prompt_and_context_wrap_prompt_in_order(
        self,
    ) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="thread-1"),
        ) as run_mock:
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={
                    "message": "hello",
                    "system_prompt": "Be concise.",
                    "chat_context": {"audience": "owner"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            run_mock.call_args.kwargs["prompt"],
            'System prompt:\nBe concise.\n\nChat context:\n{\n  "audience": "owner"\n}\n\nUser request:\nhello',
        )

    async def test_chat_accepts_context_above_old_local_limit(self) -> None:
        long_context = {"document": "x" * 15000}
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="thread-1"),
        ) as run_mock:
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={
                    "message": "hello",
                    "chat_context": long_context,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "ok")
        self.assertIn('"document"', run_mock.call_args.kwargs["prompt"])
        self.assertIn(long_context["document"], run_mock.call_args.kwargs["prompt"])

    async def test_chat_rejects_blank_system_prompt(self) -> None:
        response = await self.client.post(
            "/chat",
            headers=self.auth_headers(),
            json={"message": "hello", "system_prompt": "   "},
        )

        self.assertEqual(response.status_code, 422)

    async def test_structured_chat_accepts_system_prompt_and_checks_schema(
        self,
    ) -> None:
        response = await self.client.post(
            "/chat",
            headers=self.auth_headers(),
            json={
                "message": "hello",
                "response_format": "json",
                "schema_id": "any",
                "system_prompt": "Be concise.",
            },
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "schema_not_found")

    async def test_structured_chat_returns_structured_reply(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["status", "answer"],
            "additionalProperties": False,
        }
        schema_response = await self.client.post(
            "/schemas",
            headers=self.auth_headers(),
            json={"schema": schema},
        )
        schema_id = schema_response.json()["schema_id"]
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(
                reply='{"status":"ok","answer":"brief"}', thread_id="thread-structured"
            ),
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={
                    "message": "hello",
                    "response_format": "json",
                    "schema_id": schema_id,
                    "chat_context": {"audience": "owner"},
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["output_mode"], "json")
        self.assertEqual(payload["used_schema_id"], schema_id)
        self.assertEqual(payload["reply"]["status"], "ok")
        self.assertEqual(payload["reply"]["answer"], "brief")
        self.assertNotIn("structured_reply", payload)

    async def test_structured_chat_retries_after_invalid_json(self) -> None:
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        }
        schema_response = await self.client.post(
            "/schemas",
            headers=self.auth_headers(),
            json={"schema": schema},
        )
        schema_id = schema_response.json()["schema_id"]
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=[
                CodexRunResult(reply="not json", thread_id="thread-a"),
                CodexRunResult(reply='{"status":"ok"}', thread_id="thread-b"),
            ],
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={
                    "message": "hello",
                    "response_format": "json",
                    "schema_id": schema_id,
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["validation_retries"], 1)
        self.assertEqual(response.json()["reply"]["status"], "ok")

    async def test_structured_chat_requires_schema_id(self) -> None:
        response = await self.client.post(
            "/chat",
            headers=self.auth_headers(),
            json={"message": "hello", "response_format": "json"},
        )
        self.assertEqual(response.status_code, 422)

    async def test_strict_schema_system_prompt_usage_and_single_correction(
        self,
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"command": {"const": "OK", "type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        }
        registered = await self.client.post(
            "/schemas", headers=self.auth_headers(), json={"schema": schema}
        )
        payload = {
            "message": "respond",
            "system_prompt": "Coordinator rules",
            "response_format": "json",
            "schema_id": registered.json()["schema_id"],
        }
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=[
                CodexRunResult("invalid", "thread", {"input_tokens": 10}),
                CodexRunResult(
                    '{"response":{"command":"OK"}}', "thread", {"input_tokens": 12}
                ),
            ],
        ) as run:
            response = await self.client.post(
                "/chat", headers=self.auth_headers(), json=payload
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["usage"], {"input_tokens": 12})
        self.assertEqual(response.json()["validation_retries"], 1)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertIn("Coordinator rules", call.kwargs["prompt"])
            self.assertEqual(call.kwargs["output_schema"]["required"], ["response"])
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult("invalid", "thread"),
        ) as run:
            response = await self.client.post(
                "/chat", headers=self.auth_headers(), json=payload
            )
        self.assertEqual(run.call_count, 2)
        self.assertEqual(response.status_code, 502)
        self.assertFalse(response.json()["retryable"])

    async def test_generation_usage_and_compact_resume_reset_and_schema_change(self):
        schema = {
            "type": "object",
            "properties": {"command": {"const": "OK", "type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        }
        registered = await self.client.post(
            "/schemas", headers=self.auth_headers(), json={"schema": schema}
        )
        session = await self.client.post(
            "/sessions", headers=self.auth_headers(), json={"generation_only": True}
        )
        sid = session.json()["session_id"]
        payload = {
            "session_id": sid,
            "message": "task",
            "system_prompt": "ROLE A",
            "response_format": "json",
            "schema_id": registered.json()["schema_id"],
        }
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=[
                CodexRunResult(
                    '{"command":"OK"}',
                    "thread",
                    {"input_tokens": 10, "output_tokens": 2},
                ),
                CodexRunResult(
                    "bad json", "thread", {"input_tokens": 25, "output_tokens": 3}
                ),
                CodexRunResult(
                    '{"command":"OK"}',
                    "thread",
                    {"input_tokens": 44, "output_tokens": 5},
                ),
                CodexRunResult(
                    '{"command":"DONE"}',
                    "thread",
                    {"input_tokens": 64, "output_tokens": 7},
                ),
                CodexRunResult(
                    '{"command":"DONE"}',
                    "new-thread",
                    {"input_tokens": 9, "output_tokens": 1},
                ),
            ],
        ) as run:
            first = await self.client.post(
                "/chat", headers=self.auth_headers(), json=payload
            )
            second = await self.client.post(
                "/chat", headers=self.auth_headers(), json=payload
            )
            self.assertEqual(
                first.json()["usage"], {"input_tokens": 10, "output_tokens": 2}
            )
            self.assertEqual(
                second.json()["usage"], {"input_tokens": 34, "output_tokens": 3}
            )
            self.assertEqual(second.json()["validation_retries"], 1)
            self.assertIn("ROLE A", run.call_args_list[0].kwargs["prompt"])
            self.assertNotIn("ROLE A", run.call_args_list[1].kwargs["prompt"])
            self.assertNotIn("ROLE A", run.call_args_list[2].kwargs["prompt"])
            schema["properties"]["command"]["const"] = "DONE"
            registered = await self.client.post(
                "/schemas", headers=self.auth_headers(), json={"schema": schema}
            )
            payload.update(
                schema_id=registered.json()["schema_id"], system_prompt="ROLE B"
            )
            third = await self.client.post(
                "/chat", headers=self.auth_headers(), json=payload
            )
            self.assertEqual(third.json()["reply"], {"command": "DONE"})
            self.assertEqual(third.json()["usage"]["input_tokens"], 20)
            self.assertIn("ROLE B", run.call_args_list[3].kwargs["prompt"])
            self.assertEqual(
                run.call_args_list[3].kwargs["output_schema"]["properties"]["response"][
                    "properties"
                ]["command"]["const"],
                "DONE",
            )
            await self.client.post(
                "/sessions/" + sid + "/reset", headers=self.auth_headers()
            )
            reset = await self.client.post(
                "/chat", headers=self.auth_headers(), json=payload
            )
            self.assertEqual(reset.json()["usage"]["input_tokens"], 9)
            self.assertIn("ROLE B", run.call_args_list[4].kwargs["prompt"])
            for call in run.call_args_list:
                self.assertNotIn("JSON Schema:", call.kwargs["prompt"])
                self.assertIsNotNone(call.kwargs["output_schema"])

    async def test_structured_chat_rejects_unknown_schema(self) -> None:
        response = await self.client.post(
            "/chat",
            headers=self.auth_headers(),
            json={
                "message": "hello",
                "response_format": "json",
                "schema_id": "missing",
            },
        )
        self.assertEqual(response.status_code, 404)

    async def test_chat_reuses_existing_session(self) -> None:
        create_response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={},
        )
        self.assertEqual(create_response.status_code, 200)
        session_id = create_response.json()["session_id"]

        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="again", thread_id="thread-2"),
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"session_id": session_id, "message": "next"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["created_session"])
        self.assertEqual(response.json()["thread_id"], "thread-2")

    async def test_chat_returns_409_when_session_is_busy(self) -> None:
        create_response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={},
        )
        session_id = create_response.json()["session_id"]
        lock = self.app.state.store.acquire_session_lock(session_id)
        lock.acquire()
        try:
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"session_id": session_id, "message": "next"},
            )
        finally:
            lock.release()

        self.assertEqual(response.status_code, 409)

    async def test_chat_rejects_dangerous_sandbox(self) -> None:
        response = await self.client.post(
            "/chat",
            headers=self.auth_headers(),
            json={"message": "x", "sandbox": "danger-full-access"},
        )
        self.assertEqual(response.status_code, 422)

    async def test_create_session_accepts_workspace_write_sandbox(self) -> None:
        response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={"sandbox": "workspace-write"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sandbox"], "workspace-write")

    async def test_create_session_rejects_read_only_when_gateway_state_is_inside_workspace(
        self,
    ) -> None:
        os.environ["CODEX_API_STATE_DIR"] = str(self.allowed_root / ".codex-api")
        await self.recreate_app()

        response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={"sandbox": "read-only"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], READ_ONLY_PATH_CONFLICT_DETAIL)
        self.assertEqual(
            list((self.app.state.settings.state_dir / "sessions").glob("*.json")),
            [],
        )

    async def test_chat_accepts_read_only_sandbox(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="thread-read-only"),
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"message": "hello", "sandbox": "read-only"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sandbox"], "read-only")

    async def test_chat_rejects_new_read_only_session_when_gateway_state_is_inside_workspace(
        self,
    ) -> None:
        os.environ["CODEX_API_STATE_DIR"] = str(self.allowed_root / ".codex-api")
        await self.recreate_app()

        with patch("codex_gateway.app.run_codex") as run_mock:
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"message": "hello", "sandbox": "read-only"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], READ_ONLY_PATH_CONFLICT_DETAIL)
        self.assertFalse(run_mock.called)
        self.assertEqual(
            list((self.app.state.settings.state_dir / "sessions").glob("*.json")),
            [],
        )
        self.assertEqual(
            list((self.app.state.settings.state_dir / "transcripts").glob("*.md")),
            [],
        )

    async def test_chat_ignores_unknown_fields_and_uses_valid_sandbox(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="thread-1"),
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={
                    "message": "hello",
                    "sandbox": "read-only",
                    "workspace-allow": "read-only",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sandbox"], "read-only")

    async def test_chat_rejects_model_outside_allowlist_when_creating_session(
        self,
    ) -> None:
        self.write_models_config(["custom-a"])
        await self.recreate_app()

        response = await self.client.post(
            "/chat",
            headers=self.auth_headers(),
            json={"message": "x", "model": "custom-b"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "unsupported model")

    async def test_chat_rejects_session_overrides(self) -> None:
        create_response = await self.client.post(
            "/sessions",
            headers=self.auth_headers(),
            json={},
        )
        session_id = create_response.json()["session_id"]

        response = await self.client.post(
            "/chat",
            headers=self.auth_headers(),
            json={
                "session_id": session_id,
                "message": "next",
                "workspace": str(self.allowed_root),
            },
        )
        self.assertEqual(response.status_code, 422)

    async def test_chat_returns_503_on_codex_failure(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=CodexRunnerError("backend failed", retryable=True),
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"message": "hello"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json()["retryable"])

    async def test_chat_returns_413_on_codex_context_overflow(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=CodexRunnerError(
                "Payload Too Large: maximum context length exceeded"
            ),
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"message": "hello"},
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            response.json()["detail"],
            "request context too large; reduce message or chat_context",
        )

    async def test_chat_returns_413_with_reset_hint_for_resumed_session(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="ok", thread_id="thread-1"),
        ):
            create_response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"message": "hello"},
            )

        session_id = create_response.json()["session_id"]
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=CodexRunnerError("maximum context length exceeded"),
        ):
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"session_id": session_id, "message": "next"},
            )

        self.assertEqual(response.status_code, 413)
        self.assertIn("reset the session thread", response.json()["detail"])

    async def test_chat_rejects_resumed_read_only_session_when_gateway_state_is_inside_workspace(
        self,
    ) -> None:
        os.environ["CODEX_API_STATE_DIR"] = str(self.allowed_root / ".codex-api")
        await self.recreate_app()
        record = self.app.state.store.create_session(
            workspace=str(self.allowed_root),
            model=None,
            sandbox="read-only",
        )

        with patch("codex_gateway.app.run_codex") as run_mock:
            response = await self.client.post(
                "/chat",
                headers=self.auth_headers(),
                json={"session_id": record.session_id, "message": "next"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], READ_ONLY_PATH_CONFLICT_DETAIL)
        self.assertFalse(run_mock.called)
        self.assertEqual(
            list((self.app.state.settings.state_dir / "transcripts").glob("*.md")),
            [],
        )

    async def test_job_completes_and_exposes_result(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="done", thread_id="thread-job"),
        ):
            create_response = await self.client.post(
                "/jobs",
                headers=self.auth_headers(),
                json={"message": "run in background"},
            )

            self.assertEqual(create_response.status_code, 200)
            payload = create_response.json()
            self.assertEqual(payload["status"], "queued")
            job_id = payload["job_id"]

            for _ in range(20):
                status_response = await self.client.get(
                    f"/jobs/{job_id}",
                    headers=self.auth_headers(),
                )
                self.assertEqual(status_response.status_code, 200)
                status_payload = status_response.json()
                if status_payload["status"] == "completed":
                    break
                await asyncio.sleep(0.01)

        self.assertEqual(status_payload["status"], "completed")
        self.assertEqual(status_payload["reply"], "done")
        self.assertEqual(status_payload["thread_id"], "thread-job")
        self.assertTrue(status_payload["created_session"])

    async def test_job_accepts_message_above_old_local_limit(self) -> None:
        long_message = "x" * 15000
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="done", thread_id="thread-job"),
        ) as run_mock:
            create_response = await self.client.post(
                "/jobs",
                headers=self.auth_headers(),
                json={"message": long_message},
            )

            self.assertEqual(create_response.status_code, 200)
            job_id = create_response.json()["job_id"]

            for _ in range(20):
                status_response = await self.client.get(
                    f"/jobs/{job_id}",
                    headers=self.auth_headers(),
                )
                self.assertEqual(status_response.status_code, 200)
                status_payload = status_response.json()
                if status_payload["status"] == "completed":
                    break
                await asyncio.sleep(0.01)

        self.assertEqual(status_payload["status"], "completed")
        self.assertEqual(status_payload["reply"], "done")
        self.assertEqual(run_mock.call_args.kwargs["prompt"], long_message)

    async def test_job_accepts_system_prompt_in_raw_mode(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            return_value=CodexRunResult(reply="done", thread_id="thread-job"),
        ) as run_mock:
            create_response = await self.client.post(
                "/jobs",
                headers=self.auth_headers(),
                json={
                    "message": "run in background",
                    "system_prompt": "Be concise.",
                },
            )

            self.assertEqual(create_response.status_code, 200)
            job_id = create_response.json()["job_id"]

            for _ in range(20):
                status_response = await self.client.get(
                    f"/jobs/{job_id}",
                    headers=self.auth_headers(),
                )
                self.assertEqual(status_response.status_code, 200)
                status_payload = status_response.json()
                if status_payload["status"] == "completed":
                    break
                await asyncio.sleep(0.01)

        self.assertEqual(status_payload["status"], "completed")
        self.assertEqual(
            run_mock.call_args.kwargs["prompt"],
            "System prompt:\nBe concise.\n\nUser request:\nrun in background",
        )

    async def test_job_rejects_model_outside_allowlist_when_creating_session(
        self,
    ) -> None:
        self.write_models_config(["custom-a"])
        await self.recreate_app()

        response = await self.client.post(
            "/jobs",
            headers=self.auth_headers(),
            json={"message": "run in background", "model": "custom-b"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "unsupported model")

    async def test_job_returns_failure_payload(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=CodexRunnerError("backend failed", retryable=True),
        ):
            create_response = await self.client.post(
                "/jobs",
                headers=self.auth_headers(),
                json={"message": "run in background"},
            )
            self.assertEqual(create_response.status_code, 200)
            job_id = create_response.json()["job_id"]

            for _ in range(20):
                status_response = await self.client.get(
                    f"/jobs/{job_id}",
                    headers=self.auth_headers(),
                )
                status_payload = status_response.json()
                if status_payload["status"] == "failed":
                    break
                await asyncio.sleep(0.01)

        self.assertEqual(status_payload["status"], "failed")
        self.assertEqual(status_payload["error"]["code"], "codex_unavailable")
        self.assertTrue(status_payload["error"]["retryable"])

    async def test_job_returns_context_too_large_error_on_codex_overflow(self) -> None:
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=CodexRunnerError("prompt is too long for this model"),
        ):
            create_response = await self.client.post(
                "/jobs",
                headers=self.auth_headers(),
                json={"message": "run in background"},
            )
            self.assertEqual(create_response.status_code, 200)
            job_id = create_response.json()["job_id"]

            for _ in range(20):
                status_response = await self.client.get(
                    f"/jobs/{job_id}",
                    headers=self.auth_headers(),
                )
                status_payload = status_response.json()
                if status_payload["status"] == "failed":
                    break
                await asyncio.sleep(0.01)

        self.assertEqual(status_payload["status"], "failed")
        self.assertEqual(status_payload["error"]["code"], "context_too_large")
        self.assertFalse(status_payload["error"]["retryable"])
        self.assertIn(
            "reduce message or chat_context", status_payload["error"]["message"]
        )

    async def test_job_rejects_read_only_when_gateway_state_is_inside_workspace(
        self,
    ) -> None:
        os.environ["CODEX_API_STATE_DIR"] = str(self.allowed_root / ".codex-api")
        await self.recreate_app()

        response = await self.client.post(
            "/jobs",
            headers=self.auth_headers(),
            json={"message": "run in background", "sandbox": "read-only"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], READ_ONLY_PATH_CONFLICT_DETAIL)
        self.assertEqual(
            list((self.app.state.settings.state_dir / "jobs").glob("*.json")),
            [],
        )

    async def test_job_rejects_existing_read_only_session_before_creating_job_record(
        self,
    ) -> None:
        os.environ["CODEX_API_STATE_DIR"] = str(self.allowed_root / ".codex-api")
        await self.recreate_app()
        record = self.app.state.store.create_session(
            workspace=str(self.allowed_root),
            model=None,
            sandbox="read-only",
        )

        response = await self.client.post(
            "/jobs",
            headers=self.auth_headers(),
            json={"session_id": record.session_id, "message": "run in background"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], READ_ONLY_PATH_CONFLICT_DETAIL)
        self.assertEqual(
            list((self.app.state.settings.state_dir / "jobs").glob("*.json")),
            [],
        )

    async def test_reset_rejects_read_only_when_gateway_state_is_inside_workspace(
        self,
    ) -> None:
        os.environ["CODEX_API_STATE_DIR"] = str(self.allowed_root / ".codex-api")
        await self.recreate_app()
        record = self.app.state.store.create_session(
            workspace=str(self.allowed_root),
            model=None,
            sandbox="read-only",
        )

        response = await self.client.post(
            f"/sessions/{record.session_id}/reset",
            headers=self.auth_headers(),
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], READ_ONLY_PATH_CONFLICT_DETAIL)


class CodexRunnerTests(unittest.TestCase):
    def test_build_command_disables_interactive_approvals(self) -> None:
        session = SessionRecord(
            session_id="session-1",
            workspace="/tmp/workspace",
            model="gpt-5-codex",
            sandbox="workspace-write",
            thread_id=None,
            created_at="2026-03-20T00:00:00+00:00",
            updated_at="2026-03-20T00:00:00+00:00",
        )

        command = build_command(
            "/usr/bin/codex",
            session,
            "hello",
            "/tmp/output.txt",
            allow_network=True,
        )

        self.assertEqual(
            command[:6],
            [
                "/usr/bin/codex",
                "-a",
                "never",
                "-c",
                "sandbox.network_access=true",
                "exec",
            ],
        )
        self.assertIn("-s", command)
        self.assertEqual(command[command.index("-s") + 1], "workspace-write")

    def test_build_command_can_disable_network(self) -> None:
        session = SessionRecord(
            session_id="session-1",
            workspace="/tmp/workspace",
            model=None,
            sandbox="read-only",
            thread_id=None,
            created_at="2026-03-20T00:00:00+00:00",
            updated_at="2026-03-20T00:00:00+00:00",
        )

        command = build_command(
            "/usr/bin/codex",
            session,
            "hello",
            "/tmp/output.txt",
            allow_network=False,
        )

        self.assertEqual(
            command[:6],
            [
                "/usr/bin/codex",
                "-a",
                "never",
                "-c",
                "sandbox.network_access=false",
                "exec",
            ],
        )
        self.assertIn("-s", command)
        self.assertEqual(command[command.index("-s") + 1], "read-only")

    def test_build_subprocess_env_keeps_workspace_write_environment(self) -> None:
        session = SessionRecord(
            session_id="session-1",
            workspace="/tmp/workspace",
            model=None,
            sandbox="workspace-write",
            thread_id=None,
            created_at="2026-03-20T00:00:00+00:00",
            updated_at="2026-03-20T00:00:00+00:00",
        )

        with (
            tempfile.TemporaryDirectory() as state_dir,
            patch.dict(
                os.environ,
                {
                    "CODEX_HOME": "/tmp/original-home",
                    "CODEX_SQLITE_HOME": "/tmp/original-sqlite",
                },
                clear=True,
            ),
        ):
            environment = build_subprocess_env(session, state_dir=Path(state_dir))

        self.assertEqual(environment["CODEX_HOME"], "/tmp/original-home")
        self.assertEqual(environment["CODEX_SQLITE_HOME"], "/tmp/original-sqlite")

    def test_build_subprocess_env_isolates_read_only_and_seeds_auth(self) -> None:
        session = SessionRecord(
            session_id="session-1",
            workspace="/tmp/workspace",
            model=None,
            sandbox="read-only",
            thread_id=None,
            created_at="2026-03-20T00:00:00+00:00",
            updated_at="2026-03-20T00:00:00+00:00",
        )

        with (
            tempfile.TemporaryDirectory() as source_home,
            tempfile.TemporaryDirectory() as state_dir,
        ):
            source_auth = Path(source_home) / "auth.json"
            source_auth.write_text('{"token":"fresh"}\n', encoding="utf-8")
            managed_home = Path(state_dir) / "codex-read-only-home" / session.session_id
            managed_home.mkdir(parents=True)
            (managed_home / "auth.json").write_text(
                '{"token":"stale"}\n', encoding="utf-8"
            )
            stale_config = managed_home / "config.toml"
            stale_config.write_text('model = "gpt-5.4"\n', encoding="utf-8")
            os.utime(managed_home / "auth.json", (1, 1))
            os.utime(source_auth, None)

            with patch.dict(os.environ, {"CODEX_HOME": source_home}, clear=True):
                environment = build_subprocess_env(session, state_dir=Path(state_dir))

            self.assertEqual(environment["CODEX_HOME"], str(managed_home))
            self.assertEqual(
                environment["CODEX_SQLITE_HOME"],
                str(Path(state_dir) / "codex-read-only-sqlite" / session.session_id),
            )
            self.assertEqual(
                (managed_home / "auth.json").read_text(encoding="utf-8"),
                '{"token":"fresh"}\n',
            )
            self.assertFalse(stale_config.exists())


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.which_patcher = patch(
            "codex_gateway.config.shutil.which", return_value="/usr/bin/codex"
        )
        self.which_patcher.start()
        self.addCleanup(self.which_patcher.stop)
        self.environ_backup = os.environ.copy()
        self.addCleanup(self._restore_env)
        os.environ["CODEX_API_TOKEN"] = "test-token"
        os.environ["CODEX_API_ALLOWED_ROOT"] = "/tmp"
        os.environ["CODEX_API_DEFAULT_WORKSPACE"] = "/tmp"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.models_config_path = Path(self.temp_dir.name) / "models.json"
        os.environ["CODEX_API_MODELS_FILE"] = str(self.models_config_path)
        self.write_models_config()

    def _restore_env(self) -> None:
        os.environ.clear()
        os.environ.update(self.environ_backup)

    def write_models_config(
        self,
        models: list[str] | None = None,
        default_model: str | None = None,
    ) -> None:
        self.models_config_path.write_text(
            json.dumps(
                {
                    "models": models or [],
                    "default_model": default_model,
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_load_settings_parses_and_deduplicates_available_models(self) -> None:
        self.models_config_path.write_text(
            json.dumps(
                {
                    "models": [" custom-a ", "custom-b", " custom-a ", "", "custom-c"],
                    "default_model": None,
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        settings = load_settings()

        self.assertEqual(
            settings.available_models, ("custom-a", "custom-b", "custom-c")
        )

    def test_load_settings_rejects_default_model_outside_allowlist(self) -> None:
        self.write_models_config(["custom-a"], "custom-b")

        with self.assertRaisesRegex(RuntimeError, "default_model must be present"):
            load_settings()

    def test_load_settings_rejects_missing_models_file(self) -> None:
        os.environ["CODEX_API_MODELS_FILE"] = str(
            Path(self.temp_dir.name) / "missing.json"
        )

        with self.assertRaisesRegex(RuntimeError, "models config file not found"):
            load_settings()

    def test_load_settings_rejects_invalid_models_json(self) -> None:
        self.models_config_path.write_text("{not-json}\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "not valid JSON"):
            load_settings()

    def test_load_settings_accepts_read_only_default_sandbox(self) -> None:
        os.environ["CODEX_API_DEFAULT_SANDBOX"] = "read-only"

        settings = load_settings()

        self.assertEqual(settings.default_sandbox, "read-only")

    def test_load_settings_accepts_workspace_write_default_sandbox(self) -> None:
        os.environ["CODEX_API_DEFAULT_SANDBOX"] = "workspace-write"

        settings = load_settings()

        self.assertEqual(settings.default_sandbox, "workspace-write")

    def test_load_settings_rejects_unknown_default_sandbox(self) -> None:
        os.environ["CODEX_API_DEFAULT_SANDBOX"] = "danger-full-access"

        with self.assertRaisesRegex(RuntimeError, "must be one of"):
            load_settings()

    def test_load_settings_rejects_default_read_only_workspace_when_state_is_inside_it(
        self,
    ) -> None:
        workspace = Path(self.temp_dir.name) / "workspace"
        workspace.mkdir()
        os.environ["CODEX_API_DEFAULT_WORKSPACE"] = str(workspace)
        os.environ["CODEX_API_DEFAULT_SANDBOX"] = "read-only"
        os.environ["CODEX_API_STATE_DIR"] = str(workspace / ".codex-api")

        with self.assertRaisesRegex(
            RuntimeError, DEFAULT_READ_ONLY_PATH_CONFLICT_MESSAGE
        ):
            load_settings()


class StructuredOutputTests(unittest.TestCase):
    def test_validate_json_schema_does_not_depend_on_specific_draft_class(self) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        }

        with patch.object(
            __import__(
                "codex_gateway.structured_output", fromlist=["jsonschema"]
            ).jsonschema,
            "Draft202012Validator",
            new=None,
            create=True,
        ):
            validate_json_schema(schema)


if __name__ == "__main__":
    unittest.main()
