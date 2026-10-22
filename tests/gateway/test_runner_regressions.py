import os
import hashlib
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codex_gateway.codex_runner import CodexRunnerError, run_codex
from codex_gateway.app import _run_codex_with_context_overflow_handling
from codex_gateway.session_store import SessionRecord, utc_now
from codex_gateway.structured_output import StructuredOutputError


class RunnerRegressions(unittest.TestCase):
    def test_large_prompt_slow_stdin_reader_with_cancellation_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = root / "slow-codex"
            cli.write_text(
                "#!"
                + sys.executable
                + "\n"
                + """
import hashlib,json,pathlib,sys,time
time.sleep(.3)
data = sys.stdin.buffer.read()
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text(hashlib.sha256(data).hexdigest())
print(json.dumps({'type':'thread.started','thread_id':'slow-reader'}))
"""
            )
            cli.chmod(0o700)
            prompt = "Большой prompt с точными байтами.\n" * 15000
            expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            stamp = utc_now()
            for thread in (None, "existing-thread"):
                session = SessionRecord(
                    "probe", str(root), None, "read-only", thread, stamp, stamp
                )
                polls = []

                def cancelled():
                    polls.append(time.monotonic())
                    return False

                with patch(
                    "codex_gateway.codex_runner.build_subprocess_env",
                    return_value=os.environ.copy(),
                ):
                    started = time.monotonic()
                    result = run_codex(
                        codex_bin=str(cli),
                        session=session,
                        prompt=prompt,
                        timeout_seconds=2,
                        allow_network=False,
                        state_dir=root,
                        cancelled=cancelled,
                    )
                self.assertEqual(result.reply, expected)
                self.assertGreaterEqual(len(polls), 3)
                self.assertLess(time.monotonic() - started, 1.5)

    def test_generation_home_disables_host_and_bootstrap_system_skills(self):
        import tomllib
        from codex_gateway.codex_runner import build_subprocess_env

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "user"
            host = home / ".agents/skills/example/SKILL.md"
            system = home / ".codex/skills/.system/builtin/SKILL.md"
            for path in (host, system):
                path.parent.mkdir(parents=True)
                path.write_text("skill instructions")
            stamp = utc_now()
            session = SessionRecord(
                "probe", str(root), None, "read-only", None, stamp, stamp, True
            )
            with (
                patch("codex_gateway.codex_runner.Path.home", return_value=home),
                patch(
                    "codex_gateway.codex_runner._source_codex_home",
                    return_value=home / ".codex",
                ),
            ):
                env = build_subprocess_env(session, state_dir=root / "state")
            managed = Path(env["CODEX_HOME"])
            config = tomllib.loads((managed / "config.toml").read_text())
            entries = {
                entry["path"]: entry["enabled"] for entry in config["skills"]["config"]
            }
            self.assertEqual(entries[str(host)], False)
            self.assertEqual(
                entries[str(managed / "skills/.system/builtin/SKILL.md")], False
            )
            self.assertFalse((home / ".codex/config.toml").exists())
            self.assertEqual(host.read_text(), "skill instructions")

    def test_usage_unknown_baseline_decrease_and_persistence(self):
        from codex_gateway.app import _record_turn_usage
        from codex_gateway.codex_runner import CodexRunResult
        from codex_gateway.session_store import SessionStore

        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            record = store.create_session(
                workspace=directory, model=None, sandbox="read-only"
            )
            record.thread_id = "legacy-thread"
            self.assertIsNone(
                _record_turn_usage(
                    record,
                    CodexRunResult("ok", "legacy-thread", {"input_tokens": 100}),
                    store,
                )
            )
            record = store.get(record.session_id)
            self.assertEqual(
                _record_turn_usage(
                    record,
                    CodexRunResult("ok", "legacy-thread", {"input_tokens": 130}),
                    store,
                ),
                {"input_tokens": 30},
            )
            self.assertIsNone(
                _record_turn_usage(
                    record,
                    CodexRunResult("ok", "legacy-thread", {"input_tokens": 90}),
                    store,
                )
            )
            self.assertEqual(
                _record_turn_usage(
                    record,
                    CodexRunResult("ok", "legacy-thread", {"input_tokens": 95}),
                    store,
                ),
                {"input_tokens": 5},
            )
            self.assertIsNone(
                _record_turn_usage(
                    record, CodexRunResult("ok", "legacy-thread", None), store
                )
            )
            self.assertIsNone(
                _record_turn_usage(
                    record,
                    CodexRunResult("ok", "legacy-thread", {"input_tokens": 140}),
                    store,
                )
            )
            record = store.reset_thread(record.session_id)
            self.assertIsNone(record.cumulative_usage)
            self.assertIsNone(record.usage_thread_id)
            self.assertEqual(
                _record_turn_usage(
                    record, CodexRunResult("ok", "fresh", {"input_tokens": 12}), store
                ),
                {"input_tokens": 12},
            )

    def test_generation_only_uses_empty_workspace_and_disabled_native_tools(self):
        from codex_gateway.codex_runner import build_command

        stamp = utc_now()
        session = SessionRecord(
            "probe", "/tmp/project", None, "read-only", "thread", stamp, stamp, True
        )
        command = build_command(
            "codex", session, "prompt", "/tmp/output", allow_network=False
        )
        for name in (
            "shell_tool",
            "unified_exec",
            "apps",
            "multi_agent",
            "browser_use",
            "computer_use",
        ):
            self.assertIn(
                ["--disable", name],
                [command[index : index + 2] for index in range(len(command) - 1)],
            )
        self.assertIn("mcp_servers={}", command)
        self.assertIn('web_search="disabled"', command)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = root / "fake-codex"
            cli.write_text(
                "#!"
                + sys.executable
                + "\n"
                + """
import pathlib, sys
workspace=pathlib.Path(sys.argv[sys.argv.index('-C')+1])
assert workspace.name == 'probe'
assert workspace.parent.name == 'generation-workspaces'
assert not list(workspace.iterdir())
sys.stdin.read()
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text('done')
"""
            )
            cli.chmod(0o700)
            with patch(
                "codex_gateway.codex_runner.build_subprocess_env",
                return_value=os.environ.copy(),
            ):
                result = run_codex(
                    codex_bin=str(cli),
                    session=session,
                    prompt="text",
                    timeout_seconds=3,
                    allow_network=False,
                    state_dir=root,
                )
            self.assertEqual(result.reply, "done")

    def test_native_schema_new_and_resume_usage_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = root / "fake-codex"
            cli.write_text(
                "#!"
                + sys.executable
                + "\n"
                + """
import json, pathlib, sys
args = sys.argv
schema_path = pathlib.Path(args[args.index('--output-schema')+1])
assert json.loads(schema_path.read_text()) == {'type': 'object'}
if 'resume' in args: assert args.index('--output-schema') > args.index('resume')
pathlib.Path(__file__).with_name('schema-path').write_text(str(schema_path))
sys.stdin.read()
pathlib.Path(args[args.index('-o')+1]).write_text('{"response":{"command":"OK"}}')
print(json.dumps({'type':'thread.started','thread_id':'thread-new'}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':20,'cached_input_tokens':4,'output_tokens':5}}))
"""
            )
            cli.chmod(0o700)
            stamp = utc_now()
            for thread in (None, "thread-original"):
                session = SessionRecord(
                    "probe", str(root), None, "read-only", thread, stamp, stamp
                )
                with patch(
                    "codex_gateway.codex_runner.build_subprocess_env",
                    return_value=os.environ.copy(),
                ):
                    result = run_codex(
                        codex_bin=str(cli),
                        session=session,
                        prompt="respond",
                        timeout_seconds=3,
                        allow_network=False,
                        state_dir=root,
                        output_schema={"type": "object"},
                    )
                self.assertEqual(
                    result.usage,
                    {"input_tokens": 20, "cached_input_tokens": 4, "output_tokens": 5},
                )
                self.assertFalse(Path((root / "schema-path").read_text()).exists())

    def test_large_stdin_new_and_resumed_and_real_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = root / "fake-codex"
            cli.write_text(
                "#!"
                + sys.executable
                + "\n"
                + """
import json, pathlib, sys, time, subprocess
prompt = sys.stdin.read()
if prompt == 'hang':
    child = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    pathlib.Path(__file__).with_name('child.pid').write_text(str(child.pid))
    time.sleep(10)
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text(str(len(prompt)))
print(json.dumps({'type':'thread.started','thread_id':'thread-new'}))
"""
            )
            cli.chmod(0o700)
            stamp = utc_now()
            for thread in (None, "thread-original"):
                session = SessionRecord(
                    "probe", str(root), None, "read-only", thread, stamp, stamp
                )
                with patch(
                    "codex_gateway.codex_runner.build_subprocess_env",
                    return_value=os.environ.copy(),
                ):
                    result = run_codex(
                        codex_bin=str(cli),
                        session=session,
                        prompt="x" * 300000,
                        timeout_seconds=3,
                        allow_network=False,
                        state_dir=root,
                    )
                    self.assertEqual(result.reply, "300000")
                    self.assertEqual(result.thread_id, "thread-new")
                    start = time.monotonic()
                    with self.assertRaisesRegex(CodexRunnerError, "timed out"):
                        run_codex(
                            codex_bin=str(cli),
                            session=session,
                            prompt="hang",
                            timeout_seconds=0.1,
                            allow_network=False,
                            state_dir=root,
                        )
                    self.assertLess(time.monotonic() - start, 2)
                    child = int((root / "child.pid").read_text())
                    status = Path(f"/proc/{child}/stat")
                    if status.exists():
                        self.assertEqual(
                            status.read_text().split()[2],
                            "Z",
                            "descendant still executing",
                        )

    def test_lost_context_never_replays_followup_into_new_thread(self):
        stamp = utc_now()
        session = SessionRecord(
            "probe", "/tmp", None, "read-only", "old-thread", stamp, stamp
        )
        settings = SimpleNamespace(
            codex_bin="unused", allow_network=False, state_dir=Path("/tmp")
        )
        with patch(
            "codex_gateway.app.run_codex",
            side_effect=CodexRunnerError("session lost", retryable=True),
        ) as run:
            with self.assertRaises(StructuredOutputError) as raised:
                _run_codex_with_context_overflow_handling(
                    session=session,
                    prompt="CONTINUE",
                    settings=settings,
                    timeout_seconds=1,
                )
        self.assertEqual(run.call_count, 1)
        self.assertEqual(session.thread_id, "old-thread")
        self.assertEqual(raised.exception.code, "session_context_lost")
        self.assertEqual(raised.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
