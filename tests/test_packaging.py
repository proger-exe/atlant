from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from AurethosSDK.models.base import BaseModel
from AurethosSDK.utils.model_parser import model_parser
from atlant_coder import MODELS_PATH
from atlant_coder.cli import _headless, main


def test_packaged_prompts_load_from_unrelated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = model_parser(str(MODELS_PATH))
    assert set(settings) == {"CORE", "ARCH", "ARCH_THINK", "CODER", "CODER_THINK"}
    for model in settings.values():
        assert Path(model.system_instruction).is_absolute()
        assert BaseModel(model).settings.system_instruction.strip()


def test_readonly_config_resolves_relative_and_absolute_prompts(tmp_path, monkeypatch):
    prompts = tmp_path / "config"
    prompts.mkdir()
    prompt = prompts / "prompt.md"
    prompt.write_text("Keep the exact contract.")
    config = prompts / "models.json"
    config.write_text(
        json.dumps(
            [
                {
                    "Name": "CORE",
                    "Priority": 0,
                    "FromFile": True,
                    "SystemInstruction": "prompt.md",
                },
                {"Name": "ABS", "FromFile": True, "SystemInstruction": str(prompt)},
                {
                    "Name": "INLINE",
                    "FromFile": False,
                    "SystemInstruction": "literal/not-a-path",
                },
            ]
        )
    )
    prompt.chmod(0o444)
    config.chmod(0o444)
    monkeypatch.chdir(tmp_path)
    settings = model_parser(str(config))
    assert (
        BaseModel(settings["CORE"]).settings.system_instruction
        == "Keep the exact contract."
    )
    assert (
        BaseModel(settings["ABS"]).settings.system_instruction
        == "Keep the exact contract."
    )
    assert settings["INLINE"].system_instruction == "literal/not-a-path"


@pytest.mark.parametrize(
    "status,code",
    [
        ("completed", 0),
        ("unverified", 1),
        ("failed", 1),
        ("interrupted", 1),
        ("stopped", 1),
    ],
)
def test_headless_only_completed_is_success(status, code):
    client = SimpleNamespace(
        events=lambda *a, **k: [], snapshot=lambda _: {"status": status}
    )
    assert _headless(client, "/work", "run-id") == code


def test_headless_drains_event_pages_before_final_status(capsys):
    seen = []

    def events(run_id, after=0):
        seen.append(after)
        return [{"seq": after + 1, "kind": "stage", "message": f"message-{after + 1}"}]

    client = SimpleNamespace(
        events=events, snapshot=lambda _: {"status": "completed", "last_seq": 3}
    )
    assert _headless(client, "/work", "run-id") == 0
    assert seen == [0, 1, 2]
    assert "message-3" in capsys.readouterr().out


def test_headless_approval_detaches_without_approving(capsys):
    client = SimpleNamespace(
        events=lambda *a, **k: [], snapshot=lambda _: {"status": "waiting_approval"}
    )
    assert _headless(client, "/work with spaces", "run-id") == 2
    assert "atlant '/work with spaces' --attach run-id" in capsys.readouterr().err


def test_headless_waiting_input_detaches_for_steering(capsys):
    client = SimpleNamespace(
        events=lambda *a, **k: [], snapshot=lambda _: {"status": "waiting_input"}
    )
    assert _headless(client, "/work with spaces", "run-id") == 2
    output = capsys.readouterr()
    assert "waiting for input" in output.err
    assert "atlant '/work with spaces' --attach run-id" in output.err


def test_cli_dispatches_task_and_mode(tmp_path, monkeypatch):
    calls = []

    def start(workspace, task, mode):
        calls.append((workspace, task, mode))
        return {"id": "run-id"}

    client = SimpleNamespace(
        start=start,
        events=lambda *a, **k: [],
        snapshot=lambda _: {"status": "completed"},
    )
    monkeypatch.setattr("atlant_coder.cli.ensure_service", lambda _: client)
    assert (
        main(
            [
                str(tmp_path),
                "--headless",
                "--mode",
                "generate",
                "--task",
                "Build a parser",
            ]
        )
        == 0
    )
    assert calls == [(str(tmp_path), "Build a parser", "generate")]


@pytest.mark.parametrize(
    "args", [["--headless"], ["--task", " "], ["--attach", "id", "--task", "hello"]]
)
def test_invalid_arguments_do_not_launch_service(args, tmp_path, monkeypatch):
    def unexpected_launch(_):
        pytest.fail("invalid arguments launched a service")

    monkeypatch.setattr("atlant_coder.cli.ensure_service", unexpected_launch)
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), *args])
    assert error.value.code == 2


def test_keyboard_interrupt_detaches_without_stop(tmp_path, monkeypatch, capsys):
    def events(*args, **kwargs):
        raise KeyboardInterrupt

    client = SimpleNamespace(events=events, snapshot=lambda _: {"status": "running"})
    monkeypatch.setattr("atlant_coder.cli.ensure_service", lambda _: client)
    assert main([str(tmp_path), "--headless", "--attach", "run-id"]) == 130
    assert "--attach run-id" in capsys.readouterr().err
