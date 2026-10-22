"""Command output may contain binary bytes or partially truncated UTF-8."""

import sys

import pytest

from atlant_coder.handlers import coder


@pytest.mark.parametrize("exit_code", [0, 7])
def test_command_keeps_output_and_exit_status_with_invalid_utf8(
    tmp_path, monkeypatch, exit_code
):
    monkeypatch.setattr(coder, "_control", None)
    output = []
    error = coder._command(
        [
            sys.executable,
            "-c",
            "import os,sys; os.write(1, b'out: \\xd0\\n'); "
            "os.write(2, b'err: \\xff\\n'); sys.exit(" + str(exit_code) + ")",
        ],
        str(tmp_path),
        output=output,
    )
    assert output == ["out: \ufffd\nerr: \ufffd\n"]
    assert bool(error) == bool(exit_code)
    if exit_code:
        assert "err: \ufffd" in error
