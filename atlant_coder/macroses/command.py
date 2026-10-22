"""Subprocess execution shared by coordinator-owned tools."""

import os
import signal
import subprocess
import time


def run_command(
    command, home, timeout=300, require_tests=False, output=None, control=None
):
    if control:
        control.check_stopped()
        control.event("check", "Запуск: " + " ".join(command))
    try:
        process = subprocess.Popen(
            command,
            cwd=home,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
            env={
                k: v
                for k, v in os.environ.items()
                if k
                not in {
                    "ATLANT_SERVICE_TOKEN",
                    "CODEX_API_TOKEN",
                    "CODEX_GATEWAY_TOKEN",
                    "OPENAI_API_KEY",
                    "GEMINI_API_KEY",
                    "LLM7_TOKEN",
                    "POLLINATIONS_TOKEN",
                }
            },
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                if control:
                    control.check_stopped()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(
                        timeout=min(0.1, remaining) if control else remaining
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue
            if control:
                control.check_stopped()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        result = subprocess.CompletedProcess(
            command, process.returncode, stdout, stderr
        )
        if output is not None:
            output.append((stdout + stderr)[-16000:])
    except InterruptedError:
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if require_tests and "Ran 0 tests" in result.stdout + result.stderr:
        return "test discovery succeeded but ran zero tests"
    if result.returncode:
        return (
            "\n".join((result.stdout, result.stderr)).strip()
            or f"exit {result.returncode}"
        )[-16000:]
    return ""
