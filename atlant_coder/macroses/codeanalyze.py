import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

# Prefer the ruff binary — spawning `python -m ruff` costs ~100ms extra per call
# for interpreter startup + package import. Fall back to `-m ruff` if the bin
# isn't on PATH.
_ruff_bin = shutil.which("ruff")
_RUFF_CMD = [_ruff_bin] if _ruff_bin else [sys.executable, "-m", "ruff"]


def analyze_code(code: str, *, fix: bool = True) -> dict:
    """
    Analyzes the provided Python code (as a string) using Ruff and returns the output.

    Args:
        code (str): The Python code to analyze.

    Returns:
        dict: A dictionary containing the analazed(and fixed) code, Ruff's stdout, stderr result reports and errors.
    """
    # Create a temporary file to store the code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        temp_file_name = temp_file.name

    logger.debug(
        f"Temporary file '{temp_file.name}' created. Starting code analyzer..."
    )

    # Lint a file and start ruff check
    result = subprocess.run(
        [
            *_RUFF_CMD,
            "check",
            *(["--fix"] if fix else []),
            # "--fix-only", # проверить не получиться, ruff не выводит в jsonе информацию о фиксах
            "--output-format=json",
            temp_file_name,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    with open(temp_file_name, "r+", encoding="utf-8") as output_file:
        analyzed_code = output_file.read()

    try:
        os.remove(temp_file_name)
    except OSError:
        pass
    logger.info(
        f"Code analyzing ended with exitcode={result.returncode}, analyzied code saved, temporary file removed"
    )

    report = {
        "error": True,
        "stdout_error": True,
        "stderr_error": True,
        "exit_code": result.returncode,
        "stdout_report": [],
        "stderr_report": [],
        "stdout_report_str": "",
        "stderr_report_str": "",
        "analyzed_code": analyzed_code,
    }

    try:
        report["stdout_report"] = (
            json.loads(result.stdout) if result.stdout != "" else []
        )
    except json.JSONDecodeError:
        report["stdout_report_str"] = result.stdout

    try:
        report["stderr_report"] = (
            json.loads(result.stderr) if result.stdout != "" else []
        )
    except json.JSONDecodeError:
        report["stderr_report_str"] = result.stderr
    logger.debug(
        f"Code analyzer output: stdout_report={report['stdout_report']}, stderr_report={report['stderr_report']}"
    )

    report["stdout_error"] = False if report["stdout_report"] == [] else True
    report["stderr_error"] = False if report["stderr_report"] == [] else True

    report["error"] = (
        True
        if report["stdout_error"] or report["stderr_error"] or report["exit_code"] != 0
        else False
    )
    logger.debug(
        f"Report fiiled up, final solution: error={report['error']}, stdout_error={report['stdout_error']}, stderr_error={report['stderr_error']}"
    )

    return report
