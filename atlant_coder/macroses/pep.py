import logging
import os
import shutil
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

# See codeanalyze._RUFF_CMD — same reasoning.
_ruff_bin = shutil.which("ruff")
_RUFF_CMD = [_ruff_bin] if _ruff_bin else [sys.executable, "-m", "ruff"]


def make(code: str) -> dict:
    """Format Python-code by PEP8.

    Args:
        code: string of Python-code.

    Returns:
        dict: A dictionary with error status, stderr, stdout and Python-code string, formatted by PEP8.
    """
    # Create a temporary file to store the code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as temp_file:
        temp_file.write(code)
        temp_file.flush()
        temp_file_name = temp_file.name

    logger.debug(
        f"Temporary file '{temp_file.name}' created. Starting code formatter..."
    )

    # Lint a file and start ruff formating
    result = subprocess.run(
        [*_RUFF_CMD, "format", temp_file_name],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    with open(temp_file_name, "r+", encoding="utf-8") as output_file:
        formatted_code = output_file.read()

    try:
        os.remove(temp_file_name)
    except:
        ...
    logger.info(
        f"Code formatting ended with exitcode={result.returncode}, formatted code saved, temporary file removed"
    )

    stdout = result.stdout.lower().strip()
    stderr = result.stderr.lower().strip()

    if "reformatted" in stdout:  # if code reformated successfully
        report = {
            "file": temp_file_name,
            "changed": True,
            "error": False,
            "exit_code": result.returncode,
            "formatted_code": formatted_code,
            "stderr": stderr,
            "stdout": stdout,
        }
    elif "unchanged" in stdout:  # if code not needed to be reformated
        report = {
            "file": temp_file_name,
            "changed": False,
            "error": False,
            "exit_code": result.returncode,
            "formatted_code": formatted_code,
            "stderr": stderr,
            "stdout": stdout,
        }
    elif result.returncode != 0:  # if return code is error
        report = {
            "file": temp_file_name,
            "changed": False,
            "error": True,
            "exit_code": result.returncode,
            "stderr": stderr,
            "stdout": stdout,
        }
    else:  # if return is untracked error
        report = {
            "file": temp_file_name,
            "changed": False,
            "error": True,
            "exit_code": result.returncode,
            "stderr": stderr,
            "stdout": stdout,
        }

    logger.debug(
        f"Report filled up, final solution: error={report['error']}, changed={report['changed']}"
    )

    return report
