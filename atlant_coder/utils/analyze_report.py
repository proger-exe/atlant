"""Compact human-readable summary of ruff/black failures, for feedback to CODER."""

from typing import Optional


def _format_ruff_findings(findings: list) -> list[str]:
    lines = []
    for f in findings[:8]:
        if not isinstance(f, dict):
            continue
        code = f.get("code") or "?"
        msg = f.get("message") or ""
        loc = f.get("location") or {}
        line_no = loc.get("row") if isinstance(loc, dict) else None
        prefix = f"L{line_no}: " if line_no else ""
        lines.append(f"  [{code}] {prefix}{msg}")
    if len(findings) > 8:
        lines.append(f"  ... (+{len(findings) - 8} more)")
    return lines


def format_error_feedback(
    analyze_report: Optional[dict], format_report: Optional[dict]
) -> str:
    """Turn ruff (analyze) + black-like (format) reports into a compact prompt fragment.

    Returns an empty string when neither report has meaningful errors.
    """
    chunks: list[str] = []

    if analyze_report and analyze_report.get("error"):
        stdout = analyze_report.get("stdout_report") or []
        if isinstance(stdout, list) and stdout:
            chunks.append("ruff issues:")
            chunks.extend(_format_ruff_findings(stdout))
        stderr = (analyze_report.get("stderr_report_str") or "").strip()
        if stderr:
            trimmed = stderr[:500]
            chunks.append(f"ruff stderr: {trimmed}")

    if format_report and format_report.get("error"):
        ferr = (format_report.get("stderr") or "").strip()
        fout = (format_report.get("stdout") or "").strip()
        if ferr:
            chunks.append(f"formatter stderr: {ferr[:500]}")
        elif fout:
            chunks.append(f"formatter stdout: {fout[:500]}")

    return "\n".join(chunks)
