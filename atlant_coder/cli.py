"""User entry point; the local service owns work independently of this process."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

import requests

from atlant_coder import __version__
from atlant_coder.runtime import ensure_service
from atlant_coder.tui import AtlantApp


def _attach_command(workspace: str, run_id: str) -> str:
    return shlex.join(["atlant", workspace, "--attach", run_id])


def _headless(client, workspace: str, run_id: str) -> int:
    after = 0
    last_status = None
    while True:
        events = client.events(run_id, after=after)
        for event in events:
            after = max(after, event["seq"])
            print(f"[{event['kind']}] {event.get('message', '')}", flush=True)
        state = client.snapshot(run_id)
        status = state["status"]
        if status != last_status:
            print(f"[{status}] {state.get('stage', '')}", flush=True)
            last_status = status
        # The event API pages results; drain the final page before returning.
        if events and after < state.get("last_seq", after):
            continue
        if status in {"completed", "failed", "stopped", "interrupted", "unverified"}:
            for check in state.get("checks", []):
                print(
                    f"[check] {check.get('check', '')}: {check.get('status', 'UNKNOWN')}"
                )
            return 0 if status == "completed" else 1
        approval = state.get("pending_approval")
        if status == "waiting_approval" or (approval and "allow" not in approval):
            print(
                "Run is waiting for approval. Attach the UI to respond:",
                file=sys.stderr,
            )
            print(_attach_command(workspace, run_id), file=sys.stderr)
            return 2
        if status == "waiting_input":
            print(
                "Run is waiting for input. Attach the UI to answer:",
                file=sys.stderr,
            )
            print(_attach_command(workspace, run_id), file=sys.stderr)
            return 2
        time.sleep(0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atlant", description="Atlant coding agent and project generator"
    )
    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="existing project directory (default: current directory)",
    )
    parser.add_argument(
        "--attach", metavar="RUN_ID", help="reconnect to an existing run"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="print events and exit when the run finishes",
    )
    parser.add_argument("--task", help="start a run with this task")
    parser.add_argument(
        "--mode",
        choices=("agent", "generate"),
        default="agent",
        help="mode for --task (default: agent); otherwise select in the UI",
    )
    parser.add_argument("--version", action="version", version=f"Atlant {__version__}")
    args = parser.parse_args(argv)
    if args.attach and args.task is not None:
        parser.error("--attach and --task cannot be combined")
    if args.task is not None and not args.task.strip():
        parser.error("--task must not be blank")
    if args.headless and not (args.attach or args.task):
        parser.error("--headless requires --task or --attach")
    workspace = str(Path(args.workspace).expanduser().resolve())
    if not Path(workspace).is_dir():
        parser.error(f"workspace is not a directory: {workspace}")
    run_id = args.attach
    try:
        client = ensure_service(workspace)
        if args.task:
            run_id = client.start(workspace, args.task, args.mode)["id"]
            print(f"Run: {run_id}", flush=True)
        elif run_id:
            client.snapshot(run_id)
        if args.headless:
            return _headless(client, workspace, run_id)
        AtlantApp(client, workspace, run_id).run()
        return 0
    except KeyboardInterrupt:
        print("Detached; the run service continues independently.", file=sys.stderr)
        if run_id:
            print(_attach_command(workspace, run_id), file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        print(f"atlant: {exc}", file=sys.stderr)
        if run_id:
            print(f"Reconnect: {_attach_command(workspace, run_id)}", file=sys.stderr)
        return 1
