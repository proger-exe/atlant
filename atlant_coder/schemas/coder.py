"""Validated internal events accepted by the registered CODER handlers."""

from enum import StrEnum
from typing import Literal

from AurethosSDK.types.update import Update


class CoderAcceptedCommands(StrEnum):
    GENERATE = "GENERATE"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"
    REPLAN_CANDIDATE = "REPLAN_CANDIDATE"
    PLAN_APPROVED = "PLAN_APPROVED"
    STEER_APPROVED = "STEER_APPROVED"
    RESOLVE_APPROVED = "RESOLVE_APPROVED"
    FINAL_APPROVED = "FINAL_APPROVED"
    STOP_REQUESTED = "STOP_REQUESTED"
    MAKE = "MAKE"
    REQUEST = "REQUEST"
    PATCH = "PATCH"
    CONFLICT = "CONFLICT"
    NEEDRESUME = "NEEDRESUME"
    MADE = "MADE"
    REVIEW = "REVIEW"
    OK = "OK"
    VERIFY = "VERIFY"
    READ = "READ"
    LIST = "LIST"
    SEARCH = "SEARCH"
    RUN = "RUN"
    THINK = "THINK"
    ANSW = "ANSW"


class CoderUpdate(Update):
    to_model: Literal["CODER"] = "CODER"
    from_model: str = "CODER"
    command: CoderAcceptedCommands
