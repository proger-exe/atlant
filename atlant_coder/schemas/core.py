"""Validated internal events accepted by the registered CORE handlers."""

from enum import StrEnum
from typing import Literal

from AurethosSDK.types.update import Update


class CoreAcceptedCommands(StrEnum):
    START = "START"
    REQUEST = "REQUEST"
    ASK = "ASK"
    OK = "OK"
    MADE = "MADE"
    REVIEW_PLAN = "REVIEW_PLAN"
    RESOLVE = "RESOLVE"
    FINALIZE = "FINALIZE"
    STEER = "STEER"
    APPROVE = "APPROVE"
    REPLAN = "REPLAN"
    STOP = "STOP"
    READ = "READ"
    LIST = "LIST"
    SEARCH = "SEARCH"


class CoreUpdate(Update):
    to_model: Literal["CORE"] = "CORE"
    from_model: str = "CORE"
    command: CoreAcceptedCommands
