"""Validated internal events accepted by the registered ARCH handlers."""

from enum import StrEnum
from typing import Literal

from AurethosSDK.types.update import Update


class ArchAcceptedCommands(StrEnum):
    PLAN = "PLAN"
    REVISE = "REVISE"
    REQUEST = "REQUEST"
    MADE = "MADE"
    READ = "READ"
    LIST = "LIST"
    SEARCH = "SEARCH"


class ArchUpdate(Update):
    to_model: Literal["ARCH"] = "ARCH"
    from_model: str = "ARCH"
    command: ArchAcceptedCommands
