"""Plain data containers shared by all modules.

Keeping these in one place avoids circular imports and gives every module a
common vocabulary: an :class:`Issue` is anything we report, a :class:`Box`
is one validated YOLO object, and an :class:`ImageInfo` is a readable image.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class DatasetError(Exception):
    """Raised for problems with the *request* (bad path, bad option), not the data."""


class Severity(str, Enum):
    """How serious an issue is.

    ERROR   - will break or silently corrupt training; fix before training.
    WARNING - suspicious; may be intentional, deserves a look.
    INFO    - purely informational (e.g. an empty label file = background image).
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Issue:
    """One finding. ``path`` is relative to the dataset root ('' = whole dataset)."""

    severity: Severity
    code: str
    path: str
    message: str
    line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "path": self.path,
            "line": self.line,
            "message": self.message,
        }


@dataclass(frozen=True)
class Box:
    """A fully validated YOLO bounding box (normalised coordinates)."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line: int


@dataclass(frozen=True)
class ImageInfo:
    """A successfully decoded image."""

    path: Path
    width: int
    height: int
