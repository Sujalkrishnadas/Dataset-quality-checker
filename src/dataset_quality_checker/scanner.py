"""Filesystem discovery: find images and label files, and pair them up.

Supported layouts (nested folders are fine):

* Label next to the image:      ``data/a.jpg``  +  ``data/a.txt``
* Parallel folders (YOLO/Ultralytics style):
  ``data/images/train/a.jpg``  +  ``data/labels/train/a.txt``

Limitation: only these two conventions are recognised. Datasets that keep
labels somewhere else will be reported as "missing labels".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
)


@dataclass
class ScanResult:
    """Everything found on disk, in deterministic (sorted) order."""

    images: List[Path] = field(default_factory=list)
    label_files: List[Path] = field(default_factory=list)


def scan_dataset(root: Path) -> ScanResult:
    """Walk ``root`` and collect image files and candidate label files.

    A ``.txt`` file counts as a label file only if one of its parent folders
    is called ``labels`` or it sits next to an image of the same name; the
    second case is resolved later by :func:`find_label`. Here we collect all
    ``.txt`` files that live under a ``labels`` folder (used for orphan
    detection).
    """
    result = ScanResult()
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                result.images.append(path)
            elif suffix == ".txt" and name.lower() != "classes.txt":
                rel_parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
                if "labels" in rel_parts:
                    result.label_files.append(path)
    result.images.sort()
    result.label_files.sort()
    return result


def label_candidates(image: Path, root: Path) -> List[Path]:
    """Return possible label paths for ``image``, most likely first."""
    rel = image.relative_to(root)
    candidates: List[Path] = []
    parts = list(rel.parts)
    for i in range(len(parts) - 2, -1, -1):  # skip the file name itself
        if parts[i].lower() == "images":
            swapped = parts[:i] + ["labels"] + parts[i + 1 :]
            candidates.append((root.joinpath(*swapped)).with_suffix(".txt"))
            break
    candidates.append(image.with_suffix(".txt"))
    return candidates


def find_label(image: Path, root: Path) -> Optional[Path]:
    """Return the existing label file for ``image`` or ``None``."""
    for candidate in label_candidates(image, root):
        if candidate.is_file():
            return candidate
    return None


def relative_posix(path: Path, root: Path) -> str:
    """Path relative to ``root`` using forward slashes (stable across OSes)."""
    return path.relative_to(root).as_posix()
