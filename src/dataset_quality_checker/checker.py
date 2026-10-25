"""Orchestrator: runs every check and assembles a :class:`DatasetReport`.

This module is strictly read-only with respect to the dataset.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_quality_checker import __version__
from dataset_quality_checker.duplicate_detector import find_duplicates
from dataset_quality_checker.models import DatasetError, ImageInfo, Issue, Severity
from dataset_quality_checker.scanner import find_label, relative_posix, scan_dataset
from dataset_quality_checker.statistics import balance_issues, class_distribution, image_dimension_stats
from dataset_quality_checker.validators import check_image, parse_yolo_label

log = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    """Result of one run: settings, statistics and the list of issues."""

    root: Path
    settings: Dict[str, Any]
    statistics: Dict[str, Any]
    issues: List[Issue] = field(default_factory=list)

    def count(self, severity: Severity) -> int:
        return sum(1 for i in self.issues if i.severity == severity)

    @property
    def errors(self) -> int:
        return self.count(Severity.ERROR)

    @property
    def warnings(self) -> int:
        return self.count(Severity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": "dataset-quality-checker",
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_root": str(self.root),
            "settings": self.settings,
            "statistics": self.statistics,
            "issues": [i.to_dict() for i in self.issues],
        }


def run_checks(
    root: Path,
    *,
    num_classes: Optional[int] = None,
    min_size: int = 32,
    check_blank: bool = False,
    imbalance_ratio: float = 10.0,
    workers: Optional[int] = None,
) -> DatasetReport:
    """Validate the YOLO dataset under ``root``.

    Raises:
        DatasetError: if ``root`` is not a directory or an option is invalid.
    """
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"Dataset path is not a directory: {root}")
    if num_classes is not None and num_classes < 1:
        raise DatasetError("num_classes must be at least 1.")
    if min_size < 0:
        raise DatasetError("min_size cannot be negative.")
    if imbalance_ratio <= 1:
        raise DatasetError("imbalance_ratio must be greater than 1.")
    if workers is None:
        workers = min(4, os.cpu_count() or 1)
    if workers < 1:
        raise DatasetError("workers must be at least 1.")

    root = root.resolve()
    settings = {
        "num_classes": num_classes,
        "min_size": min_size,
        "check_blank": check_blank,
        "imbalance_ratio": imbalance_ratio,
        "workers": workers,
        "annotation_format": "yolo",
    }

    log.info("Scanning %s", root)
    scan = scan_dataset(root)
    log.info("Found %d images and %d label files in labels/ folders", len(scan.images), len(scan.label_files))
    issues: List[Issue] = []

    # 1. Image integrity and quality (parallel: decoding is the slow part).
    def _check(path: Path):
        return check_image(path, relative_posix(path, root), min_size=min_size, check_blank=check_blank)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        image_results = list(pool.map(_check, scan.images))

    infos: Dict[Path, ImageInfo] = {}
    for path, (info, found) in zip(scan.images, image_results):
        issues.extend(found)
        if info is not None:
            infos[path] = info

    # 2. Labels: pairing, format, geometry, class ids.
    class_counts: Counter = Counter()
    matched_labels = set()
    label_files_read = 0
    images_without_label = 0
    empty_labels = 0
    objects = 0
    invalid_lines = 0

    for path in scan.images:
        rel_img = relative_posix(path, root)
        label = find_label(path, root)
        if label is None:
            images_without_label += 1
            issues.append(Issue(Severity.ERROR, "LABEL_MISSING", rel_img, "No matching label file found."))
            continue
        matched_labels.add(label)
        label_files_read += 1
        info = infos.get(path)
        boxes, found = parse_yolo_label(
            label,
            relative_posix(label, root),
            num_classes=num_classes,
            image_size=(info.width, info.height) if info else None,
        )
        issues.extend(found)
        if any(i.code == "LABEL_EMPTY" for i in found):
            empty_labels += 1
        invalid_lines += sum(1 for i in found if i.severity == Severity.ERROR and i.line is not None)
        objects += len(boxes)
        class_counts.update(b.class_id for b in boxes)

    for label in scan.label_files:
        if label not in matched_labels:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "LABEL_ORPHAN",
                    relative_posix(label, root),
                    "Label file has no matching image.",
                )
            )

    # 3. Exact duplicates.
    duplicate_groups = find_duplicates(scan.images, workers=workers)
    for group in duplicate_groups:
        first = relative_posix(group[0], root)
        for extra in group[1:]:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "IMG_DUPLICATE",
                    relative_posix(extra, root),
                    f"Byte-identical duplicate of {first}.",
                )
            )

    # 4. Class balance.
    issues.extend(balance_issues(class_counts, num_classes, imbalance_ratio))

    # Stable ordering: errors first, then by code and path.
    order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    issues.sort(key=lambda i: (order[i.severity], i.code, i.path, i.line or 0))

    by_code = Counter(i.code for i in issues)
    statistics = {
        "images_found": len(scan.images),
        "images_readable": len(infos),
        "images_unreadable": len(scan.images) - len(infos),
        "images_without_label": images_without_label,
        "label_files_read": label_files_read,
        "empty_label_files": empty_labels,
        "invalid_label_lines": invalid_lines,
        "valid_objects": objects,
        "duplicate_groups": len(duplicate_groups),
        "image_dimensions": image_dimension_stats(list(infos.values())),
        "class_distribution": class_distribution(class_counts, num_classes),
        "issue_counts": {
            "errors": sum(1 for i in issues if i.severity == Severity.ERROR),
            "warnings": sum(1 for i in issues if i.severity == Severity.WARNING),
            "info": sum(1 for i in issues if i.severity == Severity.INFO),
        },
        "issues_by_code": dict(sorted(by_code.items())),
    }
    return DatasetReport(root=root, settings=settings, statistics=statistics, issues=issues)
