"""Per-file validation: image integrity/quality and YOLO label content.

Each function returns data plus a list of :class:`Issue` objects; none of
them raise for bad *data* and none of them write to disk.

Limitations (important!):

* An image that decodes fine can still be semantically useless (wrong
  content, wrong colours, mislabeled). We only check technical validity.
* Label checks verify *format and geometry*, not whether a box really
  surrounds the object it claims to.
* Only YOLO detection format (``class xc yc w h``, normalised) is supported.
  Segmentation polygons are reported as malformed lines.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

from PIL import Image, ImageStat

from dataset_quality_checker.models import Box, ImageInfo, Issue, Severity

_INT_RE = re.compile(r"^[+-]?\d+$")
_EPS = 1e-6  # tolerance for floating point noise at the image border
BLANK_STDDEV_THRESHOLD = 1.0


def check_image(
    path: Path,
    rel: str,
    *,
    min_size: int = 32,
    check_blank: bool = False,
) -> Tuple[Optional[ImageInfo], List[Issue]]:
    """Fully decode an image and run quality checks.

    ``Image.verify()`` alone misses truncated files, so we also call
    ``load()`` which decodes every pixel. This is the slowest check; it is
    run in a thread pool by the checker.

    Returns ``(info, issues)``; ``info`` is ``None`` if the image is unreadable.
    """
    issues: List[Issue] = []
    try:
        if path.stat().st_size == 0:
            return None, [Issue(Severity.ERROR, "IMG_CORRUPT", rel, "File is empty (0 bytes).")]
    except OSError as exc:
        return None, [Issue(Severity.ERROR, "IMG_CORRUPT", rel, f"Cannot access file: {exc}")]

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            blank = False
            if check_blank:
                gray = img.convert("L")
                gray.thumbnail((64, 64))
                blank = ImageStat.Stat(gray).stddev[0] < BLANK_STDDEV_THRESHOLD
    except Exception as exc:  # Pillow raises many different types for bad files
        return None, [
            Issue(Severity.ERROR, "IMG_CORRUPT", rel, f"Cannot decode image: {type(exc).__name__}: {exc}")
        ]

    if width < min_size or height < min_size:
        issues.append(
            Issue(
                Severity.WARNING,
                "IMG_TOO_SMALL",
                rel,
                f"Image is {width}x{height}px, smaller than the minimum {min_size}px.",
            )
        )
    if blank:
        issues.append(
            Issue(
                Severity.WARNING,
                "IMG_BLANK",
                rel,
                "Image is blank or nearly blank (almost no pixel variation).",
            )
        )
    return ImageInfo(path=path, width=width, height=height), issues


def parse_yolo_label(
    path: Path,
    rel: str,
    *,
    num_classes: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Box], List[Issue]]:
    """Validate one YOLO label file.

    Returns the list of fully valid boxes and the list of issues. A line with
    any *error* contributes no box (so class statistics only count objects
    that training could actually use).
    """
    issues: List[Issue] = []
    boxes: List[Box] = []

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return [], [Issue(Severity.ERROR, "LABEL_UNREADABLE", rel, "Label file is not valid UTF-8 text.")]
    except OSError as exc:
        return [], [Issue(Severity.ERROR, "LABEL_UNREADABLE", rel, f"Cannot read label file: {exc}")]

    lines = [(n, s.strip()) for n, s in enumerate(text.splitlines(), start=1)]
    lines = [(n, s) for n, s in lines if s]
    if not lines:
        return [], [
            Issue(Severity.INFO, "LABEL_EMPTY", rel, "Label file is empty (treated as a background image).")
        ]

    seen: Set[Tuple[int, float, float, float, float]] = set()

    for n, line in lines:
        tokens = line.split()
        if len(tokens) != 5:
            hint = ""
            if len(tokens) > 5 and (len(tokens) - 1) % 2 == 0:
                hint = " This looks like a segmentation polygon, which is not supported."
            issues.append(
                Issue(
                    Severity.ERROR,
                    "LABEL_MALFORMED_LINE",
                    rel,
                    f"Expected 5 values (class x_center y_center width height), found {len(tokens)}.{hint}",
                    n,
                )
            )
            continue

        # --- class id -------------------------------------------------
        cls_token = tokens[0]
        if not _INT_RE.match(cls_token):
            issues.append(
                Issue(Severity.ERROR, "CLASS_NOT_INTEGER", rel, f"Class id '{cls_token}' is not an integer.", n)
            )
            continue
        class_id = int(cls_token)
        if class_id < 0:
            issues.append(Issue(Severity.ERROR, "CLASS_NEGATIVE", rel, f"Class id {class_id} is negative.", n))
            continue
        if num_classes is not None and class_id >= num_classes:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "CLASS_OUT_OF_RANGE",
                    rel,
                    f"Class id {class_id} is outside the valid range 0..{num_classes - 1}.",
                    n,
                )
            )
            continue

        # --- coordinates ----------------------------------------------
        try:
            xc, yc, w, h = (float(t) for t in tokens[1:])
        except ValueError:
            issues.append(
                Issue(Severity.ERROR, "BOX_NOT_NUMERIC", rel, "Box coordinates are not all numbers.", n)
            )
            continue
        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            issues.append(
                Issue(Severity.ERROR, "BOX_NOT_NUMERIC", rel, "Box coordinates contain NaN or infinity.", n)
            )
            continue

        if w <= 0 or h <= 0:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "BOX_NONPOSITIVE_SIZE",
                    rel,
                    f"Box has zero or negative size (width={w}, height={h}).",
                    n,
                )
            )
            continue

        x1, x2 = xc - w / 2, xc + w / 2
        y1, y2 = yc - h / 2, yc + h / 2
        if x1 < -_EPS or y1 < -_EPS or x2 > 1 + _EPS or y2 > 1 + _EPS:
            hint = ""
            if max(xc, yc, w, h) > 1:
                hint = " Values above 1 suggest pixel coordinates instead of normalised ones."
            issues.append(
                Issue(
                    Severity.ERROR,
                    "BOX_OUT_OF_BOUNDS",
                    rel,
                    f"Box extends beyond the image (x: {x1:.4f}..{x2:.4f}, y: {y1:.4f}..{y2:.4f}).{hint}",
                    n,
                )
            )
            continue

        # --- warnings (box is usable, but suspicious) -----------------
        key = (class_id, round(xc, 6), round(yc, 6), round(w, 6), round(h, 6))
        if key in seen:
            issues.append(
                Issue(Severity.WARNING, "BOX_DUPLICATE", rel, "Identical box appears more than once in this file.", n)
            )
        seen.add(key)

        if image_size is not None:
            img_w, img_h = image_size
            if w * img_w < 1 or h * img_h < 1:
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "BOX_TINY",
                        rel,
                        f"Box is smaller than 1 pixel ({w * img_w:.2f}x{h * img_h:.2f}px).",
                        n,
                    )
                )

        boxes.append(Box(class_id, xc, yc, w, h, n))

    return boxes, issues
