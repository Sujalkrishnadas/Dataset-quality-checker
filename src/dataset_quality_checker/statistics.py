"""Dataset statistics and class-balance analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from dataset_quality_checker.models import ImageInfo, Issue, Severity


def class_distribution(counts: Mapping[int, int], num_classes: Optional[int]) -> Dict[str, Dict[str, Any]]:
    """Per-class object count and percentage. Includes zero-count classes if ``num_classes`` is known."""
    total = sum(counts.values())
    ids = set(counts)
    if num_classes is not None:
        ids |= set(range(num_classes))
    return {
        str(c): {
            "count": counts.get(c, 0),
            "percent": round(100.0 * counts.get(c, 0) / total, 2) if total else 0.0,
        }
        for c in sorted(ids)
    }


def balance_issues(
    counts: Mapping[int, int], num_classes: Optional[int], imbalance_ratio: float
) -> List[Issue]:
    """Warn about unused classes and severe imbalance.

    Imbalance = (largest class count) / (smallest *non-zero* class count).
    The threshold is a rule of thumb, not a statistical test: whether a ratio
    is harmful depends on your task and model.
    """
    issues: List[Issue] = []
    if num_classes is not None:
        unused = [c for c in range(num_classes) if counts.get(c, 0) == 0]
        if unused and sum(counts.values()) > 0:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "CLASS_UNUSED",
                    "",
                    f"Declared classes with zero valid objects: {', '.join(map(str, unused))}.",
                )
            )
    present = {c: n for c, n in counts.items() if n > 0}
    if len(present) >= 2:
        big_c = max(present, key=lambda c: present[c])
        small_c = min(present, key=lambda c: present[c])
        ratio = present[big_c] / present[small_c]
        if ratio >= imbalance_ratio:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "CLASS_IMBALANCE",
                    "",
                    f"Severe class imbalance: class {big_c} has {present[big_c]} objects, "
                    f"class {small_c} has {present[small_c]} (ratio {ratio:.1f}:1, threshold {imbalance_ratio:g}:1).",
                )
            )
    return issues


def image_dimension_stats(infos: Sequence[ImageInfo]) -> Dict[str, Any]:
    """Summary of width/height across readable images."""
    if not infos:
        return {"min_width": None, "max_width": None, "mean_width": None,
                "min_height": None, "max_height": None, "mean_height": None, "common_sizes": []}
    widths = [i.width for i in infos]
    heights = [i.height for i in infos]
    sizes = Counter(f"{i.width}x{i.height}" for i in infos)
    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": round(sum(widths) / len(widths), 1),
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": round(sum(heights) / len(heights), 1),
        "common_sizes": [{"size": s, "count": n} for s, n in sizes.most_common(5)],
    }
