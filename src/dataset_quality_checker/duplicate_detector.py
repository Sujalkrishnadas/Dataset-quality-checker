"""Exact duplicate detection using SHA-256 file hashes.

Performance trick: files can only be identical if they have the same size,
so we group by size first and hash only files that share a size with another
file. On large datasets this skips most of the hashing work.

Limitation: this finds *byte-identical* files only. The same picture
re-saved, resized, or re-compressed is NOT detected (that would need
perceptual hashing, which this project deliberately does not claim to do).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_CHUNK = 1024 * 1024


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicates(paths: Sequence[Path], workers: Optional[int] = None) -> List[List[Path]]:
    """Group byte-identical files. Returns only groups with 2+ files, sorted."""
    by_size: Dict[int, List[Path]] = defaultdict(list)
    for path in paths:
        try:
            by_size[path.stat().st_size].append(path)
        except OSError:
            continue  # unreadable files are reported elsewhere

    candidates = [p for group in by_size.values() if len(group) > 1 for p in group]
    if not candidates:
        return []

    def safe_hash(p: Path) -> Optional[str]:
        try:
            return hash_file(p)
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(safe_hash, candidates))

    by_hash: Dict[str, List[Path]] = defaultdict(list)
    for path, digest in zip(candidates, digests):
        if digest is not None:
            by_hash[digest].append(path)

    groups = [sorted(g) for g in by_hash.values() if len(g) > 1]
    groups.sort(key=lambda g: g[0])
    return groups
