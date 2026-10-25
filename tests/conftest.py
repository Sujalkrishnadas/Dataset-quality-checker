"""Shared helpers for building tiny datasets inside pytest's tmp_path."""

from pathlib import Path

import pytest
from PIL import Image


def make_image(path: Path, size=(64, 64), color=(200, 30, 30)) -> Path:
    """Save a small image with a bit of variation (so it is not 'blank')."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (240, 240, 240))
    img.paste(color, (8, 8, size[0] // 2, size[1] // 2))
    img.save(path)
    return path


def make_label(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def codes(report) -> list:
    return [i.code for i in report.issues]


@pytest.fixture
def ds(tmp_path: Path) -> Path:
    """A valid 2-image dataset using parallel images/ and labels/ folders."""
    make_image(tmp_path / "images" / "train" / "a.png", color=(200, 30, 30))
    make_label(tmp_path / "labels" / "train" / "a.txt", "0 0.25 0.25 0.4 0.4\n")
    make_image(tmp_path / "images" / "train" / "b.png", color=(30, 30, 200))
    make_label(tmp_path / "labels" / "train" / "b.txt", "1 0.25 0.25 0.4 0.4\n")
    return tmp_path
