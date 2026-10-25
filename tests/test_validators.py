"""Unit tests for label parsing and image validation."""

import pytest

from conftest import make_image, make_label
from dataset_quality_checker.validators import check_image, parse_yolo_label


def parse(tmp_path, text, **kw):
    p = make_label(tmp_path / "l.txt", text)
    return parse_yolo_label(p, "l.txt", **kw)


def test_valid_label(tmp_path):
    boxes, issues = parse(tmp_path, "0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.2 0.2\n", num_classes=3)
    assert len(boxes) == 2 and issues == []


@pytest.mark.parametrize(
    "line, code",
    [
        ("-1 0.5 0.5 0.2 0.2", "CLASS_NEGATIVE"),
        ("1.5 0.5 0.5 0.2 0.2", "CLASS_NOT_INTEGER"),
        ("cat 0.5 0.5 0.2 0.2", "CLASS_NOT_INTEGER"),
        ("5 0.5 0.5 0.2 0.2", "CLASS_OUT_OF_RANGE"),
        ("0 0.5 0.5 0 0.2", "BOX_NONPOSITIVE_SIZE"),
        ("0 0.5 0.5 0.2 -0.1", "BOX_NONPOSITIVE_SIZE"),
        ("0 0.95 0.5 0.2 0.2", "BOX_OUT_OF_BOUNDS"),
        ("0 -0.1 0.5 0.1 0.2", "BOX_OUT_OF_BOUNDS"),
        ("0 100 100 50 50", "BOX_OUT_OF_BOUNDS"),
        ("0 abc 0.5 0.2 0.2", "BOX_NOT_NUMERIC"),
        ("0 nan 0.5 0.2 0.2", "BOX_NOT_NUMERIC"),
        ("0 inf 0.5 0.2 0.2", "BOX_NOT_NUMERIC"),
        ("0 0.5 0.5", "LABEL_MALFORMED_LINE"),
        ("0 0.5 0.5 0.2 0.2 0.3", "LABEL_MALFORMED_LINE"),
    ],
)
def test_invalid_lines_are_reported_and_excluded(tmp_path, line, code):
    boxes, issues = parse(tmp_path, line + "\n", num_classes=3)
    assert boxes == []
    assert [i.code for i in issues] == [code]
    assert issues[0].line == 1


def test_box_touching_border_is_valid(tmp_path):
    boxes, issues = parse(tmp_path, "0 0.5 0.5 1.0 1.0\n")
    assert len(boxes) == 1 and issues == []


def test_class_range_not_checked_without_num_classes(tmp_path):
    boxes, issues = parse(tmp_path, "99 0.5 0.5 0.2 0.2\n")
    assert len(boxes) == 1 and issues == []


def test_empty_label_is_info(tmp_path):
    boxes, issues = parse(tmp_path, "\n  \n")
    assert boxes == [] and [i.code for i in issues] == ["LABEL_EMPTY"]
    assert issues[0].severity.value == "info"


def test_duplicate_and_tiny_boxes_are_warnings(tmp_path):
    boxes, issues = parse(
        tmp_path, "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.001 0.001\n", image_size=(100, 100)
    )
    assert len(boxes) == 3
    assert sorted(i.code for i in issues) == ["BOX_DUPLICATE", "BOX_TINY"]


def test_segmentation_polygon_gets_hint(tmp_path):
    _, issues = parse(tmp_path, "0 0.1 0.1 0.2 0.1 0.2 0.2\n")
    assert "segmentation" in issues[0].message


def test_non_utf8_label(tmp_path):
    p = tmp_path / "l.txt"
    p.write_bytes(b"\xff\xfe\x00bad")
    _, issues = parse_yolo_label(p, "l.txt")
    assert issues[0].code == "LABEL_UNREADABLE"


def test_utf8_bom_is_accepted(tmp_path):
    p = tmp_path / "l.txt"
    p.write_bytes("\ufeff0 0.5 0.5 0.2 0.2\n".encode("utf-8"))
    boxes, issues = parse_yolo_label(p, "l.txt")
    assert len(boxes) == 1 and issues == []


def test_crlf_line_endings(tmp_path):
    boxes, issues = parse(tmp_path, "0 0.5 0.5 0.2 0.2\r\n1 0.5 0.5 0.2 0.2\r\n")
    assert len(boxes) == 2 and issues == []


def test_valid_image(tmp_path):
    p = make_image(tmp_path / "a.png", size=(64, 48))
    info, issues = check_image(p, "a.png")
    assert (info.width, info.height) == (64, 48) and issues == []


def test_garbage_and_empty_images_are_corrupt(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    for p in (bad, empty):
        info, issues = check_image(p, p.name)
        assert info is None and issues[0].code == "IMG_CORRUPT"


def test_truncated_image_is_corrupt(tmp_path):
    p = make_image(tmp_path / "a.png", size=(200, 200))
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    info, issues = check_image(p, "a.png")
    assert info is None and issues[0].code == "IMG_CORRUPT"


def test_small_and_blank_images(tmp_path):
    from PIL import Image

    Image.new("RGB", (10, 10), (0, 0, 0)).save(tmp_path / "s.png")
    info, issues = check_image(tmp_path / "s.png", "s.png", min_size=32, check_blank=True)
    assert info is not None
    assert sorted(i.code for i in issues) == ["IMG_BLANK", "IMG_TOO_SMALL"]
    _, issues = check_image(tmp_path / "s.png", "s.png", min_size=32, check_blank=False)
    assert [i.code for i in issues] == ["IMG_TOO_SMALL"]
