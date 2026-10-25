"""End-to-end tests of run_checks on small hand-built datasets."""

import shutil

import pytest

from conftest import codes, make_image, make_label
from dataset_quality_checker.checker import run_checks
from dataset_quality_checker.duplicate_detector import find_duplicates
from dataset_quality_checker.models import DatasetError
from dataset_quality_checker.sample_data import generate_sample_dataset


def test_valid_dataset_has_no_issues(ds):
    report = run_checks(ds, num_classes=2)
    assert report.issues == []
    s = report.statistics
    assert s["images_found"] == 2 and s["valid_objects"] == 2
    assert s["class_distribution"]["0"]["count"] == 1


def test_missing_label(ds):
    (ds / "labels" / "train" / "a.txt").unlink()
    report = run_checks(ds)
    assert codes(report) == ["LABEL_MISSING"]
    assert report.statistics["images_without_label"] == 1


def test_label_next_to_image(tmp_path):
    make_image(tmp_path / "x" / "p.png")
    make_label(tmp_path / "x" / "p.txt", "0 0.5 0.5 0.2 0.2\n")
    assert run_checks(tmp_path).issues == []


def test_nested_folders_and_uppercase_extension(tmp_path):
    make_image(tmp_path / "images" / "a" / "b" / "c" / "PIC.png")
    p = make_image(tmp_path / "images" / "a" / "b" / "d.png")
    p.rename(p.with_name("UPPER.PNG"))
    make_label(tmp_path / "labels" / "a" / "b" / "c" / "PIC.txt", "0 0.5 0.5 0.2 0.2\n")
    make_label(tmp_path / "labels" / "a" / "b" / "UPPER.txt", "0 0.5 0.5 0.2 0.2\n")
    report = run_checks(tmp_path)
    assert report.statistics["images_found"] == 2
    assert "LABEL_MISSING" not in codes(report)


def test_corrupted_image_still_reports_label_checks(ds):
    (ds / "images" / "train" / "a.png").write_bytes(b"garbage")
    report = run_checks(ds)
    assert codes(report) == ["IMG_CORRUPT"]
    assert report.statistics["images_unreadable"] == 1


def test_invalid_annotation_counts(ds):
    make_label(ds / "labels" / "train" / "a.txt", "0 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n0 0.5 0.5 0 0\n")
    report = run_checks(ds, num_classes=2)
    assert sorted(codes(report)) == ["BOX_NONPOSITIVE_SIZE", "CLASS_OUT_OF_RANGE"]
    assert report.statistics["valid_objects"] == 2  # 1 from a.txt + 1 from b.txt
    assert report.statistics["invalid_label_lines"] == 2


def test_orphan_label(ds):
    make_label(ds / "labels" / "train" / "ghost.txt", "0 0.5 0.5 0.2 0.2\n")
    assert codes(run_checks(ds)) == ["LABEL_ORPHAN"]


def test_classes_txt_is_not_an_orphan(ds):
    make_label(ds / "labels" / "classes.txt", "cat\ndog\n")
    assert run_checks(ds).issues == []


def test_duplicate_images(ds):
    shutil.copy(ds / "images" / "train" / "a.png", ds / "images" / "train" / "a_copy.png")
    shutil.copy(ds / "labels" / "train" / "a.txt", ds / "labels" / "train" / "a_copy.txt")
    report = run_checks(ds)
    assert codes(report) == ["IMG_DUPLICATE"]
    assert report.issues[0].path.endswith("a_copy.png")
    assert report.statistics["duplicate_groups"] == 1


def test_same_size_different_content_is_not_duplicate(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"1234")
    b.write_bytes(b"1235")
    assert find_duplicates([a, b]) == []


def test_empty_label_is_info_not_error(ds):
    make_label(ds / "labels" / "train" / "a.txt", "")
    report = run_checks(ds)
    assert codes(report) == ["LABEL_EMPTY"]
    assert report.errors == 0 and report.statistics["empty_label_files"] == 1


def test_class_imbalance_and_unused(tmp_path):
    for i in range(12):
        make_image(tmp_path / "images" / f"{i}.png", color=(i * 20, 10, 10))
        make_label(tmp_path / "labels" / f"{i}.txt", "0 0.25 0.25 0.4 0.4\n")
    make_image(tmp_path / "images" / "rare.png", color=(5, 250, 5))
    make_label(tmp_path / "labels" / "rare.txt", "1 0.25 0.25 0.4 0.4\n")
    report = run_checks(tmp_path, num_classes=3, imbalance_ratio=10)
    assert sorted(codes(report)) == ["CLASS_IMBALANCE", "CLASS_UNUSED"]
    assert report.statistics["class_distribution"]["0"]["count"] == 12


def test_empty_dataset_folder(tmp_path):
    report = run_checks(tmp_path)
    assert report.issues == [] and report.statistics["images_found"] == 0
    assert report.statistics["image_dimensions"]["min_width"] is None


@pytest.mark.parametrize("kw", [{"num_classes": 0}, {"min_size": -1}, {"imbalance_ratio": 1}, {"workers": 0}])
def test_invalid_options(ds, kw):
    with pytest.raises(DatasetError):
        run_checks(ds, **kw)


def test_missing_root_raises(tmp_path):
    with pytest.raises(DatasetError):
        run_checks(tmp_path / "nope")


def test_dataset_is_never_modified(ds):
    def snapshot():
        return {p.relative_to(ds).as_posix(): p.read_bytes() for p in ds.rglob("*") if p.is_file()}

    before = snapshot()
    run_checks(ds, num_classes=2, check_blank=True)
    assert snapshot() == before


def test_generated_clean_sample_passes(tmp_path):
    root = generate_sample_dataset(tmp_path / "clean", with_defects=False)
    report = run_checks(root, num_classes=3, check_blank=True)
    assert report.issues == []


def test_generated_defective_sample_triggers_every_check(tmp_path):
    root = generate_sample_dataset(tmp_path / "bad", with_defects=True)
    report = run_checks(root, num_classes=3, check_blank=True)
    expected = {
        "IMG_CORRUPT", "IMG_TOO_SMALL", "IMG_BLANK", "IMG_DUPLICATE", "LABEL_MISSING", "LABEL_ORPHAN",
        "LABEL_EMPTY", "LABEL_MALFORMED_LINE", "CLASS_NEGATIVE", "CLASS_NOT_INTEGER", "CLASS_OUT_OF_RANGE",
        "BOX_NONPOSITIVE_SIZE", "BOX_OUT_OF_BOUNDS", "BOX_NOT_NUMERIC", "BOX_DUPLICATE", "CLASS_IMBALANCE",
    }
    assert expected <= set(codes(report))


def test_sample_generator_refuses_non_empty_folder(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        generate_sample_dataset(tmp_path)
