"""Tests for report writing and the command-line interface."""

import csv
import json

import pytest

from dataset_quality_checker.checker import run_checks
from dataset_quality_checker.cli import main
from dataset_quality_checker.models import DatasetError
from dataset_quality_checker.reports import render_html, write_reports
from dataset_quality_checker.sample_data import generate_sample_dataset


@pytest.fixture
def bad_report(tmp_path):
    root = generate_sample_dataset(tmp_path / "bad")
    return run_checks(root, num_classes=3, check_blank=True)


def test_all_formats_written_and_valid(bad_report, tmp_path):
    out = tmp_path / "out"
    paths = write_reports(bad_report, out)
    assert {p.suffix for p in paths} == {".json", ".csv", ".html"}
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert data["statistics"]["issue_counts"]["errors"] == bad_report.errors
    assert len(data["issues"]) == len(bad_report.issues)
    with open(out / "report.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["severity", "code", "path", "line", "message"]
    assert len(rows) == len(bad_report.issues) + 1
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Class distribution" in html and "IMG_CORRUPT" in html


def test_existing_reports_are_not_overwritten(bad_report, tmp_path):
    write_reports(bad_report, tmp_path, formats=["json"])
    (tmp_path / "report.json").write_text("KEEP ME")
    with pytest.raises(FileExistsError):
        write_reports(bad_report, tmp_path, formats=["json", "csv"])
    assert (tmp_path / "report.json").read_text() == "KEEP ME"
    assert not (tmp_path / "report.csv").exists()  # all-or-nothing
    write_reports(bad_report, tmp_path, formats=["json"], overwrite=True)
    assert (tmp_path / "report.json").read_text() != "KEEP ME"


def test_unknown_format_rejected(bad_report, tmp_path):
    with pytest.raises(DatasetError):
        write_reports(bad_report, tmp_path, formats=["pdf"])


def test_html_escapes_untrusted_filenames(ds):
    (ds / "labels" / "train" / "<script>alert(1)<").with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2")
    html = render_html(run_checks(ds))
    assert "<script>alert" not in html and "&lt;script&gt;" in html


def test_cli_check_clean_exit_zero(ds, tmp_path, capsys):
    assert main(["check", str(ds), "-o", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "report.html").exists()
    assert "0 error(s)" in capsys.readouterr().out


def test_cli_check_defective_exit_one(tmp_path):
    root = generate_sample_dataset(tmp_path / "bad")
    assert main(["check", str(root), "--num-classes", "3", "--no-report"]) == 1


def test_cli_fail_on_warnings(ds):
    (ds / "labels" / "train" / "ghost.txt").write_text("0 0.5 0.5 0.2 0.2")
    assert main(["check", str(ds), "--no-report"]) == 0
    assert main(["check", str(ds), "--no-report", "--fail-on-warnings"]) == 1


def test_cli_missing_dataset_exit_two(tmp_path, capsys):
    assert main(["check", str(tmp_path / "nope"), "--no-report"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_refuses_overwrite_then_allows(ds, tmp_path, capsys):
    out = str(tmp_path / "r")
    assert main(["check", str(ds), "-o", out]) == 0
    assert main(["check", str(ds), "-o", out]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert main(["check", str(ds), "-o", out, "--overwrite"]) == 0


def test_cli_rejects_bad_numbers():
    with pytest.raises(SystemExit) as exc:
        main(["check", ".", "--num-classes", "0"])
    assert exc.value.code == 2


def test_cli_sample_command(tmp_path, capsys):
    target = tmp_path / "demo"
    assert main(["sample", str(target)]) == 0
    assert (target / "images").is_dir()
    assert main(["sample", str(target)]) == 2  # refuses non-empty folder
