"""Command-line interface.

Examples::

    dqc sample sample_dataset/demo            # create a synthetic dataset
    dqc check sample_dataset/demo --num-classes 3 --check-blank
    dqc check my_data --output reports --formats json html --overwrite

Exit codes: 0 = no errors found, 1 = errors found (or warnings with
``--fail-on-warnings``), 2 = bad usage / unusable input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from dataset_quality_checker import __version__
from dataset_quality_checker.checker import DatasetReport, run_checks
from dataset_quality_checker.models import DatasetError
from dataset_quality_checker.reports import FORMATS, write_reports
from dataset_quality_checker.sample_data import NUM_CLASSES, generate_sample_dataset


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not an integer")
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _ratio(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number")
    if value <= 1:
        raise argparse.ArgumentTypeError("must be greater than 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dqc", description="Validate computer vision datasets before model training."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="Validate a YOLO-format dataset (read-only).")
    check.add_argument("dataset", type=Path, help="Dataset root folder (searched recursively).")
    check.add_argument("--num-classes", type=_positive_int, help="Number of classes; enables class-range checks.")
    check.add_argument("--min-size", type=int, default=32, help="Warn for images smaller than this many pixels (default 32).")
    check.add_argument("--check-blank", action="store_true", help="Also flag blank / nearly blank images.")
    check.add_argument("--imbalance-ratio", type=_ratio, default=10.0, help="Warn when largest/smallest class ratio reaches this (default 10).")
    check.add_argument("--workers", type=_positive_int, help="Parallel threads (default: up to 4).")
    check.add_argument("-o", "--output", type=Path, default=Path("reports"), help="Folder for reports (default ./reports).")
    check.add_argument("--name", default="report", help="Report file base name (default 'report').")
    check.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS), help="Report formats to write.")
    check.add_argument("--no-report", action="store_true", help="Print the summary only; write no files.")
    check.add_argument("--overwrite", action="store_true", help="Replace existing report files (default: refuse).")
    check.add_argument("--fail-on-warnings", action="store_true", help="Exit with code 1 if there are warnings too.")
    check.add_argument("--max-print", type=int, default=15, help="Max issues printed to the console (default 15).")
    check.add_argument("-v", "--verbose", action="store_true", help="Show progress logging.")

    sample = sub.add_parser("sample", help="Generate a small synthetic dataset.")
    sample.add_argument("folder", type=Path, help="Target folder (must not exist or be empty).")
    sample.add_argument("--clean", action="store_true", help="Generate a defect-free dataset.")
    sample.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    return parser


def _print_summary(report: DatasetReport, max_print: int) -> None:
    s = report.statistics
    c = s["issue_counts"]
    print(f"Dataset: {report.root}")
    print(f"Images: {s['images_found']} ({s['images_unreadable']} unreadable) | "
          f"Valid objects: {s['valid_objects']} | Images without label: {s['images_without_label']}")
    if s["class_distribution"]:
        dist = ", ".join(f"class {k}: {v['count']}" for k, v in s["class_distribution"].items())
        print(f"Class distribution: {dist}")
    print(f"Result: {c['errors']} error(s), {c['warnings']} warning(s), {c['info']} info note(s)")
    for issue in report.issues[:max_print]:
        where = issue.path or "(dataset)"
        if issue.line:
            where += f":{issue.line}"
        print(f"  [{issue.severity.value.upper():7}] {issue.code:22} {where} - {issue.message}")
    if len(report.issues) > max_print:
        print(f"  ... and {len(report.issues) - max_print} more (see the report files).")


def _cmd_check(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    report = run_checks(
        args.dataset,
        num_classes=args.num_classes,
        min_size=args.min_size,
        check_blank=args.check_blank,
        imbalance_ratio=args.imbalance_ratio,
        workers=args.workers,
    )
    _print_summary(report, args.max_print)
    if not args.no_report:
        written = write_reports(report, args.output, name=args.name, formats=args.formats, overwrite=args.overwrite)
        print("Reports written: " + ", ".join(str(p) for p in written))
    if report.errors or (args.fail_on_warnings and report.warnings):
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "check":
            return _cmd_check(args)
        generate_sample_dataset(args.folder, with_defects=not args.clean, seed=args.seed)
        kind = "clean" if args.clean else "defective"
        print(f"Created {kind} sample dataset in {args.folder}")
        print(f"Try: dqc check {args.folder} --num-classes {NUM_CLASSES} --check-blank")
        return 0
    except (DatasetError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
