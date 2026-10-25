# AI Dataset Quality Checker

[![tests](https://github.com/sujalkrishnadas/dataset-quality-checker/actions/workflows/tests.yml/badge.svg)](https://github.com/sujalkrishnadas/dataset-quality-checker/actions)

A lightweight, **CPU-only** command-line tool that validates computer-vision
datasets in **YOLO format** before you train a model. It finds problems,
explains them, and writes JSON, CSV and HTML reports.

* No GPU, no PyTorch, no cloud, no paid APIs. Only [Pillow](https://pypi.org/project/pillow/) is required.
* **Read-only:** it never modifies or deletes your dataset. There is deliberately no auto-repair feature.

## What it checks

| Check | Issue codes | Severity |
|---|---|---|
| Unreadable / corrupted / truncated / empty image files | `IMG_CORRUPT` | error |
| Image smaller than `--min-size` | `IMG_TOO_SMALL` | warning |
| Blank / nearly blank image (`--check-blank`) | `IMG_BLANK` | warning |
| Image with no label file | `LABEL_MISSING` | error |
| Label file with no image | `LABEL_ORPHAN` | warning |
| Empty label file (background image) | `LABEL_EMPTY` | info |
| Label file unreadable / not UTF-8 | `LABEL_UNREADABLE` | error |
| Line without exactly 5 values (incl. segmentation polygons) | `LABEL_MALFORMED_LINE` | error |
| Class id negative / not an integer / out of range | `CLASS_NEGATIVE`, `CLASS_NOT_INTEGER`, `CLASS_OUT_OF_RANGE` | error |
| Non-numeric, NaN or infinite coordinates | `BOX_NOT_NUMERIC` | error |
| Zero or negative box width/height | `BOX_NONPOSITIVE_SIZE` | error |
| Box outside the image (also catches pixel coordinates) | `BOX_OUT_OF_BOUNDS` | error |
| Identical box repeated; box under 1 pixel | `BOX_DUPLICATE`, `BOX_TINY` | warning |
| Byte-identical duplicate images (SHA-256) | `IMG_DUPLICATE` | warning |
| Severe class imbalance; declared classes never used | `CLASS_IMBALANCE`, `CLASS_UNUSED` | warning |

It also reports statistics: image count, readable/unreadable counts, valid
objects, image dimensions, per-class object counts and issue counts.

## Install (Windows PowerShell)

```powershell
git clone https://github.com/Sujalkrishnadas/dataset-quality-checker.git
cd dataset-quality-checker
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

macOS / Linux: activate with `source .venv/bin/activate`.
Requires Python 3.9 or newer.

## Usage

```powershell
# 1. Create a synthetic dataset with planted defects (no download needed)
dqc sample sample_dataset\demo

# 2. Check it
dqc check sample_dataset\demo --num-classes 3 --check-blank

# 3. Open the report
start reports\report.html
```

Check your own dataset:

```powershell
dqc check C:\data\my_dataset --num-classes 5 --output reports --name my_run
```

Useful options (`dqc check --help` lists all):

| Option | Meaning |
|---|---|
| `--num-classes N` | Enables class-range checks and unused-class detection |
| `--min-size PX` | Small-image threshold (default 32) |
| `--check-blank` | Flag blank images |
| `--imbalance-ratio R` | Imbalance warning threshold (default 10) |
| `--formats json csv html` | Choose report formats |
| `--overwrite` | Replace existing reports (default: refuse and keep them) |
| `--no-report` | Console summary only |
| `--fail-on-warnings` | Exit code 1 on warnings too (useful in CI) |
| `--workers N` | Threads for image decoding and hashing |

**Exit codes:** `0` no errors, `1` errors found, `2` bad usage or unusable input.

### Supported layouts

Nested folders and the extensions `.jpg .jpeg .png .bmp .tif .tiff .webp .gif` (any letter case) are supported. Labels are found either

* next to the image: `data/a.jpg` and `data/a.txt`, or
* in a parallel folder: `data/images/train/a.jpg` and `data/labels/train/a.txt`.

### Report overwrite behavior

Existing report files are never replaced silently. If any target file exists the
run stops with an explanation and **nothing** is written; use `--overwrite` or `--name`.

## Limitations (please read)

This tool cannot detect every possible dataset problem. A clean report does **not** mean the dataset is good.

* **Corruption:** decoding every image catches unreadable and truncated files, but not images that decode fine yet contain the wrong content.
* **Annotations:** only format and geometry are checked, never whether a box really surrounds the right object or the class is semantically correct.
* **Duplicates:** only byte-identical files are found. Resized, re-saved or re-compressed copies are not.
* **Blank check:** based on pixel variation; a legitimately uniform image (e.g. a clean background shot) is flagged.
* **Class imbalance:** a ratio rule of thumb, not a statistical test, and it counts only valid boxes.
* **Formats:** YOLO detection only. YOLO segmentation polygons are reported as malformed. COCO JSON is not implemented yet.
* **Performance:** every image is fully decoded. About 3,000 small JPEGs took ~4 s on a test machine; very large images or network drives will be slower.

## Architecture

```
src/dataset_quality_checker/
  models.py             Issue, Severity, Box, ImageInfo
  scanner.py            find images / labels, pair them
  validators.py         image + YOLO label validation
  duplicate_detector.py size-then-SHA-256 duplicate grouping
  statistics.py         class distribution, imbalance, dimensions
  checker.py            orchestrates everything -> DatasetReport
  reports.py            JSON / CSV / HTML writers
  sample_data.py        synthetic dataset generator
  cli.py                the `dqc` command
```

**Adding COCO later:** `parse_yolo_label` takes a label file and returns `(boxes, issues)`.
Add a sibling `parse_coco_json` with the same return shape, then choose the parser in
`checker.py` from a new `--format` option. Everything downstream (statistics, reports)
only sees `Box` and `Issue`, so it needs no changes.

## Example output

See [`examples/`](examples/) for the JSON, CSV and HTML reports generated from the defective sample dataset.

## Contributing

Issues and pull requests are welcome. Run `pytest` before submitting.

## License

MIT
