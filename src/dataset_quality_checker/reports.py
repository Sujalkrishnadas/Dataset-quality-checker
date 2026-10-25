"""Write reports as JSON, CSV and a self-contained HTML page.

Overwrite policy: existing report files are NEVER replaced unless the caller
passes ``overwrite=True`` (CLI: ``--overwrite``). If any target already
exists, nothing is written at all, so you never get a half-updated set.
Reports are written to a separate output folder, never into the dataset
unless you choose that folder yourself.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List

from dataset_quality_checker.checker import DatasetReport
from dataset_quality_checker.models import DatasetError

FORMATS = ("json", "csv", "html")
HTML_ISSUE_LIMIT = 2000

_CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem auto;max-width:1000px;padding:0 1rem;color:#1f2933;background:#fff}
h1{margin-bottom:.2rem} h2{margin-top:2rem;border-bottom:1px solid #d9dee3;padding-bottom:.3rem}
.muted{color:#6b7785;font-size:.9rem}
.cards{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}
.card{flex:1;min-width:140px;border:1px solid #d9dee3;border-radius:8px;padding:.8rem 1rem}
.card b{display:block;font-size:1.7rem}
.err{border-left:6px solid #c0392b}.warn{border-left:6px solid #d68910}.ok{border-left:6px solid #1e8449}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{border:1px solid #d9dee3;padding:.35rem .5rem;text-align:left;vertical-align:top}
th{background:#f3f5f7}
.sev-error{color:#c0392b;font-weight:600}.sev-warning{color:#b9770e;font-weight:600}.sev-info{color:#5d6d7e}
.bar{background:#d6eaf8;height:1rem;border-radius:3px}.barfill{background:#2e86c1;height:100%;border-radius:3px}
.box{background:#f8f9fa;border:1px solid #d9dee3;border-radius:8px;padding:.6rem 1rem;font-size:.9rem}
@media (prefers-color-scheme: dark){
body{background:#14181c;color:#e4e8ec}th{background:#222a31}th,td,.card,.box{border-color:#37424d}
.box{background:#1b2127}.bar{background:#27465f}.muted{color:#9aa7b4}}
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _csv_safe(value: str) -> str:
    """Neutralise spreadsheet formula injection (cells starting with = + @)."""
    return "'" + value if value[:1] in ("=", "+", "@") else value


def render_json(report: DatasetReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def render_csv_rows(report: DatasetReport) -> List[List[str]]:
    rows = [["severity", "code", "path", "line", "message"]]
    for i in report.issues:
        rows.append([i.severity.value, i.code, _csv_safe(i.path), "" if i.line is None else str(i.line), _csv_safe(i.message)])
    return rows


def render_html(report: DatasetReport) -> str:
    s = report.statistics
    counts = s["issue_counts"]
    dims = s["image_dimensions"]
    status_cls = "err" if counts["errors"] else ("warn" if counts["warnings"] else "ok")
    status_txt = (
        "Errors found: fix before training" if counts["errors"]
        else "No errors, but there are warnings to review" if counts["warnings"]
        else "No errors or warnings found"
    )

    def card(label: str, value: object, cls: str = "") -> str:
        return f'<div class="card {cls}"><b>{_esc(value)}</b>{_esc(label)}</div>'

    parts: List[str] = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Dataset Quality Report</title><style>{_CSS}</style></head><body>",
        "<h1>Dataset Quality Report</h1>",
        f"<p class='muted'>Dataset: {_esc(report.root)}</p>",
        f"<div class='card {status_cls}'><b>{_esc(status_txt)}</b></div>",
        "<div class='cards'>",
        card("errors", counts["errors"], "err" if counts["errors"] else "ok"),
        card("warnings", counts["warnings"], "warn" if counts["warnings"] else "ok"),
        card("info notes", counts["info"]),
        card("images found", s["images_found"]),
        card("valid objects", s["valid_objects"]),
        "</div>",
        "<h2>Statistics</h2><table>",
    ]
    stat_rows = [
        ("Images found", s["images_found"]),
        ("Readable images", s["images_readable"]),
        ("Unreadable images", s["images_unreadable"]),
        ("Images without label", s["images_without_label"]),
        ("Label files read", s["label_files_read"]),
        ("Empty label files (background images)", s["empty_label_files"]),
        ("Invalid label lines", s["invalid_label_lines"]),
        ("Valid objects (annotations)", s["valid_objects"]),
        ("Duplicate groups", s["duplicate_groups"]),
    ]
    if dims["min_width"] is not None:
        stat_rows += [
            ("Width min / mean / max", f"{dims['min_width']} / {dims['mean_width']} / {dims['max_width']}"),
            ("Height min / mean / max", f"{dims['min_height']} / {dims['mean_height']} / {dims['max_height']}"),
            ("Most common sizes", ", ".join(f"{c['size']} (x{c['count']})" for c in dims["common_sizes"])),
        ]
    parts += [f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in stat_rows]
    parts.append("</table>")

    parts.append("<h2>Class distribution</h2>")
    dist = s["class_distribution"]
    if dist:
        top = max(v["count"] for v in dist.values()) or 1
        parts.append("<table><tr><th>Class</th><th>Objects</th><th>Share</th><th style='width:50%'></th></tr>")
        for cid, v in dist.items():
            width = 100.0 * v["count"] / top
            parts.append(
                f"<tr><td>{_esc(cid)}</td><td>{v['count']}</td><td>{v['percent']}%</td>"
                f"<td><div class='bar'><div class='barfill' style='width:{width:.1f}%'></div></div></td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p class='muted'>No valid objects found.</p>")

    parts.append("<h2>Issues by type</h2>")
    if s["issues_by_code"]:
        sev_of: Dict[str, str] = {i.code: i.severity.value for i in report.issues}
        parts.append("<table><tr><th>Code</th><th>Severity</th><th>Count</th></tr>")
        for code, n in s["issues_by_code"].items():
            parts.append(f"<tr><td>{_esc(code)}</td><td class='sev-{sev_of[code]}'>{sev_of[code]}</td><td>{n}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p class='muted'>None.</p>")

    parts.append("<h2>All issues</h2>")
    if report.issues:
        parts.append("<table><tr><th>Severity</th><th>Code</th><th>File</th><th>Line</th><th>Message</th></tr>")
        for i in report.issues[:HTML_ISSUE_LIMIT]:
            parts.append(
                f"<tr><td class='sev-{i.severity.value}'>{i.severity.value}</td><td>{_esc(i.code)}</td>"
                f"<td>{_esc(i.path or '(dataset)')}</td><td>{'' if i.line is None else i.line}</td><td>{_esc(i.message)}</td></tr>"
            )
        parts.append("</table>")
        if len(report.issues) > HTML_ISSUE_LIMIT:
            parts.append(f"<p class='muted'>Showing the first {HTML_ISSUE_LIMIT} of {len(report.issues)} issues. The CSV and JSON reports contain all of them.</p>")
    else:
        parts.append("<p class='muted'>No issues found.</p>")

    parts.append(
        "<h2>What this report does not cover</h2><div class='box'>"
        "This tool checks technical validity only. It does <b>not</b> verify that boxes actually "
        "surround the right objects, that class labels are semantically correct, or that near-duplicate "
        "(resized or re-compressed) images are absent. A clean report does not guarantee a good dataset.</div>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def write_reports(
    report: DatasetReport,
    out_dir: Path,
    *,
    name: str = "report",
    formats: Iterable[str] = FORMATS,
    overwrite: bool = False,
) -> List[Path]:
    """Write the requested report formats into ``out_dir``; return the paths written.

    Raises:
        DatasetError: unknown format.
        FileExistsError: a target exists and ``overwrite`` is False (nothing is written).
    """
    formats = list(dict.fromkeys(formats))
    for fmt in formats:
        if fmt not in FORMATS:
            raise DatasetError(f"Unknown report format '{fmt}'. Choose from: {', '.join(FORMATS)}.")
    out_dir = Path(out_dir)
    targets = {fmt: out_dir / f"{name}.{fmt}" for fmt in formats}

    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Report file(s) already exist and were left untouched: "
            + ", ".join(existing)
            + ". Use --overwrite to replace them or --name to pick another name."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for fmt, path in targets.items():
        if fmt == "json":
            path.write_text(render_json(report), encoding="utf-8")
        elif fmt == "html":
            path.write_text(render_html(report), encoding="utf-8")
        else:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:  # BOM so Excel reads UTF-8
                csv.writer(handle).writerows(render_csv_rows(report))
        written.append(path)
    return written
