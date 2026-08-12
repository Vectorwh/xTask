"""Concurrency-safe Excel workbooks for accumulated xTask results."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


SUMMARY_SHEET = "执行汇总"
EXCEL_CELL_LIMIT = 32767
INVALID_SHEET_CHARS = re.compile(r"[\\/*?:\[\]]")
ACCURACY_CASE_TOKEN = re.compile(r"(?:^|[_-])acc(?:uracy)?(?:[_-]|$)", re.IGNORECASE)
# Terminal color sequences (e.g. \x1b[32mINFO\x1b[0m) leaking in from tool stderr.
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Control characters openpyxl refuses to store; mirrors
# openpyxl.cell.cell.ILLEGAL_CHARACTERS_RE.
ILLEGAL_XML_CHARS_RE = re.compile(r"[\000-\010]|[\013-\014]|[\016-\037]")
NOT_APPLICABLE = "NA"
MISSING_ACCURACY = "-"

LEADING_RESULT_FIELDS = (
    "model_name",
    "case_group",
    "case_name",
    "status",
    "finished_at",
    "execution_time_s",
)

EXCLUDED_RESULT_FIELDS = {
    "host_addr",
    "source_yaml",
    "metadata.source_file",
    "metadata.parse_time",
    "metadata.total_rows",
    "case_type",
}

PERF_SUMMARY_FIELDS = (
    "total_requests",
    "failed_requests",
    "concurrency",
    "max_concurrency",
    "request_throughput",
    "output_tps_avg",
    "output_tps_total",
    "summary_tps_total",
)
PERF_SUMMARY_HEADERS = {f"summary.{field}" for field in PERF_SUMMARY_FIELDS}
LEGACY_RESULT_FIELD_RENAMES = {
    "summary_requests": "summary.total_requests",
    "summary.output_token_throughput": "summary.output_tps_avg",
}

FAILURE_DETAIL_FIELDS = (
    "failure_category",
    "failure_reason",
    "baseline_difference",
)

CHINA_STANDARD_TIME = timezone(timedelta(hours=8))

SUMMARY_FIELDS = (
    "model_name",
    "total_runs",
    "passed",
    "failed",
    "xLLM进程异常",
    "benchmark结果与基线比对失败",
    "benchmark执行异常",
    "benchmark结果处理异常",
    "其他异常",
)


def _sheet_title(model_name: str) -> str:
    readable = INVALID_SHEET_CHARS.sub("_", str(model_name or "unknown")).strip(" '")
    readable = readable or "unknown"
    if len(readable) <= 31:
        return readable
    digest = hashlib.sha1(readable.encode("utf-8")).hexdigest()[:6]
    return f"{readable[:24]}_{digest}"


def _excel_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float)):
        return value
    # Benchmark stderr may carry ANSI colors and control characters that
    # openpyxl rejects (IllegalCharacterError), which would silently drop the
    # whole result row; strip them before truncation keeps the row writable.
    text = ANSI_ESCAPE_RE.sub("", str(value))
    text = ILLEGAL_XML_CHARS_RE.sub("", text)
    if len(text) > EXCEL_CELL_LIMIT:
        text = text[: EXCEL_CELL_LIMIT - 16] + "...[truncated]"
    if text.startswith(("=", "+", "@")):
        return "'" + text
    return text


def _new_workbook():
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    return workbook


def _load_workbook(path: Path, reset: bool):
    if reset or not path.exists():
        return _new_workbook()
    from openpyxl import load_workbook

    return load_workbook(path)


def _header_style(cell) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="1F4E78")
    cell.alignment = Alignment(horizontal="center", vertical="center")


def _style_data_cell(cell, field_name: str) -> None:
    from openpyxl.styles import Alignment, PatternFill

    cell.alignment = Alignment(
        vertical="top",
        wrap_text=field_name
        in {"failure_reason", "baseline_difference", "output_path"},
    )
    if field_name == "status":
        color = "C6EFCE" if cell.value == "pass" else "FFC7CE"
        cell.fill = PatternFill("solid", fgColor=color)


def _update_column_width(worksheet, column_index: int, value: Any) -> None:
    from openpyxl.utils import get_column_letter

    letter = get_column_letter(column_index)
    text = str(value or "")
    desired = min(max(len(text) + 2, 12), 48)
    current = worksheet.column_dimensions[letter].width or 0
    if desired > current:
        worksheet.column_dimensions[letter].width = desired


def _ensure_headers(worksheet, row: Mapping[str, Any]) -> list[str]:
    headers = [
        str(worksheet.cell(1, column).value)
        for column in range(1, worksheet.max_column + 1)
        if worksheet.cell(1, column).value is not None
    ]
    for field_name in row:
        if field_name not in headers:
            headers.append(field_name)
            cell = worksheet.cell(1, len(headers), field_name)
            _header_style(cell)
            _update_column_width(worksheet, len(headers), field_name)
    return headers


def _beijing_finished_at(value: Any) -> Any:
    if not value:
        return value
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return value
    if parsed.tzinfo is None:
        return value
    return parsed.astimezone(CHINA_STANDARD_TIME).isoformat(timespec="seconds")


def _normalized_result_field(field_name: str):
    field_name = LEGACY_RESULT_FIELD_RENAMES.get(field_name, field_name)
    if (
        field_name in EXCLUDED_RESULT_FIELDS
        or field_name.endswith(".n")
        or field_name.startswith(("parameters.ITL.", "ITL.", "metadata."))
        or (
            field_name.startswith("summary.")
            and field_name not in PERF_SUMMARY_HEADERS
        )
    ):
        return None
    if field_name.startswith("parameters."):
        field_name = field_name[len("parameters.") :]
    return ".".join(part for part in field_name.split(".") if part != "total")


def _is_blank(value: Any) -> bool:
    return value in (None, "")


def _has_accuracy_value(value: Any) -> bool:
    return not _is_blank(value) and str(value).strip().upper() != NOT_APPLICABLE


def _accuracy_case_names(rows: list[Mapping[str, Any]]) -> set[str]:
    return {
        str(values.get("case_name") or "")
        for values in rows
        if _has_accuracy_value(values.get("accuracy"))
        or ACCURACY_CASE_TOKEN.search(str(values.get("case_name") or ""))
    }


def _normalize_result_sheet(worksheet) -> None:
    source_headers = [
        (column, str(worksheet.cell(1, column).value))
        for column in range(1, worksheet.max_column + 1)
        if worksheet.cell(1, column).value is not None
    ]
    if not source_headers:
        return

    kept_headers = []
    for _, field_name in source_headers:
        normalized = _normalized_result_field(field_name)
        if normalized is not None and normalized not in kept_headers:
            kept_headers.append(normalized)
    ordered_headers = [
        field_name for field_name in LEADING_RESULT_FIELDS if field_name in kept_headers
    ]
    ordered_headers.extend(
        field_name
        for field_name in kept_headers
        if field_name not in LEADING_RESULT_FIELDS
    )
    if "accuracy" in ordered_headers and "output_path" in ordered_headers:
        ordered_headers.remove("accuracy")
        ordered_headers.insert(ordered_headers.index("output_path") + 1, "accuracy")

    rows = []
    values_changed = False
    for row_index in range(2, worksheet.max_row + 1):
        values = {}
        for column, field_name in source_headers:
            normalized = _normalized_result_field(field_name)
            if normalized is None:
                continue
            value = worksheet.cell(row_index, column).value
            if normalized not in values or value not in (None, ""):
                values[normalized] = value
        finished_at = _beijing_finished_at(values.get("finished_at"))
        if finished_at != values.get("finished_at"):
            values["finished_at"] = finished_at
            values_changed = True
        if values.get("status") == "pass":
            for field_name in ("failure_category", "failure_reason"):
                if _is_blank(values.get(field_name)):
                    values[field_name] = NOT_APPLICABLE
                    values_changed = True
        rows.append(values)

    accuracy_cases = _accuracy_case_names(rows)
    performance_headers = []
    if "accuracy" in ordered_headers:
        performance_headers = ordered_headers[ordered_headers.index("accuracy") + 1 :]
    for values in rows:
        if _is_blank(values.get("baseline_difference")):
            values["baseline_difference"] = NOT_APPLICABLE
            values_changed = True
        if "accuracy" not in ordered_headers:
            continue
        case_name = str(values.get("case_name") or "")
        if case_name in accuracy_cases:
            if _is_blank(values.get("accuracy")) or str(
                values.get("accuracy")
            ).strip().upper() == NOT_APPLICABLE:
                values["accuracy"] = MISSING_ACCURACY
                values_changed = True
            for field_name in performance_headers:
                if _is_blank(values.get(field_name)):
                    values[field_name] = NOT_APPLICABLE
                    values_changed = True
        elif _is_blank(values.get("accuracy")):
            values["accuracy"] = NOT_APPLICABLE
            values_changed = True

    current_headers = [field_name for _, field_name in source_headers]
    if current_headers == ordered_headers and not values_changed:
        _refresh_sheet_layout(worksheet)
        return

    worksheet.delete_cols(1, worksheet.max_column)
    for column_index, field_name in enumerate(ordered_headers, start=1):
        cell = worksheet.cell(1, column_index, field_name)
        _header_style(cell)
        _update_column_width(worksheet, column_index, field_name)
    for row_index, values in enumerate(rows, start=2):
        for column_index, field_name in enumerate(ordered_headers, start=1):
            value = _excel_value(values.get(field_name, ""))
            cell = worksheet.cell(row_index, column_index, value)
            _style_data_cell(cell, field_name)
            _update_column_width(worksheet, column_index, value)
    _refresh_sheet_layout(worksheet)


def _insert_grouped_row(worksheet, headers: list[str], row: Mapping[str, Any]) -> None:
    case_column = headers.index("case_name") + 1
    last_case_row = None
    for row_index in range(2, worksheet.max_row + 1):
        if worksheet.cell(row_index, case_column).value == row.get("case_name"):
            last_case_row = row_index

    if last_case_row is None:
        target_row = max(worksheet.max_row + 1, 2)
    else:
        target_row = last_case_row + 1
        worksheet.insert_rows(target_row)

    for column_index, field_name in enumerate(headers, start=1):
        value = _excel_value(row.get(field_name, ""))
        cell = worksheet.cell(target_row, column_index, value)
        _style_data_cell(cell, field_name)
        _update_column_width(worksheet, column_index, value)


def _group_result_rows_by_metric(worksheet) -> None:
    """Keep perf cases first and accuracy cases last without changing case history."""
    if worksheet.max_row < 3:
        return
    headers = [
        str(worksheet.cell(1, column).value)
        for column in range(1, worksheet.max_column + 1)
        if worksheet.cell(1, column).value is not None
    ]
    if "case_name" not in headers:
        return

    rows = [
        {
            field_name: worksheet.cell(row_index, column_index).value
            for column_index, field_name in enumerate(headers, start=1)
        }
        for row_index in range(2, worksheet.max_row + 1)
    ]
    accuracy_cases = _accuracy_case_names(rows)
    case_order: dict[tuple[str, str], int] = {}
    next_order = {"perf": 0, "acc": 0}
    decorated_rows = []
    for original_index, values in enumerate(rows):
        case_name = str(values.get("case_name") or "")
        metric_type = "acc" if case_name in accuracy_cases else "perf"
        case_key = (metric_type, case_name)
        if case_key not in case_order:
            case_order[case_key] = next_order[metric_type]
            next_order[metric_type] += 1
        decorated_rows.append(
            (
                1 if metric_type == "acc" else 0,
                case_order[case_key],
                original_index,
                values,
            )
        )

    ordered_rows = sorted(decorated_rows, key=lambda item: item[:3])
    if [item[2] for item in ordered_rows] == list(range(len(rows))):
        return

    for row_index, (_, _, _, values) in enumerate(ordered_rows, start=2):
        for column_index, field_name in enumerate(headers, start=1):
            value = _excel_value(values.get(field_name, ""))
            cell = worksheet.cell(row_index, column_index, value)
            _style_data_cell(cell, field_name)
            _update_column_width(worksheet, column_index, value)
    _refresh_sheet_layout(worksheet)


def _refresh_sheet_layout(worksheet) -> None:
    from openpyxl.styles import Border, Side
    from openpyxl.utils import get_column_letter

    side = Side(style="thin", color="B7B7B7")
    table_border = Border(left=side, right=side, top=side, bottom=side)
    for row in worksheet.iter_rows(
        min_row=1,
        max_row=worksheet.max_row,
        min_col=1,
        max_col=worksheet.max_column,
    ):
        for cell in row:
            cell.border = table_border

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = (
        f"A1:{get_column_letter(worksheet.max_column)}{worksheet.max_row}"
    )
    worksheet.row_dimensions[1].height = 24


def _summary_counts(workbook) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for worksheet in workbook.worksheets:
        if worksheet.title == SUMMARY_SHEET or worksheet.max_row < 2:
            continue
        headers = {
            str(worksheet.cell(1, column).value): column
            for column in range(1, worksheet.max_column + 1)
            if worksheet.cell(1, column).value is not None
        }
        if "status" not in headers:
            continue
        for row_index in range(2, worksheet.max_row + 1):
            model_name = worksheet.cell(
                row_index, headers.get("model_name", 1)
            ).value or worksheet.title
            status = worksheet.cell(row_index, headers["status"]).value
            category = (
                worksheet.cell(row_index, headers["failure_category"]).value
                if "failure_category" in headers
                else ""
            )
            model_counts = counts.setdefault(
                str(model_name), {field: 0 for field in SUMMARY_FIELDS[1:]}
            )
            model_counts["total_runs"] += 1
            if status == "pass":
                model_counts["passed"] += 1
            else:
                model_counts["failed"] += 1
                category = str(category or "其他异常")
                if category not in model_counts:
                    category = "其他异常"
                model_counts[category] += 1
    return counts


def _rebuild_summary_sheet(workbook) -> None:
    if SUMMARY_SHEET in workbook.sheetnames:
        workbook.remove(workbook[SUMMARY_SHEET])
    worksheet = workbook.create_sheet(SUMMARY_SHEET, 0)
    worksheet.append(SUMMARY_FIELDS)
    for cell in worksheet[1]:
        _header_style(cell)

    counts = _summary_counts(workbook)
    total = {field: 0 for field in SUMMARY_FIELDS[1:]}
    for model_name in sorted(counts):
        values = counts[model_name]
        worksheet.append([model_name] + [values[field] for field in SUMMARY_FIELDS[1:]])
        for field in SUMMARY_FIELDS[1:]:
            total[field] += values[field]
    worksheet.append(["TOTAL"] + [total[field] for field in SUMMARY_FIELDS[1:]])

    for column_index, field_name in enumerate(SUMMARY_FIELDS, start=1):
        _update_column_width(worksheet, column_index, field_name)
        for row_index in range(2, worksheet.max_row + 1):
            _update_column_width(
                worksheet, column_index, worksheet.cell(row_index, column_index).value
            )
    _refresh_sheet_layout(worksheet)


def _save_atomically(workbook, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.stem}.", suffix=".xlsx"
    )
    os.close(descriptor)
    try:
        workbook.save(temporary_name)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def append_result_workbook(
    path: Path,
    model_name: str,
    row: Mapping[str, Any],
    *,
    reset: bool = False,
) -> bool:
    """Append a grouped result row and atomically replace the workbook."""
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT,
            0o644,
        )
        with os.fdopen(lock_descriptor, "r+") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            workbook = _load_workbook(path, reset)
            try:
                for existing_sheet in workbook.worksheets:
                    if existing_sheet.title != SUMMARY_SHEET:
                        _normalize_result_sheet(existing_sheet)
                        _group_result_rows_by_metric(existing_sheet)
                title = _sheet_title(model_name)
                worksheet = (
                    workbook[title]
                    if title in workbook.sheetnames
                    else workbook.create_sheet(title)
                )
                headers = _ensure_headers(worksheet, row)
                _normalize_result_sheet(worksheet)
                headers = [
                    str(worksheet.cell(1, column).value)
                    for column in range(1, worksheet.max_column + 1)
                    if worksheet.cell(1, column).value is not None
                ]
                _insert_grouped_row(worksheet, headers, row)
                _normalize_result_sheet(worksheet)
                _group_result_rows_by_metric(worksheet)
                _refresh_sheet_layout(worksheet)
                _rebuild_summary_sheet(workbook)
                _save_atomically(workbook, path)
            finally:
                workbook.close()
        return True
    except Exception:
        logging.error("Failed to update result workbook %s", path, exc_info=True)
        return False


def normalize_result_workbook(path: Path) -> bool:
    """Apply the current result schema to an existing workbook in place."""
    path = Path(path)
    if not path.exists():
        return True
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        with os.fdopen(lock_descriptor, "r+") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            workbook = _load_workbook(path, reset=False)
            try:
                for worksheet in workbook.worksheets:
                    if worksheet.title != SUMMARY_SHEET:
                        _normalize_result_sheet(worksheet)
                        _group_result_rows_by_metric(worksheet)
                _rebuild_summary_sheet(workbook)
                _save_atomically(workbook, path)
            finally:
                workbook.close()
        return True
    except Exception:
        logging.error("Failed to normalize result workbook %s", path, exc_info=True)
        return False
