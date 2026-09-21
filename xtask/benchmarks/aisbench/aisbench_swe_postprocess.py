"""Normalize AISBench SWE-bench JSON into a small xTask result mapping.

Looks under the case output directory for official AISBench files:

- ``results/**/swebench*.json`` for eval metrics (accuracy / resolved_*)
- ``predictions/**/swebench*.json`` for infer-only evidence

Existing AISBench GSM8K acc/perf parsers are not used.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from xtask.benchmarks.aisbench.swebench import swebench_mode


_DATE_DIR = re.compile(r"^\d{8}_\d{6}$")
_RESULT_JSON_HINTS = ("accuracy", "resolved_instances", "submitted_accuracy")
_PAYLOAD_FIELDS = (
    "resolved_rate",
    "accuracy",
    "submitted_accuracy",
    "resolved_instances",
    "unresolved_instances",
    "error_instances",
    "empty_patch_instances",
    "total_instances",
    "submitted_instances",
    "completed_ratio",
    "harness_exit_code",
    "result_file",
    "predictions_file",
    "mode",
    "dataset",
)


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _as_int(value: Any) -> Optional[int]:
    number = _as_number(value)
    if number is None:
        return None
    if abs(number - round(number)) > 1e-9:
        return int(number)
    return int(round(number))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _looks_like_eval_result(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return any(key in payload for key in _RESULT_JSON_HINTS)


def _collect_json_files(root: Path) -> Iterable[Path]:
    try:
        yield from sorted(
            path
            for path in root.rglob("*.json")
            if path.is_file() and path.name != "aisbench_swe_config.py"
        )
    except OSError:
        return


def _find_eval_result(root: Path) -> Tuple[Optional[Path], Optional[Mapping[str, Any]]]:
    preferred = []
    fallback = []
    for path in _collect_json_files(root):
        relative = path.as_posix().lower()
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not _looks_like_eval_result(payload):
            continue
        if "results" in relative.split("/") or "swebench" in path.name.lower():
            preferred.append((path, payload))
        else:
            fallback.append((path, payload))
    chosen = preferred or fallback
    if not chosen:
        return None, None
    path, payload = chosen[-1]
    return path, payload


def _find_predictions(root: Path) -> Optional[Path]:
    matches = []
    for path in _collect_json_files(root):
        relative = path.as_posix().lower()
        if "prediction" in relative or path.name.lower().startswith("swebench"):
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping) and payload and not _looks_like_eval_result(
                payload
            ):
                matches.append(path)
            elif isinstance(payload, list) and payload:
                matches.append(path)
    return matches[-1] if matches else None


def _count_instances(payload: Any) -> int:
    if isinstance(payload, Mapping):
        if "instance_id" in payload or "model_patch" in payload:
            return 1
        return sum(_count_instances(item) for item in payload.values())
    if isinstance(payload, list):
        return sum(_count_instances(item) for item in payload)
    return 0


def _output_dir_for(root: Path, result_file: Optional[Path]) -> str:
    if result_file is not None:
        for parent in result_file.parents:
            if _DATE_DIR.fullmatch(parent.name):
                return str(parent)
            if parent == root:
                break
        return str(result_file.parent)
    try:
        dated = [
            item
            for item in root.iterdir()
            if item.is_dir() and _DATE_DIR.fullmatch(item.name)
        ]
    except OSError:
        dated = []
    if dated:
        return str(max(dated, key=lambda item: item.name))
    return str(root)


def _normalize_eval_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    accuracy = _as_number(
        payload.get("accuracy", payload.get("resolved_rate", payload.get("resolved")))
    )
    total = _as_int(
        payload.get("total_instances", payload.get("num_instances", payload.get("total")))
    )
    resolved = _as_int(payload.get("resolved_instances", payload.get("resolved")))
    submitted = _as_int(
        payload.get("submitted_instances", payload.get("completed_instances"))
    )
    if accuracy is None and resolved is not None and total:
        accuracy = resolved / float(total)
    completed_ratio = None
    if submitted is not None and total:
        completed_ratio = submitted / float(total)
    elif accuracy is not None:
        completed_ratio = accuracy
    return {
        "resolved_rate": accuracy,
        "accuracy": accuracy,
        "submitted_accuracy": _as_number(payload.get("submitted_accuracy")),
        "resolved_instances": resolved,
        "unresolved_instances": _as_int(payload.get("unresolved_instances")),
        "error_instances": _as_int(payload.get("error_instances")),
        "empty_patch_instances": _as_int(payload.get("empty_patch_instances")),
        "total_instances": total,
        "submitted_instances": submitted,
        "completed_ratio": completed_ratio,
        "harness_exit_code": _as_int(payload.get("harness_exit_code")),
    }


def aisbench_swe_post_process(
    task_config: Mapping[str, Any],
    output_file_path: str,
    init_entries: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], str]:
    del init_entries  # AISBench SWE writes under the case dir / work_dir.
    root = Path(output_file_path)
    mode = swebench_mode(task_config)
    result_file, payload = _find_eval_result(root)
    predictions_file = _find_predictions(root)
    result: Dict[str, Any] = {
        "mode": mode,
        "dataset": str(
            ((task_config.get("benchmark") or {}).get("swebench") or {}).get(
                "dataset", "verified_mini"
            )
        ),
        "result_file": str(result_file) if result_file else None,
        "predictions_file": str(predictions_file) if predictions_file else None,
    }
    if isinstance(payload, Mapping):
        result.update(_normalize_eval_payload(payload))
    elif predictions_file is not None:
        try:
            pred_payload = _read_json(predictions_file)
        except (OSError, json.JSONDecodeError):
            pred_payload = None
        submitted = _count_instances(pred_payload) if pred_payload is not None else 0
        result["submitted_instances"] = submitted or None
        result["completed_ratio"] = 1.0 if submitted else None
    elif mode in {"eval", "all"}:
        raise FileNotFoundError(
            f"AISBench SWE-bench eval result JSON was not found under {root}"
        )
    else:
        raise FileNotFoundError(
            f"AISBench SWE-bench predictions JSON was not found under {root}"
        )

    output_dir = _output_dir_for(root, result_file or predictions_file)
    logging.info("SWE-bench AISBench result loaded from %s", result.get("result_file"))
    return result, output_dir


def swebench_payload_fields(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        field: result[field]
        for field in _PAYLOAD_FIELDS
        if field in result and result[field] is not None
    }
