"""Parse VBench artifacts into xTask's result dictionary."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_DATE_DIR = re.compile(r"^\d{8}_\d{6}$")


def _new_date_dir(output_path: Path, init_entries: Iterable[str]) -> Optional[Path]:
    initial = set(init_entries or [])
    try:
        candidates = [
            item
            for item in output_path.iterdir()
            if item.is_dir() and _DATE_DIR.fullmatch(item.name) and item.name not in initial
        ]
    except OSError:
        return None
    return max(candidates, key=lambda item: item.name) if candidates else None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_values(value: Any, names: Iterable[str]) -> List[Any]:
    wanted = {_normal_key(name) for name in names}
    found: List[Any] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if _normal_key(key) in wanted:
                found.append(nested)
            found.extend(_find_values(nested, names))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_find_values(nested, names))
    return found


def _first_number(value: Any, names: Iterable[str]) -> Optional[float]:
    """Return a numeric score, preferring explicit fields at the current level.

    VBench score files may contain both a final ``total_score`` and unrelated
    nested ``total`` counters.  Exact fields therefore take precedence over
    the recursive compatibility lookup used for older result layouts.
    """
    names = tuple(names)
    if isinstance(value, dict):
        normalized = {_normal_key(key): nested for key, nested in value.items()}
        for name in names:
            candidate = normalized.get(_normal_key(name))
            number = _as_number(candidate)
            if number is not None:
                return number
    for candidate in _find_values(value, names):
        number = _as_number(candidate)
        if number is not None:
            return number
    return None


def _as_number(candidate: Any) -> Optional[float]:
    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        return float(candidate)
    try:
        if isinstance(candidate, str) and candidate.strip():
            return float(candidate)
    except ValueError:
        pass
    return None


def _find_score_file(date_dir: Path, suffix: str) -> Optional[Path]:
    matches = sorted(date_dir.rglob(f"*{suffix}"))
    return matches[0] if matches else None


def _load_generation_metrics(date_dir: Path) -> Dict[str, Any]:
    path = date_dir / "generation_metrics.json"
    if not path.is_file():
        matches = sorted(date_dir.rglob("generation_metrics.json"))
        path = matches[0] if matches else path
    if not path.is_file():
        return {}
    try:
        value = _read_json(path)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        logging.warning("Unable to parse VBench generation metrics: %s", path, exc_info=True)
        return {}


def vbench_result_post_process(
    task_config: Dict[str, Any],
    output_path: str,
    init_entries: Iterable[str],
) -> Tuple[Dict[str, Any], str]:
    """Return VBench quality scores and the newly created case date directory."""

    from xtask.benchmarks.vbench.text_video.config import is_text_video_task

    if is_text_video_task(task_config):
        from xtask.benchmarks.vbench.text_video.reporting import post_process

        return post_process(task_config, output_path, init_entries)
    root = Path(output_path)
    date_dir = _new_date_dir(root, init_entries)
    if date_dir is None:
        raise FileNotFoundError(
            f"VBench did not create a YYYYMMDD_HHMMSS result directory under {root}"
        )

    generation = _load_generation_metrics(date_dir)
    final_score_file = _find_score_file(date_dir, "_final_score.json")
    eval_result_file = _find_score_file(date_dir, "_eval_results.json")
    final_score: Dict[str, Any] = {}
    eval_result: Any = None
    if final_score_file is not None:
        try:
            value = _read_json(final_score_file)
            final_score = value if isinstance(value, dict) else {"value": value}
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to parse VBench final score: {final_score_file}") from exc
    if eval_result_file is not None:
        try:
            eval_result = _read_json(eval_result_file)
        except (OSError, json.JSONDecodeError):
            logging.warning("Unable to parse VBench evaluation results: %s", eval_result_file, exc_info=True)

    result: Dict[str, Any] = {
        "generated": generation.get("generated"),
        "resumed": generation.get("resumed"),
        "total": generation.get("total"),
        "failed": generation.get("failed", 0),
        "failures": generation.get("failures", []),
        "generation": generation,
        "raw_scores": final_score.get("raw", final_score.get("raw_scores", {})),
        "normalized_weighted": final_score.get(
            "normalized_weighted", final_score.get("normalized", {})
        ),
        "quality_score": _first_number(final_score, ("quality_score", "quality")),
        "i2v_score": _first_number(final_score, ("i2v_score", "i2v")),
        "total_score": _first_number(final_score, ("total_score", "total")),
        "temporal_flickering": _first_number(
            final_score, ("temporal_flickering", "temporal_flicker")
        ),
        "final_score_file": str(final_score_file) if final_score_file else "",
        "eval_results_file": str(eval_result_file) if eval_result_file else "",
        "eval_results": eval_result,
    }
    # A generation-only run is valid without a final score.  A quality run
    # must produce a score file so a missing/partial evaluation cannot pass.
    mode = str(((task_config.get("benchmark") or {}).get("vbench") or {}).get("mode", "generate_and_evaluate")).lower()
    if mode in {"evaluate", "generate_and_evaluate"} and final_score_file is None:
        raise FileNotFoundError(f"VBench final score JSON was not found under {date_dir}")
    return result, str(date_dir)
