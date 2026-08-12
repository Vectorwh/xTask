"""Shared helpers for resolving case YAML paths and model directory names."""

from __future__ import annotations

import re
from pathlib import Path

_SEARCH_CASE_SUFFIXES = (
    re.compile(r"_sg\d+$", re.IGNORECASE),
    re.compile(r"_bs-[^/]+$", re.IGNORECASE),
)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _base_case_names(case_name: str):
    names = [case_name]
    current = case_name
    for pattern in _SEARCH_CASE_SUFFIXES:
        stripped = pattern.sub("", current)
        if stripped != current and stripped not in names:
            names.append(stripped)
            current = stripped
    return names


def resolve_source_yaml(task_path, task_config):
    """Locate the source case YAML used to derive model_name."""
    task_config = task_config or {}
    metadata = task_config.get("_xtask") or {}
    configured_source = metadata.get("source_case_file")
    if configured_source:
        source = Path(str(configured_source)).resolve()
        if source.is_file():
            return source

    source = Path(task_path).resolve() if task_path else None
    tests_root = (Path.cwd() / "tests").resolve()
    if source and source.is_file() and _is_under(source, tests_root):
        return source

    case_name = str(task_config.get("case_name") or "")
    if case_name:
        try:
            from xtask.tasks.task_parser import find_case_file
        except Exception:
            find_case_file = None
        if find_case_file is not None:
            for candidate_name in _base_case_names(case_name):
                try:
                    candidate = find_case_file(candidate_name, str(tests_root))
                except Exception:
                    candidate = None
                if candidate:
                    return Path(candidate).resolve()
    return source if source and source.is_file() else None


def resolve_model_name(task_path=None, task_config=None):
    """Resolve model name the same way write_result / Excel summary does."""
    task_config = task_config or {}
    source_yaml = resolve_source_yaml(task_path, task_config)
    if source_yaml is not None:
        return str(source_yaml.parent.name or "unknown")
    if task_path:
        return Path(task_path).parent.name or "unknown"
    return "unknown"
