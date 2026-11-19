"""Shared task gate for VBench cases.

This module is the only metrics-independent gate used by the result
processor; the scheduler keeps its own ``_is_vbench_task`` (same
``benchmark_type`` semantics).

Two YAML shapes are recognised:

- New format (aligned with plain accuracy cases): ``metrics: acc`` plus the
  ``benchmark.vbench`` block (``benchmark.benchmark_type: vbench``).
- Legacy format: ``metrics: vbench`` plus the same ``benchmark.vbench`` block.

The gate never looks at ``metrics``: VBench is identified by
``benchmark.benchmark_type: vbench`` (case-insensitive) with a mapping-shaped
``benchmark.vbench`` block. Plain accuracy cases (gpqa/gedit), AISBench
SWE-bench and perf cases stay out because their ``benchmark_type`` differs.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

VBENCH_BENCHMARK_TYPE = "vbench"


def is_vbench_task(task_config: Optional[Mapping[str, Any]]) -> bool:
    """Return True only for explicit VBench YAML.

    Both formats require ``benchmark.benchmark_type: vbench`` plus a mapping
    ``benchmark.vbench`` config block. The ``metrics`` root field is ignored
    on purpose so the new ``metrics: acc`` authoring style and the legacy
    ``metrics: vbench`` style stay supported.
    """
    if not isinstance(task_config, Mapping):
        return False
    benchmark = task_config.get("benchmark") or {}
    if not isinstance(benchmark, Mapping):
        return False
    if str(benchmark.get("benchmark_type", "")).strip().lower() != VBENCH_BENCHMARK_TYPE:
        return False
    return isinstance(benchmark.get("vbench"), Mapping)