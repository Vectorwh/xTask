"""Isolated helpers for AISBench SWE-bench tasks.

This module is the only gate used by the scheduler and result processor.
Existing ``metrics: perf`` / ``metrics: acc`` AISBench YAML must not enter
the SWE path even if a leftover ``benchmark.swebench`` block is present.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional


SWEBENCH_METRIC = "swebench"
DEFAULT_SWEBENCH_TIMEOUT_S = 86400
DEFAULT_SWEBENCH_DATASET = "verified_mini"
DEFAULT_SWEBENCH_MODE = "all"
DEFAULT_SWEBENCH_SPLIT = "test"
DEFAULT_SWEBENCH_STEP_LIMIT = 200
DEFAULT_SWEBENCH_BATCH_SIZE = 1
DEFAULT_SWEBENCH_THRESHOLD = 0.10
ALLOWED_SWEBENCH_DATASETS = frozenset(
    {
        "verified_mini",
        "verified",
        "lite",
        "full",
        "multilingual",
        "multilingual_mini",
    }
)
ALLOWED_SWEBENCH_MODES = frozenset({"infer", "eval", "all"})
_AGENT_CONTAINER_TIMEOUT = re.compile(r"^[1-9][0-9]*[smh]$")


def _metrics_include_swebench(task_config: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(task_config, Mapping):
        return False
    metrics = task_config.get("metrics")
    if isinstance(metrics, str):
        return metrics.strip().lower() == SWEBENCH_METRIC
    if isinstance(metrics, (list, tuple)):
        return any(str(item).strip().lower() == SWEBENCH_METRIC for item in metrics)
    return False


def swebench_config(task_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(task_config, Mapping):
        return {}
    benchmark = task_config.get("benchmark") or {}
    if not isinstance(benchmark, Mapping):
        return {}
    config = benchmark.get("swebench") or {}
    return dict(config) if isinstance(config, Mapping) else {}


def is_aisbench_swebench_task(task_config: Optional[Mapping[str, Any]]) -> bool:
    """Return True only for explicit AISBench SWE-bench YAML.

    All three must hold: ``metrics: swebench``, ``benchmark_type: aisbench``,
    and a mapping ``benchmark.swebench``. Perf/acc/vbench cases stay out.
    """
    if not _metrics_include_swebench(task_config):
        return False
    if not isinstance(task_config, Mapping):
        return False
    benchmark = task_config.get("benchmark") or {}
    if not isinstance(benchmark, Mapping):
        return False
    if str(benchmark.get("benchmark_type", "")).lower() != "aisbench":
        return False
    return isinstance(benchmark.get("swebench"), Mapping)


def swebench_mode(task_config: Optional[Mapping[str, Any]]) -> str:
    mode = str(swebench_config(task_config).get("mode") or DEFAULT_SWEBENCH_MODE).lower()
    return mode if mode in ALLOWED_SWEBENCH_MODES else DEFAULT_SWEBENCH_MODE


def is_swebench_eval_only(task_config: Optional[Mapping[str, Any]]) -> bool:
    return is_aisbench_swebench_task(task_config) and swebench_mode(task_config) == "eval"


def swebench_timeout_s(task_config: Optional[Mapping[str, Any]]) -> int:
    raw = swebench_config(task_config).get("timeout", DEFAULT_SWEBENCH_TIMEOUT_S)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_SWEBENCH_TIMEOUT_S
    return value if value > 0 else DEFAULT_SWEBENCH_TIMEOUT_S


def swebench_runner(task_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    runner = swebench_config(task_config).get("runner") or {}
    return dict(runner) if isinstance(runner, Mapping) else {}


def swebench_agent_container_timeout(task_config: Optional[Mapping[str, Any]]) -> str:
    """Opt-in mini-swe-agent Docker lifetime (GNU sleep syntax)."""
    raw = swebench_runner(task_config).get("agent_container_timeout")
    if raw is None or raw == "":
        return ""
    if not isinstance(raw, str) or not _AGENT_CONTAINER_TIMEOUT.fullmatch(raw):
        raise ValueError(
            "benchmark.swebench.runner.agent_container_timeout must be "
            "a positive duration such as '4h'"
        )
    return raw


def swebench_direct_output_enabled(task_config: Optional[Mapping[str, Any]]) -> bool:
    """Use the case root for new experiments, without changing legacy reuse.

    AISBench's ``--reuse`` searches its work directory for earlier experiments.
    Keep its existing ``outputs/default`` location in that mode so historical
    predictions stay available. Also honor an explicitly supplied work dir.
    """
    config = swebench_config(task_config)
    if bool(config.get("reuse", True)):
        return False
    extra_args = config.get("extra_args") or ()
    if not isinstance(extra_args, (list, tuple)):
        return True  # The script generator validates extra_args separately.
    return not any(
        str(arg) in {"-w", "--work-dir"}
        or str(arg).startswith(("-w=", "--work-dir="))
        for arg in extra_args
    )


# Keep old setup fields readable, but author new x86 setup options directly in
# benchmark.swebench.runner alongside host/work_dir/install_dir.
_RUNNER_SETUP_FIELDS = frozenset({
    "python", "install_dir", "timeout", "skip", "source_root",
    "aisbench_source", "mini_swe_agent_source", "swebench_harness_source",
    "mini_swe_agent_git", "swebench_harness_git", "runtime_container",
})


def swebench_setup(task_config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    config = swebench_config(task_config)
    old = config.get("setup") or {}
    if not isinstance(old, Mapping):
        raise ValueError("benchmark.swebench.setup must be a mapping")
    merged = dict(old)
    runner = swebench_runner(task_config)
    merged.update({key: runner[key] for key in _RUNNER_SETUP_FIELDS if key in runner})
    return merged


def swebench_dataset_name(task_config: Optional[Mapping[str, Any]]) -> str:
    config = swebench_config(task_config)
    name = str(
        config.get("dataset")
        or config.get("dataset_name")
        or DEFAULT_SWEBENCH_DATASET
    ).strip()
    return name or DEFAULT_SWEBENCH_DATASET


def validate_aisbench_swebench_task(task_config: Mapping[str, Any]) -> None:
    if not is_aisbench_swebench_task(task_config):
        raise ValueError(
            "AISBench SWE-bench requires metrics: swebench, "
            "benchmark.benchmark_type: aisbench, and benchmark.swebench"
        )
    config = swebench_config(task_config)
    mode = str(config.get("mode") or DEFAULT_SWEBENCH_MODE).lower()
    if mode not in ALLOWED_SWEBENCH_MODES:
        raise ValueError(
            f"benchmark.swebench.mode must be one of {sorted(ALLOWED_SWEBENCH_MODES)}: {mode!r}"
        )
    dataset = swebench_dataset_name(task_config)
    if dataset not in ALLOWED_SWEBENCH_DATASETS:
        raise ValueError(
            f"benchmark.swebench.dataset must be one of "
            f"{sorted(ALLOWED_SWEBENCH_DATASETS)}: {dataset!r}"
        )
    runner = swebench_runner(task_config)
    host = str(runner.get("host") or "").strip()
    if not host:
        raise ValueError(
            "benchmark.swebench.runner.host is required; AISBench infer/eval "
            "must run on a dedicated x86 host, not the NPU serving container"
        )
    if mode in {"infer", "all"}:
        model = str(
            config.get("model") or task_config.get("served_model_name") or ""
        ).strip()
        if not model:
            raise ValueError(
                "AISBench SWE-bench infer requires served_model_name or "
                "benchmark.swebench.model"
            )
