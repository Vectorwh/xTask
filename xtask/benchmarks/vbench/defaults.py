"""Centralized VBench default paths shared by every vbench case.

Case YAMLs may omit the ``benchmark.vbench`` path fields; the defaults below
define one stable layout:

- Shared dataset inputs, prepared once and read-only for all cases.  The
  ``/export/home/models/`` mount exists in every runner container, so host
  and container address these paths identically:
  ``<DEPENDENCE_PACKAGE_PATH>/vbench_data/full_info.json``
  ``<DEPENDENCE_PACKAGE_PATH>/vbench_data/data/``
- Project-local cache, inside the xTask project so the workspace mount
  exposes it as ``/workspace/outputs/vbench_data/cache`` in the runner:
  ``<project>/outputs/vbench_data/cache``
- The evaluation script is framework source code, versioned alongside the
  generation driver.  It lives next to ``run_i2v.py`` and reaches the
  runner through the same workspace mount:
  ``<project>/xtask/benchmarks/vbench/evaluate_vbench_i2v_bench.py``

Explicit ``benchmark.vbench`` values always override these defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from xtask.utils.config_parser import get_xtask_config_data

# Fallback when DEPENDENCE_PACKAGE_PATH is not configured.
_FALLBACK_RESOURCE_ROOT = "/export/home/models/xllm_test_060"


def _dependence_package_path() -> str:
    """Return the configured shared resource root (host path)."""
    return (get_xtask_config_data().get("ENV_CONFIG") or {}).get(
        "DEPENDENCE_PACKAGE_PATH", _FALLBACK_RESOURCE_ROOT
    ).strip() or _FALLBACK_RESOURCE_ROOT


def get_vbench_data_home() -> str:
    """Read-only dataset home shared by every VBench case on this host."""
    return _dependence_package_path() + "/vbench_data"


def get_default_full_info() -> str:
    """Full-info manifest inside the shared dataset home."""
    return get_vbench_data_home() + "/full_info.json"


def get_default_image_root() -> str:
    """Reference images referenced by the full-info manifest."""
    return get_vbench_data_home() + "/data"


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _project_root() -> Path:
    # .../<project>/xtask/benchmarks/vbench/defaults.py -> <project>
    return Path(__file__).resolve().parents[3]


def default_cache_dir() -> str:
    """Shared VBench cache directory inside the xTask project."""
    return str(_project_root() / "outputs" / "vbench_data" / "cache")


def default_evaluation_script() -> str:
    """Framework-owned evaluation script, versioned with the source tree."""
    return str(_module_dir() / "evaluate_vbench_i2v_bench.py")


# ---------------------------------------------------------------------------
# Evaluation-time environment defaults
# ---------------------------------------------------------------------------
# ``evaluate_vbench_i2v_bench.py`` resolves two code/data roots through env
# vars (falling back to weinan5's machine).  Centralizing them here lets the
# runner container work without per-case YAML configuration:
#   VBENCH2_DATASETS -> directory containing the ``vbench2_beta_i2v`` package
#                       (i2v dimensions + dataset images + full_info)
#   AIS_THIRD_PARTY  -> directory containing the ``vbench`` package
#                       (ais_bench NPU-adapted quality dimensions)
def get_default_vbench2_datasets() -> str:
    return _dependence_package_path() + "/benchmark/vbench2"


def get_default_ais_third_party() -> str:
    return _dependence_package_path() + "/benchmark/vbench2"
