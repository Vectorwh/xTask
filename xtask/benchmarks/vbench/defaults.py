"""Centralized VBench default paths shared by every vbench case.

Case YAMLs may omit the ``benchmark.vbench`` path fields; the defaults below
define one stable layout:

- One canonical asset tree, provisioned once per host and read-only for all
  cases (both WAN I2V and JoyAI text-video consume it).  The
  ``/export/home/models/`` mount exists in every runner container, so host
  and container address these paths identically:
  ``<DEPENDENCE_PACKAGE_PATH>/mmm_bench/Video/vbench2_beta_i2v``
  └─ full_info.json + data/          (I2V prompts & reference images)
  └─ clip/ + pretrained_models/     (offline scoring-model weights)
  The directory names are stable interfaces — never rename them (a rename
  broke every vbench case once when a snapshot directory was moved).
- The per-run scoring-model cache (``VBENCH_CACHE_DIR``) is a symlink view
  over the canonical tree; ``ensure_cache_layout()`` rebuilds it on demand
  so no host needs a hand-maintained link farm.
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
    """Canonical asset tree shared by every VBench case on this host.

    ``Video/vbench2_beta_i2v`` is the single real copy of the I2V dataset
    (full_info + reference images) and the scoring-model weights (clip/ +
    pretrained_models/).  Both WAN I2V and JoyAI video cases consume this
    one tree; no per-case or per-workspace copies are maintained.  The
    directory name is a stable interface — do not rename.
    """
    return _dependence_package_path() + "/mmm_bench/Video/vbench2_beta_i2v"


def get_default_full_info() -> str:
    """Full-info manifest inside the shared asset tree."""
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
    """Scoring-model cache root, resolved without hand-maintained symlinks.

    The canonical asset tree (``Video/vbench2_beta_i2v``) carries the
    weights under ``pretrained_models/`` plus ``clip/``.  Rather than
    pointing at any hand-built link view, this returns a view directory
    rooted inside the canonical tree; the evaluation drivers call
    ``ensure_cache_layout()`` to materialize it as a symlink view on first
    use (idempotent), so the only thing that must exist on the host is the
    canonical tree itself.
    """
    return _dependence_package_path() + "/mmm_bench/Video/vbench2_beta_i2v/vbench_cache"


def default_evaluation_script() -> str:
    """Framework-owned evaluation script, versioned with the source tree."""
    return str(_module_dir() / "evaluate_vbench_i2v_bench.py")


def ensure_cache_layout(cache_dir: str, data_home: str) -> None:
    """Idempotently materialize the VBench scoring-model cache layout.

    Both the WAN I2V evaluation driver and the JoyAI text-video evaluation
    (via the shared ``vbench`` package) resolve their scoring models through
    ``VBENCH_CACHE_DIR`` with a flat layout (``clip_model/``, ``dino_model/``,
    ...).  The canonical asset tree stores them under ``pretrained_models/``
    and ``clip/`` instead, so the cache is a symlink view built on demand.
    Only missing entries are created (resume-safe); existing entries —
    including any a case yaml deliberately pinned — are left untouched.
    """
    if not data_home or not os.path.isdir(os.path.join(data_home, "pretrained_models")):
        return

    cache = cache_dir

    def _link(name, target):
        dst = os.path.join(cache, name)
        if os.path.lexists(dst):
            return
        if not os.path.exists(target):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(target, dst)

    pm = os.path.join(data_home, "pretrained_models")
    # ais_bench vbench-side model directories
    for model in ("aesthetic_model", "amt_model", "pyiqa_model", "raft_model"):
        _link(model, os.path.join(pm, model))
    # CLIP
    for weight in ("ViT-B-32.pt", "ViT-L-14.pt"):
        _link(os.path.join("clip_model", weight), os.path.join(data_home, "clip", weight))
    # DINO (vbench side reads from dino_model/ directly)
    _link(
        os.path.join("dino_model", "dino_vitbase16_pretrain.pth"),
        os.path.join(pm, "torch", "hub", "checkpoints", "dino_vitbase16_pretrain.pth"),
    )
    _link(
        os.path.join("dino_model", "facebookresearch_dino_main"),
        os.path.join(pm, "torch", "hub", "facebookresearch_dino_main"),
    )
    # dreamsim (i2v dimensions)
    for weight in (
        "clip_vitb16_pretrain.pth.tar",
        "dino_vitb16_pretrain.pth",
        "open_clip_vitb16_pretrain.pth.tar",
    ):
        _link(os.path.join("dreamsim_model", weight), os.path.join(pm, "dreamsim_model", weight))
    _link(os.path.join("dreamsim_model", "ensemble_lora"), os.path.join(pm, "dreamsim_model", "ensemble_lora"))
    # torch.hub (camera_motion's co-tracker; TORCH_HOME points here)
    _link(
        os.path.join("torch", "hub", "facebookresearch_co-tracker_main"),
        os.path.join(pm, "torch", "hub", "facebookresearch_co-tracker_main"),
    )
    _link(
        os.path.join("torch", "hub", "facebookresearch_dino_main"),
        os.path.join(pm, "torch", "hub", "facebookresearch_dino_main"),
    )
    # torch.hub checkpoints: the co-tracker hubconf calls
    # load_state_dict_from_url(pretrained=True); without a cached copy it
    # would try a network download and hang offline environments.
    _link(os.path.join("torch", "hub", "checkpoints"), os.path.join(pm, "torch", "hub", "checkpoints"))
    trusted = os.path.join(cache, "torch", "hub", "trusted_list")
    if not os.path.exists(trusted):
        os.makedirs(os.path.dirname(trusted), exist_ok=True)
        with open(trusted, "w", encoding="utf-8") as handle:
            handle.write("facebookresearch_co-tracker\nfacebookresearch_dino\n")


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
