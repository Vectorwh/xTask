# Copyright 2026 The xTask Authors.
"""Basic T2V quality dimensions via official VBench compute functions."""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path

from .artifacts import new_date_dir, redact, sha256, write_json
from .config import normalize_config, result_issues


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if hasattr(value, "item"):
        return sanitize(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("VBench per-video result contains NaN/Inf")
    return value


def evaluate(directory: Path, records, settings):
    # This function runs only in the new isolated evaluation process. A CPU
    # scorer must not autoload NPU extensions from the generator's environment.
    if settings["device"] == "cpu" or str(settings["device"]).startswith("cuda"):
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    code_root = settings.get("code_root")
    if code_root:
        sys.path.insert(0, str(code_root))
    from vbench.utils import CACHE_DIR, init_submodules
    dimensions = settings["dimensions"]
    scorer_provenance = {"torch_version": getattr(torch, "__version__", "unknown"),
                         "device": settings["device"], "compute_sources": {}, "model_assets": {}}
    # Fail before official init_submodules can invoke wget/git. Evaluation
    # dependencies and weights must be prepared explicitly, never mutate them
    # or access networks during performance/regression workflows.
    if settings["local"]:
        cache = Path(CACHE_DIR)
        required = {
            "subject_consistency": ["dino_model/facebookresearch_dino_main", "dino_model/dino_vitbase16_pretrain.pth"],
            "background_consistency": ["clip_model/ViT-B-32.pt"],
            "motion_smoothness": ["amt_model/amt-s.pth"],
            "dynamic_degree": ["raft_model/models/raft-things.pth"],
            "aesthetic_quality": ["clip_model/ViT-L-14.pt", "aesthetic_model/emb_reader"],
            "imaging_quality": ["pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"],
        }
        missing = [str(cache / item) for dim in dimensions for item in required.get(dim, []) if not (cache / item).exists()]
        if missing:
            raise FileNotFoundError("Missing offline VBench scoring assets: " + ", ".join(missing))
        for dim in dimensions:
            for item in required.get(dim, []):
                asset = cache / item
                if asset.is_file():
                    scorer_provenance["model_assets"][item] = sha256(asset)
                elif asset.is_dir():
                    # Code/model directories retain a bounded file-hash manifest.
                    scorer_provenance["model_assets"][item] = {
                        str(path.relative_to(asset)): sha256(path)
                        for path in sorted(asset.rglob("*")) if path.is_file()
                        and path.suffix in {".pth", ".pt", ".py"}
                    }
    manifest = directory / "vbench_full_info.json"
    write_json(manifest, [{"prompt_en": row["prompt"], "dimension": dimensions,
                           "video_list": [str((directory / row["video_path"]).resolve())]} for row in records])
    from .decoder import prepare_decoder

    decoder = prepare_decoder(directory / records[0]["video_path"])
    modules = init_submodules(dimensions, local=settings["local"], read_frame=False)
    scores = {}
    for dimension in dimensions:
        module = importlib.import_module("vbench." + dimension)
        compute = getattr(module, "compute_" + dimension)
        module_source = getattr(module, "__file__", None)
        scorer_provenance["compute_sources"][dimension] = sha256(Path(module_source)) if module_source else "unknown"
        value, details = compute(str(manifest), torch.device(settings["device"]), modules[dimension])
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"VBench {dimension} returned a non-finite score")
        # Keep per-video evidence, not only one aggregate. Empty details cannot pass.
        if not isinstance(details, list) or not details:
            raise ValueError(f"VBench {dimension} returned no per-video evidence")
        expected = {str((directory / row["video_path"]).resolve()) for row in records}
        observed = [str(Path(row.get("video_path", "")).resolve()) for row in details if isinstance(row, dict)]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError(f"VBench {dimension} per-video coverage missing or duplicated")
        write_json(directory / (dimension + "_details.json"), sanitize(details))
        scores[dimension] = score
        write_json(directory / "video_quality.json", {"profile": "vbench_basic_t2v", "scores": scores,
                   "standard_full_benchmark": False, "total_score": None, "video_decoder": decoder,
                   "scorer_provenance": scorer_provenance,
                   "note": "Per-dimension official functions on explicit custom dataset; not VBench-I2V/full-16 total"})
    return scores


def evaluate_archive(config, output_root):
    config = normalize_config(config)
    source = Path(config["source_run"]).resolve()
    directory = new_date_dir(output_root)
    write_json(directory / "config.json", redact(config))
    payload = {"schema_version": 1, "adapter": "text_video", "task": config["task"], "scenario": config["scenario"], "complete": False, "expected_requests": 0,
               "completed_requests": 0, "failed": 0, "video_quality": {}, "state": "running"}
    try:
        previous = json.loads((source / "text_video_result.json").read_text())
        if previous.get("complete") is not True or previous.get("failed") != 0:
            raise ValueError("Cannot evaluate an incomplete generation archive")
        if previous.get("task") != config["task"] or previous.get("scenario") != config["scenario"]:
            raise ValueError("Source archive task/scenario mismatch")
        records = json.loads((source / "requests.json").read_text())
        if len(records) != previous.get("expected_requests") or any(not row.get("success") for row in records):
            raise ValueError("Source requests incomplete")
        from .artifacts import sha256
        from .media import validate_media
        for row in records:
            path = (source / row["video_path"]).resolve()
            if source not in path.parents or sha256(path) != row["video_sha256"]:
                raise ValueError("Source video escaped archive or hash changed")
            validate_media(path, row["parameters"], config["media"])
            row["video_path"] = str(path)
        payload.update(previous, state="running", video_quality={})
        payload["manifest_sha256"] = sha256(source / "manifest.json")
        payload["offline_evaluation"] = True
        payload["performance"] = {}  # Offline scoring is not a fresh throughput measurement.
        payload["performance_by_arm"] = {}
        payload["source_run"] = os.path.relpath(source, directory)
        payload.pop("issues", None)
        for name in ("negative_control", "embedding_parity", "processing_error", "execution_error"):
            payload.pop(name, None)
        semantic = config["checks"].get("negative_control", {})
        if semantic.get("backend") == "benchmark_semantics":
            from .semantic_models import preflight_checkpoints, preflight_environment

            preflight_checkpoints(semantic)
            preflight_environment(semantic, directory)
        if config["evaluation"]["enabled"]:
            payload["video_quality"] = evaluate(directory, [r for r in records if r["arm"] != "reference"], config["evaluation"])
        if config["checks"]:
            from .checks import run_checks
            payload.update(run_checks(directory, records, config))
        for row in records:
            row["video_path"] = os.path.relpath(row["video_path"], directory)
        write_json(directory / "requests.json", records)
        payload["complete"] = True
    except Exception as exc:
        payload.update(complete=False, processing_error=str(exc))
    payload["state"] = "completed"
    payload["issues"] = result_issues(payload, config)
    write_json(directory / "text_video_result.json", payload)
    return payload, directory
