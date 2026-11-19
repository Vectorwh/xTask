# Copyright 2026 The xTask Authors.
"""Validate the opt-in video adapter without changing legacy VBench defaults."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, Dict, List

BASIC_DIMENSIONS = (
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
)
VIDEO_DIMENSIONS = BASIC_DIMENSIONS + ("temporal_flickering",)
SCENARIOS = {"text", "cfg_pair", "embedding", "embedding_pair"}
PERFORMANCE_METRICS = {
    "performance.e2el_avg_ms", "performance.e2el_p90_ms", "performance.e2el_p99_ms",
    "performance.request_throughput", "performance.video_throughput",
}


def is_text_video_task(task_config: Mapping[str, Any]) -> bool:
    if not isinstance(task_config, Mapping):
        return False
    benchmark = task_config.get("benchmark") or {}
    if not isinstance(benchmark, Mapping):
        return False
    block = benchmark.get("vbench") or {}
    return (str(benchmark.get("benchmark_type", "")).lower() == "vbench"
            and isinstance(block, Mapping) and isinstance(block.get("task"), str)
            and block.get("task") in {"t2v", "t2av"})


def _number(value, name, minimum=0, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'>' if positive else '>='} {minimum}")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark.vbench must be a mapping")
    config = copy.deepcopy(dict(raw))
    allowed_fields = {"task", "scenario", "manifest", "source_manifest", "source_run", "mode", "name", "model",
                      "endpoint", "reference_endpoint", "api_key", "metrics", "generation", "evaluation", "media",
                      "checks", "gates", "embedding_dim", "task_snapshot", "image_home", "image_name",
                      "container_home", "container_image", "cache_dir", "vbench2_datasets", "ais_third_party",
                      "vbench_data_home", "provenance"}
    if set(config) - allowed_fields:
        raise ValueError(f"Unknown text-video configuration fields: {sorted(set(config) - allowed_fields)}")
    if config.get("metrics", "acc") not in {"acc", "perf"}:
        raise ValueError("Text-video quality and performance use separate metrics: acc or perf")
    if config.get("task") not in {"t2v", "t2av"}:
        raise ValueError("Text video adapter requires task: t2v or t2av")
    config.setdefault("mode", "generate_and_evaluate")
    if config["mode"] not in {"generate", "evaluate", "generate_and_evaluate"}:
        raise ValueError("Unsupported text video mode")
    config.setdefault("scenario", "text")
    if config["scenario"] not in SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}")
    if not isinstance(config.get("manifest"), str) or not config["manifest"]:
        raise ValueError("Text video adapter requires an explicit manifest; no I2V default dataset")
    if config["mode"] == "evaluate" and not config.get("source_run"):
        raise ValueError("evaluate mode requires source_run (an archived text-video execution)")
    config.setdefault("name", "text_video")
    generation = dict(config.get("generation") or {})
    allowed = {"size", "num_frames", "fps", "num_inference_steps", "guidance_scale",
               "flow_shift", "seed", "sampling_rate", "num_videos_per_prompt",
               "request_timeout", "limit", "concurrency", "repeats", "warmups"}
    unknown = set(generation) - allowed
    if unknown:
        raise ValueError(f"Unsupported text-video generation fields: {sorted(unknown)}")
    for field in ("size", "num_frames", "fps", "num_inference_steps", "seed", "guidance_scale"):
        if field not in generation:
            raise ValueError(f"generation.{field} must be explicit")
    try:
        width, height = [int(x) for x in str(generation["size"]).split("*")]
    except (TypeError, ValueError) as exc:
        raise ValueError("generation.size must be width*height") from exc
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("JoyOV2 canvas must be positive multiples of 32; no silent rounding")
    frames = _integer(generation["num_frames"], "generation.num_frames")
    if frames % 4 != 1:
        raise ValueError("JoyOV2 generation.num_frames must be 4n+1")
    _number(generation["fps"], "generation.fps", positive=True)
    _number(generation["guidance_scale"], "generation.guidance_scale", positive=True)
    _integer(generation["num_inference_steps"], "generation.num_inference_steps")
    _integer(generation["seed"], "generation.seed", 0)
    for name, default in (("concurrency", 1), ("repeats", 1), ("limit", 0), ("warmups", 0)):
        _integer(generation.setdefault(name, default), "generation." + name, 0 if name in {"limit", "warmups"} else 1)
    if generation.setdefault("num_videos_per_prompt", 1) != 1:
        raise ValueError("JoyOV2 currently supports one video per request")
    _number(generation.setdefault("request_timeout", 1800), "generation.request_timeout", positive=True)
    if "flow_shift" in generation:
        _number(generation["flow_shift"], "generation.flow_shift", positive=True)
    if config["task"] == "t2av":
        _integer(generation.get("sampling_rate"), "T2AV generation.sampling_rate (must match audio VAE)")
    elif "sampling_rate" in generation:
        _integer(generation["sampling_rate"], "generation.sampling_rate")
    if config["scenario"] == "cfg_pair" and generation["guidance_scale"] <= 1:
        raise ValueError("cfg_pair requires guidance_scale > 1; keep it identical in both arms")
    config["generation"] = generation
    if config["scenario"] in {"cfg_pair", "embedding_pair"} and generation["concurrency"] != 1:
        raise ValueError("Paired diagnostics require concurrency=1 for comparable conditions; use separate perf cases")
    media = dict(config.get("media") or {})
    if set(media) - {"audio", "duration_tolerance_s", "fps_tolerance", "channels", "decode_timeout", "ffmpeg", "ffprobe"}:
        raise ValueError("Unknown media validation fields")
    media.setdefault("audio", "required" if config["task"] == "t2av" else "forbidden")
    if config.get("metrics") == "perf" and config["task"] == "t2v" and media["audio"] != "forbidden":
        raise ValueError("Pure T2V performance requires no audio; measure audio-enabled service as T2AV")
    if media["audio"] not in {"required", "forbidden", "optional"} or (config["task"] == "t2av" and media["audio"] != "required"):
        raise ValueError("T2AV requires audio; T2V may explicitly allow audio for visual-only evaluation")
    for name, default in (("duration_tolerance_s", .25), ("fps_tolerance", .01), ("decode_timeout", 120)):
        _number(media.setdefault(name, default), "media." + name, positive=name == "decode_timeout")
    if "channels" in media:
        _integer(media["channels"], "media.channels")
    config["media"] = media
    evaluation = dict(config.get("evaluation") or {})
    if set(evaluation) - {"enabled", "dimensions", "device", "local", "code_root", "baseline", "threshold"}:
        raise ValueError("Unknown T2V evaluation fields; VABench belongs to stage two")
    if evaluation.get("baseline") is not None or evaluation.get("threshold") is not None:
        raise ValueError("Text-video scores have per-dimension gates, not the legacy I2V total baseline")
    evaluation.setdefault("enabled", config["mode"] != "generate" and str(config.get("metrics", "acc")) != "perf")
    if config["mode"] == "generate" and evaluation["enabled"]:
        raise ValueError("mode: generate cannot silently execute quality evaluation; use generate_and_evaluate")
    if not isinstance(evaluation["enabled"], bool):
        raise ValueError("evaluation.enabled must be boolean")
    dimensions = evaluation.setdefault("dimensions", list(BASIC_DIMENSIONS))
    if (not isinstance(dimensions, list) or not dimensions or len(set(dimensions)) != len(dimensions)
            or any(dim not in VIDEO_DIMENSIONS for dim in dimensions)):
        raise ValueError("Stage one custom T2V supports only basic visual dimensions (plus temporal_flickering); no fake full-16 score")
    evaluation.setdefault("device", "cpu")
    if evaluation.setdefault("local", True) is not True:
        raise ValueError("Stage-one evaluation requires local: true with prepared scoring assets; no online weight downloads")
    config["evaluation"] = evaluation
    if config["mode"] == "evaluate" and not evaluation["enabled"] and not config.get("checks"):
        raise ValueError("evaluate mode needs an enabled scorer/check; cannot claim evaluation with no evaluation workload")
    checks = config.setdefault("checks", {})
    if not isinstance(checks, Mapping) or set(checks) - {"negative_control", "embedding_parity"}:
        raise ValueError("checks only supports negative_control and embedding_parity")
    for name, specification in checks.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"checks.{name} must be a mapping")
        if specification.get("backend") == "benchmark_semantics" and name == "negative_control":
            from .negative import normalize_spec

            checks[name] = normalize_spec(specification)
            if "audio" in checks[name] and config["task"] != "t2av":
                raise ValueError("Audio negative-control assessment requires task: t2av")
        elif specification.get("backend") == "decoded_media" and name == "embedding_parity":
            if set(specification) - {"backend", "video_size", "audio_sample_rate", "timeout"}:
                raise ValueError("Unknown decoded-media parity fields")
            _integer(specification.get("video_size", 64), "embedding_parity.video_size")
            _integer(specification.get("audio_sample_rate", 16000), "embedding_parity.audio_sample_rate")
        else:
            if set(specification) - {"backend", "command", "timeout"} or specification.get("backend", "command") != "command":
                raise ValueError(f"Unknown checks.{name} evaluator fields")
            command = specification.get("command")
            if (not isinstance(command, list) or not command
                    or any(not isinstance(arg, str) or not arg for arg in command)):
                raise ValueError(f"checks.{name}.command must be a nonempty argv list; no implicit judge")
        _number(specification.get("timeout", 600), f"checks.{name}.timeout", positive=True)
        expected_scenario = "cfg_pair" if name == "negative_control" else "embedding_pair"
        if config["scenario"] != expected_scenario:
            raise ValueError(f"checks.{name} requires scenario: {expected_scenario}")
    required_check = {"cfg_pair": "negative_control", "embedding_pair": "embedding_parity"}.get(config["scenario"])
    if config["mode"] != "generate" and required_check and required_check not in checks:
        raise ValueError(f"Quality/evaluation scenario requires checks.{required_check}; use mode: generate for request-only diagnostics")
    if config.get("metrics") == "perf":
        if config["mode"] != "generate" or evaluation["enabled"] or checks:
            raise ValueError("Performance cases must generate fresh outputs without quality/judge workloads")
    gates = config.setdefault("gates", [])
    if not isinstance(gates, list):
        raise ValueError("gates must be a list")
    for gate in gates:
        if not isinstance(gate, Mapping) or gate.get("compare") not in {"gte", "lte"}:
            raise ValueError("Text-video gates require compare: gte or lte")
        metric = gate.get("metric", "")
        valid = metric in PERFORMANCE_METRICS
        valid |= metric.startswith("video_quality.") and metric.split(".", 1)[1] in dimensions and evaluation["enabled"]
        valid |= any(metric.startswith(name + ".") for name in checks)
        if metric.startswith("negative_control.") and checks.get("negative_control", {}).get("backend") == "benchmark_semantics":
            from .negative import METRICS

            if metric.split(".", 1)[1] not in METRICS:
                raise ValueError(f"Unknown benchmark_semantics metric: {metric}")
        if metric.startswith("performance.") and config["scenario"] in {"cfg_pair", "embedding_pair"}:
            raise ValueError("Paired diagnostics must not gate mixed reference/target throughput; use independent performance cases")
        if not valid:
            raise ValueError(f"Unknown/unavailable gate metric: {metric}")
        _number(gate.get("baseline"), "gate.baseline")
        tolerance = _number(gate.get("threshold", 0), "gate.threshold")
        if tolerance > 1:
            raise ValueError("gate.threshold must be a relative tolerance <= 1")
    if config["mode"] == "evaluate" and any(gate["metric"].startswith("performance.") for gate in gates):
        raise ValueError("Offline evaluation cannot verify fresh generation throughput/latency")
    return config


def metric_value(result, metric):
    value = result
    for name in metric.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    return value


def result_issues(payload: Any, config: Dict[str, Any]) -> List[Dict[str, str]]:
    if not isinstance(payload, Mapping):
        return [{"category": "processing", "reason": "Missing text-video execution result"}]
    issues = []
    expected = payload.get("expected_requests")
    completed = payload.get("completed_requests")
    counts_valid = (isinstance(expected, int) and not isinstance(expected, bool) and expected > 0
                    and isinstance(completed, int) and not isinstance(completed, bool))
    if (payload.get("complete") is not True or payload.get("failed") != 0
            or not counts_valid
            or completed != expected):
        issues.append({"category": "execution", "reason": str(payload.get("execution_error") or "Failed/incomplete text-video requests or media")})
    if payload.get("processing_error"):
        issues.append({"category": "processing", "reason": str(payload["processing_error"])})
    if payload.get("task") not in (None, config["task"]) or payload.get("scenario") not in (None, config["scenario"]):
        issues.append({"category": "processing", "reason": "Result task/scenario differs from effective configuration"})
    if config["evaluation"]["enabled"]:
        for dim in config["evaluation"]["dimensions"]:
            value = (payload.get("video_quality") or {}).get(dim)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                issues.append({"category": "processing", "reason": f"Missing/non-finite video quality dimension: {dim}"})
    for name in config["checks"]:
        block = payload.get(name) or {}
        if not isinstance(block, Mapping) or block.get("complete") is not True:
            issues.append({"category": "processing", "reason": f"Missing/incomplete {name} check"})
            continue
        if name == "negative_control" and config["checks"][name].get("backend") == "benchmark_semantics":
            assessment = block.get("assessment")
            if (not isinstance(assessment, Mapping) or assessment.get("status") not in {"pass", "fail", "inconclusive"}
                    or assessment.get("policy_profile") != config["checks"][name]["profile"]
                    or assessment.get("provisional") is not True
                    or assessment.get("passed") is not (assessment.get("status") == "pass")):
                issues.append({"category": "processing", "reason": "Missing/inconsistent semantic negative-control assessment"})
            else:
                violations = assessment.get("violations")
                if not isinstance(violations, list) or any(not isinstance(v, Mapping) or v.get("category") not in {"baseline", "processing"} or not isinstance(v.get("reason"), str) for v in violations):
                    issues.append({"category": "processing", "reason": "Invalid negative-control policy violations"})
                else:
                    issues.extend(dict(v) for v in violations)
                    if assessment.get("status") != "pass" and not violations:
                        issues.append({"category": "processing", "reason": "Negative-control assessment did not pass"})
    for gate in config["gates"]:
        actual = metric_value(payload, gate["metric"])
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
            issues.append({"category": "processing", "reason": f"Missing/non-finite metric {gate['metric']}"})
            continue
        baseline, tolerance = gate["baseline"], gate.get("threshold", 0)
        bound = baseline * (1 + tolerance if gate["compare"] == "lte" else 1 - tolerance)
        if (actual > bound if gate["compare"] == "lte" else actual < bound):
            issues.append({"category": "baseline", "reason": f"{gate['metric']}={actual} violates {gate['compare']} {bound}"})
    return issues
