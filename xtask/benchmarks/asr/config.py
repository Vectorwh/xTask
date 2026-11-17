"""Validate ASR configuration and evaluate all per-run baseline gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List

METRICS = frozenset({
    "error_rate_pct", "error_count", "e2el_avg_ms", "e2el_p90_ms",
    "e2el_p99_ms", "first_text_avg_ms", "first_text_p99_ms",
    "legacy_ttft_avg_ms", "legacy_ttft_p99_ms", "request_throughput",
    "audio_throughput", "legacy_rtf", "failed_requests", "failure_rate",
})


def is_asr_task(task_config: Dict[str, Any]) -> bool:
    return str((task_config.get("benchmark") or {}).get("benchmark_type", "")).lower() == "asr"


def _number(value: Any, name: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark.asr must be a mapping")
    config = dict(raw)
    allowed_fields = {
        "backend", "api", "dataset_name", "dataset_archive", "infer_archive",
        "concurrency", "repeats", "limit", "expected_samples", "stream",
        "use_context", "simulate_realtime", "cache_condition", "scoring_profile",
        "generation", "timeout", "max_audio_seconds", "warmup", "gates",
        "server_addr", "api_key", "model", "provenance", "task_snapshot", "auto_assess",
    }
    unknown = set(config) - allowed_fields
    if unknown:
        raise ValueError(f"Unknown benchmark.asr fields: {sorted(unknown)}")
    if config.get("backend", "legacy_infer") != "legacy_infer":
        raise ValueError("Stage 1 supports only benchmark.asr.backend: legacy_infer")
    config["backend"] = "legacy_infer"
    config.setdefault("api", "audio")
    if config["api"] not in {"audio", "chat"}:
        raise ValueError("benchmark.asr.api must be audio or chat")
    config.setdefault("dataset_name", "ta3_test0925_quiet")
    if not isinstance(config["dataset_name"], str) or config["dataset_name"] in {"", ".", ".."} or any(
        symbol in config["dataset_name"] for symbol in ("/", "\\")
    ):
        raise ValueError("benchmark.asr.dataset_name must be a safe directory name")
    levels = config.get("concurrency", [1, 2, 5, 10])
    levels = [levels] if isinstance(levels, int) and not isinstance(levels, bool) else levels
    if not isinstance(levels, list) or not levels or any(
        isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in levels
    ) or len(set(levels)) != len(levels):
        raise ValueError("benchmark.asr.concurrency must contain distinct positive integers")
    config["concurrency"] = levels
    for name, default, minimum in (("repeats", 1, 1), ("limit", 0, 0), ("expected_samples", 1162, 1)):
        value = config.setdefault(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"benchmark.asr.{name} must be an integer >= {minimum}")
    for name, default in (("stream", True), ("use_context", True), ("simulate_realtime", False)):
        if not isinstance(config.setdefault(name, default), bool):
            raise ValueError(f"benchmark.asr.{name} must be boolean")
    config.setdefault("cache_condition", "service_lifetime")
    if config["cache_condition"] != "service_lifetime":
        raise ValueError("Legacy runner cannot enforce a cold cache; use separate xTask cases to restart services")
    if config.get("scoring_profile", "joyai_legacy") != "joyai_legacy":
        raise ValueError("Stage 1 preserves only the joyai_legacy scoring profile")
    config["scoring_profile"] = "joyai_legacy"
    if not isinstance(config.get("generation", {}), Mapping):
        raise ValueError("benchmark.asr.generation must be a mapping")
    generation = dict(config.get("generation") or {})
    allowed = {"temperature", "repetition_penalty", "language", "top_p", "top_k",
               "max_completion_tokens", "max_tokens", "hotwords", "chat_prompt", "max_context_chars"}
    if set(generation) - allowed:
        raise ValueError(f"Unsupported legacy generation fields: {sorted(set(generation) - allowed)}")
    generation.setdefault("temperature", 0.0)
    generation.setdefault("repetition_penalty", 1.05)
    generation.setdefault("language", "zh")
    generation.setdefault("max_tokens", 256)
    generation.setdefault("max_context_chars", 450)
    _number(generation["temperature"], "generation.temperature")
    if generation["temperature"] > 2:
        raise ValueError("generation.temperature must be <= 2")
    if _number(generation["repetition_penalty"], "generation.repetition_penalty") <= 0:
        raise ValueError("generation.repetition_penalty must be > 0")
    if generation.get("top_p") is not None and not 0 < _number(generation["top_p"], "generation.top_p") <= 1:
        raise ValueError("generation.top_p must be in (0, 1]")
    for key in ("max_tokens", "max_completion_tokens", "max_context_chars", "top_k"):
        value = generation.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"generation.{key} must be an integer")
    for key in ("max_tokens", "max_completion_tokens"):
        if generation.get(key) is not None and generation[key] <= 0:
            raise ValueError(f"generation.{key} must be positive")
    if generation.get("top_k") is not None and generation["top_k"] != -1 and generation["top_k"] <= 0:
        raise ValueError("generation.top_k must be -1 or positive")
    if generation["max_context_chars"] < 0:
        raise ValueError("generation.max_context_chars must be nonnegative")
    for key in ("language", "hotwords", "chat_prompt"):
        if generation.get(key) is not None and not isinstance(generation[key], str):
            raise ValueError(f"generation.{key} must be a string")
    if config["api"] == "chat" and (
        generation["temperature"] != 0 or any(generation.get(k) is not None for k in ("top_p", "top_k", "max_completion_tokens"))
    ):
        raise ValueError("Legacy chat fixes temperature=0 and does not consume audio-only generation parameters")
    config["generation"] = generation
    config.setdefault("timeout", 300.0)
    if _number(config["timeout"], "benchmark.asr.timeout") <= 0:
        raise ValueError("benchmark.asr.timeout must be positive")
    config.setdefault("max_audio_seconds", 30.0)
    if _number(config["max_audio_seconds"], "benchmark.asr.max_audio_seconds") <= 0:
        raise ValueError("benchmark.asr.max_audio_seconds must be positive")
    if not isinstance(config.get("warmup", {}), Mapping):
        raise ValueError("benchmark.asr.warmup must be a mapping")
    warmup = dict(config.get("warmup") or {})
    warmup.setdefault("synthetic_requests", 3)
    warmup.setdefault("exclude_first", 0)
    warmup.setdefault("api", "audio")
    if set(warmup) - {"synthetic_requests", "exclude_first", "api"}:
        raise ValueError("Unknown ASR warmup fields")
    for key in ("synthetic_requests", "exclude_first"):
        value = warmup[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"warmup.{key} must be a nonnegative integer")
    if warmup["synthetic_requests"] > 3 or warmup["api"] not in {"audio", "chat"}:
        raise ValueError("warmup supports up to three packaged audios and api audio/chat")
    config["warmup"] = warmup
    gates = config.get("gates", [])
    if not isinstance(gates, list):
        raise ValueError("benchmark.asr.gates must be a list")
    for gate in gates:
        if not isinstance(gate, Mapping) or gate.get("metric") not in METRICS:
            raise ValueError(f"ASR gate.metric must be one of {sorted(METRICS)}")
        compare = gate.get("compare")
        if compare not in {"lt", "lte", "gt", "gte"}:
            raise ValueError("ASR gate.compare must be lt/lte/gt/gte")
        _number(gate.get("baseline"), "ASR gate.baseline")
        threshold = _number(gate.get("threshold", 0), "ASR gate.threshold")
        if threshold > 1:
            raise ValueError("ASR gate.threshold must be in [0, 1]")
    if config.get("auto_assess") is not None:
        settings = config["auto_assess"]
        if not isinstance(settings, Mapping) or set(settings) - {"metric", "max", "low", "high"}:
            raise ValueError("ASR auto_assess must contain metric/max/low/high")
        if settings.get("metric") not in {"e2el_avg_ms", "e2el_p90_ms", "e2el_p99_ms", "first_text_p99_ms"}:
            raise ValueError("ASR auto_assess.metric must be an explicit latency metric")
        if _number(settings.get("max"), "ASR auto_assess.max") <= 0:
            raise ValueError("ASR auto_assess.max must be positive")
        low, high = settings.get("low", 0), settings.get("high")
        if any(isinstance(n, bool) or not isinstance(n, int) for n in (low, high)) or low < 0 or high <= low:
            raise ValueError("ASR auto_assess bounds must be integers: 0 <= low < high")
        config["auto_assess"] = {**settings, "low": low, "high": high}
    return config


def result_issues(payload: Any, task_config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Require execution, completeness and every metric gate in every run."""
    if not isinstance(payload, Mapping):
        return [{"category": "processing", "reason": "Missing ASR result"}]
    try:
        config = normalize_config((task_config.get("benchmark") or {}).get("asr") or {})
    except (TypeError, ValueError) as exc:
        return [{"category": "processing", "reason": f"Invalid ASR configuration: {exc}"}]
    issues = []
    runs = payload.get("runs")
    expected_runs = len(config["concurrency"]) * config["repeats"]
    if not isinstance(runs, list) or len(runs) != expected_runs:
        issues.append({"category": "execution", "reason": f"ASR expected {expected_runs} runs; received {len(runs) if isinstance(runs, list) else 0}"})
        runs = runs if isinstance(runs, list) else []
    expected_pairs = {(c, n) for c in config["concurrency"] for n in range(1, config["repeats"] + 1)}
    observed_pairs = []
    for run in runs:
        if not isinstance(run, Mapping):
            issues.append({"category": "processing", "reason": "Invalid ASR run record"})
            continue
        label = str(run.get("run_id", "unknown"))
        pair = (run.get("concurrency"), run.get("repeat"))
        if all(isinstance(n, int) and not isinstance(n, bool) for n in pair):
            observed_pairs.append(pair)
        else:
            issues.append({"category": "processing", "reason": f"{label}: invalid concurrency/repeat"})
        expected_count = min(config["expected_samples"], config["limit"]) if config["limit"] else config["expected_samples"]
        if (run.get("execution_error") or run.get("failed_requests") != 0 or run.get("complete") is not True
                or run.get("expected_requests") != expected_count or run.get("completed_requests") != expected_count):
            issues.append({"category": "execution", "reason": f"{label}: {run.get('execution_error') or 'failed or incomplete ASR requests'}"})
        score = run.get("score")
        valid_score = isinstance(score, Mapping) and score.get("profile") == "joyai_legacy"
        if valid_score:
            counts = [score.get(key) for key in ("N", "C", "S", "D", "I")]
            valid_score = all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts)
            if valid_score:
                n, correct, substitutions, deletions, insertions = counts
                valid_score = n > 0 and correct + substitutions + deletions == n
        if not valid_score:
            issues.append({"category": "processing", "reason": f"{label}: missing/invalid ASR score"})
        metrics = run.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            issues.append({"category": "processing", "reason": f"{label}: invalid ASR metrics"})
            metrics = {}
        for gate in config.get("gates", []):
            name, compare = gate["metric"], gate["compare"]
            actual = metrics.get(name)
            baseline, tolerance = float(gate["baseline"]), float(gate.get("threshold", 0))
            bound = baseline * (1 + tolerance if compare in {"lt", "lte"} else 1 - tolerance)
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
                issues.append({"category": "processing", "reason": f"{label}: missing metric {name}"})
                continue
            passed = {"lt": actual < bound, "lte": actual <= bound, "gt": actual > bound, "gte": actual >= bound}[compare]
            if not passed:
                issues.append({"category": "baseline", "reason": f"{label}: {name}={actual:.6g} must be {compare} {bound:.6g} (baseline={baseline:.6g}, tolerance={tolerance:g})"})
    if set(observed_pairs) != expected_pairs or len(set(observed_pairs)) != len(observed_pairs):
        issues.append({"category": "execution", "reason": "ASR concurrency/repeat run coverage mismatch"})
    if payload.get("execution_error"):
        issues.append({"category": "execution", "reason": str(payload["execution_error"])})
    return issues
