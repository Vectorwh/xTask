"""ASR-native summaries shared by Excel, search and optional dashboard adapters."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any, Dict, List

from xtask.benchmarks.asr.config import METRICS, is_asr_task, normalize_config


def finite_metrics(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): float(number) for key, number in value.items()
        if not isinstance(number, bool) and isinstance(number, (int, float))
        and math.isfinite(number)
    }


def summarize_runs(runs: Any) -> Dict[str, Any]:
    """Describe repetitions per concurrency, never average distinct workloads."""
    groups: Dict[int, List[Mapping[str, Any]]] = {}
    for run in runs if isinstance(runs, list) else []:
        if not isinstance(run, Mapping):
            continue
        concurrency = run.get("concurrency")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
            continue
        groups.setdefault(concurrency, []).append(run)
    summaries = {}
    for concurrency, records in sorted(groups.items()):
        valid = []
        for run in records:
            score = run.get("score")
            n = score.get("N") if isinstance(score, Mapping) else None
            if (run.get("complete") is True and run.get("failed_requests") == 0
                    and not run.get("execution_error") and isinstance(n, int)
                    and not isinstance(n, bool) and n > 0):
                valid.append(run)
        values: Dict[str, List[float]] = {}
        for run in valid:
            for name, number in finite_metrics(run.get("metrics")).items():
                values.setdefault(name, []).append(number)
        summaries[f"c{concurrency}"] = {
            "concurrency": concurrency, "total_runs": len(records), "valid_runs": len(valid),
            "invalid_runs": len(records) - len(valid),
            "run_ids": [str(run.get("run_id", "")) for run in records],
            "metrics": {
                name: {"count": len(numbers), "mean": statistics.mean(numbers),
                       "median": statistics.median(numbers), "min": min(numbers),
                       "max": max(numbers), "stddev": statistics.pstdev(numbers)}
                for name, numbers in sorted(values.items())
            },
        }
    return summaries


def search_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Return median run metrics; qualify names if multiple loads were tested."""
    payload = result.get("benchmark_asr_result")
    if not isinstance(payload, Mapping):
        return {}
    output = {}
    groups = summarize_runs(payload.get("runs"))
    for label, group in groups.items():
        for metric, stats in group["metrics"].items():
            if metric in METRICS:
                output[f"{label}.{metric}"] = stats["median"]
                if len(groups) == 1:
                    output[metric] = stats["median"]
    return output


def auto_assess_settings(task_config: Mapping[str, Any]) -> Dict[str, Any]:
    """ASR auto-assess requires its own latency ceiling, not LLM TPOT."""
    config = normalize_config((task_config.get("benchmark") or {}).get("asr") or {})
    settings = config.get("auto_assess")
    if not settings:
        raise ValueError("ASR auto_assess requires benchmark.asr.auto_assess metric/max/bounds")
    if len(config["concurrency"]) != 1:
        raise ValueError("ASR auto_assess requires exactly one concurrency per trial")
    if not settings["low"] < config["concurrency"][0] <= settings["high"]:
        raise ValueError("ASR auto_assess starting concurrency must satisfy low < concurrency <= high")
    return {**settings, "concurrency": config["concurrency"][0]}


def report_to_dashboard(result: Dict[str, Any], task_config: Dict[str, Any]) -> Any:
    """Use the optional reporter transport, without its LLM SLA/accuracy rules.

    ASR metrics use toolSpecificConfig (an existing extension field). The server
    UI may not yet display these extensions; local spooled payloads retain them.
    """
    if not is_asr_task(task_config):
        return None
    from bench_reporter.integration import _hardware_count, _post_or_spool
    from bench_reporter.payload import build_payload

    payload = result.get("benchmark_asr_result")
    if not isinstance(payload, Mapping):
        return None
    metadata = {
        "model_name": task_config.get("served_model_name") or "unknown",
        "model_type": "asr", "model_version": task_config.get("engine_version") or "unknown",
        "framework": task_config.get("engine_type", "xllm"),
        "scenario": task_config.get("case_name"),
        "hardware_count": _hardware_count(task_config),
    }
    # Keep unknown hardware/version unknown, rather than inventing a TP count.
    reports = []
    for run in payload.get("runs", []):
        if not isinstance(run, Mapping):
            continue
        metrics = finite_metrics(run.get("metrics"))
        common = {
            "e2e_avg_ms": metrics.get("e2el_avg_ms"), "e2e_p90_ms": metrics.get("e2el_p90_ms"),
            "e2e_p99_ms": metrics.get("e2el_p99_ms"),
            "request_per_sec": metrics.get("request_throughput"),
            "total_requests": run.get("expected_requests"),
            "failed_requests": run.get("failed_requests"),
            "concurrency": run.get("concurrency"), "max_concurrency": run.get("concurrency"),
        }
        source = f"{result.get('benchmark_output_path', '')}/{run.get('run_id', '')}"
        report = build_payload(common, "xtask", metadata, source)
        report["hardware"]["gpuCount"] = metadata["hardware_count"]
        report["sweepResults"][0]["runId"] = run.get("run_id")
        # No fake accuracy, TPOT, token counts or generic LLM SLA.
        report["toolSpecificConfig"].pop("sla", None)
        report["toolSpecificConfig"]["asr"] = {
            "scoringProfile": payload.get("scoring_profile"),
            "runId": run.get("run_id"), "repeat": run.get("repeat"),
            "complete": run.get("complete"), "metrics": metrics,
            "score": {key: value for key, value in (run.get("score") or {}).items()
                      if key in {"N", "C", "S", "D", "I", "unit", "error_count", "error_rate_pct"}},
            "casePassed": result.get("success") is True,
            "gateIssues": result.get("asr_gate_issues") or payload.get("gate_issues", []),
            "artifactHost": result.get("artifact_host"),
            "serviceEndpoint": result.get("service_endpoint"),
        }
        if run.get("completed_requests") is not None:
            report["sweepResults"][0]["successfulRequests"] = max(
                0, run["completed_requests"] - (run.get("failed_requests") or 0)
            )
        reports.append(report)
    from pathlib import Path
    from xtask.benchmarks.asr.artifacts import write_json
    archive = result.get("benchmark_output_path")
    if archive and Path(archive).is_dir():
        write_json(Path(archive) / "dashboard_payloads.json", reports)
    return [_post_or_spool(report) for report in reports]
