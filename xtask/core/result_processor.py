import inspect
import json
import logging
import math
import os
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from xtask.benchmarks.base_benchmark import ResultSummary
from xtask.core.model_identity import resolve_model_name, resolve_source_yaml
from xtask.core.xllm_service_config import get_xllm_service_config
from xtask.core.result_workbook import (
    MISSING_ACCURACY,
    PERF_SUMMARY_FIELDS,
    append_result_workbook,
)


CORE_RESULT_FIELDS = (
    "model_name",
    "case_group",
    "case_name",
    "status",
    "finished_at",
    "execution_time_s",
    "assess_test_type",
    "benchmark_type",
    "failure_category",
    "failure_reason",
    "baseline_difference",
    "output_path",
)

OMITTED_BENCHMARK_FIELDS = {
    "metadata.source_file",
    "metadata.parse_time",
    "metadata.total_rows",
}

FAILURE_XLLM = "xLLM进程异常"
FAILURE_BASELINE = "benchmark结果与基线比对失败"
FAILURE_BENCHMARK = "benchmark执行异常"
FAILURE_PROCESSING = "benchmark结果处理异常"
FAILURE_OTHER = "其他异常"
CHINA_STANDARD_TIME = timezone(timedelta(hours=8))
CASE_GROUP_WORKBOOK_PATH_ATTR = "_xtask_case_group_result_path"
CASE_GROUP_WORKBOOK_INITIALIZED_ATTR = "_xtask_case_group_result_initialized"
DEFAULT_PERF_BASELINE_METRIC = "output_tps_avg"
PERF_BASELINE_METRICS = frozenset(field.name for field in fields(ResultSummary))
PERF_BAND_COMPARE = "band"
PERF_RELATIONAL_COMPARES = frozenset({"lt", "lte", "gt", "gte"})
PERF_ALLOWED_COMPARES = frozenset({PERF_BAND_COMPARE}) | PERF_RELATIONAL_COMPARES


def check_accuracy_within_threshold(baseline, threshold, result_value):
    """
    Check if the actual test value is within the allowed fluctuation range of the baseline data
    
    Args:
        baseline (float): Baseline data value
        threshold (float): Allowed fluctuation range percentage (decimal between 0-1)
        result_value (float): Actual test value
        
    Returns:
        status: bool = True / Fasle
    """
    ret = False
    # Parameter validation
    if not isinstance(baseline, (int, float)) or baseline <= 0:
        raise ValueError("baseline must be a positive number")
    
    if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
        raise ValueError("threshold must be a decimal between 0-1")
    
    if not isinstance(result_value, (int, float)):
        raise ValueError("acc_result must be a number")
    
    # Calculate upper and lower bounds
    lower_bound = baseline * (1 - threshold)
    upper_bound = baseline * (1 + threshold)
    
    # Check if within range,
    is_valid = (lower_bound <= result_value <= upper_bound)
    
    # Calculate deviation percentage
    deviation = ((result_value - baseline) / baseline) * 100
    
    # Generate result message
    if is_valid:
        logging.info(f"✓ Test passed: Actual value {result_value:.4f} is within allowed range [{lower_bound:.4f}, {upper_bound:.4f}]")
        return True
    else:
        if result_value < lower_bound:
            logging.error(f"✗ Test failed: Actual value {result_value:.4f} is below lower bound {lower_bound:.4f} ({deviation:.2f}%)")
        else:
            logging.error(f"✗ Test failed: Actual value {result_value:.4f} is above upper bound {upper_bound:.4f} ({deviation:.2f}%)")
        return ret


def _as_positive_number(value, field_name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive number")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return value


def _as_threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("threshold must be a decimal between 0-1")
    if value < 0 or value > 1:
        raise ValueError("threshold must be a decimal between 0-1")
    return float(value)


def _perf_config(task_config):
    return ((task_config.get("benchmark") or {}).get("perf") or {})


def _has_perf_baseline_gate(perf_config):
    if not isinstance(perf_config, Mapping):
        return False
    metrics = perf_config.get("baseline_metrics")
    if isinstance(metrics, list) and metrics:
        return True
    return bool(perf_config.get("baseline"))


def _parse_one_perf_baseline_rule(item, index):
    if not isinstance(item, Mapping):
        raise ValueError(f"baseline_metrics[{index}] must be a mapping")
    metric = str(item.get("metric") or "").strip()
    if metric not in PERF_BASELINE_METRICS:
        raise ValueError(
            f"baseline_metrics[{index}].metric must be one of: "
            f"{', '.join(sorted(PERF_BASELINE_METRICS))}"
        )
    baseline = _as_positive_number(
        item.get("baseline"), f"baseline_metrics[{index}].baseline"
    )
    compare_raw = item.get("compare")
    threshold_raw = item.get("threshold")
    has_compare = compare_raw not in (None, "")
    has_threshold = threshold_raw is not None and threshold_raw != ""
    if has_compare:
        compare = str(compare_raw).strip().lower()
        if compare not in PERF_ALLOWED_COMPARES:
            raise ValueError(
                f"baseline_metrics[{index}].compare must be one of: "
                f"{', '.join(sorted(PERF_ALLOWED_COMPARES))}"
            )
    else:
        compare = PERF_BAND_COMPARE if has_threshold else None
    if compare is None:
        raise ValueError(
            f"baseline_metrics[{index}] must set threshold or compare"
        )
    if compare in PERF_RELATIONAL_COMPARES:
        if has_threshold:
            raise ValueError(
                f"baseline_metrics[{index}] cannot set both threshold and compare"
            )
        return {
            "metric": metric,
            "baseline": baseline,
            "compare": compare,
            "threshold": None,
        }
    if not has_threshold:
        raise ValueError(
            f"baseline_metrics[{index}] band compare requires threshold"
        )
    return {
        "metric": metric,
        "baseline": baseline,
        "compare": PERF_BAND_COMPARE,
        "threshold": _as_threshold(threshold_raw),
    }


def _legacy_perf_baseline_rule(perf_config):
    baseline = _as_positive_number(perf_config.get("baseline"), "baseline")
    threshold = _as_threshold(perf_config.get("threshold"))
    return {
        "metric": DEFAULT_PERF_BASELINE_METRIC,
        "baseline": baseline,
        "compare": PERF_BAND_COMPARE,
        "threshold": threshold,
    }


def parse_perf_baseline_rules(perf_config):
    """Return perf baseline rules. Empty means no gate. Raises ValueError if invalid."""
    if not isinstance(perf_config, Mapping):
        return []
    raw_metrics = perf_config.get("baseline_metrics")
    if raw_metrics is None:
        if not perf_config.get("baseline"):
            return []
        return [_legacy_perf_baseline_rule(perf_config)]
    if not isinstance(raw_metrics, list):
        raise ValueError("benchmark.perf.baseline_metrics must be a list")
    if not raw_metrics:
        if not perf_config.get("baseline"):
            return []
        return [_legacy_perf_baseline_rule(perf_config)]
    return [
        _parse_one_perf_baseline_rule(item, index)
        for index, item in enumerate(raw_metrics)
    ]


def _perf_rule_actual(summary, metric):
    if summary is None:
        return None
    actual = getattr(summary, metric, None)
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return None
    return actual


def _perf_rule_holds(actual, rule):
    baseline = rule["baseline"]
    compare = rule["compare"]
    if compare == PERF_BAND_COMPARE:
        lower = baseline * (1 - rule["threshold"])
        upper = baseline * (1 + rule["threshold"])
        return lower <= actual <= upper
    if compare == "lt":
        return actual < baseline
    if compare == "lte":
        return actual <= baseline
    if compare == "gt":
        return actual > baseline
    if compare == "gte":
        return actual >= baseline
    raise ValueError(f"unsupported compare: {compare}")


def _log_perf_rule_result(actual, rule, passed):
    metric = rule["metric"]
    baseline = rule["baseline"]
    if rule["compare"] == PERF_BAND_COMPARE:
        threshold = rule["threshold"]
        lower = baseline * (1 - threshold)
        upper = baseline * (1 + threshold)
        bound = f"{lower:.4f}, {upper:.4f}"
        if passed:
            logging.info(
                "✓ Test passed: %s %s is within allowed range [%s]",
                metric,
                f"{actual:.4f}",
                bound,
            )
            return
        deviation = ((actual - baseline) / baseline) * 100
        if actual < lower:
            logging.error(
                "✗ Test failed: %s %s is below lower bound %s (%.2f%%)",
                metric,
                f"{actual:.4f}",
                f"{lower:.4f}",
                deviation,
            )
        else:
            logging.error(
                "✗ Test failed: %s %s is above upper bound %s (%.2f%%)",
                metric,
                f"{actual:.4f}",
                f"{upper:.4f}",
                deviation,
            )
        return
    if rule["compare"] == "lt":
        bound = f"< {baseline:.4f}"
    elif rule["compare"] == "lte":
        bound = f"<= {baseline:.4f}"
    elif rule["compare"] == "gt":
        bound = f"> {baseline:.4f}"
    else:
        bound = f">= {baseline:.4f}"
    if passed:
        logging.info(
            "✓ Test passed: %s %s is within allowed range [%s]",
            metric,
            f"{actual:.4f}",
            bound,
        )
    else:
        logging.error(
            "✗ Test failed: %s %s does not satisfy %s",
            metric,
            f"{actual:.4f}",
            bound,
        )


def _evaluate_perf_baseline_rules(summary, rules):
    if not rules:
        return True
    if summary is None:
        return False
    passed = True
    for rule in rules:
        actual = _perf_rule_actual(summary, rule["metric"])
        if actual is None:
            logging.error(
                "✗ Test failed: missing perf summary field %s", rule["metric"]
            )
            passed = False
            continue
        rule_passed = _perf_rule_holds(actual, rule)
        _log_perf_rule_result(actual, rule, rule_passed)
        if not rule_passed:
            passed = False
    return passed


def _perf_rule_failure_reason(actual, rule):
    metric = rule["metric"]
    baseline = rule["baseline"]
    if rule["compare"] == PERF_BAND_COMPARE:
        threshold = rule["threshold"]
        try:
            lower = baseline * (1 - threshold)
            upper = baseline * (1 + threshold)
            return (
                f"{metric}={actual}, baseline={baseline}, "
                f"allowed=[{lower:.4f}, {upper:.4f}]"
            )
        except (TypeError, ValueError):
            return (
                f"{metric}={actual}, baseline={baseline}, threshold={threshold}"
            )
    return (
        f"{metric}={actual}, baseline={baseline}, compare={rule['compare']}"
    )


def _perf_rule_difference(actual, rule):
    metric = rule["metric"]
    baseline = rule["baseline"]
    compare = rule["compare"]
    try:
        actual_value = float(actual)
        baseline_value = float(baseline)
        difference = actual_value - baseline_value
        deviation = difference / baseline_value * 100
    except (TypeError, ValueError, ZeroDivisionError):
        if compare == PERF_BAND_COMPARE:
            return (
                f"{metric}: actual={actual}, baseline={baseline}, "
                f"threshold={rule.get('threshold')}"
            )
        return (
            f"{metric}: actual={actual}, baseline={baseline}, compare={compare}"
        )
    if compare == PERF_BAND_COMPARE:
        threshold_value = float(rule["threshold"])
        lower_bound = baseline_value * (1 - threshold_value)
        upper_bound = baseline_value * (1 + threshold_value)
        if actual_value < lower_bound:
            comparison = f"below lower bound {lower_bound:.4f}"
        elif actual_value > upper_bound:
            comparison = f"above upper bound {upper_bound:.4f}"
        else:
            comparison = (
                f"within allowed range [{lower_bound:.4f}, {upper_bound:.4f}]"
            )
        return (
            f"{metric}: actual={actual_value:.4f}, "
            f"baseline={baseline_value:.4f}, difference={difference:+.4f}, "
            f"deviation={deviation:+.2f}%, {comparison}"
        )
    return (
        f"{metric}: actual={actual_value:.4f}, "
        f"baseline={baseline_value:.4f}, difference={difference:+.4f}, "
        f"deviation={deviation:+.2f}%, compare={compare}"
    )


def _failed_perf_baseline_rules(result, task_config):
    try:
        rules = parse_perf_baseline_rules(_perf_config(task_config))
    except ValueError:
        return None
    if not rules:
        return None
    summary = getattr(result.get("benchmark_perf_result"), "summary", None)
    failed = []
    for rule in rules:
        actual = _perf_rule_actual(summary, rule["metric"])
        if actual is None or not _perf_rule_holds(actual, rule):
            failed.append((actual, rule))
    return failed


def _swebench_harness_failure(swe_result):
    """Distinguish per-instance harness errors from valid unresolved patches."""
    for field in ("harness_exit_code", "error_instances"):
        value = swe_result.get(field)
        if value is None:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            return f"Invalid SWE-bench {field}: {value!r}"
        if count < 0 or (field == "harness_exit_code" and count != 0) or (
            field == "error_instances" and count > 0
        ):
            return f"SWE-bench {field}={count}"
    return None


def verify_result(result, task_config):
    ret = False
    acc_flag = False
    logging.debug("entry function: %s"% inspect.currentframe().f_code.co_name)

    from xtask.benchmarks.asr.config import is_asr_task, result_issues

    if is_asr_task(task_config):
        issues = result_issues(result.get("benchmark_asr_result"), task_config)
        if result.get("archive_error"):
            issues.append({"category": "processing", "reason": str(result["archive_error"])})
        if result.get("benchmark_success") is False and not issues:
            issues.append({"category": "execution", "reason": "ASR benchmark command failed"})
        result["asr_gate_issues"] = issues
        if issues:
            logging.error("ASR verification failed: %s", "; ".join(issue["reason"] for issue in issues))
        return not issues

    if result.get("benchmark_success") is False:
        logging.error("✗ Test failed: benchmark command did not complete successfully")
        return False

    if _is_vbench_case(task_config):
        vbench_result = result.get("benchmark_vbench_result")
        if not isinstance(vbench_result, Mapping):
            logging.error("✗ Test failed: missing VBench result")
            return False
        try:
            failed = int(vbench_result.get("failed", 0) or 0)
        except (TypeError, ValueError):
            failed = 1
        if failed > 0:
            logging.error("✗ Test failed: VBench contains %s failed videos", failed)
            return False
        mode = str(
            ((task_config.get("benchmark") or {}).get("vbench") or {}).get(
                "mode", "generate_and_evaluate"
            )
        ).lower()
        actual = vbench_result.get("total_score")
        if mode in {"evaluate", "generate_and_evaluate"} and actual is None:
            logging.error("✗ Test failed: VBench final total score is missing")
            return False
        baseline = _vbench_baseline(task_config)
        if baseline is None or actual is None:
            return True
        baseline_value, threshold = baseline
        try:
            actual_value = float(actual)
        except (TypeError, ValueError):
            logging.error("✗ Test failed: invalid VBench total score: %r", actual)
            return False
        return check_accuracy_within_threshold(baseline_value, threshold, actual_value)

    from xtask.benchmarks.aisbench.swebench import (
        is_aisbench_swebench_task,
        swebench_mode,
    )

    if is_aisbench_swebench_task(task_config):
        swe_result = result.get("benchmark_swebench_result")
        if not isinstance(swe_result, Mapping):
            logging.error("✗ Test failed: missing SWE-bench result")
            return False
        mode = swebench_mode(task_config)
        if mode in {"eval", "all"}:
            harness_failure = _swebench_harness_failure(swe_result)
            if harness_failure:
                logging.error("✗ Test failed: %s", harness_failure)
                return False
            if swe_result.get("resolved_rate") is None:
                logging.error("✗ Test failed: SWE-bench resolved_rate is missing")
                return False
        if mode == "infer" and not (
            swe_result.get("predictions_file")
            or swe_result.get("submitted_instances")
            or swe_result.get("completed_ratio") is not None
        ):
            logging.error("✗ Test failed: SWE-bench infer produced no predictions")
            return False
        baseline = _swebench_baseline(task_config)
        actual = swe_result.get("resolved_rate")
        if baseline is None or actual is None:
            return True
        baseline_value, threshold = baseline
        try:
            actual_value = float(actual)
        except (TypeError, ValueError):
            logging.error("✗ Test failed: invalid SWE-bench resolved_rate: %r", actual)
            return False
        return check_accuracy_within_threshold(baseline_value, threshold, actual_value)

    if "acc" in task_config.get("metrics"):
        baseline = task_config.get("benchmark").get("acc").get("baseline")
        if baseline:
            threshold = task_config.get("benchmark").get("acc").get("threshold")
            benchmark_acc_result = float(result.get("benchmark_acc_result"))
            if check_accuracy_within_threshold(baseline, threshold, benchmark_acc_result):
                acc_flag = True

    if "perf" in task_config.get("metrics"):
        perf_result = result.get("benchmark_perf_result")
        if perf_result is None or getattr(perf_result, "summary", None) is None:
            logging.error("✗ Test failed: missing perf result summary")
            return False

        try:
            rules = parse_perf_baseline_rules(_perf_config(task_config))
        except ValueError as exc:
            logging.error("✗ Test failed: invalid perf baseline config: %s", exc)
            return False
        quality_passed = "acc" not in task_config.get("metrics") or acc_flag
        if not rules:
            return quality_passed
        if _evaluate_perf_baseline_rules(perf_result.summary, rules):
            return quality_passed
    else:
        return acc_flag

    return ret


def _execution_time_seconds(result):
    try:
        value = float(result.get("execution_time", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value >= 0 else 0.0


def _resolve_host_addr(task_config):
    deployment = task_config.get("deployment") or {}
    control = get_xllm_service_config(task_config)
    candidates = (
        control.get("host"),
        deployment.get("master_host"),
        task_config.get("host_addr"),
        os.getenv("LOCAL_HOST_IP"),
    )
    return next((str(value) for value in candidates if value), "")


def _resolve_execution_host(task_config, args=None):
    cli_host = getattr(args, "host_addr", None) if args is not None else None
    return str(cli_host or _resolve_host_addr(task_config))


def _relative_output_path(result):
    raw_path = result.get("benchmark_output_path") or result.get("output_file_path")
    if not raw_path:
        return ""

    path = Path(str(raw_path))
    try:
        return path.resolve().relative_to((Path.cwd() / "outputs").resolve()).as_posix()
    except ValueError:
        parts = path.parts
        try:
            outputs_index = len(parts) - 1 - tuple(reversed(parts)).index("outputs")
        except ValueError:
            return path.as_posix().strip("/")
        return Path(*parts[outputs_index + 1 :]).as_posix()


def _result_location(result, execution_host):
    execution_host = result.get("artifact_host") or execution_host
    relative_path = _relative_output_path(result).strip("/")
    if execution_host and relative_path:
        return f"{execution_host}/{relative_path}"
    return execution_host or relative_path


def _resolve_task_config(result, task_config):
    result_config = result.get("task_config")
    if isinstance(result_config, dict):
        return result_config
    return task_config if isinstance(task_config, dict) else {}


def _resolve_source_yaml(task_path, task_config):
    return resolve_source_yaml(task_path, task_config)


def _contains_metric(task_config, metric_name):
    metrics = task_config.get("metrics") or []
    return metric_name in metrics


def _is_vbench_case(task_config):
    """VBench gate: task gate first (new format), legacy metric as fallback.

    New-format VBench YAML carries ``metrics: acc`` with the config in
    ``benchmark.vbench`` (``benchmark_type: vbench``), so the metrics list no
    longer identifies these cases; the legacy ``metrics: vbench`` style is
    still honoured.
    """
    from xtask.benchmarks.vbench.task_gate import is_vbench_task

    return is_vbench_task(task_config) or _contains_metric(task_config, "vbench")


def _vbench_baseline_pair(benchmark):
    """Extract ``(baseline, threshold)`` from a ``benchmark.vbench`` block."""
    vbench = (benchmark.get("vbench") or {}) if isinstance(benchmark, Mapping) else {}
    evaluation = vbench.get("evaluation") or {}
    baseline = evaluation.get("baseline", vbench.get("baseline"))
    if baseline in (None, ""):
        return None
    return baseline, evaluation.get("threshold", vbench.get("threshold", 0.1))


def _configured_baseline(task_config):
    # A SWE-bench task may ship metrics: acc with the SWE config in
    # benchmark.acc (new format) or metrics: swebench with benchmark.swebench
    # (legacy). Decide before the generic acc/perf loop, otherwise the acc
    # branch would read benchmark_acc_result, which SWE runs never produce.
    from xtask.benchmarks.aisbench.swebench import (
        is_aisbench_swebench_task,
        swebench_config,
    )

    if is_aisbench_swebench_task(task_config):
        swebench = swebench_config(task_config)
        baseline = swebench.get("baseline")
        if baseline not in (None, ""):
            return "swebench", baseline, swebench.get("threshold", 0.1)
        return None
    benchmark = task_config.get("benchmark") or {}
    # A VBench task (benchmark_type: vbench) owns benchmark.vbench regardless
    # of metrics (new format uses metrics: acc, legacy uses metrics: vbench).
    # Decide before the generic acc/perf loop: a new-format VBench YAML would
    # otherwise hit the acc branch with an empty benchmark.acc block and lose
    # its baseline.
    if _is_vbench_case(task_config):
        pair = _vbench_baseline_pair(benchmark)
        if pair is not None:
            return ("vbench",) + pair
        return None
    for metric_name in ("acc", "perf"):
        if not _contains_metric(task_config, metric_name):
            continue
        metric_config = benchmark.get(metric_name) or {}
        if metric_name == "perf" and _has_perf_baseline_gate(metric_config):
            return metric_name, metric_config.get("baseline"), metric_config.get("threshold")
        baseline = metric_config.get("baseline")
        if baseline:
            return metric_name, baseline, metric_config.get("threshold")
    # A vbench block may be present as metadata in a non-VBench task.  Only
    # an explicit legacy VBench metric is allowed to select its baseline
    # here (VBench tasks were already handled above). SWE-bench tasks (both
    # YAML formats) were also handled above.
    if not _contains_metric(task_config, "vbench"):
        return None
    pair = _vbench_baseline_pair(benchmark)
    if pair is not None:
        return ("vbench",) + pair
    return None


def _vbench_baseline(task_config):
    configured = _configured_baseline(task_config)
    if configured is None or configured[0] != "vbench":
        return None
    try:
        baseline = float(configured[1])
        threshold = float(configured[2])
    except (TypeError, ValueError):
        return None
    if baseline <= 0 or threshold < 0 or threshold > 1:
        return None
    return baseline, threshold


def _swebench_baseline(task_config):
    configured = _configured_baseline(task_config)
    if configured is None or configured[0] != "swebench":
        return None
    try:
        baseline = float(configured[1])
        threshold = float(configured[2])
    except (TypeError, ValueError):
        return None
    if baseline <= 0 or threshold < 0 or threshold > 1:
        return None
    return baseline, threshold


def _baseline_failure_reason(result, task_config):
    configured = _configured_baseline(task_config)
    if configured is None:
        return "Benchmark result verification failed"
    metric_name, baseline, threshold = configured
    if metric_name == "acc":
        actual = result.get("benchmark_acc_result")
        metric_label = "accuracy"
    elif metric_name == "vbench":
        actual = (result.get("benchmark_vbench_result") or {}).get("total_score")
        metric_label = "vbench_total_score"
    elif metric_name == "swebench":
        actual = (result.get("benchmark_swebench_result") or {}).get("resolved_rate")
        metric_label = "swebench_resolved_rate"
    else:
        failed = _failed_perf_baseline_rules(result, task_config)
        if not failed:
            return "Benchmark result verification failed"
        return "; ".join(
            _perf_rule_failure_reason(actual, rule) for actual, rule in failed
        )
    try:
        lower = float(baseline) * (1 - float(threshold))
        upper = float(baseline) * (1 + float(threshold))
        return (
            f"{metric_label}={actual}, baseline={baseline}, "
            f"allowed=[{lower:.4f}, {upper:.4f}]"
        )
    except (TypeError, ValueError):
        return f"{metric_label}={actual}, baseline={baseline}, threshold={threshold}"


def _baseline_difference(result, task_config):
    from xtask.benchmarks.asr.config import is_asr_task, result_issues

    if is_asr_task(task_config):
        return "; ".join(
            issue["reason"]
            for issue in result_issues(result.get("benchmark_asr_result"), task_config)
            if issue["category"] == "baseline"
        )
    configured = _configured_baseline(task_config)
    if configured is None:
        return ""

    metric_name, baseline, threshold = configured
    if metric_name == "acc":
        actual = result.get("benchmark_acc_result")
        metric_label = "accuracy"
    elif metric_name == "vbench":
        actual = (result.get("benchmark_vbench_result") or {}).get("total_score")
        metric_label = "vbench_total_score"
    elif metric_name == "swebench":
        actual = (result.get("benchmark_swebench_result") or {}).get("resolved_rate")
        metric_label = "swebench_resolved_rate"
    else:
        failed = _failed_perf_baseline_rules(result, task_config)
        if not failed:
            return ""
        return "; ".join(
            _perf_rule_difference(actual, rule) for actual, rule in failed
        )

    try:
        actual_value = float(actual)
        baseline_value = float(baseline)
        threshold_value = float(threshold)
        lower_bound = baseline_value * (1 - threshold_value)
        upper_bound = baseline_value * (1 + threshold_value)
        difference = actual_value - baseline_value
        deviation = difference / baseline_value * 100
    except (TypeError, ValueError, ZeroDivisionError):
        return (
            f"{metric_label}: actual={actual}, baseline={baseline}, "
            f"threshold={threshold}"
        )

    if actual_value < lower_bound:
        comparison = f"below lower bound {lower_bound:.4f}"
    elif actual_value > upper_bound:
        comparison = f"above upper bound {upper_bound:.4f}"
    else:
        comparison = f"within allowed range [{lower_bound:.4f}, {upper_bound:.4f}]"
    return (
        f"{metric_label}: actual={actual_value:.4f}, "
        f"baseline={baseline_value:.4f}, difference={difference:+.4f}, "
        f"deviation={deviation:+.2f}%, {comparison}"
    )


def _baseline_result_failed(result, task_config):
    configured = _configured_baseline(task_config)
    if configured is None:
        return None
    metric_name, baseline, threshold = configured
    if metric_name == "acc":
        actual = result.get("benchmark_acc_result")
    elif metric_name == "vbench":
        actual = (result.get("benchmark_vbench_result") or {}).get("total_score")
    elif metric_name == "swebench":
        actual = (result.get("benchmark_swebench_result") or {}).get("resolved_rate")
    else:
        failed = _failed_perf_baseline_rules(result, task_config)
        if failed is None:
            return None
        return bool(failed)
    try:
        actual_value = float(actual)
        baseline_value = float(baseline)
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return None
    lower = baseline_value * (1 - threshold_value)
    upper = baseline_value * (1 + threshold_value)
    return not lower <= actual_value <= upper


def _classify_failure(result, task_config):
    from xtask.benchmarks.aisbench.swebench import is_aisbench_swebench_task

    if result.get("success"):
        return "", ""

    message = str(result.get("message") or "")
    benchmark_stderr = str(result.get("benchmark_stderr") or "")
    combined = f"{message}\n{benchmark_stderr}".lower()
    engine = str(task_config.get("engine_type") or "").lower()
    xllm_service_failure = engine == "xllm" and any(
        marker in combined
        for marker in (
            "startup command failed",
            "service health check failed",
            "failed readiness check",
            "pd control service startup failed",
            "xllm control service startup failed",
            "scheduler session has no benchmark runtime",
            "reused service health check failed",
        )
    )
    if "npu process abnormal" in combined or xllm_service_failure:
        reason = (
            benchmark_stderr
            if "npu process abnormal" in benchmark_stderr.lower()
            else message
        )
        return FAILURE_XLLM, reason or "xLLM process or service health check failed"
    from xtask.benchmarks.asr.config import is_asr_task, result_issues

    if is_asr_task(task_config):
        if result.get("archive_error"):
            return FAILURE_PROCESSING, str(result["archive_error"])
        issues = result.get("asr_gate_issues") or result_issues(result.get("benchmark_asr_result"), task_config)
        category = next(
            (name for name in ("execution", "processing", "baseline")
             if any(issue["category"] == name for issue in issues)),
            "processing",
        )
        labels = {
            "execution": FAILURE_BENCHMARK,
            "processing": FAILURE_PROCESSING,
            "baseline": FAILURE_BASELINE,
        }
        return labels[category], "; ".join(issue["reason"] for issue in issues) or message

    if result.get("benchmark_success") is False:
        reason = message
        if benchmark_stderr and benchmark_stderr not in reason:
            reason = f"{reason}: {benchmark_stderr}" if reason else benchmark_stderr
        return FAILURE_BENCHMARK, reason or "Benchmark command failed"

    if result.get("benchmark_success") is True and (
        _contains_metric(task_config, "swebench")
        or is_aisbench_swebench_task(task_config)
    ):
        # New-format SWE YAML carries metrics: acc, so the explicit task gate
        # must run before any _contains_metric fallback (legacy format).
        if is_aisbench_swebench_task(task_config):
            swe_result = result.get("benchmark_swebench_result")
            if isinstance(swe_result, Mapping):
                harness_failure = _swebench_harness_failure(swe_result)
                if harness_failure:
                    return FAILURE_BENCHMARK, harness_failure

    if (
        result.get("benchmark_success") is True
        and _baseline_result_failed(result, task_config) is True
    ):
        return FAILURE_BASELINE, _baseline_failure_reason(result, task_config)

    if result.get("benchmark_success") is True:
        if _is_vbench_case(task_config):
            raw_vbench_result = result.get("benchmark_vbench_result")
            vbench_result = raw_vbench_result if isinstance(raw_vbench_result, Mapping) else {}
            try:
                failed_videos = int(vbench_result.get("failed", 0) or 0)
            except (TypeError, ValueError):
                failed_videos = 1
            missing_result = (
                not isinstance(raw_vbench_result, Mapping)
                or failed_videos > 0
                or (
                    str(((task_config.get("benchmark") or {}).get("vbench") or {}).get("mode", "generate_and_evaluate")).lower()
                    in {"evaluate", "generate_and_evaluate"}
                    and vbench_result.get("total_score") is None
                )
            )
        elif is_aisbench_swebench_task(task_config) or _contains_metric(
            task_config, "swebench"
        ):
            # New-format SWE YAML carries metrics: acc; the task gate must win
            # over the _contains_metric fallback used by the legacy format.
            from xtask.benchmarks.aisbench.swebench import swebench_mode

            raw_swe_result = result.get("benchmark_swebench_result")
            swe_result = raw_swe_result if isinstance(raw_swe_result, Mapping) else {}
            mode = swebench_mode(task_config)
            if not isinstance(raw_swe_result, Mapping):
                missing_result = True
            elif mode in {"eval", "all"}:
                missing_result = swe_result.get("resolved_rate") is None
            else:
                missing_result = not (
                    swe_result.get("predictions_file")
                    or swe_result.get("submitted_instances")
                    or swe_result.get("completed_ratio") is not None
                )
        else:
            missing_result = (
                result.get("benchmark_acc_result") in (None, "")
                if _contains_metric(task_config, "acc")
                else getattr(result.get("benchmark_perf_result"), "summary", None) is None
            )
        if missing_result or "verification failed" in message.lower():
            return FAILURE_PROCESSING, message or "Benchmark result could not be processed"
        return FAILURE_OTHER, message or "Failure occurred after benchmark processing"
    return FAILURE_OTHER, message or "Unknown scheduler failure"


def _result_payload(result, task_config=None):
    vbench_result = result.get("benchmark_vbench_result")
    task_config = task_config or result.get("task_config") or {}
    benchmark_type = str(
        (task_config.get("benchmark") or {}).get("benchmark_type", "")
    ).lower()
    if benchmark_type == "asr":
        payload = result.get("benchmark_asr_result") or {}
        fields = {
            "backend": payload.get("backend"),
            "scoring_profile": payload.get("scoring_profile"),
            "config_file": payload.get("config_file"),
            "result_file": payload.get("result_file"),
        }
        from xtask.benchmarks.asr.reporting import summarize_runs
        fields["summary_by_concurrency"] = {
            label: {key: value for key, value in group.items() if key != "run_ids"}
            for label, group in summarize_runs(payload.get("runs")).items()
        }
        # Keep request texts/audio and large lists in artifacts, not Excel cells.
        score_fields = {"N", "C", "S", "D", "I", "error_count", "error_rate_pct", "unit"}
        for run in payload.get("runs", []):
            label = str(run.get("run_id", "run"))
            fields[label] = {
                "complete": run.get("complete"),
                "failed_requests": run.get("failed_requests"),
                "score": {
                    key: value for key, value in (run.get("score") or {}).items()
                    if key in score_fields
                },
                "metrics": run.get("metrics", {}),
            }
        return fields
    if benchmark_type == "vbench" and isinstance(vbench_result, Mapping):
        # Keep the workbook bounded.  VBench's per-video evaluation payload
        # and latency list belong in the archived JSON files, not in one Excel
        # cell (Excel rejects values over 32K and the row becomes unreadable).
        fields = (
            "generated",
            "resumed",
            "total",
            "failed",
            "quality_score",
            "i2v_score",
            "total_score",
            "temporal_flickering",
            "raw_scores",
            "normalized_weighted",
            "final_score_file",
            "eval_results_file",
        )
        return {field: vbench_result.get(field) for field in fields if field in vbench_result}
    swe_result = result.get("benchmark_swebench_result")
    if benchmark_type == "aisbench" and isinstance(swe_result, Mapping):
        from xtask.benchmarks.aisbench.aisbench_swe_postprocess import (
            swebench_payload_fields,
        )
        from xtask.benchmarks.aisbench.swebench import is_aisbench_swebench_task

        if is_aisbench_swebench_task(task_config):
            return swebench_payload_fields(swe_result)
    perf_result = result.get("benchmark_perf_result")
    if perf_result is None:
        return {}

    parameters = getattr(perf_result, "parameters", None)
    summary = getattr(perf_result, "summary", None)
    payload = {}
    if isinstance(parameters, Mapping):
        payload["parameters"] = parameters
    if summary is not None:
        summary_payload = {}
        for field_name in PERF_SUMMARY_FIELDS:
            if isinstance(summary, Mapping):
                if field_name in summary:
                    summary_payload[field_name] = summary[field_name]
            elif hasattr(summary, field_name):
                summary_payload[field_name] = getattr(summary, field_name)
        payload["summary"] = summary_payload
    return payload


def _flatten_mapping(value: Any, target: dict, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if (
                path in OMITTED_BENCHMARK_FIELDS
                or path.endswith(".n")
                or path == "parameters.ITL"
            ):
                continue
            _flatten_mapping(nested, target, path)
        return
    if prefix.startswith("parameters.") and hasattr(value, "__dict__"):
        _flatten_mapping(vars(value), target, prefix)
        return
    if isinstance(value, (list, tuple)):
        field_name = (
            prefix[len("parameters.") :]
            if prefix.startswith("parameters.")
            else prefix
        )
        field_name = ".".join(
            part for part in field_name.split(".") if part != "total"
        )
        if not field_name:
            # Stripping the "total" segment can leave an empty name (the
            # VBench ``total`` video counter); an empty column header would
            # break the workbook layout, so drop the field instead.
            return
        target[field_name] = json.dumps(value, ensure_ascii=False, default=str)
        return
    field_name = (
        prefix[len("parameters.") :]
        if prefix.startswith("parameters.")
        else prefix
    )
    field_name = ".".join(
        part for part in field_name.split(".") if part != "total"
    )
    if not field_name:
        # Same as above: never emit an empty column header.
        return
    target[field_name] = value


def _accuracy_excel_value(value):
    if value is None:
        return MISSING_ACCURACY
    if isinstance(value, str) and value.strip().upper() in {
        "",
        "NA",
        "N/A",
        "NAN",
        "NONE",
    }:
        return MISSING_ACCURACY
    try:
        if not math.isfinite(float(value)):
            return MISSING_ACCURACY
    except (TypeError, ValueError):
        pass
    return value


def _build_result_row(result, task_config, task_path=None, args=None):
    task_config = _resolve_task_config(result, task_config)

    case_name = task_config.get("case_name")
    if not case_name and task_path:
        case_name = Path(task_path).stem

    model_name = resolve_model_name(task_path, task_config)
    success = bool(result.get("success"))
    failure_category, failure_reason = _classify_failure(result, task_config)
    # New-format VBench YAML carries metrics: acc, but its Excel columns come
    # from _result_payload (benchmark_type: vbench), not the accuracy column.
    from xtask.benchmarks.asr.config import is_asr_task

    is_accuracy = _contains_metric(task_config, "acc") and not (
        _is_vbench_case(task_config) or is_asr_task(task_config)
    )
    benchmark = task_config.get("benchmark") or {}
    execution_host = _resolve_execution_host(task_config, args)
    finished_at = datetime.now(tz=CHINA_STANDARD_TIME).isoformat(timespec="seconds")
    row = {
        "model_name": str(model_name or "unknown"),
        "case_group": str(getattr(args, "case_group", "") or ""),
        "case_name": str(case_name or "unknown"),
        "status": "pass" if success else "fail",
        "finished_at": finished_at,
        "execution_time_s": round(_execution_time_seconds(result), 2),
        "assess_test_type": str(getattr(args, "assess_test_type", "none") or "none"),
        "benchmark_type": str(benchmark.get("benchmark_type") or ""),
        "failure_category": failure_category or ("NA" if success else ""),
        "failure_reason": failure_reason or ("NA" if success else ""),
        "baseline_difference": (
            _baseline_difference(result, task_config)
            if failure_category == FAILURE_BASELINE
            else "NA"
        ) or "NA",
        "output_path": _result_location(result, execution_host),
    }
    if is_accuracy:
        row["accuracy"] = _accuracy_excel_value(
            result.get("benchmark_acc_result")
        )
    else:
        row["accuracy"] = "NA"
        _flatten_mapping(_result_payload(result, task_config), row)
    return model_name, row


def _current_case_group_timestamp():
    return datetime.now(tz=CHINA_STANDARD_TIME).strftime("%Y%m%d_%H%M%S")


def _case_group_workbook_path(outputs, case_group, args):
    existing_path = getattr(args, CASE_GROUP_WORKBOOK_PATH_ATTR, None)
    if existing_path:
        return Path(existing_path)

    group_name = str(case_group).lower()
    filename = f"{group_name}_{_current_case_group_timestamp()}.xlsx"
    workbook_path = outputs / group_name / filename
    setattr(args, CASE_GROUP_WORKBOOK_PATH_ATTR, str(workbook_path))
    return workbook_path


def write_result(result, task_config=None, task_path=None, args=None):
    """Append one scheduler result to global and optional case-group workbooks."""
    logging.debug("entry function: %s", inspect.currentframe().f_code.co_name)
    from xtask.utils.secrets import redact_tree
    result = redact_tree(result)
    task_config = redact_tree(task_config)
    model_name, row = _build_result_row(
        result, task_config or {}, task_path, args=args
    )
    outputs = Path.cwd() / "outputs"
    history_written = append_result_workbook(
        outputs / "result.xlsx", model_name, row
    )

    group_written = True
    case_group = getattr(args, "case_group", None) if args is not None else None
    if case_group:
        reset = not bool(
            getattr(args, CASE_GROUP_WORKBOOK_INITIALIZED_ATTR, False)
        )
        group_written = append_result_workbook(
            _case_group_workbook_path(outputs, case_group, args),
            model_name,
            row,
            reset=reset,
        )
        if group_written:
            setattr(args, CASE_GROUP_WORKBOOK_INITIALIZED_ATTR, True)

    logging.info(
        "Case report: %s, %s, %.2fs, %s, %s",
        row["case_name"],
        row["status"],
        row["execution_time_s"],
        row["output_path"],
        row["finished_at"],
    )
    return history_written and group_written
