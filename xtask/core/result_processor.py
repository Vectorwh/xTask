import inspect
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from xtask.core.model_identity import resolve_model_name, resolve_source_yaml
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


def verify_result(result, task_config):
    ret = False
    acc_flag = False
    logging.debug("entry function: %s"% inspect.currentframe().f_code.co_name)

    if result.get("benchmark_success") is False:
        logging.error("✗ Test failed: benchmark command did not complete successfully")
        return False

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

        baseline = task_config.get("benchmark").get("perf").get("baseline")
        if baseline:
            threshold = task_config.get("benchmark").get("perf").get("threshold")
            benchmark_perf_result = float(perf_result.summary.output_tps_avg)
            if check_accuracy_within_threshold(baseline, threshold, benchmark_perf_result):
                return True
        else:
            return True
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
    pd_control = deployment.get("pd_control") or {}
    candidates = (
        pd_control.get("host"),
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


def _configured_baseline(task_config):
    benchmark = task_config.get("benchmark") or {}
    for metric_name in ("acc", "perf"):
        if not _contains_metric(task_config, metric_name):
            continue
        metric_config = benchmark.get(metric_name) or {}
        baseline = metric_config.get("baseline")
        if baseline:
            return metric_name, baseline, metric_config.get("threshold")
    return None


def _baseline_failure_reason(result, task_config):
    configured = _configured_baseline(task_config)
    if configured is None:
        return "Benchmark result verification failed"
    metric_name, baseline, threshold = configured
    if metric_name == "acc":
        actual = result.get("benchmark_acc_result")
        metric_label = "accuracy"
    else:
        summary = getattr(result.get("benchmark_perf_result"), "summary", None)
        actual = getattr(summary, "output_tps_avg", None)
        metric_label = "output_tps_avg"
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
    configured = _configured_baseline(task_config)
    if configured is None:
        return ""

    metric_name, baseline, threshold = configured
    if metric_name == "acc":
        actual = result.get("benchmark_acc_result")
        metric_label = "accuracy"
    else:
        summary = getattr(result.get("benchmark_perf_result"), "summary", None)
        actual = getattr(summary, "output_tps_avg", None)
        metric_label = "output_tps_avg"

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
    else:
        summary = getattr(result.get("benchmark_perf_result"), "summary", None)
        actual = getattr(summary, "output_tps_avg", None)
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

    if result.get("benchmark_success") is False:
        reason = message
        if benchmark_stderr and benchmark_stderr not in reason:
            reason = f"{reason}: {benchmark_stderr}" if reason else benchmark_stderr
        return FAILURE_BENCHMARK, reason or "Benchmark command failed"

    if (
        result.get("benchmark_success") is True
        and _baseline_result_failed(result, task_config) is True
    ):
        return FAILURE_BASELINE, _baseline_failure_reason(result, task_config)

    if result.get("benchmark_success") is True:
        missing_result = (
            result.get("benchmark_acc_result") in (None, "")
            if _contains_metric(task_config, "acc")
            else getattr(result.get("benchmark_perf_result"), "summary", None) is None
        )
        if missing_result or "verification failed" in message.lower():
            return FAILURE_PROCESSING, message or "Benchmark result could not be processed"
        return FAILURE_OTHER, message or "Failure occurred after benchmark processing"
    return FAILURE_OTHER, message or "Unknown scheduler failure"


def _result_payload(result):
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
    is_accuracy = _contains_metric(task_config, "acc")
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
        _flatten_mapping(_result_payload(result), row)
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
