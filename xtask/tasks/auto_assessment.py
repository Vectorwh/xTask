"""Dynamic batch-size decisions for auto-assessment runs."""

import logging
from pathlib import Path

from xtask.tasks.task_parser import gen_group_autoassess_task, get_tasks


def _auto_assess_base_case(result, current_task_key):
    generated_case = result.get("task_config", {}).get("case_name")
    if not generated_case and current_task_key:
        generated_case = Path(current_task_key).stem
    base_case, separator, _ = str(generated_case or "").rpartition("_bs-")
    if not separator or not base_case:
        raise ValueError(
            f"Cannot determine auto-assess base case from '{generated_case}'"
        )
    return base_case


def _insert_tasks_after_current(keys_to_process, current_task_key, new_tasks):
    new_keys = [key for key in new_tasks if key not in keys_to_process]
    if not new_keys:
        return
    try:
        insert_at = keys_to_process.index(current_task_key) + 1
    except ValueError:
        keys_to_process.extend(new_keys)
        return
    keys_to_process[insert_at:insert_at] = new_keys


def process_auto_assessment(
    args,
    result,
    config_para,
    auto_assess_threshold,
    target_ttft,
    target_tpot,
    tasks,
    keys_to_process,
    xtask_config,
    current_task_key=None,
    output_path=None,
):
    """Evaluate one trial and append the next batch-size task when needed."""
    is_auto_perf = (
        getattr(args, "assess_test_type", None) == "auto_assess"
        and "perf" in result.get("task_config", {}).get("metrics", {})
    )
    if not is_auto_perf:
        return None
    auto_assess_threshold.pop("stop_reason", None)

    if result.get("scheduler_error"):
        logging.error(
            "Stopping auto-assess after scheduler session failure: %s",
            result.get("message") or "unknown scheduler error",
        )
        auto_assess_threshold["stop_reason"] = "scheduler_error"
        return "break"

    perf_config = result["task_config"].get("benchmark", {}).get("perf")
    if not perf_config:
        logging.error(
            "Performance config ('perf') not found in benchmark. "
            "Breaking auto-assess loop."
        )
        auto_assess_threshold["stop_reason"] = "missing_performance_config"
        return "break"

    current_batch = perf_config.get("batch_size", 1)
    if not config_para:
        assess_config = perf_config.get("bs_assess", {})
        auto_assess_threshold["lower"] = assess_config.get("low_thrld", 0)
        auto_assess_threshold["upper"] = assess_config.get(
            "high_thrld", current_batch * 2
        )
        logging.info(
            "Initializing auto-assess. Batch Size Range: [%s, %s]",
            auto_assess_threshold["lower"],
            auto_assess_threshold["upper"],
        )

    auto_assess_threshold["curr"] = current_batch
    benchmark_result = result.get("benchmark_perf_result")
    if not benchmark_result or not hasattr(benchmark_result, "summary"):
        logging.error("Benchmark result or summary is missing. Skipping assessment.")
        auto_assess_threshold["stop_reason"] = "missing_performance_summary"
        return "continue"

    try:
        result_ttft = float(benchmark_result.summary.ttft_p90)
        result_tpot = float(benchmark_result.summary.tpot_p90)
    except (AttributeError, TypeError, ValueError) as error:
        logging.error("Failed to parse benchmark results: %s", error)
        auto_assess_threshold["stop_reason"] = "invalid_performance_summary"
        return "continue"

    logging.info(
        "Result Comparison -> TTFT: %.2f (Target: %s) | "
        "TPOT: %.2f (Target: %s)",
        result_ttft,
        target_ttft,
        result_tpot,
        target_tpot,
    )
    logging.info(
        "Current Thresholds -> Lower: %s, Upper: %s, Curr: %s",
        auto_assess_threshold.get("lower"),
        auto_assess_threshold.get("upper"),
        auto_assess_threshold.get("curr"),
    )

    if abs(result_tpot - target_tpot) <= 1:
        logging.info(
            "SUCCESS: Reached target performance! Final Batch Size: %s",
            auto_assess_threshold["curr"],
        )
        auto_assess_threshold["stop_reason"] = "tpot_target_reached"
        return "break"

    if result_tpot < target_tpot and result_ttft < target_ttft:
        auto_assess_threshold["lower"] = auto_assess_threshold["curr"]
        logging.info(
            "Performance is better than target. Trying higher batch size "
            "(Moving lower bound up)."
        )
    elif result_tpot > target_tpot or result_ttft >= target_ttft:
        auto_assess_threshold["upper"] = auto_assess_threshold["curr"]
        logging.info(
            "Performance is worse than target. Trying lower batch size "
            "(Moving upper bound down)."
        )

    gap = auto_assess_threshold["upper"] - auto_assess_threshold["lower"]
    if abs(gap) <= 1:
        logging.info(
            "Search converged. Best Batch Size found: %s",
            auto_assess_threshold["curr"],
        )
        auto_assess_threshold["stop_reason"] = "interval_converged"
        return "break"

    next_batch = (
        auto_assess_threshold["lower"] + auto_assess_threshold["upper"]
    ) // 2
    auto_assess_threshold["curr"] = next_batch
    config_para["batch_size"] = next_batch
    logging.info("Next Step: Calculated new Batch Size -> %s", next_batch)

    if current_task_key and output_path:
        new_tasks = {}
        base_case = _auto_assess_base_case(result, current_task_key)
        generated = gen_group_autoassess_task(
            base_case,
            str(Path(output_path).parent),
            new_tasks,
            config_para,
        )
        if not generated:
            new_tasks = {}
    else:
        # Keep direct callers compatible; the main workflow always supplies
        # the current task so a case group advances only that base case.
        new_tasks = get_tasks(args, xtask_config, config_para)
    if new_tasks:
        tasks.update(new_tasks)
        _insert_tasks_after_current(
            keys_to_process, current_task_key, new_tasks
        )
        logging.info("Added %s new task(s) to the queue.", len(new_tasks))
    else:
        logging.warning("get_tasks returned no new tasks.")
        auto_assess_threshold["stop_reason"] = "task_generation_failed"

    return None
