import sys
import logging
import traceback

from xtask.utils.logger import init_logger
from xtask.utils.arg_parser import ArgumentChecker
from xtask.utils.config_parser import get_xtask_config_data
from xtask.utils.openme import OpenMeClient
from xtask.tasks.task_parser import get_tasks
from xtask.core.scheduler import run_scheduler
from xtask.core.result_processor import verify_result, write_result


def process_auto_assessment(args, result, config_para, auto_assess_threshold,
                            target_ttft, target_tpot, tasks, keys_to_process, xtask_config):
    """
    处理自动评估逻辑：
    1. 解析性能测试结果。
    2. 根据目标值（TTFT/TPOT）进行二分查找策略调整。
    3. 生成新的任务并加入队列。

    :return: str - 返回控制指令："break" (退出循环), "continue" (跳过当前迭代), None (正常执行完毕)
    """

    # 1. 检查是否需要进行 Auto Assess
    # 如果不是 auto_assess 模式，或者结果中没有 perf 指标，则直接返回
    if not (getattr(args, 'assess_test_type', None) == 'auto_assess'
            and "perf" in result.get("task_config", {}).get("metrics", {})):
        return None

    # 2. 获取配置
    benchmark_dict = result["task_config"].get("benchmark", {})
    perf_cfg = benchmark_dict.get("perf")

    # 3. 配置校验
    if not perf_cfg:
        logging.error("Performance config ('perf') not found in benchmark. Breaking auto-assess loop.")
        return "break"

    curr_batch = perf_cfg.get("batch_size", 1)

    # 4. 初始化阈值范围 (仅在首次执行)
    if not config_para:
        bs_lowthrld = 0
        bs_highthrld = curr_batch * 2

        bs_assess_para = perf_cfg.get("bs_assess", {})
        # 如果yaml脚本中配置了batchsize搜索上下限，使用脚本配置
        if bs_assess_para:
            bs_lowthrld = bs_assess_para.get("low_thrld", 0)
            bs_highthrld = bs_assess_para.get("high_thrld", (curr_batch * 2))

        auto_assess_threshold["upper"] = bs_highthrld
        auto_assess_threshold["lower"] = bs_lowthrld
        # 初始化 lower，防止后续计算出错
        if "lower" not in auto_assess_threshold:
            auto_assess_threshold["lower"] = bs_lowthrld

        logging.info(f"Initializing auto-assess. Batch Size Range: [{auto_assess_threshold['lower']}, {auto_assess_threshold['upper']}]")

    auto_assess_threshold["curr"] = curr_batch

    # 5. 获取测试结果
    benchmark_result = result.get("benchmark_perf_result")
    if not benchmark_result or not hasattr(benchmark_result, 'summary'):
        logging.error("Benchmark result or summary is missing. Skipping assessment.")
        return "continue"

    try:
        # result_ttft = float(benchmark_result.summary.ttft_avg)
        # result_tpot = float(benchmark_result.summary.tpot_avg)
        # result_ttft = float(benchmark_result.summary.ttft_p99)
        # result_tpot = float(benchmark_result.summary.tpot_p99)
        result_ttft = float(benchmark_result.summary.ttft_p90)
        result_tpot = float(benchmark_result.summary.tpot_p90)
    except (AttributeError, TypeError, ValueError) as e:
        logging.error(f"Failed to parse benchmark results: {e}")
        return "continue"

    # 6. 关键打印：当前结果 vs 目标
    logging.info(f"Result Comparison -> TTFT: {result_ttft:.2f} (Target: {target_ttft}) | "
                 f"TPOT: {result_tpot:.2f} (Target: {target_tpot})")
    logging.info(f"Current Thresholds -> Lower: {auto_assess_threshold.get('lower')}, "
                 f"Upper: {auto_assess_threshold.get('upper')}, Curr: {auto_assess_threshold.get('curr')}")

    # 7. 结果判断与阈值调整
    is_ttft_ok = abs(result_ttft - target_ttft) <= 100
    is_tpot_ok = abs(result_tpot - target_tpot) <= 1

    # if is_ttft_ok and is_tpot_ok:
    if is_tpot_ok:
        logging.info(f"SUCCESS: Reached target performance! Final Batch Size: {auto_assess_threshold['curr']}")
        return "break"

    # 性能优于目标 (延迟低)，尝试增大 batch_size (移动下界向上)
    # elif result_ttft < target_ttft or result_tpot < target_tpot:
    # elif result_ttft < target_ttft 同时 ttft小于底线遍历范围上升
    # elif result_tpot < target_tpot
    elif result_tpot < target_tpot and result_ttft < target_ttft:
        auto_assess_threshold["lower"] = auto_assess_threshold["curr"]
        logging.info("Performance is better than target. Trying higher batch size (Moving lower bound up).")

    # 性能差于目标 (延迟高)，尝试减小 batch_size (移动上界向下)
    # elif result_ttft > target_ttft or result_tpot > target_tpot:
    # result_tpot > target_tpot 或 ttft 大于底线遍历范围下降
    # elif result_tpot > target_tpot
    elif result_tpot > target_tpot or result_ttft >= target_ttft:
        auto_assess_threshold["upper"] = auto_assess_threshold["curr"]
        logging.info("Performance is worse than target. Trying lower batch size (Moving upper bound down).")

    # 8. 检查二分查找是否收敛
    gap = auto_assess_threshold["upper"] - auto_assess_threshold["lower"]
    if abs(gap) <= 1:
        logging.info(f"Search converged. Best Batch Size found: {auto_assess_threshold['curr']}")
        return "break"

    # 9. 计算新的 batch_size 并生成任务
    next_batch = (auto_assess_threshold["lower"] + auto_assess_threshold["upper"]) // 2
    auto_assess_threshold["curr"] = next_batch
    config_para["batch_size"] = next_batch

    logging.info(f"Next Step: Calculated new Batch Size -> {next_batch}")

    # 生成新任务
    new_task = get_tasks(args, xtask_config, config_para)
    if new_task:
        tasks.update(new_task)
        keys_to_process.extend(new_task.keys())
        logging.info(f"Added {len(new_task)} new task(s) to the queue.")
    else:
        logging.warning("get_tasks returned no new tasks.")

    return None

def main():
    """Main function for command line testing"""
    try:
        init_logger()
        checker = ArgumentChecker()
        # Parse command line arguments
        args = checker.parse_arguments()
        args_result = checker.validate_arguments(args)
        checker.print_validation_result(args_result)
        xtask_config = get_xtask_config_data()
        logging.debug("args_result is: %s"% args_result)
        if args_result.is_valid:
            openme_client = OpenMeClient("main", "A3-ARM", getattr(args, 'host_addr', '0.0.0.0'))

            config_para = {}
            target_ttft = getattr(args, 'target_ttft', 600000)
            target_tpot = getattr(args, 'target_tpot', 100)
            auto_assess_threshold = {"lower": 0, "curr": 0, "upper": 0}
            tasks = get_tasks(args, xtask_config, config_para)
            keys_to_process = list(tasks.keys())

            index = 0
            while index < len(keys_to_process):
                current_task_key = keys_to_process[index]
                index += 1
                logging.info(
                    f"======== Processing task [{index}/{len(keys_to_process)}]: {current_task_key} ========")

                try:
                    # 1. 检查任务是否存在
                    if current_task_key not in tasks:
                        logging.warning(f"Task key '{current_task_key}' not found in tasks dictionary. Skipping.")
                        continue

                    # 2. 执行任务
                    logging.info(f"Running scheduler for task: {current_task_key} ...")
                    result = run_scheduler(current_task_key, tasks[current_task_key], args)
                    openme_client.AddCaseResult(result)

                    # 调用封装好的自动评估函数
                    action = process_auto_assessment(
                        args=args,
                        result=result,
                        config_para=config_para,
                        auto_assess_threshold=auto_assess_threshold,
                        target_ttft=target_ttft,
                        target_tpot=target_tpot,
                        tasks=tasks,
                        keys_to_process=keys_to_process,
                        xtask_config=xtask_config
                    )

                    # 根据返回值控制循环流程
                    if action == "break":
                        break
                    elif action == "continue":
                        continue

                except Exception as e:
                    logging.error(f"Failed to run scheduler for task '{current_task_key}': {str(e)}")
                    traceback.print_exc()
            # 发送京me通报
            # openme_client.SendJUEMsgCard()
        else:
            sys.exit(1)
    except Exception as e:
        logging.error(f"Argument validation failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
