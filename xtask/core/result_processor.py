import os
import inspect
import logging
import stat
from datetime import datetime, timezone


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

    if "acc" in task_config.get("metrics"):
        baseline = task_config.get("benchmark").get("acc").get("baseline")
        if baseline:
            threshold = task_config.get("benchmark").get("acc").get("threshold")
            benchmark_acc_result = float(result.get("benchmark_acc_result"))
            if check_accuracy_within_threshold(baseline, threshold, benchmark_acc_result):
                acc_flag = True

    if "perf" in task_config.get("metrics"):
        baseline = task_config.get("benchmark").get("perf").get("baseline")
        if baseline:
            threshold = task_config.get("benchmark").get("perf").get("threshold")
            benchmark_perf_result = float(result.get("benchmark_perf_result").summary.output_token_throughput)
            if check_accuracy_within_threshold(baseline, threshold, benchmark_perf_result):
                return True
        else:
            return True
    else:
        return acc_flag

    return ret


def write_result(result, task_config):
    logging.debug("entry function: %s"% inspect.currentframe().f_code.co_name)
    case_name = task_config.get("case_name")
    current_datetime = datetime.now(tz=timezone.utc)
    date_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    local_host_ip = os.getenv('LOCAL_HOST_IP')
    results_path = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    result_file = os.path.join(results_path, 'result.csv')
    first_row = 'case_name,results,precision,max_deviation,time_cost,local_host_ip,date_time'
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    if not os.path.exists(result_file):
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(result_file, flags, modes), 'w') as f:
            f.write(first_row + '\n')

    if result.get("success"):
        result_ret = "pass"
    else:
        result_ret = "fail"
    error_num = result.get("error_num")
    time_cost = result.get("execution_time")
    logging.info('Case report: %s, %s, %s, %.2f, %s, %s', case_name, result_ret, error_num,
                 time_cost / 1000, local_host_ip, date_time)
    case_flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(result_file, case_flags, modes), 'a+') as case_file:
        case_file.write('%s,%s,%s,%.2f,%s,%s\n' % (case_name, result_ret, error_num, time_cost / 1000,
                                                      local_host_ip, date_time))
        case_file.close()
