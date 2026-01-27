import inspect
import logging


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
    
    # Check if within range
    is_valid = lower_bound <= result_value <= upper_bound
    
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
            benchmark_perf_result = float(result.get("benchmark_acc_result").output_tokens_avg)
            if check_accuracy_within_threshold(baseline, threshold, benchmark_perf_result):
                return True
        else:
            return True
    else:
        return acc_flag

    return ret


def write_result(result):
    logging.debug("entry function: %s"% inspect.currentframe().f_code.co_name)
