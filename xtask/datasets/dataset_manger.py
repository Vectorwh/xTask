import os


def get_dataset_type(dataset_name):
    if dataset_name and dataset_name.lower() == "gsm8k":
        datasettype = "GSM"
    
    return datasettype


def preprocess_dataset(dataset_name, task_config, output_path):
    process_dataset_cmd = list()
    if dataset_name == "gsm8k" and task_config.get("benchmark").get("benchmark_type") == "aisbench":
        process_dataset_cmd, dataset_conf_type = preprocess_gsm8k(process_dataset_cmd, task_config, output_path)
    elif dataset_name == "ceavl" and task_config.get("benchmark").get("benchmark_type") == "aisbench":
        process_dataset_cmd, dataset_conf_type = preprocess_ceavl(process_dataset_cmd, task_config, output_path)
    
    return process_dataset_cmd, dataset_conf_type


def preprocess_gsm8k(process_dataset_cmd, task_config, output_path):
    from xtask.datasets.gsm8k.process_gsm8k import process_dataset_gsm8k
    from xtask.utils.container_manager import path_mapping_in_container

    batch_size = task_config.get("benchmark").get("perf").get("batch_size")
    input_len = task_config.get("benchmark").get("perf").get("input_len")
    dataset_type = get_dataset_type(task_config.get("dataset_name"))
    model_path = task_config.get("model_path", "")
    output_file = path_mapping_in_container(process_dataset_gsm8k(batch_size, input_len, dataset_type, 
                                                            model_path, output_path))
    dataset_conf_type = "gsm8k_gen_0_shot_cot_str_perf"

    process_dataset_cmd.append('\necho "开始生成 gsm8k dataset：$AIS_BENCH_PATH/datasets/gsm8k"')
    process_dataset_cmd.append(f'mv {output_file} $AIS_BENCH_PATH/datasets/gsm8k/test.jsonl')

    return process_dataset_cmd, dataset_conf_type


def preprocess_ceavl(process_dataset_cmd, task_config, output_path):
    pass

