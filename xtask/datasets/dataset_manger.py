import os


def get_dataset_type(dataset_name):
    if dataset_name and dataset_name.lower() == "gsm8k":
        datasettype = "GSM"
    
    return datasettype


def preprocess_dataset(dataset_name, task_config, output_path):
    dataset_conf_type = ""
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
    num_prompts = task_config.get("benchmark").get("perf").get("num_prompts", batch_size)
    model_path = path_mapping_in_container(task_config.get("model_path", ""))

    # dataset_type = get_dataset_type(task_config.get("dataset_name"))
    # model_path = task_config.get("model_path", "")
    # 数据集的固定长度构造放到容器里Run_aisbench的流程执行
    # output_file = path_mapping_in_container(process_dataset_gsm8k(batch_size, input_len, dataset_type,
    #                                                     model_path, output_path, num_prompts))

    # 容器里生成固定长度数据集
    dataset_conf_type = "gsm8k_gen_0_shot_cot_str_perf"
    process_dataset_cmd.append('\necho "开始生成 gsm8k dataset：$AIS_BENCH_PATH/datasets/gsm8k"')
    output_path_in_container = path_mapping_in_container(output_path)
    process_file = os.path.join(output_path_in_container, 'process_dataset.py')
    input_data_file = os.path.join(output_path_in_container, 'GSM8K.jsonl')
    process_dataset_cmd.append(f'cp -f /workspace/xtask/benchmarks/aisbench/run_benchmark/process_dataset.py {process_file}')
    process_dataset_cmd.append(f'cp -f /workspace/xtask/datasets/gsm8k/GSM8K.jsonl {input_data_file}')
    process_dataset_cmd.append(f'python {process_file} --bs={num_prompts} --inputlen={input_len} --datasettype=GSM8K --datapath={output_path_in_container} --modelpath={model_path}')
    output_file_path = os.path.join(output_path_in_container, f'GSM8K-in{input_len}-bs{num_prompts}.jsonl')

    # 拷贝生成的固定长度数据集到ais_bench数据集目录
    process_dataset_cmd.append(f'cp -f {output_file_path} $AIS_BENCH_PATH/datasets/gsm8k/test.jsonl')

    return process_dataset_cmd, dataset_conf_type


def preprocess_ceavl(process_dataset_cmd, task_config, output_path):
    pass

def preprocess_shareGPT_V3(process_dataset_cmd, task_config, output_path):
    pass
