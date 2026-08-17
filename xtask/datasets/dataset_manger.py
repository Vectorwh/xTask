import os
import shlex


DEFAULT_SHAREGPT_DATASET = (
    "$AIS_BENCH_PATH/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
)


def _perf_config(task_config):
    return (task_config.get("benchmark") or {}).get("perf") or {}


def is_sharegpt_fixed_length_enabled(dataset_name, task_config):
    if str(dataset_name or "").lower() != "sharegpt":
        return False
    benchmark = task_config.get("benchmark") or {}
    if benchmark.get("benchmark_type") != "aisbench":
        return False
    perf_config = benchmark.get("perf") or {}
    return (
        str(perf_config.get("infer_mode", "")).lower() == "last"
        and "input_len" in perf_config
    )


def get_sharegpt_fixed_output_path(task_config, output_path):
    perf_config = _perf_config(task_config)
    input_len = perf_config.get("input_len")
    num_prompts = perf_config.get("num_prompts", perf_config.get("batch_size"))
    return os.path.join(
        output_path, f"ShareGPT-in{input_len}-n{num_prompts}-last.json"
    )


def get_dataset_type(dataset_name):
    if dataset_name and dataset_name.lower() == "gsm8k":
        datasettype = "GSM"
    
    return datasettype


def preprocess_dataset(dataset_name, task_config, output_path):
    dataset_conf_type = ""
    process_dataset_cmd = list()
    benchmark_type = (task_config.get("benchmark") or {}).get("benchmark_type")
    dataset_name = str(dataset_name or "").lower()
    if dataset_name == "gsm8k" and benchmark_type == "aisbench":
        process_dataset_cmd, dataset_conf_type = preprocess_gsm8k(process_dataset_cmd, task_config, output_path)
    elif is_sharegpt_fixed_length_enabled(dataset_name, task_config):
        process_dataset_cmd, dataset_conf_type = preprocess_sharegpt(
            process_dataset_cmd, task_config, output_path
        )
    elif dataset_name == "ceavl" and benchmark_type == "aisbench":
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


def preprocess_sharegpt(process_dataset_cmd, task_config, output_path):
    from xtask.utils.container_manager import path_mapping_in_container

    perf_config = _perf_config(task_config)
    input_len = perf_config.get("input_len")
    batch_size = perf_config.get("batch_size")
    num_prompts = perf_config.get("num_prompts", batch_size)
    if isinstance(input_len, bool) or not isinstance(input_len, int) or input_len <= 0:
        raise ValueError("ShareGPT benchmark.perf.input_len must be a positive integer")
    if (
        isinstance(num_prompts, bool)
        or not isinstance(num_prompts, int)
        or num_prompts <= 0
    ):
        raise ValueError("ShareGPT benchmark.perf.num_prompts must be a positive integer")

    model_path = path_mapping_in_container(task_config.get("model_path", ""))
    configured_source = perf_config.get("dataset_path") or task_config.get(
        "dataset_path", ""
    )
    output_path_in_container = path_mapping_in_container(output_path)
    process_file = os.path.join(output_path_in_container, "process_dataset.py")
    output_file = path_mapping_in_container(
        get_sharegpt_fixed_output_path(task_config, output_path)
    )

    process_dataset_cmd.append(
        '\necho "开始生成 infer_mode=last 的固定长度 ShareGPT dataset"'
    )
    process_dataset_cmd.append(
        "cp -f "
        "/workspace/xtask/benchmarks/aisbench/run_benchmark/process_dataset.py "
        f"{shlex.quote(process_file)}"
    )
    if configured_source:
        source_path = path_mapping_in_container(str(configured_source))
        process_dataset_cmd.append(
            f"SHAREGPT_SOURCE={shlex.quote(source_path)}"
        )
    else:
        process_dataset_cmd.append(f'SHAREGPT_SOURCE="{DEFAULT_SHAREGPT_DATASET}"')
    process_dataset_cmd.extend(
        [
            'if [ ! -f "$SHAREGPT_SOURCE" ]; then',
            '    echo "错误：ShareGPT数据集不存在 $SHAREGPT_SOURCE" >&2',
            "    exit 1",
            "fi",
            'if ! python -c "import jinja2" >/dev/null 2>&1; then',
            '    echo "安装ShareGPT chat template渲染依赖jinja2"',
            "    if ! python -m pip install 'jinja2>=3.1,<4'; then",
            '        echo "错误：安装jinja2失败，无法严格计算ShareGPT输入长度" >&2',
            "        exit 1",
            "    fi",
            "fi",
        ]
    )
    process_command = (
        "python "
        f"{shlex.quote(process_file)} "
        f"--bs={num_prompts} "
        f"--inputlen={input_len} "
        "--datasettype=SHAREGPT "
        f"--datapath={shlex.quote(output_path_in_container)} "
        f"--modelpath={shlex.quote(model_path)} "
        '--sourcepath="$SHAREGPT_SOURCE" '
        f"--outputpath={shlex.quote(output_file)} "
        "--infermode=last"
    )
    process_dataset_cmd.extend(
        [
            f"if ! {process_command}; then",
            '    echo "错误：ShareGPT固定长度数据集生成失败" >&2',
            "    exit 1",
            "fi",
        ]
    )

    dataset_conf_type = perf_config.get("dataset_conf") or "sharegpt_gen"
    return process_dataset_cmd, dataset_conf_type


def preprocess_ceavl(process_dataset_cmd, task_config, output_path):
    pass

def preprocess_shareGPT_V3(process_dataset_cmd, task_config, output_path):
    return preprocess_sharegpt(process_dataset_cmd, task_config, output_path)
