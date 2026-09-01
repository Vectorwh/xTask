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


def is_sharegpt_prefix_cache_enabled(dataset_name, task_config):
    """Return True only for the explicit ShareGPT prefix-cache opt-in."""
    if str(dataset_name or "").lower() != "sharegpt":
        return False
    benchmark = task_config.get("benchmark") or {}
    if benchmark.get("benchmark_type") != "aisbench":
        return False
    prefix_config = (benchmark.get("perf") or {}).get("prefix_cache")
    return isinstance(prefix_config, dict) and prefix_config.get("enabled") is True


def get_sharegpt_fixed_output_path(task_config, output_path):
    perf_config = _perf_config(task_config)
    input_len = perf_config.get("input_len")
    num_prompts = perf_config.get("num_prompts", perf_config.get("batch_size"))
    return os.path.join(
        output_path, f"ShareGPT-in{input_len}-n{num_prompts}-last.json"
    )


def get_sharegpt_prefix_cache_output_path(task_config, output_path):
    perf_config = _perf_config(task_config)
    prefix_config = perf_config.get("prefix_cache") or {}
    input_len = perf_config.get("input_len")
    prefix_len = prefix_config.get("prefix_len")
    prefix_groups = prefix_config.get("prefix_groups", 4)
    prefix_assembly = prefix_config.get("prefix_assembly", "single")
    prefix_min_units = prefix_config.get("prefix_min_units")
    num_prompts = perf_config.get("num_prompts", perf_config.get("batch_size"))
    assembly_suffix = "" if prefix_assembly == "single" else f"-a{prefix_assembly}"
    return os.path.join(
        output_path,
        f"ShareGPT-prefix-cache-i{input_len}-p{prefix_len}{assembly_suffix}-"
        f"g{prefix_groups}-n{num_prompts}.json",
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
    elif is_sharegpt_prefix_cache_enabled(dataset_name, task_config):
        process_dataset_cmd, dataset_conf_type = preprocess_sharegpt_prefix_cache(
            process_dataset_cmd, task_config, output_path
        )
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


def preprocess_sharegpt_prefix_cache(process_dataset_cmd, task_config, output_path):
    """Generate a case-local ShareGPT dataset for prefix-cache measurements."""
    from xtask.utils.container_manager import path_mapping_in_container

    perf_config = _perf_config(task_config)
    prefix_config = perf_config.get("prefix_cache") or {}
    infer_mode = str(perf_config.get("infer_mode", "")).lower()
    input_len = perf_config.get("input_len")
    prefix_len = prefix_config.get("prefix_len")
    prefix_groups = prefix_config.get("prefix_groups", 4)
    num_prompts = perf_config.get("num_prompts", perf_config.get("batch_size"))
    seed_requests = prefix_config.get("seed_requests", True)
    prefix_order = prefix_config.get("order", "grouped")
    prefix_source = prefix_config.get("prefix_source", "dataset")
    prefix_assembly = prefix_config.get("prefix_assembly", "single")
    prefix_min_units = prefix_config.get("prefix_min_units")

    for name, value in (
        ("perf.input_len", input_len),
        ("prefix_cache.prefix_len", prefix_len),
        ("prefix_cache.prefix_groups", prefix_groups),
        ("perf.num_prompts", num_prompts),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"ShareGPT {name} must be a positive integer")
    if infer_mode != "last":
        raise ValueError(
            "ShareGPT prefix-cache preprocessing only supports infer_mode=last"
        )
    if prefix_len >= input_len:
        raise ValueError(
            "ShareGPT prefix_cache.prefix_len must be smaller than perf.input_len"
        )
    if prefix_source != "dataset":
        raise ValueError(
            "ShareGPT prefix_cache.prefix_source currently supports only 'dataset'"
        )
    if prefix_assembly not in {"single", "packed"}:
        raise ValueError(
            "ShareGPT prefix_cache.prefix_assembly must be single or packed"
        )
    if prefix_groups > num_prompts:
        raise ValueError("ShareGPT prefix_cache.prefix_groups cannot exceed num_prompts")
    if not isinstance(seed_requests, bool):
        raise ValueError("ShareGPT prefix_cache.seed_requests must be a boolean")
    if prefix_order not in {"grouped", "interleaved"}:
        raise ValueError("ShareGPT prefix_cache.order must be grouped or interleaved")
    if seed_requests:
        num_warmups = perf_config.get("num_warmups", 0)
        if (
            isinstance(num_warmups, bool)
            or not isinstance(num_warmups, int)
            or num_warmups < prefix_groups
        ):
            raise ValueError(
                "ShareGPT prefix-cache seed_requests requires num_warmups >= "
                "prefix_groups so every prefix group is warmed up"
            )

    model_path = path_mapping_in_container(task_config.get("model_path", ""))
    configured_source = perf_config.get("dataset_path") or task_config.get(
        "dataset_path", ""
    )
    output_path_in_container = path_mapping_in_container(output_path)
    process_file = os.path.join(output_path_in_container, "process_dataset.py")
    output_file = path_mapping_in_container(
        get_sharegpt_prefix_cache_output_path(task_config, output_path)
    )

    process_dataset_cmd.append(
        '\necho "开始生成 ShareGPT Prefix Cache 测试数据集"'
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
            '        echo "错误：安装jinja2失败，无法严格构造ShareGPT Prefix Cache数据集" >&2',
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
        f"--datasettype=SHAREGPT_PREFIX_CACHE "
        f"--datapath={shlex.quote(output_path_in_container)} "
        f"--modelpath={shlex.quote(model_path)} "
        '--sourcepath="$SHAREGPT_SOURCE" '
        f"--outputpath={shlex.quote(output_file)} "
        f"--infermode={infer_mode} "
        f"--prefix-len={prefix_len} "
        f"--prefix-groups={prefix_groups} "
        f"--prefix-source={shlex.quote(prefix_source)} "
        f"--prefix-assembly={shlex.quote(prefix_assembly)} "
        f"--prefix-seed-requests={1 if seed_requests else 0} "
        f"--prefix-order={shlex.quote(prefix_order)}"
    )
    if prefix_min_units is not None:
        process_command += f" --prefix-min-units={prefix_min_units}"
    process_dataset_cmd.extend(
        [
            f"if ! {process_command}; then",
            '    echo "错误：ShareGPT Prefix Cache数据集生成失败" >&2',
            "    exit 1",
            "fi",
        ]
    )

    # The generator creates the isolated AISBench config after this command is
    # assembled, so the normal dataset_conf remains available to that layer.
    return process_dataset_cmd, ""


def preprocess_ceavl(process_dataset_cmd, task_config, output_path):
    pass

def preprocess_shareGPT_V3(process_dataset_cmd, task_config, output_path):
    return preprocess_sharegpt(process_dataset_cmd, task_config, output_path)
