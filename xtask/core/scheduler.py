import inspect
import logging
import os
import json
import time
import types
import yaml
import copy
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from xtask.utils.container_manager import ContainerManager, ContainerConfig
from xtask.utils.config_parser import get_xtask_config_data
from xtask.serv_fmk.serv_common import wait_serv_start
from xtask.core.result_processor import verify_result, write_result
from xtask.core.xllm_service_config import (
    get_xllm_service_config,
    normalize_xllm_service_config,
    service_container_nodes,
    service_etcd_location,
    service_port_claims,
)


class Json_Reader:
    """XLLM配置管理类"""
    
    @staticmethod
    def load_config(config_path) -> Dict[str, Any]:
        """
        加载XLLM配置文件
        
        Returns:
            XLLM配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logging.error(f"XLLM config not found: {config_path}")
            raise FileNotFoundError(f"XLLM config not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logging.info("Loaded XLLM configuration")
            return config
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse XLLM config: {e}")
            raise
        except Exception as e:
            logging.error(f"Failed to load XLLM config: {e}")
            raise
    
    @staticmethod
    def get_container_config(config_data: Dict[str, Any], engine: str = "xllm") -> Dict[str, Any]:
        """
        从XLLM配置中获取容器配置
        
        Args:
            config_data: JSON配置
            framework: 框架名称
            
        Returns:
            容器配置字典
        """
        cfg_data = config_data.get(f"{engine.upper()}_CONFIG", {})
        
        # 构建容器配置
        container_config = {
            "engine_type": engine,
            "container_home": cfg_data.get("image_home", 'xllm/xllm-ai'),
            "container_image": cfg_data.get("default_image_name", "latest"),
            "runtime": "docker",
            "gpu_ids": [],  # 默认为空，可根据需要添加
            "ports": {
                "14830": "14830",  # 默认端口，与task.yaml中的node_port对应
                "8000": "8000"     # 额外的API端口
            },
            "working_dir": "/workspace",
            "model_dir": "./models",
            "data_dir": "./data",
            "output_dir": "./outputs",
            "log_dir": "./logs",
            "environment": {
                "XLLM_VERSION": cfg_data.get("xllm_version", "docker"),
                "IMAGE_HOME": cfg_data.get("image_home", "xllm/xllm-ai"),
                "ENABLE_PROXY": cfg_data.get("enable_proxy", "False"),
                "https_proxy": cfg_data.get("https_proxy_url", ""),
                "http_proxy": cfg_data.get("http_proxy_url", ""),
                "no_proxy": cfg_data.get("no_proxy", "localhost,127.0.0.1,0.0.0.0,11.87.0.0/16,::1"),
                "WORKING_DIR": "/workspace"
            },
            "serving_container_name": cfg_data.get("serving_container_name", ""),
            "serving_script_path": cfg_data.get("serving_script_path", ""),
            "startup_command": "",  # 启动命令（根据任务动态生成）
            "startup_timeout": 300,
            "benchmark_command": "",  # 基准测试命令（根据任务动态生成）
            "benchmark_timeout": cfg_data.get("benchmark_timeout", 7200),
            "network_mode": "host",
            "privileged": True,
            "auto_remove": False,  # 默认不自动删除，便于调试
            "detach": True,
            "stdin_open": True,
            "tty": True
        }
        
        # 如果有代理配置，添加到环境变量
        if cfg_data.get("enable_proxy", "False") == "True":
            container_config["environment"].update({
                "https_proxy": cfg_data.get("https_proxy_url", ""),
                "http_proxy": cfg_data.get("http_proxy_url", ""),
            })
        
        logging.info(f"Generated container config for framework: {engine}")
        return container_config


class TaskConfig:
    """任务配置类"""
    
    @staticmethod
    def load_task_config(task_yaml_path: str) -> Dict[str, Any]:
        """
        加载任务YAML配置文件
        
        Args:
            task_yaml_path: 任务YAML文件路径
            
        Returns:
            任务配置字典
        """
        try:
            with open(task_yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                logging.error("Task config must be a YAML mapping: %s", task_yaml_path)
                return {}

            # PD YAML authoring is hierarchical only.  After validation, the
            # topology is normalized to the scheduler's canonical flat runtime
            # representation before any stage inspects deployment.instances.
            from xtask.core.service_group import (
                normalize_pd_deployment,
                validate_pd_deployment_authoring,
            )

            validate_pd_deployment_authoring(config)
            normalize_pd_deployment(config)
            
            logging.debug(f"Loaded task config from: {task_yaml_path}")
            return config if config else {}
        except Exception as e:
            logging.error(f"Failed to load task config: {e}")
            return {}
    
    @staticmethod
    def extract_framework(task_config: Dict[str, Any]) -> str:
        """
        从任务配置中提取框架信息
        
        Args:
            task_config: 任务配置
            
        Returns:
            框架名称
        """
        # 从served_model_name中推断框架
        served_model_name = task_config.get("served_model_name", "").lower()
        
        if "xllm" in served_model_name or "llm" in served_model_name:
            return "xllm"
        elif "tensorrt" in served_model_name or "trt" in served_model_name:
            return "tensorrt"
        elif "pytorch" in served_model_name or "torch" in served_model_name:
            return "pytorch"
        elif "tensorflow" in served_model_name or "tf" in served_model_name:
            return "tensorflow"
        elif "ascend" in served_model_name:
            return "ascend"
        else:
            # 默认为XLLM
            return "xllm"


class ContainerConfigBuilder:
    """容器配置构建器"""
    
    @staticmethod
    def create_container_config(
        engine_config: Dict[str, Any],
        task_config: Dict[str, Any],
        task_yaml_path: str
    ) -> ContainerConfig:
        """
        创建容器配置
        
        Args:
            framework_config: 框架配置
            task_config: 任务配置
            task_yaml_path: 任务YAML文件路径
            
        Returns:
            容器配置
        """
        # 获取容器镜像
        container_home = engine_config.get("container_home", "xllm/xllm-ai")
        container_image = engine_config.get("container_image", "latest")
        
        # 构建容器名称（使用case_name或生成唯一名称）
        case_name = task_config.get("case_name", "")
        if case_name:
            container_name = f"{case_name}-{int(time.time())}"
        else:
            container_name = f"xllm-task-{int(time.time())}"
        
        # 获取工作目录
        work_dir = engine_config.get("working_dir", "/workspace")
        
        # 构建挂载卷
        from xtask.utils.container_manager import ensure_host_bind_mount

        volumes = dict(
            get_xtask_config_data().get("DOCKER_CONFIG").get("volumes") or {}
        )
        from xtask.utils.resource_sync import (
            apply_dependence_package_mount,
            resolve_resource_sync_roots,
        )

        local_source, _remote_target = resolve_resource_sync_roots(task_config)
        apply_dependence_package_mount(volumes, local_source)

        host_paths = [task_config.get("model_path", "")]
        ext_params = task_config.get("serving_extensiable_param") or {}
        if isinstance(ext_params, dict):
            for key in ("draft_model", "draft_model_path"):
                host_paths.append(ext_params.get(key))
        for host_path in host_paths:
            ensure_host_bind_mount(volumes, host_path)
        
        # 构建环境变量
        environment = {
            "TASK_CONFIG_PATH": f"{work_dir}/config/{os.path.basename(task_yaml_path)}",
            "FRAMEWORK": engine_config.get("engine_type", "xllm"),
            "WORKING_DIR": work_dir,
            "MODEL_PATH": task_config.get("model_path", ""),
            "SERVED_MODEL_NAME": task_config.get("served_model_name", ""),
            "START_DEVICE_ID": str(task_config.get("start_device_id", 0)),
            "MAX_MEMORY_UTILIZATION": str(task_config.get("max_memory_utiliization", 0.9)),
            "NODES": str(task_config.get("num_devices", 1)),
            "WORLD_SIZE": str(task_config.get("world_size", 1)),
            "NODE_ADDR": task_config.get("host_addr", "0.0.0.0"),
            "NODE_PORT": str(task_config.get("port", 14830))
        }
        
        # 添加框架配置中的环境变量
        environment.update(engine_config.get("environment", {}))
        
        # 构建容器配置
        container_config = ContainerConfig(
            image_home=container_home,
            image_name=container_image,
            container_name=container_name,
            runtime=engine_config.get("runtime", "docker"),
            gpu_ids=engine_config.get("gpu_ids", []),
            ports=engine_config.get("ports", {}),
            volumes=volumes,
            environment=environment,
            working_dir=work_dir,
            entrypoint=engine_config.get("entrypoint"),
            command=engine_config.get("command"),
            network_mode=engine_config.get("network_mode", "host"),
            ipc_mode=task_config.get("ipc_mode", engine_config.get("ipc_mode")),
            shm_size=task_config.get("shm_size", engine_config.get("shm_size")),
            privileged=engine_config.get("privileged", True),
            auto_remove=engine_config.get("auto_remove", False),
            detach=True,
            stdin_open=engine_config.get("stdin_open", True),
            tty=engine_config.get("tty", True)
        )
        
        logging.info(f"Created container config: {container_name}")
        return container_config


def generate_benchmark_command(
    task_config: Dict[str, Any],
    output_file_path: str,
    service_plans=None,
) -> str:
    """Generate only the benchmark command for an already running service."""
    from xtask.core.external_service import is_external_service_enabled
    from xtask.core.service_group import is_multi_host_deployment

    def _bench_task_config_with_endpoint():
        # Live-target host/port were already applied. Do not derive from
        # leftover deployment.instances when external_service is on.
        if is_external_service_enabled(task_config):
            return task_config
        if not is_multi_host_deployment(task_config):
            return task_config
        from xtask.core.service_group import get_benchmark_endpoint, resolve_master_host

        endpoint = get_benchmark_endpoint(
            task_config,
            service_plans or task_config.get("_service_plans") or [],
        )
        master_host = resolve_master_host(task_config)
        port = task_config.get("port", 8800)
        if "://" in endpoint:
            host_port = endpoint.split("://", 1)[1]
            if ":" in host_port:
                master_host, port_s = host_port.split(":", 1)
                port = int(port_s.split("/")[0])
        bench_cfg = copy.deepcopy(task_config)
        bench_cfg["host_addr"] = master_host
        bench_cfg["port"] = port
        return bench_cfg

    benchmark = task_config.get("benchmark") or {}
    benchmark_type = benchmark.get("benchmark_type")
    if benchmark_type == "aisbench":
        from xtask.benchmarks.aisbench.swebench import is_aisbench_swebench_task

        if is_aisbench_swebench_task(task_config):
            from xtask.benchmarks.aisbench.generate_aisbench_swe_script import (
                generate_aisbench_swe_script,
            )

            return generate_aisbench_swe_script(
                task_config=_bench_task_config_with_endpoint(),
                output_file_path=output_file_path,
            )
        from xtask.benchmarks.aisbench.generate_aisbench_script import generate_aisbench_script

        return generate_aisbench_script(
            task_config=_bench_task_config_with_endpoint(),
            output_file_path=output_file_path,
            output_shell_file="run_aisbench_test.sh",
            python_exec="python",
        )
    if benchmark_type == "vllm_benchmark":
        return ""
    if benchmark_type == "evalscope":
        from xtask.benchmarks.evalscope.generate_evalscope_script import generate_evalscope_script

        return generate_evalscope_script(
            task_config=_bench_task_config_with_endpoint(),
            output_file_path=output_file_path,
            output_shell_file="run_evalscope_test.sh",
            python_exec="python",
        )
    if benchmark_type == "image_edit":
        # gedit image-edit pipeline reuses the EvalScope runtime; the generator
        # module is provided by a parallel workstream, import lazily so the
        # scheduler stays importable before that file exists.
        from xtask.benchmarks.evalscope.generate_image_edit_script import (
            generate_image_edit_script,
        )

        return generate_image_edit_script(
            task_config=_bench_task_config_with_endpoint(),
            output_file_path=output_file_path,
            output_shell_file="run_gedit_perf.sh",
            python_exec="python",
        )
    if benchmark_type == "asr":
        from xtask.benchmarks.asr.generate_asr_script import generate_asr_script

        return generate_asr_script(_bench_task_config_with_endpoint(), output_file_path)
    if benchmark_type == "vbench":
        from xtask.benchmarks.vbench.generate_vbench_script import generate_vbench_script

        return generate_vbench_script(
            task_config=_bench_task_config_with_endpoint(),
            output_file_path=output_file_path,
            output_shell_file="run_vbench_test.sh",
            python_exec="python3",
        )

    logging.error("benchmark_type: %s is not supported yet.", benchmark_type)
    return ""


def generate_commands(
    engine_config: Dict[str, Any], 
    task_config: Dict[str, Any],
    task_yaml_path: str,
    output_file_path: str
) -> Tuple[str, str]:
    """
    生成启动命令和基准测试命令
    
    Args:
        framework_config: 框架配置
        task_config: 任务配置
        task_yaml_path: 任务YAML文件路径
        
    Returns:
        (启动命令, 基准测试命令)
    """
    from xtask.core.external_service import is_external_service_enabled
    from xtask.core.service_group import is_multi_host_deployment

    work_dir = engine_config.get("working_dir", "/workspace")
    config_file = f"{work_dir}/config/{os.path.basename(task_yaml_path)}"
    startup_cmd = ""
    benchmark_cmd = ""

    # External mode must not build xLLM launch plans even if a copied YAML
    # still contains deployment.instances / xllm_service.
    if is_external_service_enabled(task_config):
        benchmark_cmd = generate_benchmark_command(
            task_config,
            output_file_path,
            service_plans=[],
        )
        return startup_cmd, benchmark_cmd

    # gedit self-service mode: services are started by the dedicated
    # gedit lifecycle (xtask/core/gedit_service.py); skip legacy xLLM
    # startup-script generation entirely.
    from xtask.core.gedit_service import is_gedit_services_enabled

    if is_gedit_services_enabled(task_config):
        benchmark_cmd = generate_benchmark_command(
            task_config,
            output_file_path,
            service_plans=[],
        )
        return startup_cmd, benchmark_cmd

    # deployment.instances（含单机单节点）：按实例生成启动脚本
    if is_multi_host_deployment(task_config):
        engine_type = engine_config.get("engine_type")
        if engine_type == "xllm":
            from xtask.core.service_group import (
                build_service_plans,
                sync_top_level_from_plans,
            )
            from xtask.serv_fmk.xllm_serv.generate_xllm_serv import (
                generate_xllm_serv_commands,
            )

            plans = build_service_plans(task_config, output_file_path)
            sync_top_level_from_plans(task_config, plans)
            startup_parts = []
            for plan in plans:
                cmd = generate_xllm_serv_commands(
                    plan.instance_config,
                    engine_config,
                    plan.output_dir,
                    instance={"name": plan.instance_name, "role": plan.role},
                )
                plan.startup_cmd = cmd
                startup_parts.append(f"{plan.instance_name}@{plan.host}:{cmd}")
            startup_cmd = "\n".join(startup_parts)
            # Stash plans on task_config transiently for load stage via return side-channel
            task_config["_service_plans"] = plans
        elif engine_type in ("vllm-ascend", "mindie"):
            # TODO: multi-host service-group startup for non-xLLM engines.
            logging.error(
                "Multi-host deployment is only implemented for xLLM so far; "
                "engine_type=%s is TODO.",
                engine_type,
            )
            startup_cmd = ""
        else:
            logging.error("engine_type: %s is invalid." % engine_type)
    elif engine_config.get("engine_type") == "xllm":
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_serv_commands

        startup_cmd = generate_xllm_serv_commands(task_config, engine_config, output_file_path)

    elif engine_config.get("engine_type") == "vllm-ascend":
        from xtask.serv_fmk.vllm_ascend_serv.generate_vllm_ascend_serv \
            import generate_vllm_ascend_serv_commands

        startup_cmd = generate_vllm_ascend_serv_commands(task_config, engine_config)

    elif engine_config.get("engine_type") == "mindie":
        from xtask.serv_fmk.mindie_serv.generate_mindie_serv \
            import generate_mindie_serv_commands

        startup_cmd = generate_mindie_serv_commands(task_config, engine_config)

    else:
        logging.error("engine_type: %s is invalid."% engine_config.get("engine_type"))
    
    benchmark_cmd = generate_benchmark_command(
        task_config,
        output_file_path,
        service_plans=task_config.get("_service_plans") or [],
    )
    
    return startup_cmd, benchmark_cmd

def find_benchmark_rst_directory(directory_path=".", init_entries=list()):   
    # 日期目录的正则表达式模式 (格式: YYYYMMDD_HHMMSS)
    date_pattern = re.compile(r'^\d{8}_\d{6}$')
    
    # 获取目录下的所有条目
    try:
        entries = os.listdir(directory_path)
    except FileNotFoundError:
        logging.info(f"错误: 目录 '{directory_path}' 不存在")
        return False
    except PermissionError:
        logging.info(f"错误: 没有权限访问目录 '{directory_path}'")
        return False
    
    from xtask.utils.container_manager import path_mapping_in_container

    # 过滤出符合日期格式的目录
    date_dirs = []
    for entry in entries:
        entry_path = os.path.join(directory_path, entry)
        if os.path.isdir(entry_path) and date_pattern.match(entry) and entry not in init_entries:  
            date_dirs.append(path_mapping_in_container(entry_path))
    
    # 如果没有找到日期目录
    if not date_dirs:
        logging.info("未找到日期格式的目录 (格式: YYYYMMDD_HHMMSS)")
    
    return date_dirs

def _should_run_perf_warmup(task_config):
    if _is_asr_task(task_config):
        return False  # ASR has explicit synthetic warmup and exclusion rules.
    benchmark = task_config.get("benchmark") or {}
    perf_cfg = benchmark.get("perf") or {}
    if not perf_cfg.get("warmup_paras"):
        return False
    if benchmark.get("benchmark_type") == "evalscope":
        return True
    if benchmark.get("benchmark_type") == "aisbench":
        from xtask.datasets.dataset_manger import is_custom_dataset_warmup_enabled

        return is_custom_dataset_warmup_enabled(
            task_config.get("dataset_name"), task_config
        )
    return False


def _build_warmup_task_config(task_config):
    """Copy formal perf config and overlay warmup_paras, matching EvalScope."""
    warmup_task_config = copy.deepcopy(task_config)
    warmup_paras = (task_config.get("benchmark") or {}).get("perf", {}).get(
        "warmup_paras", {}
    ) or {}
    warmup_task_config["benchmark"]["perf"].update(warmup_paras)
    warmup_task_config["benchmark"]["perf"].pop("warmup_paras", None)
    return warmup_task_config, warmup_paras


def _build_warmup_command(task_config, output_file_path):
    """构建预热命令：深拷贝配置并合并 warmup 参数"""
    warmup_task_config, warmup_paras = _build_warmup_task_config(task_config)
    benchmark_type = (task_config.get("benchmark") or {}).get("benchmark_type")
    if benchmark_type == "aisbench":
        from xtask.benchmarks.aisbench.generate_aisbench_script import (
            generate_aisbench_script,
        )
        from xtask.datasets.dataset_manger import is_custom_dataset_enabled

        if is_custom_dataset_enabled(
            task_config.get("dataset_name"), task_config
        ) and "num_warmups" not in warmup_paras:
            warmup_task_config["benchmark"]["perf"]["num_warmups"] = 0
        return generate_aisbench_script(
            task_config=warmup_task_config,
            output_file_path=output_file_path,
            output_shell_file="run_aisbench_warmup.sh",
            python_exec="python",
            config_dir_name="aisbench_configs_warmup",
        )

    from xtask.benchmarks.evalscope.generate_evalscope_script import generate_evalscope_script
    return generate_evalscope_script(
        task_config=warmup_task_config,
        output_file_path=output_file_path,
        output_shell_file="run_evalscope_warmup.sh",
        python_exec="python",
    )

def _execute_warmup(container_manager, container_config, engine_config, warmup_cmd):
    """执行预热并记录结果"""
    result = container_manager.exec_in_container(
        container_config.container_name,
        warmup_cmd,
        timeout=engine_config.get("benchmark_timeout", 3600),
    )
    if result.success:
        logging.info("Warmup completed successfully.")
    else:
        logging.error(f"Warmup failed with exit code {result.exit_code}")

def _execute_benchmark_with_npu_guard(container_manager, container_config,
                                      task_config, engine_config, benchmark_cmd,
                                      npu_card_ids=None, service_runtimes=None,
                                      guard_container_manager=None,
                                      guard_container_config=None):
    """执行 benchmark，并在执行期间守护 NPU 进程状态。

    单机时守护当前容器；多机时若传入 service_runtimes，则为每个实例容器
    启动一个 NpuProcessGuard，共享同一个 abort_event。任一 NPU 进程异常时都会
    置位 abort_event，中断 benchmark 执行，返回带失败原因的结果。
    """
    import threading
    from xtask.utils.watchdog import NpuProcessGuard

    guards = []
    abort_event = None

    if service_runtimes:
        abort_event = threading.Event()
        check_interval = engine_config.get("npu_guard_interval", 180.0)
        startup_grace = engine_config.get("npu_guard_startup_grace", 0.0)
        for runtime in service_runtimes:
            guard = NpuProcessGuard(
                npu_card_ids=list(runtime.plan.device_ids),
                abort_event=abort_event,
                check_interval=check_interval,
                startup_grace=startup_grace,
                command_executor=lambda cmd, rt=runtime: (
                    rt.container_manager.exec_in_container(
                        rt.container_config.container_name,
                        cmd,
                        # npu-smi proc-mem 查询：驱动繁忙时 10s 偏紧（与
                        # NpuProcessGuard.NPU_SMI_TIMEOUT 保持一致），放宽到 30s
                        timeout=30,
                        capture_output=True,
                    )
                ),
            )
            guards.append(guard)
            guard.start()
    else:
        if npu_card_ids is None:
            start_device = task_config.get("start_device_id", 0)
            npu_card_ids = [
                (start_device + i) for i in range(0, task_config.get("num_devices", 1))
            ]
        guard_manager = guard_container_manager or container_manager
        guard_config = guard_container_config or container_config
        guard_kwargs = {
            "npu_card_ids": list(npu_card_ids),
            "check_interval": engine_config.get("npu_guard_interval", 180.0),
            "startup_grace": engine_config.get("npu_guard_startup_grace", 0.0),
        }
        if guard_container_manager is not None and guard_container_config is not None:
            guard_kwargs["command_executor"] = lambda cmd: guard_manager.exec_in_container(
                # npu-smi proc-mem 查询：驱动繁忙时 10s 偏紧（与
                # NpuProcessGuard.NPU_SMI_TIMEOUT 保持一致），放宽到 30s
                guard_config.container_name, cmd, timeout=30, capture_output=True
            )
        else:
            guard_kwargs["container_name"] = guard_config.container_name
        guard = NpuProcessGuard(**guard_kwargs)
        guards.append(guard)
        guard.start()
        abort_event = guard.abort_event

    try:
        # Per-case YAML ``benchmark_timeout`` wins; otherwise keep the engine
        # default (xllm_config.json: 7200).  Long VBench mini runs (107 videos)
        # set this explicitly; other cases are unchanged when the key is absent.
        timeout = (task_config or {}).get("benchmark_timeout")
        if timeout in (None, ""):
            timeout = engine_config.get("benchmark_timeout", 3600)
        benchmark_result = container_manager.exec_in_container(
            container_config.container_name,
            benchmark_cmd,
            timeout=int(timeout),
            abort_event=abort_event,
            abort_poll_interval=engine_config.get("npu_guard_abort_poll_interval", 120.0),
        )
    finally:
        for guard in guards:
            guard.stop()

    aborted_reason = None
    for guard in guards:
        if guard.aborted_reason:
            aborted_reason = guard.aborted_reason
            break

    if abort_event and abort_event.is_set() and aborted_reason:
        logging.error(f"benchmark 执行被中断：NPU进程状态异常（{aborted_reason}）")
        if not benchmark_result.stderr:
            benchmark_result.stderr = f"NPU process abnormal: {aborted_reason}"
        else:
            benchmark_result.stderr = f"NPU process abnormal: {aborted_reason}\n{benchmark_result.stderr}"

    return benchmark_result

def _cleanup_warmup_results(container_manager, container_config, output_file_path, init_entries):
    """清理预热阶段产生的结果目录"""
    warmup_rst = find_benchmark_rst_directory(output_file_path, init_entries)
    if not warmup_rst:
        return
    
    clear_cmd = ";".join(f"rm -rf {folder}" for folder in warmup_rst)
    container_manager.exec_in_container(
        container_config.container_name, clear_cmd, timeout=30
    )
 
def _clear_pre_benchmark_rst(container_manager, container_config, task_config, output_file_path):
  
    from xtask.utils.container_manager import path_mapping_in_container

    # 删除上次用例输出目录下的log、用例.json文件
    output_location = path_mapping_in_container(output_file_path)
    result_json = os.path.join(output_location, (task_config.get("case_name").lower() + ".json"))
    xllm_log = os.path.join(output_location, "log")

    process_cmd = f"rm -rf {result_json};rm -rf {xllm_log}"
    container_manager.exec_in_container(
        container_config.container_name, process_cmd, timeout=30
    )


def _make_runtime_path_host_accessible(runtime, runtime_path) -> bool:
    """Allow the invoking host user to archive container-created artifacts."""
    manager = runtime.container_manager
    container_config = getattr(runtime, "container_config", None)
    if container_config is not None and hasattr(manager, "exec_in_container"):
        from xtask.utils.container_manager import path_mapping_in_container

        container_path = path_mapping_in_container(str(runtime_path))
        readable = manager.exec_in_container(
            container_config.container_name,
            f"test ! -e {shlex.quote(container_path)} || "
            f"chmod -R a+rwX {shlex.quote(container_path)}",
            timeout=30,
        )
        if not readable.success:
            logging.warning(
                "Failed to make runtime path host-accessible for %s on %s: %s",
                runtime.plan.instance_name,
                runtime.plan.host,
                readable.stderr or readable.stdout,
            )
            return False
    return True


def _download_remote_runtime_path(runtime, remote_path, local_path) -> bool:
    """Make container-created artifacts host-accessible, then download them."""
    if not _make_runtime_path_host_accessible(runtime, remote_path):
        return False
    if runtime.plan.is_local:
        return True

    manager = runtime.container_manager
    return bool(manager.executor.download(str(remote_path), str(local_path)))


def _archive_service_group_logs(runtimes, output_file_path, benchmarkpath):
    """Archive the latest per-instance logs under the benchmark result directory."""
    archive_root = Path(output_file_path) / benchmarkpath / "xllm_log"

    for runtime in runtimes or []:
        source_log_dir = Path(runtime.plan.output_dir) / "log"
        if runtime.plan.is_local:
            if not _make_runtime_path_host_accessible(runtime, source_log_dir):
                continue
        else:
            try:
                downloaded = _download_remote_runtime_path(
                    runtime,
                    str(source_log_dir), str(source_log_dir)
                )
                if not downloaded:
                    logging.warning(
                        "Failed to refresh logs for %s from %s before archiving",
                        runtime.plan.instance_name,
                        runtime.plan.host,
                    )
            except Exception:
                logging.warning(
                    "Failed to refresh logs for %s from %s before archiving",
                    runtime.plan.instance_name,
                    runtime.plan.host,
                    exc_info=True,
                )

        if not source_log_dir.is_dir():
            logging.warning(
                "Skip missing instance log directory during archive: %s",
                source_log_dir,
            )
            continue

        destination = archive_root / runtime.plan.instance_name / "log"
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_log_dir, destination)


def _copy_archive_entry(source: Path, destination: Path) -> None:
    """Replace one archived file or directory with the latest source."""
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def _copy_non_log_runtime_artifacts(
    source_dir: Path,
    destination_dir: Path,
    *,
    local_overlay_only: bool = False,
    container_manager=None,
    container_config=None,
) -> None:
    """Copy runtime artifacts without replacing the trial-specific ``log/``.

    A reusable service may execute multiple benchmark trials.  Each trial archives
    its own clean ``log/`` immediately after the benchmark, while startup scripts,
    kernel metadata and shutdown logs are service-lifecycle artifacts shared by
    those trials.  They are added after service shutdown without overwriting the
    already archived trial log.

    Host-side copies may still fail on root-owned files (for example
    ``kernel_meta/`` temp dirs).  When a container handle is available, retry the
    individual copy as root inside the container before emitting a warning.
    """
    if not source_dir.is_dir():
        return

    for source in source_dir.iterdir():
        if source.name in {"log", ".log_before_shutdown"}:
            continue
        if local_overlay_only and not (
            source.name == "shutdown_log" or source.is_file() or source.is_symlink()
        ):
            continue
        destination = destination_dir / source.name
        try:
            _copy_archive_entry(source, destination)
            continue
        except Exception:
            if _copy_entry_via_container(
                container_manager, container_config, source, destination
            ):
                continue
            logging.warning(
                "Failed to archive runtime artifact: %s",
                source,
                exc_info=True,
            )


def _archive_service_group_runtime_artifacts(
    runtimes, output_file_path, benchmarkpath
):
    """Complete a service-group archive after shutdown logs have been generated.

    Remote instance directories are downloaded into a temporary location so their
    host-generated artifacts (for example ``kernel_meta/``) are preserved without
    replacing the local live directory.  Locally generated ``shutdown_log/`` and
    startup scripts are then overlaid.  The existing archived ``log/`` is kept so
    every auto-assess trial retains the log captured for that specific trial.
    """
    archive_root = Path(output_file_path) / benchmarkpath / "xllm_log"

    for runtime in runtimes or []:
        plan = getattr(runtime, "plan", None)
        if plan is None or not getattr(plan, "output_dir", None):
            continue
        source_dir = Path(plan.output_dir)
        destination_dir = archive_root / plan.instance_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        try:
            if runtime.plan.is_local:
                if not _make_runtime_path_host_accessible(runtime, source_dir):
                    continue
                _copy_non_log_runtime_artifacts(
                    source_dir,
                    destination_dir,
                    container_manager=runtime.container_manager,
                    container_config=getattr(runtime, "container_config", None),
                )
                continue

            with tempfile.TemporaryDirectory(
                prefix="xtask_runtime_archive_"
            ) as temporary_directory:
                remote_snapshot = (
                    Path(temporary_directory) / runtime.plan.instance_name
                )
                downloaded = _download_remote_runtime_path(
                    runtime,
                    str(source_dir), str(remote_snapshot)
                )
                if downloaded:
                    _copy_non_log_runtime_artifacts(
                        remote_snapshot,
                        destination_dir,
                        container_manager=runtime.container_manager,
                        container_config=getattr(runtime, "container_config", None),
                    )
                else:
                    logging.warning(
                        "Failed to download complete runtime artifacts for %s from %s",
                        runtime.plan.instance_name,
                        runtime.plan.host,
                    )

            # Remote shutdown output is downloaded into the local runtime directory
            # during cleanup and is not uploaded back to the remote host.  Overlay
            # that directory, plus locally generated startup/config files.
            _copy_non_log_runtime_artifacts(
                source_dir,
                destination_dir,
                local_overlay_only=True,
                container_manager=runtime.container_manager,
                container_config=getattr(runtime, "container_config", None),
            )
        except Exception:
            logging.warning(
                "Failed to archive complete runtime artifacts for %s from %s",
                runtime.plan.instance_name,
                runtime.plan.host,
                exc_info=True,
            )


def _record_service_group_archive_path(state, benchmarkpath) -> None:
    """Remember a benchmark directory that needs post-shutdown artifacts."""
    if not state.get("service_runtimes") or not benchmarkpath:
        return
    archive_paths = state.setdefault("service_group_archive_paths", [])
    if benchmarkpath not in archive_paths:
        archive_paths.append(benchmarkpath)
    control_service = state.get("xllm_control_service")
    if getattr(control_service, "is_split", False):
        try:
            control_service.collect_logs()
            control_service.archive_logs(
                state.get("output_file_path") or str(Path(control_service.local_root).parent.parent),
                benchmarkpath,
            )
        except Exception:
            logging.warning("Failed to snapshot this trial's Service logs", exc_info=True)


def _post_process_benchmark_rst(
    container_manager,
    container_config,
    task_config,
    output_file_path,
    benchmarkpath,
    service_runtimes=None,
):
    from xtask.utils.container_manager import path_mapping_in_container

    # benchmark结束后处理，在对应日期目录再放一份benchmark结果和xLLM日志
    output_location = path_mapping_in_container(output_file_path)
    benchmark_location = path_mapping_in_container(
        os.path.join(output_file_path, benchmarkpath)
    )
    xllmlog_location = os.path.join(benchmark_location, "xllm_log")
    result_json = os.path.join(
        output_location, task_config.get("case_name").lower() + ".json"
    )
    archived_result_json = os.path.join(
        benchmark_location, os.path.basename(result_json)
    )
    start_sh = os.path.join(output_location, "start_xllm_serv.sh")
    xllm_log = os.path.join(output_location, "log")

    if service_runtimes and os.path.exists(result_json):
        from xtask.core.service_group import get_benchmark_runtime

        runner = get_benchmark_runtime(service_runtimes)
        if runner is not None and not runner.plan.is_local:
            runner.container_manager.executor.upload(result_json, result_json)

    process_cmd = (
        f"cp -f {result_json} {benchmark_location}; "
        f"chmod a+r {archived_result_json}; "
        f"cp -f {start_sh} {benchmark_location}; "
        f"if [ -d {xllm_log} ]; then "
        f"cp -rf {xllm_log} {xllmlog_location}; "
        f"chmod -R a+rwX {xllmlog_location}; "
        "fi"
    )
    container_manager.exec_in_container(
        container_config.container_name, process_cmd, timeout=30
    )
    _archive_service_group_logs(
        service_runtimes, output_file_path, benchmarkpath
    )


def _find_new_benchmark_date_dir(output_file_path: Path, init_entries):
    """Return the newest date directory the benchmark created during this run.

    A benchmark that started but failed midway may have created its
    ``YYYYMMDD_HHMMSS`` directory without post-processing capturing it; that
    directory is reused for the fallback archive instead of creating another
    one.  ``init_entries`` holds the case directory listing captured before the
    benchmark ran, so directories from previous runs are never reused.
    """
    date_pattern = re.compile(r"^\d{8}_\d{6}$")
    try:
        candidates = [
            entry
            for entry in output_file_path.iterdir()
            if entry.is_dir()
            and date_pattern.match(entry.name)
            and entry.name not in init_entries
        ]
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda entry: entry.name, reverse=True)
    return candidates[0]


def _create_fallback_date_dir(output_file_path: Path, init_entries) -> Optional[Path]:
    """Create a ``YYYYMMDD_HHMMSS`` directory matching the benchmark layout."""
    try:
        output_file_path.mkdir(parents=True, exist_ok=True)
        candidate_time = datetime.now()
        for _ in range(10):
            dirname = candidate_time.strftime("%Y%m%d_%H%M%S")
            date_dir = output_file_path / dirname
            if dirname in init_entries or date_dir.exists():
                candidate_time += timedelta(seconds=1)
                continue
            date_dir.mkdir()
            return date_dir
        logging.warning(
            "Failed to allocate a fallback date directory under %s",
            output_file_path,
        )
    except OSError:
        logging.warning(
            "Failed to create fallback date directory under %s",
            output_file_path,
            exc_info=True,
        )
    return None


def _chown_via_running_container(container_manager, container_config, host_path) -> bool:
    """chown a container-created path to the control-host user before download."""
    try:
        if container_manager is None or container_config is None:
            return False
        if not container_manager.container_is_running(
            container_config.container_name
        ):
            return False
        from xtask.utils.container_manager import path_mapping_in_container

        container_path = path_mapping_in_container(str(host_path))
        uid = os.getuid()
        gid = os.getgid()
        result = container_manager.exec_in_container(
            container_config.container_name,
            f"test ! -e {shlex.quote(container_path)} || "
            f"chown -R {uid}:{gid} {shlex.quote(container_path)}",
            timeout=30,
        )
        return bool(getattr(result, "success", False))
    except Exception:
        logging.debug(
            "chown through container failed for %s", host_path, exc_info=True
        )
        return False


def _chmod_via_running_container(container_manager, container_config, host_path) -> bool:
    """chmod a container-created path only while its container is still running.

    Returns True when the chmod command was issued successfully (or the path
    does not exist yet).  False means no usable running container was available
    or the chmod itself failed.
    """
    try:
        if container_manager is None or container_config is None:
            return False
        if not container_manager.container_is_running(
            container_config.container_name
        ):
            return False
        from xtask.utils.container_manager import path_mapping_in_container

        container_path = path_mapping_in_container(str(host_path))
        result = container_manager.exec_in_container(
            container_config.container_name,
            f"test ! -e {shlex.quote(container_path)} || "
            f"chmod -R a+rwX {shlex.quote(container_path)}",
            timeout=30,
        )
        return bool(getattr(result, "success", False))
    except Exception:
        logging.debug(
            "chmod through container failed for %s", host_path, exc_info=True
        )
        return False


def _iter_running_container_handles(context):
    """Yield (container_manager, container_config) pairs still usable for exec."""
    state = context.state
    seen = set()
    for runtime in state.get("service_runtimes") or []:
        manager = getattr(runtime, "container_manager", None)
        config = getattr(runtime, "container_config", None)
        name = getattr(config, "container_name", None)
        if manager is None or config is None or name in seen:
            continue
        seen.add(name)
        yield manager, config
    manager = state.get("container_manager")
    config = state.get("container_config")
    name = getattr(config, "container_name", None) if config is not None else None
    if manager is not None and config is not None and name not in seen:
        yield manager, config


def _make_path_host_writable(context, host_path: Path) -> bool:
    """Make a container-created path writable by the host user when possible."""
    if os.access(host_path, os.W_OK):
        return True
    for manager, config in _iter_running_container_handles(context):
        if _chmod_via_running_container(manager, config, host_path):
            if os.access(host_path, os.W_OK):
                return True
    return os.access(host_path, os.W_OK)


def _first_running_container_handle(context):
    for manager, config in _iter_running_container_handles(context):
        try:
            if manager.container_is_running(config.container_name):
                return manager, config
        except Exception:
            continue
    return None, None


def _copy_entry_via_container(
    container_manager, container_config, source: Path, destination: Path
) -> bool:
    """Copy one file/directory into a root-owned date dir from inside the container."""
    if container_manager is None or container_config is None or not source.exists():
        return False
    try:
        if not container_manager.container_is_running(
            container_config.container_name
        ):
            return False
        from xtask.utils.container_manager import path_mapping_in_container

        source_path = path_mapping_in_container(str(source))
        destination_path = path_mapping_in_container(str(destination))
        parent_path = path_mapping_in_container(str(destination.parent))
        result = container_manager.exec_in_container(
            container_config.container_name,
            (
                f"test ! -e {shlex.quote(source_path)} || "
                f"(mkdir -p {shlex.quote(parent_path)} && "
                f"rm -rf {shlex.quote(destination_path)} && "
                f"cp -a {shlex.quote(source_path)} {shlex.quote(destination_path)} && "
                f"chmod -R a+rwX {shlex.quote(destination_path)})"
            ),
            timeout=60,
        )
        return bool(getattr(result, "success", False))
    except Exception:
        logging.warning(
            "Failed to archive %s into %s via container",
            source,
            destination,
            exc_info=True,
        )
        return False


def _copy_fallback_entry(
    source: Path,
    destination: Path,
    *,
    container_manager=None,
    container_config=None,
    prefer_container: bool = False,
) -> None:
    if not source.exists():
        return
    if prefer_container and _copy_entry_via_container(
        container_manager, container_config, source, destination
    ):
        return
    try:
        _copy_archive_entry(source, destination)
        return
    except Exception:
        if prefer_container:
            logging.warning(
                "Failed to archive %s into %s", source, destination, exc_info=True
            )
            return
        # Host cannot write into a root-owned aisbench date dir; retry as root
        # inside the still-running container instead of creating another date dir.
        if _copy_entry_via_container(
            container_manager, container_config, source, destination
        ):
            return
        logging.warning(
            "Failed to archive %s into %s", source, destination, exc_info=True
        )


def _archive_fallback_runtime_artifacts(
    runtime,
    date_dir: Path,
    *,
    container_manager=None,
    container_config=None,
    prefer_container: bool = False,
) -> None:
    """Mirror the service-group archive layout for one instance.

    Layout matches ``_archive_service_group_logs`` plus
    ``_archive_service_group_runtime_artifacts``: instance logs land in
    ``<date_dir>/xllm_log/<instance>/log`` and startup scripts, kernel
    metadata and shutdown logs land alongside them.
    """
    source_dir = Path(runtime.plan.output_dir)
    destination_dir = date_dir / "xllm_log" / runtime.plan.instance_name
    copy_manager = container_manager or runtime.container_manager
    copy_config = container_config or getattr(runtime, "container_config", None)

    if runtime.plan.is_local:
        # chmod the whole instance directory so host-side copies of kernel_meta
        # and startup scripts succeed; log-only chmod left root temp files
        # unreadable and produced archive warnings.
        _chmod_via_running_container(
            runtime.container_manager,
            getattr(runtime, "container_config", None),
            source_dir,
        )
    else:
        # Prefer refreshing remote logs while the remote container is still
        # alive (auto-assess trial failures keep the service running).  The
        # normal cleanup path already downloads them before container removal.
        try:
            manager = runtime.container_manager
            runtime_config = getattr(runtime, "container_config", None)
            if (
                manager is not None
                and runtime_config is not None
                and manager.container_is_running(runtime_config.container_name)
            ):
                _download_remote_runtime_path(
                    runtime, str(source_dir / "log"), str(source_dir / "log")
                )
                _make_runtime_path_host_accessible(runtime, source_dir)
        except Exception:
            logging.warning(
                "Failed to refresh logs for %s from %s during fallback archive",
                runtime.plan.instance_name,
                runtime.plan.host,
                exc_info=True,
            )

    _copy_fallback_entry(
        source_dir / "log",
        destination_dir / "log",
        container_manager=copy_manager,
        container_config=copy_config,
        prefer_container=prefer_container,
    )
    _copy_non_log_runtime_artifacts(
        source_dir,
        destination_dir,
        container_manager=copy_manager,
        container_config=copy_config,
    )


def _ensure_benchmark_date_dir(context) -> None:
    """Archive case artifacts into a dated directory the benchmark never finished.

    Prefer reusing a ``YYYYMMDD_HHMMSS`` directory aisbench/evalscope already
    created in this run.  Only create a new date directory when no such
    directory exists.  When the reused directory is root-owned, chmod it for
    host writes while a container is still running; if the host still cannot
    write, copy artifacts from inside the container instead of creating another
    date directory.
    """
    result = context.result
    state = context.state
    if result.get("benchmark_output_path"):
        return

    output_file_path = Path(context.output_file_path)
    init_entries = set(state.get("init_entries") or [])

    date_dir = None
    if state.get("init_entries") is not None:
        date_dir = _find_new_benchmark_date_dir(output_file_path, init_entries)
    if date_dir is None:
        # No benchmark date directory in this run; create one only then.
        date_dir = _create_fallback_date_dir(output_file_path, init_entries)
    if date_dir is None:
        return

    # Always keep the reused/created date dir.  Make it host-writable when
    # possible; otherwise archive through the container as root.
    host_writable = _make_path_host_writable(context, date_dir)
    copy_manager, copy_config = _first_running_container_handle(context)
    prefer_container = not host_writable
    if prefer_container and copy_manager is None:
        logging.warning(
            "Benchmark date directory %s is not writable by the host user and "
            "no running container is available for in-container archive; "
            "reusing it anyway without creating another date directory",
            date_dir,
        )

    runtimes = state.get("service_runtimes") or []
    if runtimes:
        archived_runtime = False
        for runtime in runtimes:
            plan = getattr(runtime, "plan", None)
            if plan is None or not getattr(plan, "output_dir", None):
                continue
            _archive_fallback_runtime_artifacts(
                runtime,
                date_dir,
                container_manager=copy_manager,
                container_config=copy_config,
                prefer_container=prefer_container,
            )
            archived_runtime = True
        # Keep parity with successful benchmark trials: shutdown logs and
        # other service-lifecycle artifacts are overlaid at service shutdown.
        if archived_runtime:
            _record_service_group_archive_path(state, str(date_dir))
    else:
        log_dir = output_file_path / "log"
        _chmod_via_running_container(
            state.get("container_manager"), state.get("container_config"), log_dir
        )
        _copy_fallback_entry(
            log_dir,
            date_dir / "xllm_log",
            container_manager=copy_manager or state.get("container_manager"),
            container_config=copy_config or state.get("container_config"),
            prefer_container=prefer_container,
        )

    task_config = result.get("task_config") or state.get("task_config") or {}
    case_name = str(task_config.get("case_name") or "").lower()
    root_files = [output_file_path / "start_xllm_serv.sh"]
    if case_name:
        root_files.insert(0, output_file_path / f"{case_name}.json")
    for source in root_files:
        _copy_fallback_entry(
            source,
            date_dir / source.name,
            container_manager=copy_manager or state.get("container_manager"),
            container_config=copy_config or state.get("container_config"),
            prefer_container=prefer_container,
        )

    result["benchmark_output_path"] = str(date_dir)
    logging.info(
        "Benchmark date directory missing; archived case artifacts to %s",
        date_dir,
    )


def _sync_asr_trial_inputs(runtime, output_file_path: str) -> bool:
    """Refresh only executable inputs, never remove previous dated trials."""
    root = Path(output_file_path)
    executor = runtime.container_manager.executor
    ready = executor.run(f"mkdir -p {shlex.quote(str(root))}", timeout=30)
    if not ready.success:
        return False
    for source in (root / "run_asr_test.sh", root / "asr_inputs"):
        if not source.exists() or not executor.upload(str(source), str(source)):
            return False
    return True


def _download_asr_runner_results(context, runtime) -> None:
    """Download only this invocation's new date directories, not the case root.

    Replacing the entire case root through HostExecutor.download would delete
    locally generated startup and shutdown evidence from other service nodes.
    """
    from xtask.benchmarks.asr.artifacts import DATE_DIR
    from xtask.utils.container_manager import path_mapping_in_container

    root = Path(context.output_file_path).resolve()
    inside = path_mapping_in_container(str(root))
    code = (
        "import json,pathlib; p=pathlib.Path(" + repr(inside) + "); "
        "print(json.dumps(sorted(x.name for x in p.iterdir() if x.is_dir())))"
    )
    manager, config = runtime.container_manager, runtime.container_config
    listed = manager.exec_in_container(config.container_name, "python3 -c " + shlex.quote(code), timeout=30)
    if not listed.success:
        raise RuntimeError(f"Unable to list ASR results on {runtime.plan.host}")
    names = json.loads(listed.stdout)
    if not isinstance(names, list):
        raise ValueError("Remote ASR result listing must be a list")
    previous = set(context.state.get("init_entries") or [])
    for name in names:
        if not isinstance(name, str) or not DATE_DIR.fullmatch(name) or name in previous:
            continue
        path = root / name
        if not _chown_via_running_container(manager, config, path):
            raise PermissionError(f"Cannot restore ASR archive ownership on {runtime.plan.host}: {name}")
        if not manager.executor.download(str(path), str(path)):
            raise RuntimeError(f"Failed to download ASR date archive {name} from {runtime.plan.host}")


def _archive_asr_execution(context) -> None:
    """Complete this ASR date archive before container removal and at finalize."""
    state, result = context.state, context.result
    task_config = result.get("task_config") or state.get("task_config") or {}
    if not _is_asr_task(task_config):
        return
    from xtask.benchmarks.asr.artifacts import archive_execution, asr_result_post_process
    from xtask.utils import get_host_ip

    root = Path(context.output_file_path).resolve()
    directory = Path(result["benchmark_output_path"]).resolve() if result.get("benchmark_output_path") else None
    if directory is None:
        directory = _find_new_benchmark_date_dir(root, set(state.get("init_entries") or []))
    if directory is None:
        directory = _create_fallback_date_dir(root, set(state.get("init_entries") or []))
    if directory is None:
        raise RuntimeError("Unable to allocate ASR date archive")
    if directory.parent != root:
        raise ValueError("ASR date archive must be a direct child of its case output")
    result["benchmark_output_path"] = str(directory)
    # Container outputs may be root-owned; restore only this run's ownership.
    if not os.access(directory, os.W_OK):
        for manager, config in _iter_running_container_handles(context):
            if _chown_via_running_container(manager, config, directory) and os.access(directory, os.W_OK):
                break
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"ASR date archive is not writable: {directory}")

    if result.get("benchmark_asr_result") is not None:
        from xtask.benchmarks.asr.config import result_issues
        issues = result_issues(result["benchmark_asr_result"], task_config)
        if issues and result.get("success") is True:
            result["success"] = False
            result["asr_gate_issues"] = issues
            result["message"] = "; ".join(issue["reason"] for issue in issues)

    # Results live on the control/client host, not on the external model server.
    if not result.get("artifact_host"):
        result["artifact_host"] = os.getenv("LOCAL_HOST_IP") or get_host_ip()
    result["artifact_path"] = str(directory)
    endpoint = ((task_config.get("benchmark") or {}).get("asr") or {}).get("server_addr")
    if not endpoint and state.get("external_service"):
        from xtask.core.external_service import resolve_external_location
        endpoint = resolve_external_location(task_config).url
    result["service_endpoint"] = endpoint or f"http://{task_config.get('host_addr', '127.0.0.1')}:{task_config.get('port', 8018)}"

    # A timeout/SIGKILL may leave only partial requests. Preserve evidence and
    # recover a failed summary, never synthesize a successful score.
    if result.get("benchmark_asr_result") is None:
        payload, _ = asr_result_post_process(task_config, str(root), [
            item.name for item in root.iterdir() if item.name != directory.name
        ])
        result["benchmark_asr_result"] = payload
        if payload.get("success") is False:
            result["success"] = False

    captured = state.setdefault("asr_log_archives", set())
    if str(directory) not in captured:
        runtimes = state.get("service_runtimes") or []
        if runtimes:
            _archive_service_group_logs(runtimes, str(root), str(directory))
            _record_service_group_archive_path(state, str(directory))
        elif not state.get("external_service"):
            _copy_fallback_entry(root / "log", directory / "xllm_log")
            _copy_fallback_entry(root / "start_xllm_serv.sh", directory / "start_xllm_serv.sh")
        captured.add(str(directory))
    archive_execution(
        directory, root, task_config, result,
        source_yaml=getattr(context, "task_yaml_path", ""),
    )


def _try_archive_asr_execution(context) -> None:
    """Report archive failures without losing the original benchmark evidence."""
    try:
        _archive_asr_execution(context)
    except Exception as exc:
        context.result["archive_error"] = str(exc)
        context.result["success"] = False
        context.result["message"] = f"ASR execution archive failed: {exc}"
        logging.error(context.result["message"], exc_info=True)


def _initial_result() -> Dict[str, Any]:
    return {
        "success": False,
        "scheduler_error": False,
        "message": "",
        "container_id": None,
        "task_config": None,
        "framework_config": None,
        "startup_command": "",
        "benchmark_command": "",
        "execution_time": 0,
        "logs": "",
    }


def _fail_stage(context, message):
    context.result["message"] = message
    context.complete = True
    logging.error(message)


def _load_task_and_engine(context):
    logging.info(f"Loading task config from: {context.task_yaml_path}")
    task_config = TaskConfig.load_task_config(context.task_yaml_path)
    if not task_config:
        _fail_stage(
            context,
            f"Failed to load task config from {context.task_yaml_path}",
        )
        return

    if context.args:
        host_addr_override = getattr(context.args, "host_addr", None)
        if host_addr_override:
            from xtask.core.service_group import apply_host_addr_override

            original_addr = task_config.get("host_addr")
            apply_host_addr_override(task_config, host_addr_override)
            logging.info(
                f"Overriding host_addr from '{original_addr}' to "
                f"'{host_addr_override}'"
            )

    from xtask.core.external_service import (
        apply_external_service_to_task_config,
        is_external_service_enabled,
        topology_fields_ignored,
        validate_external_service_config,
    )

    external_service = is_external_service_enabled(task_config)
    if external_service:
        try:
            validate_external_service_config(task_config)
            apply_external_service_to_task_config(task_config)
        except Exception as exc:
            _fail_stage(context, f"external_service configuration error: {exc}")
            return
        ignored = topology_fields_ignored(task_config)
        if ignored:
            logging.warning(
                "external_service is enabled; ignored topology fields %s "
                "(xLLM / xllm_service will not be started)",
                ", ".join(ignored),
            )

    source_task_config = copy.deepcopy(task_config)
    context.result["task_config"] = task_config
    context.result["output_file_path"] = context.output_file_path
    if _is_asr_task(task_config):
        # Input staging/config errors can occur before ordinary state setup.
        # Cleanup still needs enough identity to create a failed date archive.
        os.makedirs(context.output_file_path, exist_ok=True)
        context.state.update(task_config=task_config, external_service=external_service,
                             init_entries=os.listdir(context.output_file_path))
    engine = TaskConfig.extract_framework(task_config)
    logging.info(f"Detected engine type: {engine}")

    try:
        config_path = os.path.join(os.getcwd(), "config", f"{engine}_config.json")
        config_data = Json_Reader.load_config(config_path)
        engine_config = Json_Reader.get_container_config(config_data, engine)
    except Exception as exc:
        _fail_stage(context, f"Failed to load XLLM config: {exc}")
        return

    # gedit self-service mode: deployment.encoder/dit/judge (image_edit /
    # gedit aisbench acc) starts multiple xLLM services per case.  Build
    # specs now, point benchmark addressing at them, and keep them in state
    # for the container/start/cleanup stages.  Zero impact on other cases:
    # the gate requires a gedit benchmark type AND deployment services AND
    # external_service off.
    gedit_specs = []
    if not external_service:
        from xtask.core.gedit_service import (
            apply_gedit_services_to_task_config,
            build_gedit_service_specs,
            is_gedit_services_enabled,
        )

        if is_gedit_services_enabled(task_config):
            try:
                gedit_specs = build_gedit_service_specs(task_config)
                apply_gedit_services_to_task_config(task_config, gedit_specs)
            except Exception as exc:
                _fail_stage(
                    context,
                    f"deployment gedit services configuration error: {exc}",
                )
                return

    startup_cmd, benchmark_cmd = generate_commands(
        engine_config,
        task_config,
        context.task_yaml_path,
        context.output_file_path,
    )
    context.result["framework_config"] = engine_config
    context.result["startup_command"] = startup_cmd
    context.result["benchmark_command"] = benchmark_cmd

    from xtask.core.service_group import is_multi_host_deployment

    service_plans = task_config.pop("_service_plans", None)
    multi_host = False if external_service else is_multi_host_deployment(task_config)
    context.state.update(
        task_config=task_config,
        source_task_config=source_task_config,
        engine=engine,
        config_data=config_data,
        engine_config=engine_config,
        startup_cmd=startup_cmd,
        benchmark_cmd=benchmark_cmd,
        multi_host=multi_host,
        external_service=external_service,
        service_plans=service_plans or [],
        pd_disaggregated=(
            False
            if external_service
            else (
                task_config.get("deployment", {}).get("mode") == "pd_disaggregated"
                or any(
                    str(instance.get("role", "")).lower() in {"prefill", "decode"}
                    for instance in (task_config.get("deployment", {}).get("instances") or [])
                )
            )
        ),
        service_routed=(
            False
            if external_service
            else (task_config.get("deployment", {}).get("mode") == "service_routed")
        ),
    )
    if external_service:
        context.result["external_service"] = True
    if multi_host and not service_plans:
        _fail_stage(
            context,
            "Multi-host deployment detected but service plans were not generated",
        )
    context.state.update(gedit_specs=gedit_specs)


def _create_container(context, container_manager, container_config):
    logging.info(f"Creating new container: {container_config.container_name}")
    success, container_id = container_manager.create_container()
    if not success:
        _fail_stage(context, f"Failed to create container: {container_id}")
        return False
    context.result["container_id"] = container_id
    return True


def _is_asr_task(task_config):
    from xtask.benchmarks.asr.config import is_asr_task

    return is_asr_task(task_config)


def _is_vbench_task(task_config):
    return str((task_config.get("benchmark") or {}).get("benchmark_type", "")).lower() == "vbench"


def _is_aisbench_swebench_task(task_config):
    from xtask.benchmarks.aisbench.swebench import is_aisbench_swebench_task

    return is_aisbench_swebench_task(task_config)


def _is_swebench_eval_only(task_config):
    from xtask.benchmarks.aisbench.swebench import is_swebench_eval_only

    return is_swebench_eval_only(task_config)


def _vbench_container_config(context, service_manager, service_config):
    """Build an isolated VBench runner config for the benchmark host.

    The xLLM service container remains the owner of NPU processes.  The runner
    only needs HTTP access and the shared workspace/data mounts, so it is kept
    out of ``service_runtimes`` and has its own cleanup handle.
    """
    from xtask.utils.container_manager import path_mapping_in_container
    from xtask.benchmarks.vbench.defaults import (
        get_default_ais_third_party,
        get_default_vbench2_datasets,
        default_cache_dir,
    )

    task_config = context.state["task_config"]
    vbench = (task_config.get("benchmark") or {}).get("vbench") or {}
    output_path = os.path.abspath(context.output_file_path)
    case_name = str(task_config.get("case_name") or "vbench").replace("/", "-")
    image_home = vbench.get("image_home") or vbench.get("container_home") or "quay.io/jd_xllm/xllm-ai"
    image_name = vbench.get("image_name") or vbench.get("container_image") or "8.1.rc1-a3-openeuler22.03-py3.10"
    volumes = dict(get_xtask_config_data().get("DOCKER_CONFIG", {}).get("volumes") or {})
    from xtask.utils.resource_sync import (
        apply_dependence_package_mount,
        resolve_resource_sync_roots,
    )
    from xtask.core.service_group import get_benchmark_runtime

    local_source, remote_target = resolve_resource_sync_roots(task_config)
    runner = get_benchmark_runtime(context.state.get("service_runtimes") or [])
    dependence_host = (
        remote_target
        if runner is not None and not runner.plan.is_local
        else local_source
    )
    apply_dependence_package_mount(volumes, dependence_host)
    # Keep an explicit case-output bind even when the workspace root is also
    # mounted.  This makes the contract clear for remote runner containers.
    output_inside = path_mapping_in_container(output_path)
    volumes[output_path] = output_inside

    # Paths outside the configured model/workspace mounts need an explicit
    # bind.  Do not create missing dataset paths here; the generated script
    # performs the user-facing existence checks.
    config = vbench if isinstance(vbench, dict) else {}
    for key in ("full_info", "dataset_path", "image_root", "cache_dir", "evaluation_script", "evaluate_script"):
        value = config.get(key)
        if not value:
            continue
        normalized_value = os.path.normpath(str(value))
        # Values already expressed in the container namespace (the default
        # evaluation script is one) must remain image/workspace paths.  Treating
        # them as host paths would create a bind mount that hides image files.
        if any(
            normalized_value == prefix or normalized_value.startswith(prefix + "/")
            for prefix in ("/workspace", "/xllm_test_060", "/models", "/outputs")
        ):
            continue
        host_path = os.path.abspath(str(value))
        mapped = path_mapping_in_container(host_path)
        if mapped == host_path and host_path not in volumes:
            volumes[host_path] = host_path

    return ContainerConfig(
        image_home=image_home,
        image_name=image_name,
        container_name=f"{case_name}-vbench-{int(time.time())}",
        runtime=service_config.get("runtime", "docker"),
        gpu_ids=[],
        ports={},
        volumes=volumes,
        environment={
            "VBENCH_CACHE_DIR": path_mapping_in_container(
                str(config.get("cache_dir") or default_cache_dir())
            ),
            "WORKING_DIR": "/workspace",
            # Evaluation-time code/data roots consumed by
            # evaluate_vbench_i2v_bench.py.  Explicit benchmark.vbench
            # env_* values win over these shared defaults.
            "VBENCH2_DATASETS": str(
                config.get("vbench2_datasets") or get_default_vbench2_datasets()
            ),
            "AIS_THIRD_PARTY": str(
                config.get("ais_third_party") or get_default_ais_third_party()
            ),
            # camera_motion / dino go through torch.hub.load, which resolves
            # repos and checkpoints under TORCH_HOME.  Point it at the shared
            # VBench cache so the runner stays fully offline.
            "TORCH_HOME": path_mapping_in_container(
                str(config.get("cache_dir") or default_cache_dir())
            )
            + "/torch",
        },
        working_dir="/workspace",
        network_mode=service_config.get("network_mode", "host"),
        ipc_mode=service_config.get("ipc_mode"),
        shm_size=service_config.get("shm_size"),
        privileged=service_config.get("privileged", True),
        auto_remove=False,
        detach=True,
        stdin_open=True,
        tty=True,
    )


def _prepare_vbench_container(context, service_manager, service_config):
    """Create the optional VBench runner without changing service runtime state."""
    manager = None
    container_created = False
    try:
        vbench_config = _vbench_container_config(context, service_manager, service_config)
        manager = ContainerManager(
            vbench_config,
            logging.getLogger(),
            executor=(
                getattr(service_manager, "executor", None)
                if service_manager is not None
                else None
            ),
        )
        image_url = f"{vbench_config.image_home}:{vbench_config.image_name}"
        if not manager.image_exists(image_url) and not manager.pull_image(image_url):
            _fail_stage(context, f"Failed to pull VBench image {image_url}")
            return
        success, container_id = manager.create_container()
        if not success:
            _fail_stage(context, f"Failed to create VBench runner: {container_id}")
            return
        container_created = True
        if not manager.wait_for_container(vbench_config.container_name, timeout=60):
            _fail_stage(context, "VBench runner failed to become ready")
            return
        context.state.update(
            vbench_container_manager=manager,
            vbench_container_config=vbench_config,
            vbench_container_id=container_id,
        )
        context.result["benchmark_container_id"] = container_id
        logging.info("VBench runner is ready: %s", vbench_config.container_name)
    except Exception as exc:
        _fail_stage(context, f"VBench runner preparation failed: {exc}")
    finally:
        # The scheduler cleanup stage cannot see a manager until preparation
        # succeeds.  Remove a partially created runner here on every failure.
        if manager is not None and container_created and not context.state.get(
            "vbench_container_manager"
        ):
            try:
                manager.cleanup()
            except Exception:
                logging.warning(
                    "Failed to clean up incomplete VBench runner %s",
                    getattr(getattr(manager, "config", None), "container_name", ""),
                    exc_info=True,
                )


def _prepare_external_client_container(context):
    """Start only the benchmark client. Do not create xLLM serving containers.

    AISBench/EvalScope reuse the engine image as a client. VBench uses its
    dedicated runner image and does not keep a dummy serving container.
    SWE-bench infer/eval runs on a dedicated x86 host via AISBench, so it
    does not use the xLLM image as a client.
    """
    state = context.state
    task_config = state["task_config"]
    engine_config = state["engine_config"]
    if _is_aisbench_swebench_task(task_config):
        os.makedirs(context.output_file_path, exist_ok=True)
        state["init_entries"] = os.listdir(context.output_file_path)
        logging.info(
            "SWE-bench: skip AISBench client container; infer/eval run on x86 host"
        )
        return
    if _is_vbench_task(task_config):
        _prepare_vbench_container(context, None, engine_config)
        if context.complete:
            return
        manager = state.get("vbench_container_manager")
        config = state.get("vbench_container_config")
        if manager is None or config is None:
            _fail_stage(
                context,
                "Failed to create VBench runner for external_service",
            )
            return
        state["container_manager"] = manager
        state["container_config"] = config
        context.result["container_id"] = state.get("vbench_container_id")
        state["init_entries"] = os.listdir(context.output_file_path)
        logging.info(
            "external_service: using dedicated VBench runner %s",
            config.container_name,
        )
        return

    _prepare_single_host_client_container(context)


def _prepare_single_host_client_container(context):
    """Create the local client/serving container used by single-host cases.

    Shared by E2E single-host prepare and AISBench/EvalScope external_service.
    """
    state = context.state
    task_config = state["task_config"]
    engine_config = state["engine_config"]
    container_config = ContainerConfigBuilder.create_container_config(
        engine_config, task_config, context.task_yaml_path
    )
    container_manager = ContainerManager(container_config, logging.getLogger())
    state.update(
        container_config=container_config,
        container_manager=container_manager,
    )

    if not container_manager.check_runtime_available():
        _fail_stage(
            context,
            f"Container runtime {container_config.runtime} not available",
        )
        return

    image_url = f"{container_config.image_home}:{container_config.image_name}"
    serving_name = engine_config.get("serving_container_name")
    if serving_name and container_manager.container_exists(serving_name):
        logging.info(f"Container {serving_name} already exists")
        if container_manager.container_is_running(serving_name):
            logging.info(f"Container {serving_name} is already running")
            context.result["container_id"] = serving_name
            container_config.container_name = serving_name
    elif not container_manager.image_exists(image_url):
        logging.info(f"Image {image_url} not found locally, pulling...")
        if not container_manager.pull_image(image_url):
            _fail_stage(context, f"Failed to pull image {image_url}")
            return

        if container_manager.container_exists(container_config.container_name):
            logging.info(f"Container {container_config.container_name} already exists")
            if container_manager.container_is_running(container_config.container_name):
                logging.info(
                    f"Container {container_config.container_name} is already running"
                )
                context.result["container_id"] = container_config.container_name
            else:
                logging.info(
                    f"Starting existing container {container_config.container_name}"
                )
                if container_manager.start_container(container_config.container_name):
                    context.result["container_id"] = container_config.container_name
                else:
                    _fail_stage(
                        context,
                        "Failed to start existing container "
                        f"{container_config.container_name}",
                    )
                    return
        elif not _create_container(context, container_manager, container_config):
            return
    elif not _create_container(context, container_manager, container_config):
        return

    logging.info(
        f"Waiting for container {container_config.container_name} to be ready..."
    )
    if not container_manager.wait_for_container(
        container_config.container_name, timeout=60
    ):
        _fail_stage(
            context,
            f"Container {container_config.container_name} failed to start within timeout",
        )
        return
    state["init_entries"] = os.listdir(context.output_file_path)
    if _is_vbench_task(task_config):
        _prepare_vbench_container(context, container_manager, engine_config)


def _prepare_container(context):
    state = context.state
    if state.get("external_service"):
        _prepare_external_client_container(context)
        return
    if _is_swebench_eval_only(state.get("task_config") or {}):
        os.makedirs(context.output_file_path, exist_ok=True)
        state["init_entries"] = os.listdir(context.output_file_path)
        logging.info("SWE-bench eval-only: skip xLLM serving container")
        return
    if state.get("multi_host"):
        _prepare_container_multi_host(context)
        return
    if state.get("gedit_specs"):
        _prepare_gedit_service_containers(context)
        return

    _prepare_single_host_client_container(context)


def _gedit_service_container_config(
    engine_config: Dict[str, Any],
    task_config: Dict[str, Any],
    task_yaml_path: str,
    spec,
):
    """Build a ContainerConfig for one gedit service container.

    Reuses ContainerConfigBuilder for the image/runtime baseline, then adds
    the NPU driver mounts the gedit xLLM binary needs (aclInit 507008 core
    dump without them) and the per-service device list.
    """
    from xtask.core.gedit_service import DEFAULT_SERVICE_VOLUMES, VENDORS_MOUNT_TARGET
    from xtask.utils.container_manager import ContainerConfig, ensure_host_bind_mount

    base_config = ContainerConfigBuilder.create_container_config(
        engine_config, task_config, task_yaml_path
    )
    volumes = dict(base_config.volumes or {})
    volumes.update(copy.deepcopy(DEFAULT_SERVICE_VOLUMES))
    vendors_source = os.path.join(os.getcwd(), "pkg", "xllm", "vendors", "custom_xllm_math")
    if os.path.isdir(vendors_source):
        volumes[vendors_source] = VENDORS_MOUNT_TARGET
    else:
        raise RuntimeError(
            "gedit services require the xLLM vendors directory "
            f"{vendors_source} (build xLLM or copy pkg/xllm/vendors first)"
        )
    for host_path, container_path in (spec.extra_volumes or {}).items():
        volumes[host_path] = container_path
    ensure_host_bind_mount(volumes, spec.model_path)
    ensure_host_bind_mount(volumes, spec.xllm_binary)

    environment = dict(base_config.environment or {})
    environment.setdefault("GEDIT_SERVICE_NAME", spec.name)

    import time as _time

    container_config = ContainerConfig(
        image_home=base_config.image_home,
        image_name=base_config.image_name,
        container_name=f"{task_config.get('case_name', 'gedit')}-{spec.name}-{int(_time.time())}",
        runtime=base_config.runtime,
        gpu_ids=list(spec.device_ids),
        ports={},
        volumes=volumes,
        environment=environment,
        working_dir=base_config.working_dir,
        entrypoint=base_config.entrypoint,
        command=base_config.command,
        network_mode="host",
        ipc_mode="host",
        shm_size=base_config.shm_size,
        privileged=True,
        auto_remove=False,
        detach=True,
        stdin_open=base_config.stdin_open,
        tty=base_config.tty,
    )
    return container_config


def _prepare_gedit_service_containers(context):
    """Create per-service containers plus one benchmark client container.

    gedit self-service mode (case 36/37): one container per
    deployment.encoder/dit/judge entry runs the xllm binary directly; the
    benchmark client reuses the single-host client container path so script
    generation, dependency download and post-processing stay unchanged.
    """
    from xtask.core.gedit_service import render_gedit_startup_script
    from xtask.utils.container_manager import ContainerManager

    state = context.state
    task_config = state["task_config"]
    engine_config = state["engine_config"]
    specs = state["gedit_specs"]

    # 1. Benchmark client container first (standard single-host path).
    _prepare_single_host_client_container(context)
    if context.complete:
        return
    if not context.result.get("container_id"):
        _fail_stage(context, "gedit: benchmark client container was not created")
        return

    service_runtimes = []
    for spec in specs:
        try:
            container_config = _gedit_service_container_config(
                engine_config, task_config, context.task_yaml_path, spec
            )
        except Exception as exc:
            _fail_stage(
                context,
                f"gedit service {spec.name!r} container config failed: {exc}",
            )
            state["gedit_service_runtimes"] = service_runtimes
            return

        manager = ContainerManager(container_config, logging.getLogger())
        image_url = f"{container_config.image_home}:{container_config.image_name}"
        if not manager.image_exists(image_url):
            logging.info("Image %s not found locally, pulling...", image_url)
            if not manager.pull_image(image_url):
                _fail_stage(
                    context,
                    f"gedit service {spec.name!r}: failed to pull image {image_url}",
                )
                state["gedit_service_runtimes"] = service_runtimes
                return
        success, container_id = manager.create_container()
        if not success:
            _fail_stage(
                context,
                f"gedit service {spec.name!r}: failed to create container: {container_id}",
            )
            state["gedit_service_runtimes"] = service_runtimes
            return
        if not manager.wait_for_container(container_config.container_name, timeout=60):
            _fail_stage(
                context,
                f"gedit service {spec.name!r} container failed to become ready",
            )
            state["gedit_service_runtimes"] = service_runtimes
            return

        output_dir = os.path.join(context.output_file_path, "xllm_log", f"gedit_{spec.name}")
        try:
            script_path = render_gedit_startup_script(spec, output_dir)
        except Exception as exc:
            _fail_stage(
                context,
                f"gedit service {spec.name!r} startup script generation failed: {exc}",
            )
            state["gedit_service_runtimes"] = service_runtimes
            return

        service_runtimes.append(
            {
                "spec": spec,
                "container_name": container_config.container_name,
                "container_id": container_id,
                "manager": manager,
                "config": container_config,
                "output_dir": output_dir,
                "startup_script": script_path,
            }
        )
        logging.info(
            "gedit service %s: container %s ready (devices=%s ports=%s)",
            spec.name,
            container_config.container_name,
            spec.device_ids,
            spec.ports,
        )

    state["gedit_service_runtimes"] = service_runtimes


def _prepare_container_multi_host(context):
    """Create one container per deployment instance (local or SSH)."""
    from xtask.core.service_group import create_runtime_for_plan
    from xtask.utils.resource_sync import sync_required_resources_to_hosts
    from xtask.utils.workspace_sync import sync_workspace_to_hosts  # pyright: ignore[reportMissingImports]

    state = context.state
    engine_config = state["engine_config"]
    task_config = state["task_config"]
    plans = state["service_plans"]
    runtimes = []
    instances_meta = {}

    # Fill required host resources before any container is created, then sync xTask.
    deployment = task_config.get("deployment") or {}
    try:
        state["resource_sync_results"] = sync_required_resources_to_hosts(
            task_config,
            plans,
        )
    except Exception as exc:
        _fail_stage(context, f"Multi-host resource sync failed: {exc}")
        return

    if deployment.get("sync_workspace", True):
        try:
            # VBench output can contain hundreds of MP4 files and model
            # caches. ASR similarly stages large runner-only audio archives.
            # Service workers need the normal workspace, not these datasets.
            extra_paths = []
            if not (_is_vbench_task(task_config) or _is_asr_task(task_config)):
                extra_paths = [context.output_file_path]
            sync_workspace_to_hosts(
                [plan.host for plan in plans],
                extra_paths=extra_paths,
            )
            if _is_vbench_task(task_config) or _is_asr_task(task_config):
                runner_plan = next(
                    (plan for plan in plans if plan.is_benchmark_runner),
                    plans[0] if plans else None,
                )
                if runner_plan is not None:
                    sync_workspace_to_hosts(
                        [runner_plan.host],
                        extra_paths=[context.output_file_path],
                    )
        except Exception as exc:
            _fail_stage(context, f"Multi-host workspace sync failed: {exc}")
            return

    for plan in plans:
        logging.info(
            "Preparing container for instance %s on host %s (local=%s)",
            plan.instance_name,
            plan.host,
            plan.is_local,
        )
        runtime = create_runtime_for_plan(
            plan,
            engine_config,
            context.task_yaml_path,
            ContainerConfigBuilder,
        )
        manager = runtime.container_manager
        if not manager.check_runtime_available():
            _fail_stage(
                context,
                f"Container runtime not available on host {plan.host}",
            )
            state["service_runtimes"] = runtimes
            return

        image_url = (
            f"{runtime.container_config.image_home}:"
            f"{runtime.container_config.image_name}"
        )
        if not manager.image_exists(image_url):
            logging.info("Image %s not found on %s, pulling...", image_url, plan.host)
            if not manager.pull_image(image_url):
                _fail_stage(context, f"Failed to pull image {image_url} on {plan.host}")
                state["service_runtimes"] = runtimes
                return

        try:
            from xtask.utils.hccn_config import prepare_runtime_hccn
            prepare_runtime_hccn(plan, runtime.container_config, manager.executor)
        except (ValueError, OSError) as exc:
            _fail_stage(context, str(exc))
            state["service_runtimes"] = runtimes
            return

        success, container_id = manager.create_container()
        if not success:
            _fail_stage(
                context,
                f"Failed to create container for {plan.instance_name}: {container_id}",
            )
            state["service_runtimes"] = runtimes
            return
        runtime.container_id = container_id
        if not manager.wait_for_container(
            runtime.container_config.container_name, timeout=60
        ):
            _fail_stage(
                context,
                f"Container for {plan.instance_name} failed to become ready",
            )
            state["service_runtimes"] = runtimes
            return

        # 容器就绪、xLLM 未启动：此刻宿主 HBM 即无引擎基线水位。
        # 清理阶段据此判断"释放干净"（见 _wait_npu_hbm_released），
        # 代替对硬件/驱动版本敏感的固定阈值。失败不阻塞（返回 0，
        # 清理时回退环境变量）。
        _sample_npu_hbm_baseline(runtime)

        # 启动前预清理 stale SHM：别的任务（或上次崩溃）残留的
        # xllm_<chip>_* 布局与本用例不同时，xLLM 启动即报
        # "incompatible shared memory layout" 直接崩溃，事后清理救不回
        # 本次启动。容器内以 root 删除（宿主用户删不掉 root 属主对象）；
        # kill_xllm=False：本容器还没有 xllm 进程，且共享 PID 命名空间下
        # pkill 可能误伤别人活着的 xllm 服务；rm 前逐个 fuser 检查，
        # 只删无主对象。
        _cleanup_xllm_shm_in_container(manager, kill_xllm=False)

        runtimes.append(runtime)
        instances_meta[plan.instance_name] = {
            "host": plan.host,
            "role": plan.role,
            "container_id": container_id,
            "container_name": runtime.container_config.container_name,
            "rank_start": plan.rank_start,
            "device_ids": plan.device_ids,
        }

    # Compatibility: keep top-level container_* pointing at benchmark runner.
    from xtask.core.service_group import get_benchmark_runtime

    runner = get_benchmark_runtime(runtimes)
    state["service_runtimes"] = runtimes
    state["container_manager"] = runner.container_manager if runner else None
    state["container_config"] = runner.container_config if runner else None
    if runner:
        context.result["container_id"] = runner.container_id
    context.result["instances"] = instances_meta
    state["init_entries"] = os.listdir(context.output_file_path)
    if _is_vbench_task(task_config) and runner is not None:
        state["vbench_runner_runtime"] = runner
        _prepare_vbench_container(context, runner.container_manager, engine_config)


def _uses_xllm_control_service(task_config):
    deployment = task_config.get("deployment") or {}
    mode = deployment.get("mode")
    if mode in {"pd_disaggregated", "service_routed"}:
        return True
    return any(
        str(instance.get("role", "")).lower() in {"prefill", "decode"}
        for instance in (deployment.get("instances") or [])
    )


def _xllm_control_host(task_config):
    if not _uses_xllm_control_service(task_config):
        return None
    deployment = task_config.get("deployment") or {}
    host, _port = service_etcd_location(task_config)
    return str(host) if host else None


def _download_xllm_command(install_service, task_config=None):
    value = 1 if install_service else 0
    package = ((task_config or {}).get("deployment") or {}).get("xllm_package") or {}
    environment = [f"XLLM_INSTALL_SERVICE={value}"]
    for key, variable in (("release_pack", "XLLM_RELEASE_PACK"), ("etcd_bin_dir", "XLLM_ETCD_BIN_DIR")):
        setting = package.get(key) or os.environ.get(variable)
        if setting:
            environment.append(variable + "=" + shlex.quote(str(setting)))
    return " ".join(environment) + " bash /workspace/tools/download_xllm.sh;"


def _prepare_runtime_dependencies(context):
    state = context.state
    if state.get("external_service"):
        _prepare_external_runtime_dependencies(context)
        return
    if state.get("multi_host"):
        _prepare_runtime_dependencies_multi_host(context)
        return
    if state.get("gedit_specs"):
        # gedit self-service: the client container only needs benchmark tools;
        # services run the standalone xllm binary (no wheel install needed).
        _prepare_external_runtime_dependencies(context)
        return

    task_config = state["task_config"]
    engine_config = state["engine_config"]
    if _is_swebench_eval_only(task_config):
        logging.info("SWE-bench eval-only: skip xLLM/AISBench container dependency prep")
        return
    container_config = state["container_config"]
    container_manager = state["container_manager"]
    timeout = engine_config.get("startup_timeout", 300)

    compilation_cmd = _download_xllm_command(
        _uses_xllm_control_service(task_config), task_config
    )
    logging.info(f"Starting the compilation task: {compilation_cmd}")
    download_result = container_manager.exec_in_container(
        container_config.container_name, compilation_cmd, timeout=timeout
    )
    if not download_result.success:
        detail = download_result.stderr or download_result.stdout or "unknown error"
        _fail_stage(context, f"download_xllm failed: {detail}")
        return

    benchmark_type = task_config.get("benchmark").get("benchmark_type")
    down_benchmark_cmd = ""
    if benchmark_type == "aisbench" and not _is_aisbench_swebench_task(task_config):
        down_benchmark_cmd = "bash /workspace/tools/download_ais_bench.sh;"
    elif benchmark_type == "evalscope":
        down_benchmark_cmd = "bash /workspace/tools/download_evalscope.sh;"
    elif benchmark_type == "image_edit":
        # image_edit reuses the same offline evalscope 1.14.0 wheel closure.
        down_benchmark_cmd = "bash /workspace/tools/download_evalscope.sh;"
    if down_benchmark_cmd:
        logging.info(f"Starting download the benchmark tool: {down_benchmark_cmd}")
        benchmark_download_result = container_manager.exec_in_container(
            container_config.container_name, down_benchmark_cmd, timeout=timeout
        )
        if not benchmark_download_result.success:
            detail = benchmark_download_result.stderr or benchmark_download_result.stdout or "unknown error"
            _fail_stage(context, f"benchmark dependency preparation failed: {detail}")
            return

    model_config_path = os.path.join(task_config.get("model_path"), "config.json")
    if not os.path.exists(model_config_path):
        logging.error(f"model config file: {model_config_path} is not exist")
        context.complete = True
        return

    with open(model_config_path, "r") as config_file:
        transformers_version = json.load(config_file).get("transformers_version")
    # Transformers is a text-LLM dependency; DiT-style multimodal models
    # (Wan2.2 etc.) do not use it and pin no version in config.json.
    # Skip the install instead of forcing an upgrade of the container env.
    if transformers_version:
        install_result = container_manager.exec_in_container(
            container_config.container_name,
            f"pip install transformers=={transformers_version} --upgrade",
            timeout=timeout,
        )
        if not install_result.success:
            logging.error(
                "transformer upgrade to %s failed,stdout:%s,stderr:%s.",
                transformers_version,
                install_result.stdout,
                install_result.stderr,
            )
            install_result = container_manager.exec_in_container(
                container_config.container_name,
                "pip install --upgrade transformers;",
                timeout=timeout,
            )
            if not install_result.success:
                logging.error(
                    "transformer upgrade failed,stdout:%s,stderr:%s.",
                    install_result.stdout,
                    install_result.stderr,
                )

    _clear_pre_benchmark_rst(
        container_manager,
        container_config,
        task_config,
        context.output_file_path,
    )


def _prepare_runtime_dependencies_multi_host(context):
    from xtask.core.ranktable_manager import prepare_multihost_ranktables
    from xtask.core.service_group import get_benchmark_runtime, run_on_runtimes_parallel
    from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_serv_commands
    from xtask.utils.hccl_ranktable import RanktableError

    state = context.state
    task_config = state["task_config"]
    engine_config = state["engine_config"]
    runtimes = state.get("service_runtimes") or []
    plans = state.get("service_plans") or []
    timeout = engine_config.get("startup_timeout", 300)

    control_runtimes = []
    if _uses_xllm_control_service(task_config):
        try:
            control_runtimes = service_container_nodes(task_config, runtimes)
        except (ValueError, TypeError) as exc:
            _fail_stage(context, str(exc))
            return

    # Auto-generate / merge / inject HCCL ranktables before downloading deps.
    # Workspace is already synced and containers are up; scripts may still lack path.
    try:
        injected = prepare_multihost_ranktables(
            task_config,
            plans,
            context.output_file_path,
            runtimes=runtimes,
        )
        if injected:
            logging.info("Auto ranktable groups injected: %s", injected)
            for plan in plans:
                cmd = generate_xllm_serv_commands(
                    plan.instance_config,
                    engine_config,
                    plan.output_dir,
                    instance={"name": plan.instance_name, "role": plan.role},
                )
                plan.startup_cmd = cmd
            state["startup_cmd"] = "\n".join(
                f"{plan.instance_name}@{plan.host}:{plan.startup_cmd}" for plan in plans
            )
    except RanktableError as exc:
        _fail_stage(context, f"Multi-host ranktable preparation failed: {exc}")
        return
    except Exception as exc:
        _fail_stage(context, f"Multi-host ranktable preparation failed: {exc}")
        return

    model_config_path = os.path.join(task_config.get("model_path"), "config.json")
    if not os.path.exists(model_config_path):
        logging.error(f"model config file: {model_config_path} is not exist")
        context.complete = True
        return

    with open(model_config_path, "r") as config_file:
        transformers_version = json.load(config_file).get("transformers_version")

    benchmark_type = task_config.get("benchmark").get("benchmark_type")
    if benchmark_type == "aisbench" and not _is_aisbench_swebench_task(task_config):
        down_benchmark_cmd = "bash /workspace/tools/download_ais_bench.sh;"
    elif benchmark_type == "evalscope":
        down_benchmark_cmd = "bash /workspace/tools/download_evalscope.sh;"
    elif benchmark_type == "image_edit":
        # image_edit reuses the same offline evalscope 1.14.0 wheel closure.
        down_benchmark_cmd = "bash /workspace/tools/download_evalscope.sh;"
    else:
        down_benchmark_cmd = ""

    def _prepare_one(runtime):
        name = runtime.container_config.container_name
        manager = runtime.container_manager
        logging.info(
            "Preparing runtime deps for %s on %s",
            runtime.plan.instance_name,
            runtime.plan.host,
        )
        download_cmd = _download_xllm_command(any(runtime is item for item in control_runtimes), task_config)
        result = manager.exec_in_container(
            name, download_cmd, timeout=timeout
        )
        if not result.success:
            raise RuntimeError(
                f"download_xllm failed on {runtime.plan.host}: {result.stderr}"
            )
        if down_benchmark_cmd and runtime.plan.is_benchmark_runner:
            result = manager.exec_in_container(
                name, down_benchmark_cmd, timeout=timeout
            )
            if not result.success:
                raise RuntimeError(
                    f"download benchmark failed on {runtime.plan.host}: {result.stderr}"
                )
        # Same rationale as the single-host path: DiT-style multimodal models
        # pin no transformers version; skip rather than upgrade the env.
        if transformers_version:
            install_result = manager.exec_in_container(
                name,
                f"pip install transformers=={transformers_version} --upgrade",
                timeout=timeout,
            )
            if not install_result.success:
                manager.exec_in_container(
                    name, "pip install --upgrade transformers;", timeout=timeout
                )
        else:
            # DiT-style multimodal models pin no transformers version; skipping
            # keeps the image's preinstalled version intact.  Log so a missing
            # field is visible instead of silently doing nothing.
            logging.info(
                "No transformers_version in %s; skip transformers install",
                model_config_path,
            )
        return True

    # Containers on one physical host share the bind-mounted /workspace. Run
    # dependency preparation serially per host so download/extract operations
    # cannot race, while different hosts still prepare in parallel.
    runtimes_by_host = {}
    for runtime in runtimes:
        runtimes_by_host.setdefault(str(runtime.plan.host), []).append(runtime)

    host_representatives = [members[0] for members in runtimes_by_host.values()]

    def _prepare_host(representative):
        for runtime in runtimes_by_host[str(representative.plan.host)]:
            try:
                _prepare_one(runtime)
            except Exception as exc:
                raise RuntimeError(
                    f"{runtime.plan.instance_name}: {exc}"
                ) from exc
        return True

    outcomes = run_on_runtimes_parallel(host_representatives, _prepare_host)
    for runtime, _ok, exc in outcomes:
        if exc is not None:
            _fail_stage(
                context,
                f"Runtime dependency prep failed for {runtime.plan.instance_name}: {exc}",
            )
            return

    runner = get_benchmark_runtime(runtimes)
    if runner:
        _clear_pre_benchmark_rst(
            runner.container_manager,
            runner.container_config,
            task_config,
            context.output_file_path,
        )


def _prepare_external_runtime_dependencies(context):
    """Install benchmark tools only. Never download/build xLLM in this mode."""
    state = context.state
    task_config = state["task_config"]
    engine_config = state["engine_config"]
    if _is_aisbench_swebench_task(task_config):
        logging.info("SWE-bench: skip AISBench install in the xLLM/client container")
        return
    container_config = state.get("container_config")
    container_manager = state.get("container_manager")
    if container_manager is None or container_config is None:
        _fail_stage(context, "external_service has no client container")
        return
    timeout = engine_config.get("startup_timeout", 300)
    benchmark_type = (task_config.get("benchmark") or {}).get("benchmark_type")
    down_benchmark_cmd = ""
    if benchmark_type == "aisbench":
        down_benchmark_cmd = "bash /workspace/tools/download_ais_bench.sh;"
    elif benchmark_type == "evalscope":
        down_benchmark_cmd = "bash /workspace/tools/download_evalscope.sh;"
    elif benchmark_type == "image_edit":
        # image_edit reuses the same offline evalscope 1.14.0 wheel closure.
        down_benchmark_cmd = "bash /workspace/tools/download_evalscope.sh;"
    if down_benchmark_cmd:
        logging.info(
            "external_service: preparing benchmark tools: %s", down_benchmark_cmd
        )
        benchmark_download_result = container_manager.exec_in_container(
            container_config.container_name, down_benchmark_cmd, timeout=timeout
        )
        if not benchmark_download_result.success:
            detail = (
                benchmark_download_result.stderr
                or benchmark_download_result.stdout
                or "unknown error"
            )
            _fail_stage(context, f"benchmark dependency preparation failed: {detail}")
            return
    if task_config.get("case_name"):
        _clear_pre_benchmark_rst(
            container_manager,
            container_config,
            task_config,
            context.output_file_path,
        )


def _http_status_is_success(stdout: str) -> bool:
    code = (stdout or "").strip()
    return len(code) >= 3 and code[:3].isdigit() and code.startswith("2")


def _probe_external_service(context) -> bool:
    """Health-check a caller-provided live endpoint. Never start xLLM."""
    from xtask.core.external_service import (
        is_vbench_evaluate_only,
        resolve_external_origin,
        resolve_health_path,
    )
    from xtask.core.service_group import get_benchmark_endpoint

    state = context.state
    task_config = state["task_config"]
    if is_vbench_evaluate_only(task_config) or _is_swebench_eval_only(task_config):
        logging.info(
            "external_service: evaluate-only SWE/VBench task, skip live endpoint probe"
        )
        state["service_healthy"] = True
        return True

    health_path = resolve_health_path(task_config)
    if health_path is None:
        logging.info("external_service: health_path is empty, skip probe")
        state["service_healthy"] = True
        return True

    manager = state.get("container_manager")
    container_config = state.get("container_config")
    if manager is None or container_config is None:
        if _is_aisbench_swebench_task(task_config):
            from xtask.utils.host_executor import LocalHostExecutor

            try:
                origin = get_benchmark_endpoint(
                    task_config, state.get("service_plans") or []
                )
            except Exception:
                origin = resolve_external_origin(task_config)
            check_url = origin.rstrip("/") + health_path
            check_cmd = (
                f"curl --noproxy '*' -sS -o /dev/null -w '%{{http_code}}' "
                f"--max-time 10 {shlex.quote(check_url)}"
            )
            logging.info(
                "SWE-bench: probing live endpoint from scheduler host %s", check_url
            )
            health_result = LocalHostExecutor().run(check_cmd, timeout=30)
            state["health_result"] = health_result
            healthy = bool(
                health_result.success and _http_status_is_success(health_result.stdout)
            )
            state["service_healthy"] = healthy
            if healthy:
                logging.info("SWE-bench endpoint is healthy: %s", check_url)
                return True
            message = (
                f"SWE-bench health check failed for {check_url}: "
                f"{health_result.stderr or health_result.stdout}"
            )
            context.result["message"] = message
            logging.error(message)
            return False
        _fail_stage(context, "external_service has no client container for health probe")
        state["service_healthy"] = False
        return False

    try:
        origin = get_benchmark_endpoint(task_config, state.get("service_plans") or [])
    except Exception:
        origin = resolve_external_origin(task_config)
    check_url = origin.rstrip("/") + health_path
    check_cmd = (
        f"curl --noproxy '*' -sS -o /dev/null -w '%{{http_code}}' "
        f"--max-time 10 {shlex.quote(check_url)}"
    )
    logging.info("external_service: probing live endpoint %s", check_url)
    health_result = manager.exec_in_container(
        container_config.container_name, check_cmd, timeout=30
    )
    state["health_result"] = health_result
    healthy = bool(
        health_result.success and _http_status_is_success(health_result.stdout)
    )
    state["service_healthy"] = healthy
    if healthy:
        logging.info("external_service endpoint is healthy: %s", check_url)
        return True
    message = (
        f"external_service health check failed for {check_url}: "
        f"{health_result.stderr or health_result.stdout}"
    )
    context.result["message"] = message
    logging.error(message)
    return False


def _cleanup_external_client(context):
    """Remove only the client/VBench runner. Never stop the live service."""
    state = context.state
    if _is_asr_task(state.get("task_config") or {}):
        _try_archive_asr_execution(context)
    try:
        _ensure_benchmark_date_dir(context)
    except Exception:
        logging.error(
            "Failed to prepare fallback result directory during cleanup",
            exc_info=True,
        )
    engine_config = state.get("engine_config")
    should_clean = bool(
        engine_config is not None
        and not engine_config.get("serving_container_name")
        and get_xtask_config_data().get("ENV_CONFIG").get("CLEAN_RESULTS")
    )
    if not should_clean:
        return
    vbench_mgr = state.get("vbench_container_manager")
    container_manager = state.get("container_manager")
    cleaned = set()
    for manager in (vbench_mgr, container_manager):
        if manager is None or id(manager) in cleaned:
            continue
        try:
            manager.cleanup()
        except Exception:
            logging.error("Failed to clean external_service client container", exc_info=True)
        cleaned.add(id(manager))


def _cleanup_gedit_services(context):
    """Stop gedit service containers plus the benchmark client container.

    Archives service logs into the case output's xllm_log/gedit_<name>/log
    before removal (mirrors the service-group archive layout), waits for the
    xllm processes to exit, then removes the containers.  CLEAN_RESULTS=false
    keeps everything alive for debugging, matching the other paths.
    """
    from xtask.utils.container_manager import path_mapping_in_container

    state = context.state
    try:
        _ensure_benchmark_date_dir(context)
    except Exception:
        logging.error(
            "Failed to prepare fallback result directory during cleanup",
            exc_info=True,
        )

    engine_config = state.get("engine_config")
    should_clean = bool(
        engine_config is not None
        and not engine_config.get("serving_container_name")
        and get_xtask_config_data().get("ENV_CONFIG").get("CLEAN_RESULTS")
    )

    service_runtimes = state.get("gedit_service_runtimes") or []
    for entry in service_runtimes:
        manager = entry.get("manager")
        config = entry.get("config")
        spec = entry.get("spec")
        if manager is None or config is None:
            continue
        try:
            if not manager.container_is_running(config.container_name):
                continue
        except Exception:
            logging.warning(
                "gedit service %s: cannot probe container state; proceeding "
                "to cleanup",
                spec.name if spec else "?",
                exc_info=True,
            )
        # Archive logs while the container is still alive.
        try:
            log_source = path_mapping_in_container(
                os.path.join(entry["output_dir"], "log")
            )
            manager.exec_in_container(
                config.container_name,
                f"chmod -R a+rwX {shlex.quote(log_source)} 2>/dev/null || true",
                timeout=30,
            )
        except Exception:
            logging.warning(
                "gedit service %s: log chmod failed before archive",
                spec.name if spec else "?",
                exc_info=True,
            )
        if should_clean:
            try:
                manager.exec_in_container(
                    config.container_name,
                    "pkill -TERM -x xllm 2>/dev/null; sleep 1; "
                    "pkill -KILL -x xllm 2>/dev/null; true",
                    timeout=30,
                )
            except Exception:
                logging.warning(
                    "gedit service %s: xllm terminate failed",
                    spec.name if spec else "?",
                    exc_info=True,
                )
            try:
                manager.exec_in_container(
                    config.container_name,
                    "rm -f /dev/shm/xllm_* 2>/dev/null; true",
                    timeout=30,
                )
            except Exception:
                logging.warning(
                    "gedit service %s: SHM cleanup failed",
                    spec.name if spec else "?",
                    exc_info=True,
                )
        try:
            manager.cleanup()
        except Exception:
            logging.error(
                "gedit service %s: container cleanup failed",
                spec.name if spec else "?",
                exc_info=True,
            )

    if should_clean:
        container_manager = state.get("container_manager")
        if container_manager is not None:
            try:
                container_manager.cleanup()
            except Exception:
                logging.error(
                    "gedit client container cleanup failed", exc_info=True
                )


def _start_and_wait_service(context):
    state = context.state
    if state.get("external_service"):
        _probe_external_service(context)
        return
    if _is_swebench_eval_only(state.get("task_config") or {}):
        logging.info("SWE-bench eval-only: skip xLLM startup")
        state["service_healthy"] = True
        return
    if state.get("multi_host"):
        _start_and_wait_service_multi_host(context)
        return
    if state.get("gedit_specs"):
        _start_and_wait_gedit_services(context)
        return

    task_config = state["task_config"]
    engine_config = state["engine_config"]
    container_config = state["container_config"]
    container_manager = state["container_manager"]
    startup_cmd = state["startup_cmd"]

    logging.info(f"Executing startup command: {startup_cmd}")
    startup_result = container_manager.exec_in_container(
        container_config.container_name,
        startup_cmd,
        timeout=engine_config.get("startup_timeout", 300),
    )
    state["startup_result"] = startup_result
    if not startup_result.success:
        context.result["message"] = f"Startup command failed: {startup_result.stderr}"
        logging.error(context.result["message"])
    else:
        logging.debug("service startup message is: %s", {startup_result.stderr})

    logging.info("Waiting for service to start...")
    serving_wait_time = (
        task_config.get("serving_wait_time")
        if task_config.get("serving_wait_time")
        else state["config_data"].get("XLLM_CONFIG").get("serving_wait_time")
    )
    time.sleep(serving_wait_time)
    if wait_serv_start(
        state["engine"], task_config, serving_wait_time, context.output_file_path
    ):
        logging.error(f"{state['engine']} serv startup failed.")

    check_cmd = (
        f"curl -i http://{task_config.get('host_addr', '0.0.0.0')}:"
        f"{task_config.get('port', 14830)}/health"
    )
    health_result = container_manager.exec_in_container(
        container_config.container_name, check_cmd, timeout=30
    )
    state["health_result"] = health_result
    state["service_healthy"] = (
        health_result.success and "ok" in health_result.stdout.lower()
    )
    if state["service_healthy"]:
        logging.info("Service is healthy and running")
    else:
        context.result["message"] = (
            f"Service health check failed: {health_result.stderr}"
        )
        logging.error(context.result["message"])


def _start_and_wait_gedit_services(context):
    """Start every gedit service container and wait for HTTP health.

    Each service's script is executed inside its own container (host network),
    then the scheduler polls each service's /health endpoint from the
    benchmark client container until all report 200.  DiT (multi-rank) opens
    its master port last, so the wait covers the slowest service.
    """
    from xtask.core.gedit_service import gedit_health_url
    from xtask.utils.container_manager import path_mapping_in_container

    state = context.state
    service_runtimes = state.get("gedit_service_runtimes") or []
    client_config = state.get("container_config")
    client_manager = state.get("container_manager")
    if not service_runtimes:
        _fail_stage(context, "gedit: no service runtimes were prepared")
        state["service_healthy"] = False
        return
    if client_manager is None or client_config is None:
        _fail_stage(context, "gedit: no benchmark client container for health probe")
        state["service_healthy"] = False
        return

    # ---- 1. Execute every startup script ----
    for entry in service_runtimes:
        spec = entry["spec"]
        manager = entry["manager"]
        container_name = entry["container_name"]
        script_dir = path_mapping_in_container(entry["output_dir"])
        script_name = os.path.basename(entry["startup_script"])
        command = (
            f"rm -rf {script_dir}/log; "
            f"cd {script_dir}; bash ./{script_name}"
        )
        logging.info(
            "gedit service %s: starting with command: %s", spec.name, command
        )
        result = manager.exec_in_container(
            container_name, command, timeout=120
        )
        entry["startup_result"] = result
        if not result.success:
            detail = (
                f"gedit service {spec.name!r} startup command failed: "
                f"{result.stderr or result.stdout}"
            )
            context.result["message"] = detail
            logging.error(detail)
            state["startup_result"] = result
            state["service_healthy"] = False
            return

    state["startup_result"] = service_runtimes[0].get("startup_result")

    # ---- 2. Poll each service health endpoint from the client container ----
    serving_wait_time = (
        (state.get("task_config") or {}).get("serving_wait_time")
        or (state.get("config_data") or {}).get("XLLM_CONFIG", {}).get(
            "serving_wait_time", 30
        )
    )
    try:
        time.sleep(int(serving_wait_time))
    except (TypeError, ValueError):
        time.sleep(30)

    def _probe_health(url: str) -> bool:
        check_cmd = (
            f"curl --noproxy '*' -s -o /dev/null -w '%{{http_code}}' "
            f"--max-time 10 {shlex.quote(url)}"
        )
        probe = client_manager.exec_in_container(
            client_config.container_name, check_cmd, timeout=30
        )
        return bool(probe.success and _http_status_is_success(probe.stdout))

    all_healthy = True
    health_results = {}
    for entry in service_runtimes:
        spec = entry["spec"]
        url = gedit_health_url(spec)
        timeout_s = int(spec.readiness_timeout or 900)
        deadline = time.monotonic() + timeout_s
        healthy = False
        while time.monotonic() < deadline:
            if _probe_health(url):
                healthy = True
                break
            # Early-abort when the xllm process died (log grows then stops).
            log_dir = path_mapping_in_container(
                os.path.join(entry["output_dir"], "log")
            )
            alive = entry["manager"].exec_in_container(
                entry["container_name"],
                "pgrep -x xllm >/dev/null || pgrep -f '/[x]llm' >/dev/null",
                timeout=30,
            )
            if not alive.success:
                logging.error(
                    "gedit service %s: xllm process is not running; aborting "
                    "health wait (see %s)",
                    spec.name,
                    os.path.join(entry["output_dir"], "log"),
                )
                break
            time.sleep(10)
        health_results[spec.name] = healthy
        if healthy:
            logging.info("gedit service %s is healthy: %s", spec.name, url)
        else:
            all_healthy = False
            message = (
                f"gedit service {spec.name!r} health check failed for {url} "
                f"within {timeout_s}s; check container "
                f"{entry['container_name']} logs under {entry['output_dir']}/log"
            )
            context.result["message"] = message
            logging.error(message)

    state["health_result"] = health_results
    state["service_healthy"] = all_healthy
    if all_healthy:
        logging.info("All gedit services are healthy")


def _wait_instance_ready(runtime, serving_wait_time, engine):
    """Wait for one multi-host node; only an instance master owns service readiness."""
    from xtask.serv_fmk.xllm_serv.generate_xllm_serv import wait_xllm_serv_start

    plan = runtime.plan
    readiness = plan.readiness or {}
    role = str(plan.role).lower()
    log_file = os.path.join(plan.output_dir, "log", "node_0.log")

    if readiness.get("type") == "command":
        cmd = readiness.get("command") or "pgrep -f xllm"
        result = runtime.container_manager.exec_in_container(
            runtime.container_config.container_name, cmd, timeout=30
        )
        return result.success

    if readiness.get("type") == "http":
        url = readiness.get("url")
        result = runtime.container_manager.exec_in_container(
            runtime.container_config.container_name,
            f"curl -s -o /dev/null -w '%{{http_code}}' {url}",
            timeout=30,
        )
        return result.success and "200" in (result.stdout or "")

    keywords = readiness.get("success_keywords")
    if keywords is None:
        is_pd_instance = bool(plan.instance_config.get("enable_disagg_pd"))
        is_routed_instance = bool(
            plan.instance_config.get("enable_service_routing")
        )
        # A routed logical group's service owner is its rank-0 runtime. Host
        # comparison is insufficient because multiple nodes may share one host.
        is_grouped_service = is_pd_instance or is_routed_instance
        is_group_master = not is_grouped_service or plan.rank_start == 0
        if is_grouped_service and not is_group_master:
            # A multi-node logical instance registers once from its group
            # master. Other nodes are distributed workers and never emit the
            # service registration/BRPC readiness keywords.
            result = runtime.container_manager.exec_in_container(
                runtime.container_config.container_name,
                "pgrep -f '[x]llm'",
                timeout=30,
            )
            return result.success
        if role in {"master", "standalone", "prefill"} or is_grouped_service:
            keywords = [
                "Brpc Server started on port",
                "Success register instance to etcd",
                "服务启动完成",
            ]
        else:
            # A regular multi-host worker has no public HTTP endpoint, but its
            # rank-0 log provides a stable readiness signal.  A one-shot pgrep
            # can miss the launcher-to-binary transition and falsely fail an
            # otherwise healthy worker.
            keywords = ["DistributeWorker server started"]

    if engine == "xllm":
        poll_in_container = not plan.is_local or getattr(
            plan, "isolate_ipc", False
        )
        if poll_in_container:
            from xtask.utils.container_manager import path_mapping_in_container

            keyword_pattern = "|".join(keywords)
            # Colocated local runtimes also poll inside their own containers.
            # Concurrent host-side watchdog processes share global state and can
            # miss readiness events from sibling runtimes.
            container_log_file = path_mapping_in_container(
                os.path.join(plan.output_dir, "log", "node_0.log")
            )
            ready_timeout = int(
                plan.instance_config.get("serving_ready_timeout")
                or plan.instance_config.get("serving_timeout")
                or 320
            )
            # The console-script launcher replaces itself with the packaged
            # binary, so the process may be reported as either ``xllm`` or by
            # its site-packages path.  Give the launcher a short window to
            # create its log before treating a missing process as a failure.
            process_probe = (
                "(pgrep -x xllm >/dev/null || "
                "pgrep -f '/[x]llm/[x]llm' >/dev/null || "
                "pgrep -f '/bin/[x]llm' >/dev/null)"
            )
            wait_body = (
                "for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do "
                f"grep -Eq {shlex.quote(keyword_pattern)} "
                f"{shlex.quote(container_log_file)} 2>/dev/null && exit 0; "
                "sleep 1; done; "
                f"while ! grep -Eq {shlex.quote(keyword_pattern)} "
                f"{shlex.quote(container_log_file)} 2>/dev/null; do "
                f"{process_probe} || exit 1; sleep 1; done"
            )
            command = wait_body
            result = runtime.container_manager.exec_in_container(
                runtime.container_config.container_name,
                command,
                timeout=ready_timeout + 30,
            )
            if not result.success:
                logging.error(
                    "Readiness command failed for instance %s on %s: "
                    "exit_code=%s, stdout=%s, stderr=%s",
                    plan.instance_name,
                    plan.host,
                    getattr(result, "exit_code", None),
                    (getattr(result, "stdout", "") or "").strip(),
                    (getattr(result, "stderr", "") or "").strip(),
                )
            return result.success
        wait_xllm_serv_start(
            plan.instance_config,
            serving_wait_time,
            plan.output_dir,
            log_file=log_file,
            success_keywords=keywords,
        )
        return True
    # TODO: non-xLLM multi-host readiness
    logging.warning("Multi-host readiness for engine=%s is TODO", engine)
    return True


def _pd_listen_ports(plan) -> list:
    """Ports this PD instance will bind on its host.

    Derived from the merged launch plan (the same values the startup script
    uses), not hardcoded defaults. Empty when the runtime is not a PD
    data-plane instance so non-PD callers are a no-op.
    """
    if plan is None:
        return []
    cfg = getattr(plan, "instance_config", None)
    if not isinstance(cfg, dict):
        cfg = {}
    role = str(getattr(plan, "role", "") or "").lower()
    if not cfg.get("enable_disagg_pd") and role not in {"prefill", "decode"}:
        return []

    ports = []

    def _add_port(raw):
        try:
            port = int(raw)
        except (TypeError, ValueError):
            return
        if 1 <= port <= 65535:
            ports.append(port)

    disagg = cfg.get("disagg_pd_port")
    if disagg is None:
        ext = cfg.get("serving_extensiable_param") or {}
        if isinstance(ext, dict):
            disagg = ext.get("disagg_pd_port")
    if disagg is None:
        disagg = 7777
    _add_port(disagg)

    try:
        transfer_base = int(cfg.get("transfer_port_base", 26000) or 26000)
    except (TypeError, ValueError):
        transfer_base = 26000
    raw_rank = getattr(plan, "rank_start", None)
    if raw_rank is None:
        raw_rank = cfg.get("rank_start", 0)
    try:
        rank_start = int(raw_rank)
    except (TypeError, ValueError):
        rank_start = 0
    devices = getattr(plan, "device_ids", None)
    if isinstance(devices, (list, tuple)) and devices:
        n_ranks = len(devices)
    else:
        n_ranks = 1
    for offset in range(n_ranks):
        _add_port(transfer_base + rank_start + offset)
    return sorted(set(ports))


_LISTEN_PORT_PROBE_PATH = "PATH=/usr/sbin:/usr/bin:/sbin:/bin"


def _pd_listen_ports_probe_cmd(ports) -> str:
    """Host-side read-only LISTEN probe for the given TCP ports."""
    want = ",".join(str(int(port)) for port in ports)
    path = _LISTEN_PORT_PROBE_PATH
    return (
        f"{path} command -v ss >/dev/null 2>&1 || {{ echo UNREADABLE; exit 0; }}; "
        f"{path} ss -lnt 2>/dev/null | awk -v want="
        + shlex.quote(want)
        + r""" '
BEGIN { n=split(want, a, ","); for (i = 1; i <= n; i++) w[a[i]] = 1 }
$1 ~ /LISTEN/ {
  nfields = split($4, p, ":")
  port = p[nfields]
  gsub(/[^0-9]/, "", port)
  if (port != "" && port in w && !seen[port]++) print port
}'"""
    )


def _read_pd_listen_ports_in_use(runtime, ports) -> Optional[list]:
    """Return LISTEN subset of ``ports``. None means the probe is unusable."""
    if not ports:
        return []
    host = getattr(getattr(runtime, "plan", None), "host", "?")
    cmd = _pd_listen_ports_probe_cmd(ports)
    try:
        result = runtime.container_manager.executor.run(cmd, timeout=20)
    except Exception as exc:
        logging.warning("PD listen-port probe error on %s: %s", host, exc)
        return None
    out = getattr(result, "stdout", None) or ""
    if not isinstance(out, str):
        out = ""
    out = out.strip()
    if "UNREADABLE" in out:
        stderr = getattr(result, "stderr", None) or ""
        if not isinstance(stderr, str):
            stderr = ""
        logging.debug(
            "Listen-port probe unreadable on %s: stdout=%r stderr=%r exit_code=%s",
            host,
            out,
            stderr.strip(),
            getattr(result, "exit_code", None),
        )
        return None
    busy = []
    wanted = {int(port) for port in ports}
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            port = int(line)
            if port in wanted:
                busy.append(port)
    return sorted(set(busy))


def _wait_pd_listen_ports_released(runtime, timeout_s=None, interval_s=1) -> None:
    """Wait until this PD instance's DisaggPD/Mooncake ports leave LISTEN.

    Timeout (default 30s, XTASK_PD_PORT_RELEASE_TIMEOUT) logs a warning and
    proceeds; it does not fail the case or pkill. Already-free ports return
    after one probe with no extra sleep.
    """
    plan = getattr(runtime, "plan", None)
    host = getattr(plan, "host", "?")
    name = getattr(plan, "instance_name", "?")
    try:
        ports = _pd_listen_ports(plan)
        if not ports:
            return
        if timeout_s is None:
            try:
                timeout_s = max(
                    1, int(os.environ.get("XTASK_PD_PORT_RELEASE_TIMEOUT", "30"))
                )
            except ValueError:
                timeout_s = 30
        deadline = time.monotonic() + timeout_s
        last_busy = None
        while time.monotonic() < deadline:
            busy = _read_pd_listen_ports_in_use(runtime, ports)
            if busy is None:
                logging.warning(
                    "PD listen-port probe unreadable on %s (%s); skipping wait",
                    host,
                    name,
                )
                return
            if not busy:
                if last_busy:
                    logging.info(
                        "PD listen ports released on %s (%s)", host, name
                    )
                return
            if busy != last_busy:
                logging.info(
                    "Waiting for PD listen ports on %s (%s): %s",
                    host,
                    name,
                    busy,
                )
            last_busy = busy
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval_s, remaining))
        logging.warning(
            "PD listen ports still in use on %s (%s) after %ss (%s); "
            "proceeding anyway",
            host,
            name,
            timeout_s,
            last_busy,
        )
    except Exception:
        logging.warning(
            "PD listen-port wait failed on %s (%s); proceeding anyway",
            host,
            name,
            exc_info=True,
        )


def _wait_pd_listen_ports_before_start(state, runtimes) -> None:
    """PD-only gate: wait for this case's listen ports before P/D start."""
    if not state.get("pd_disaggregated"):
        return
    from xtask.core.service_group import run_on_runtimes_parallel

    run_on_runtimes_parallel(runtimes or [], _wait_pd_listen_ports_released)


def _hccl_listen_ports(plan) -> list:
    """Return HCCL bind ports (hccl_if_port + device_id) for this runtime plan.

    xLLM exports ``hccl_if_port`` as ``HCCL_IF_BASE_PORT``; each local rank on
    device *d* binds ``base + d`` (see Ascend HCCL ``EI0019`` bind errors).
    """
    if plan is None:
        return []
    cfg = getattr(plan, "instance_config", None)
    if not isinstance(cfg, dict):
        cfg = {}
    if str(cfg.get("communication_backend", "hccl")).lower() != "hccl":
        return []
    try:
        base = int(cfg.get("hccl_if_port", 43439))
    except (TypeError, ValueError):
        return []
    devices = getattr(plan, "device_ids", None)
    if not isinstance(devices, (list, tuple)) or not devices:
        return []
    ports = []
    for raw in devices:
        try:
            device_id = int(raw)
        except (TypeError, ValueError):
            continue
        port = base + device_id
        if 1 <= port <= 65535:
            ports.append(port)
    return sorted(set(ports))


def _wait_hccl_listen_ports_released(runtime, timeout_s=None, interval_s=1) -> None:
    """Wait until this instance's HCCL ports leave LISTEN.

    Timeout (default 30s, XTASK_HCCL_PORT_RELEASE_TIMEOUT) logs a warning and
    proceeds; it does not fail the case or pkill. Already-free ports return
    after one probe with no extra sleep.
    """
    plan = getattr(runtime, "plan", None)
    host = getattr(plan, "host", "?")
    name = getattr(plan, "instance_name", "?")
    try:
        ports = _hccl_listen_ports(plan)
        if not ports:
            return
        if timeout_s is None:
            try:
                timeout_s = max(
                    1, int(os.environ.get("XTASK_HCCL_PORT_RELEASE_TIMEOUT", "30"))
                )
            except ValueError:
                timeout_s = 30
        deadline = time.monotonic() + timeout_s
        last_busy = None
        while time.monotonic() < deadline:
            busy = _read_pd_listen_ports_in_use(runtime, ports)
            if busy is None:
                logging.warning(
                    "HCCL listen-port probe unreadable on %s (%s); skipping wait",
                    host,
                    name,
                )
                return
            if not busy:
                if last_busy:
                    logging.info(
                        "HCCL listen ports released on %s (%s)", host, name
                    )
                return
            if busy != last_busy:
                logging.info(
                    "Waiting for HCCL listen ports on %s (%s): %s",
                    host,
                    name,
                    busy,
                )
            last_busy = busy
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval_s, remaining))
        logging.warning(
            "HCCL listen ports still in use on %s (%s) after %ss (%s); "
            "proceeding anyway",
            host,
            name,
            timeout_s,
            last_busy,
        )
    except Exception:
        logging.warning(
            "HCCL listen-port wait failed on %s (%s); proceeding anyway",
            host,
            name,
            exc_info=True,
        )


def _wait_hccl_listen_ports_before_start(state, runtimes) -> None:
    """Multi-host gate: wait for HCCL ports before xLLM start on each runtime."""
    if not state.get("multi_host"):
        return
    from xtask.core.service_group import run_on_runtimes_parallel

    run_on_runtimes_parallel(runtimes or [], _wait_hccl_listen_ports_released)


def _tcpstore_listen_ports(plan) -> list:
    """Return the xLLM c10d TCPStore rendezvous window for this runtime plan.

    xLLM derives fixed TCPStore ports from the group master address
    (``master_node_addr`` = ``master_host:progress_conn_port``) plus one
    window per process group (world/tp/dp/moe-tp/moe-ep/...; see xllm
    collective_communicator.cpp ``create_process_groups``). The exact window
    depends on the parallel topology, so use a deliberately generous upper
    bound: probing a few extra free ports costs one ``ss`` invocation.
    Single-rank groups use ephemeral ports, so world_size 1 has no window.
    """
    if plan is None:
        return []
    cfg = getattr(plan, "instance_config", None)
    if not isinstance(cfg, dict):
        cfg = {}
    try:
        base = int(cfg.get("progress_conn_port", 9590))
    except (TypeError, ValueError):
        return []
    if not 1 <= base <= 65535:
        return []
    world_size = len(getattr(plan, "device_ids", None) or []) or 1
    try:
        world_size = max(world_size, int(cfg.get("world_size", 0) or 0))
    except (TypeError, ValueError):
        pass
    if world_size <= 1:
        return []
    # Worst observed window: world +1 (world group), tp window (<= world),
    # single-rank reservation (2*world), dp/dcp windows (<= 2*world),
    # moe windows (<= ep + 3*moe_tp <= world + 3). 5*world + 16 covers all
    # of them with margin (e.g. 16-card dp4/ep16 decode binds <= master+60).
    width = 5 * world_size + 16
    upper = min(base + width, 65535)
    if upper <= base:
        return []
    return list(range(base + 1, upper + 1))


def _wait_tcpstore_listen_ports_released(runtime, timeout_s=None, interval_s=1) -> None:
    """Wait until this instance's xLLM TCPStore window leaves LISTEN.

    A leftover TCPStore listener from an abnormally exited previous case
    makes the next same-topology case abort during HCCL process-group
    creation with ``c10d::SocketError`` EADDRINUSE (fixed ports such as
    ``progress_conn_port + 40``; one observed incident held the port for
    4-15 minutes after SIGKILL). Timeout (default 600s,
    XTASK_TCPSTORE_PORT_RELEASE_TIMEOUT) logs a warning and proceeds; it
    does not fail the case or pkill. Already-free ports return after one
    probe with no extra sleep.
    """
    plan = getattr(runtime, "plan", None)
    host = getattr(plan, "host", "?")
    name = getattr(plan, "instance_name", "?")
    try:
        ports = _tcpstore_listen_ports(plan)
        if not ports:
            return
        if timeout_s is None:
            try:
                timeout_s = max(
                    1,
                    int(
                        os.environ.get(
                            "XTASK_TCPSTORE_PORT_RELEASE_TIMEOUT", "600"
                        )
                    ),
                )
            except ValueError:
                timeout_s = 600
        deadline = time.monotonic() + timeout_s
        last_busy = None
        while time.monotonic() < deadline:
            busy = _read_pd_listen_ports_in_use(runtime, ports)
            if busy is None:
                logging.warning(
                    "TCPStore listen-port probe unreadable on %s (%s); "
                    "skipping wait",
                    host,
                    name,
                )
                return
            if not busy:
                if last_busy:
                    logging.info(
                        "TCPStore listen ports released on %s (%s)", host, name
                    )
                return
            if busy != last_busy:
                logging.info(
                    "Waiting for xLLM TCPStore ports on %s (%s): %s",
                    host,
                    name,
                    busy,
                )
            last_busy = busy
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval_s, remaining))
        logging.warning(
            "TCPStore listen ports still in use on %s (%s) after %ss (%s); "
            "proceeding anyway",
            host,
            name,
            timeout_s,
            last_busy,
        )
    except Exception:
        logging.warning(
            "TCPStore listen-port wait failed on %s (%s); proceeding anyway",
            host,
            name,
            exc_info=True,
        )


def _wait_tcpstore_listen_ports_before_start(state, runtimes) -> None:
    """Multi-host gate: wait for xLLM TCPStore ports before xLLM start."""
    if not state.get("multi_host"):
        return
    from xtask.core.service_group import run_on_runtimes_parallel

    run_on_runtimes_parallel(runtimes or [], _wait_tcpstore_listen_ports_released)


_RESERVED_PORTS_PATH = "/proc/sys/net/ipv4/ip_local_reserved_ports"


def _parse_reserved_ports(value) -> set:
    """Parse an ip_local_reserved_ports value ("a,b-c,d") into a port set."""
    ports = set()
    if not isinstance(value, str):
        return ports
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, _, hi = item.partition("-")
            try:
                lo_i, hi_i = int(lo), int(hi)
            except ValueError:
                continue
            if 1 <= lo_i <= hi_i <= 65535:
                ports.update(range(lo_i, hi_i + 1))
        else:
            try:
                port = int(item)
            except ValueError:
                continue
            if 1 <= port <= 65535:
                ports.add(port)
    return ports


def _format_reserved_ports(ports) -> str:
    """Format a port set as a compact sorted comma list with ranges."""
    ordered = sorted({int(port) for port in ports if 1 <= int(port) <= 65535})
    parts = []
    index = 0
    while index < len(ordered):
        start = ordered[index]
        while index + 1 < len(ordered) and ordered[index + 1] == ordered[index] + 1:
            index += 1
        end = ordered[index]
        parts.append(str(start) if start == end else f"{start}-{end}")
        index += 1
    return ",".join(parts)


def _control_plane_ports(task_config) -> set:
    """PD/service control ports (etcd, xservice) from the task config."""
    ports = set()
    if not isinstance(task_config, dict):
        return ports
    deployment = task_config.get("deployment")
    if not isinstance(deployment, dict):
        return ports
    if not get_xllm_service_config(task_config):
        return ports
    return {port for _host, port, _description in service_port_claims(task_config)}


def _xllm_fixed_ports(plan, task_config=None) -> list:
    """Fixed ports this xLLM instance binds on its host.

    Covers the group master (``progress_conn_port``), the c10d TCPStore
    windows derived from it in xllm's ``create_process_groups`` (the
    YAML-invisible ports that caused EADDRINUSE incidents), PD
    disaggregation/transfer ports, HCCL IF ports, per-rank node service
    ports, and the control plane (etcd/xservice). Used only for
    ``ip_local_reserved_ports``; explicit binds are unaffected.
    """
    if plan is None:
        return []
    cfg = getattr(plan, "instance_config", None)
    if not isinstance(cfg, dict):
        cfg = {}
    ports = set()
    # Group master + derived TCPStore window.
    try:
        base = int(cfg.get("progress_conn_port", 9590))
    except (TypeError, ValueError):
        base = 0
    if 1 <= base <= 65535:
        ports.add(base)
    ports.update(_tcpstore_listen_ports(plan))
    # PD disagg/transfer and HCCL IF ports.
    ports.update(_pd_listen_ports(plan))
    ports.update(_hccl_listen_ports(plan))
    # Per-rank node service ports (START_PORT + local rank).
    devices = getattr(plan, "device_ids", None)
    rank_count = (
        len(devices) if isinstance(devices, (list, tuple)) and devices else 1
    )
    try:
        node_base = int(cfg.get("port", 14830) or 14830)
    except (TypeError, ValueError):
        node_base = 14830
    for offset in range(rank_count):
        port = node_base + offset
        if 1 <= port <= 65535:
            ports.add(port)
    # Reserve Service/etcd listeners only on their configured host.
    if task_config and get_xllm_service_config(task_config):
        ports.update(
            port for host, port, _description in service_port_claims(task_config)
            if str(host) == str(getattr(plan, "host", ""))
        )
    return sorted(ports)


def _reserve_ports_on_runtime(runtime, ports, timeout=20) -> None:
    """Merge ports into the host's ip_local_reserved_ports (best effort).

    Runs via the runtime's own privileged host-network container, so the
    write targets the host network namespace. The current value is always
    read first and merged (never overwritten), so other services'
    reservations survive. Any failure logs a warning and proceeds; this
    gate must never fail the case.
    """
    plan = getattr(runtime, "plan", None)
    host = getattr(plan, "host", "?")
    name = getattr(plan, "instance_name", "?")
    if not ports:
        return
    try:
        manager = runtime.container_manager
        read = manager.exec_in_container(
            cmd=f"cat {_RESERVED_PORTS_PATH}", timeout=timeout
        )
        if not read.success:
            logging.warning(
                "Cannot read %s on %s (%s); skipping xLLM port reservation",
                _RESERVED_PORTS_PATH,
                host,
                name,
            )
            return
        current = _parse_reserved_ports(read.stdout or "")
        wanted = current | set(ports)
        if wanted == current:
            return
        value = _format_reserved_ports(wanted)
        write = manager.exec_in_container(
            cmd=f"printf '%s' {shlex.quote(value)} > {_RESERVED_PORTS_PATH}",
            timeout=timeout,
        )
        if write.success:
            logging.info(
                "Reserved xLLM fixed ports on %s (%s): %s", host, name, value
            )
        else:
            logging.warning(
                "Failed to reserve xLLM ports on %s (%s): %s",
                host,
                name,
                (write.stderr or write.stdout or "unknown error")[:200],
            )
    except Exception:
        logging.warning(
            "xLLM port reservation failed on %s (%s); proceeding anyway",
            host,
            name,
            exc_info=True,
        )


def _reserve_xllm_ports_before_start(state, runtimes) -> None:
    """Multi-host gate: reserve xLLM fixed ports before xLLM start.

    Ephemeral source ports drawn from a wide ip_local_port_range can
    collide with xLLM's fixed TCPStore ports and abort process-group
    creation with EADDRINUSE (observed 2026-09-08/19/21/22). Reserving
    those ports stops the kernel from auto-assigning them while leaving
    explicit binds untouched. Re-established at every case start, so the
    protection also survives host reboots. Disable with
    XTASK_RESERVE_XLLM_PORTS=0.
    """
    flag = str(os.environ.get("XTASK_RESERVE_XLLM_PORTS", "1"))
    if flag.strip().lower() in {"0", "false", "no", "off"}:
        return
    task_config = state.get("task_config") if isinstance(state, dict) else None
    by_host = {}
    for runtime in runtimes or []:
        plan = getattr(runtime, "plan", None)
        host = str(getattr(plan, "host", "?"))
        ports = _xllm_fixed_ports(plan, task_config)
        if not ports:
            continue
        entry = by_host.setdefault(host, {"runtime": runtime, "ports": set()})
        entry["ports"].update(ports)
    for entry in by_host.values():
        _reserve_ports_on_runtime(entry["runtime"], sorted(entry["ports"]))


def _start_and_wait_service_multi_host(context):
    from xtask.core.service_group import (
        get_benchmark_endpoint,
        get_benchmark_runtime,
        run_on_runtimes_parallel,
    )

    state = context.state
    task_config = state["task_config"]
    _reserve_xllm_ports_before_start(state, state.get("service_runtimes") or [])
    if state.get("pd_disaggregated") or state.get("service_routed"):
        from xtask.core.xllm_service_cluster import XllmServiceCluster
        from xtask.core.pd_control_service import XllmControlService, build_xllm_service_config, begin_service_run

        begin_service_run(task_config)
        control_config = build_xllm_service_config(task_config)
        try:
            if control_config.service_binary_profile == "legacy":
                runtime = service_container_nodes(task_config, state["service_runtimes"])[0]
                control_service = XllmControlService(
                    runtime, control_config, task_config.get("model_path", ""),
                    context.output_file_path,
                )
            else:
                control_service = XllmServiceCluster(
                    task_config, state["service_runtimes"], context.output_file_path
                )
            # Store before start so even a partially-started cluster has an
            # owner during scheduler failure cleanup.
            state["xllm_control_service"] = control_service
            control_service.start_with_retry()
        except Exception as exc:
            _fail_stage(context, f"xLLM control service startup failed: {exc}")
            return
    engine_config = state["engine_config"]
    runtimes = state.get("service_runtimes") or []
    serving_wait_time = (
        task_config.get("serving_wait_time")
        if task_config.get("serving_wait_time")
        else state["config_data"].get("XLLM_CONFIG").get("serving_wait_time")
    )
    # Fixed Service and Engine ports were reserved before control startup.
    # PD leftover LISTEN on disagg_pd_port / transfer_listen_port prevents
    # pairing and keeps HTTP 28888 closed. Wait only for this case's ports.
    _wait_pd_listen_ports_before_start(state, runtimes)
    # HCCL leftover LISTEN on hccl_if_port + device_id prevents
    # HcclGetRootInfo on the next case sharing the same host/base port.
    _wait_hccl_listen_ports_before_start(state, runtimes)
    # Leftover c10d TCPStore LISTEN (progress_conn_port + process-group
    # window offsets, e.g. 8813) aborts process-group creation on the next
    # same-topology case with c10d::SocketError EADDRINUSE.
    _wait_tcpstore_listen_ports_before_start(state, runtimes)

    def _start_one(runtime):
        cmd = runtime.plan.startup_cmd
        # Sync generated scripts to remote host when paths are not shared.
        # Wipe remote dir first: scp -r into an existing dir nests as name/name/.
        if not runtime.plan.is_local:
            remote_dir = runtime.plan.output_dir
            remote_parent = os.path.dirname(remote_dir.rstrip(os.sep))
            # Do not mkdir remote_dir itself: scp -r into an existing dir nests name/name.
            wipe = runtime.container_manager.executor.run(
                f"chmod -R u+w {remote_dir} 2>/dev/null || true; "
                f"rm -rf {remote_dir}; mkdir -p {remote_parent}",
                timeout=120,
            )
            if not wipe.success:
                raise RuntimeError(
                    f"Failed to prepare remote output dir {remote_dir}: "
                    f"{wipe.stderr or wipe.stdout}"
                )
            if not runtime.container_manager.executor.upload(
                runtime.plan.output_dir, remote_dir
            ):
                raise RuntimeError(
                    f"Failed to upload scripts for {runtime.plan.instance_name} "
                    f"to {runtime.plan.host}:{remote_dir}"
                )
        logging.info(
            "Starting instance %s on %s: %s",
            runtime.plan.instance_name,
            runtime.plan.host,
            cmd,
        )
        result = runtime.container_manager.exec_in_container(
            runtime.container_config.container_name,
            cmd,
            timeout=engine_config.get("startup_timeout", 300),
        )
        if not result.success:
            raise RuntimeError(
                f"Startup failed for {runtime.plan.instance_name}: {result.stderr}"
            )
        return result

    # Parallel startup (no ordered groups in v1).
    start_outcomes = run_on_runtimes_parallel(runtimes, _start_one)
    for runtime, _result, exc in start_outcomes:
        if exc is not None:
            _fail_stage(context, str(exc))
            state["service_healthy"] = False
            return

    time.sleep(serving_wait_time)

    def _wait_one(runtime):
        ok = _wait_instance_ready(runtime, serving_wait_time, state["engine"])
        if not ok:
            raise RuntimeError(
                f"Instance {runtime.plan.instance_name} readiness check failed"
            )
        return True

    wait_outcomes = run_on_runtimes_parallel(runtimes, _wait_one)
    for runtime, _ok, exc in wait_outcomes:
        if exc is not None:
            _fail_stage(context, str(exc))
            state["service_healthy"] = False
            return

    # The unified xllm-service endpoint may open only after routed instances
    # register with etcd. Wait for it before the endpoint health check.
    control_service = state.get("xllm_control_service") or state.get(
        "pd_control_service"
    )
    if control_service is not None:
        if not control_service.wait_for_http_endpoint(timeout=90):
            _fail_stage(
                context,
                f"xLLM-Service endpoint not ready: "
                f"{getattr(control_service, 'benchmark_endpoint', None) or get_benchmark_endpoint(task_config, state['service_plans'])}; "
                f"{getattr(control_service, 'last_error', '')}",
            )
            state["service_healthy"] = False
            return

    endpoint = get_benchmark_endpoint(task_config, state["service_plans"])
    if getattr(control_service, "is_split", False):
        state["service_healthy"] = True  # wait_for_http_endpoint checked roles/models
        try:
            control_service.collect_logs()
        except Exception:
            logging.warning("Failed to collect live Service logs", exc_info=True)
        logging.info("Router inference endpoint is healthy: %s", endpoint)
    else:
        health_path = (task_config.get("deployment") or {}).get("benchmark", {}).get(
            "health_path", "/health"
        )
        check_url = endpoint.rstrip("/") + health_path
        runner = get_benchmark_runtime(runtimes)
        check_cmd = f"curl --noproxy '*' -i {check_url}"
        health_result = runner.container_manager.exec_in_container(
            runner.container_config.container_name, check_cmd, timeout=30
        )
        state["health_result"] = health_result
        state["service_healthy"] = (
            health_result.success and "ok" in health_result.stdout.lower()
        )
        if not state["service_healthy"]:
            _fail_stage(context, f"Service health check failed for {check_url}: {health_result.stderr}")
            return

    # Pull remote instance logs back to the control host for aggregation.
    for runtime in runtimes:
        if runtime.plan.is_local:
            continue
        remote_log = os.path.join(runtime.plan.output_dir, "log")
        try:
            if not _download_remote_runtime_path(
                runtime, remote_log, remote_log
            ):
                logging.warning(
                    "Failed to download logs for %s from %s",
                    runtime.plan.instance_name,
                    runtime.plan.host,
                )
        except Exception:
            logging.warning(
                "Failed to download logs for %s from %s",
                runtime.plan.instance_name,
                runtime.plan.host,
                exc_info=True,
            )


def _run_aisbench_swe_and_process(context):
    """Run AISBench SWE-bench on the x86 host and verify resolved_rate.

    Isolated from the AISBench GSM8K acc/perf container path: no NPU guard,
    no download_ais_bench, and no acc/perf postprocess.
    """
    from xtask.benchmarks.aisbench.aisbench_swe_postprocess import (
        aisbench_swe_post_process,
    )
    from xtask.benchmarks.aisbench.generate_aisbench_swe_script import (
        execute_aisbench_swe_benchmark,
    )
    from xtask.benchmarks.aisbench.swebench import (
        swebench_direct_output_enabled,
        swebench_timeout_s,
    )

    state = context.state
    task_config = state["task_config"]
    os.makedirs(context.output_file_path, exist_ok=True)
    init_entries = os.listdir(context.output_file_path)
    state["init_entries"] = init_entries
    if not _chown_via_running_container(
        state.get("container_manager"),
        state.get("container_config"),
        context.output_file_path,
    ):
        logging.warning(
            "Failed to chown SWE-bench output directory before download: %s",
            context.output_file_path,
        )
    logging.info("Executing AISBench SWE-bench on the configured x86 runner")
    benchmark_result = execute_aisbench_swe_benchmark(
        task_config,
        context.output_file_path,
        timeout=swebench_timeout_s(task_config),
    )
    state["benchmark_result"] = benchmark_result
    context.result.update(
        benchmark_success=benchmark_result.success,
        benchmark_exit_code=benchmark_result.exit_code,
        benchmark_stdout=benchmark_result.stdout,
        benchmark_stderr=benchmark_result.stderr,
        benchmark_execution_time=benchmark_result.execution_time,
    )
    if benchmark_result.success:
        try:
            swe_result, output_dir = aisbench_swe_post_process(
                task_config,
                context.output_file_path,
                init_entries,
                require_new_direct_run=swebench_direct_output_enabled(task_config),
            )
            context.result["benchmark_swebench_result"] = swe_result
            if output_dir:
                context.result["benchmark_output_path"] = output_dir
                _record_service_group_archive_path(state, output_dir)
                # SWE-bench runs AISBench on a separate x86 runner. Archive the
                # serving logs now, while xLLM is still live; shutdown only
                # overlays lifecycle artifacts and deliberately skips log/.
                try:
                    _archive_service_group_logs(
                        state.get("service_runtimes") or [],
                        context.output_file_path,
                        output_dir,
                    )
                except Exception:
                    logging.warning(
                        "Failed to archive live xLLM logs for SWE-bench: %s",
                        output_dir,
                        exc_info=True,
                    )
        except Exception as exc:
            context.result["benchmark_swebench_result"] = None
            context.result["message"] = f"SWE-bench result processing failed: {exc}"
            logging.error(context.result["message"], exc_info=True)
    else:
        context.result["message"] = (
            f"Benchmark command failed with exit code {benchmark_result.exit_code}"
        )
        if benchmark_result.stdout:
            logging.error("SWE-bench stdout:\n%s", benchmark_result.stdout)
        if benchmark_result.stderr:
            logging.error("SWE-bench stderr:\n%s", benchmark_result.stderr)
        logging.error(context.result["message"])

    container_manager = state.get("container_manager")
    container_id = context.result.get("container_id")
    if container_manager is not None and container_id:
        try:
            context.result["logs"] = container_manager.get_container_logs(
                container_id, tail=100
            )
        except Exception:
            logging.warning("Failed to collect xLLM logs for SWE-bench case", exc_info=True)
            context.result["logs"] = ""
    else:
        context.result["logs"] = ""

    context.result["success"] = False
    if verify_result(context.result, task_config):
        context.result["success"] = True
        if not context.result.get("message"):
            context.result["message"] = (
                f"Task scheduled successfully in container {container_id}"
                if container_id
                else "SWE-bench task completed successfully"
            )
    else:
        if benchmark_result.success and not context.result.get("message"):
            context.result["message"] = "Benchmark result verification failed"
        logging.error("verify result failed.")


def _run_benchmark_and_process(context):
    state = context.state
    if not state["service_healthy"]:
        return

    task_config = state["task_config"]
    if _is_aisbench_swebench_task(task_config):
        _run_aisbench_swe_and_process(context)
        return

    engine_config = state["engine_config"]
    container_config = state["container_config"]
    container_manager = state["container_manager"]
    benchmark_cmd = state["benchmark_cmd"]
    init_entries = state["init_entries"]
    benchmark_container_manager = state.get("vbench_container_manager") or container_manager
    benchmark_container_config = state.get("vbench_container_config") or container_config
    logging.info(f"Executing benchmark command: {benchmark_cmd}")

    benchmark_result = None
    if benchmark_cmd:
        if _should_run_perf_warmup(task_config):
            warmup_cmd = _build_warmup_command(task_config, context.output_file_path)
            _execute_warmup(
                container_manager, container_config, engine_config, warmup_cmd
            )
            _cleanup_warmup_results(
                container_manager,
                container_config,
                context.output_file_path,
                init_entries,
            )
            time.sleep(10)

        init_entries = os.listdir(context.output_file_path)
        state["init_entries"] = init_entries
        service_runtimes = state.get("service_runtimes") if state.get("multi_host") else None
        if state.get("external_service"):
            benchmark_result = benchmark_container_manager.exec_in_container(
                benchmark_container_config.container_name,
                benchmark_cmd,
                timeout=engine_config.get("benchmark_timeout", 3600),
            )
        else:
            benchmark_result = _execute_benchmark_with_npu_guard(
                benchmark_container_manager,
                benchmark_container_config,
                task_config,
                engine_config,
                benchmark_cmd,
                service_runtimes=service_runtimes,
                guard_container_manager=container_manager if _is_vbench_task(task_config) else None,
                guard_container_config=container_config if _is_vbench_task(task_config) else None,
            )
        state["benchmark_result"] = benchmark_result
        context.result.update(
            benchmark_success=benchmark_result.success,
            benchmark_exit_code=benchmark_result.exit_code,
            benchmark_stdout=benchmark_result.stdout,
            benchmark_stderr=benchmark_result.stderr,
            benchmark_execution_time=benchmark_result.execution_time,
        )

        remote_runner = state.get("session_remote_benchmark_runner")
        if remote_runner is None and _is_asr_task(task_config) and state.get("multi_host"):
            from xtask.core.service_group import get_benchmark_runtime
            candidate = get_benchmark_runtime(state.get("service_runtimes") or [])
            if candidate is not None and not candidate.plan.is_local:
                remote_runner = candidate
        if remote_runner is not None:
            if _is_asr_task(task_config):
                try:
                    _download_asr_runner_results(context, remote_runner)
                except Exception as exc:
                    context.result["archive_error"] = f"ASR result transfer failed: {exc}"
                    logging.error(context.result["archive_error"], exc_info=True)
            elif not _download_remote_runtime_path(
                remote_runner,
                context.output_file_path, context.output_file_path
            ):
                raise RuntimeError(
                    f"Failed to download benchmark result from "
                    f"{remote_runner.plan.host}"
                )

        vbench_runner = state.get("vbench_runner_runtime")
        if _is_vbench_task(task_config) and benchmark_container_manager is not None:
            from xtask.utils.container_manager import path_mapping_in_container

            # The VBench runner usually writes as root.  Make local output
            # readable before host-side post-processing as well as download.
            benchmark_container_manager.exec_in_container(
                benchmark_container_config.container_name,
                f"chmod -R a+rwX {shlex.quote(path_mapping_in_container(context.output_file_path))} 2>/dev/null || true",
                timeout=60,
            )
            if (
                vbench_runner is not None
                and not vbench_runner.plan.is_local
            ):
                # The VBench container is separate from service_runtimes, so
                # remote generated videos/results need an explicit download.
                if not benchmark_container_manager.executor.download(
                    context.output_file_path, context.output_file_path
                ):
                    raise RuntimeError(
                        f"Failed to download VBench output from {vbench_runner.plan.host}"
                    )

        if _is_asr_task(task_config):
            from xtask.benchmarks.asr.artifacts import asr_result_post_process

            try:
                date_dir = _find_new_benchmark_date_dir(Path(context.output_file_path), set(init_entries))
                if date_dir is not None and not os.access(date_dir, os.W_OK):
                    _chown_via_running_container(benchmark_container_manager, benchmark_container_config, date_dir)
                context.result["benchmark_asr_result"], output_dir = asr_result_post_process(
                    task_config, context.output_file_path, init_entries
                )
                context.result["benchmark_output_path"] = output_dir
                _record_service_group_archive_path(state, output_dir)
            except Exception as exc:
                context.result["message"] = f"ASR result processing failed: {exc}"
                logging.error(context.result["message"], exc_info=True)

        if benchmark_result.success:
            logging.info("perf benchmark completed successfully.")
            if _is_asr_task(task_config):
                pass  # ASR quality and performance were processed together above.
            elif _is_vbench_task(task_config):
                from xtask.benchmarks.vbench.vbench_postprocess import (
                    vbench_result_post_process,
                )

                try:
                    context.result["benchmark_vbench_result"], output_dir = (
                        vbench_result_post_process(
                            task_config, context.output_file_path, init_entries
                        )
                    )
                except Exception as exc:
                    context.result["benchmark_vbench_result"] = None
                    context.result["message"] = f"VBench result processing failed: {exc}"
                    logging.error(context.result["message"], exc_info=True)
                    output_dir = ""
                if output_dir:
                    context.result["benchmark_output_path"] = output_dir
                if output_dir:
                    _record_service_group_archive_path(state, output_dir)
            # NOTE: AISBench SWE-bench tasks (metrics: acc + benchmark.acc in
            # the new format, or metrics: swebench legacy) never reach this
            # chain: _run_benchmark_and_process routes them to
            # _run_aisbench_swe_and_process and returns before the benchmark
            # execution above. This generic acc branch must stay SWE-free.
            elif "acc" in task_config.get("metrics"):
                logging.info("Start to check acc benchmark result.")
                from xtask.benchmarks.base_benchmark import acc_result_postprocess

                context.result["benchmark_acc_result"], output_dir = (
                    acc_result_postprocess(
                        task_config, context.output_file_path, init_entries
                    )
                )
                if output_dir:
                    context.result["benchmark_output_path"] = output_dir
                _post_process_benchmark_rst(
                    container_manager,
                    container_config,
                    task_config,
                    context.output_file_path,
                    output_dir,
                    service_runtimes=state.get("service_runtimes"),
                )
                _record_service_group_archive_path(state, output_dir)
            elif "protocol" in task_config.get("metrics"):
                logging.info("Start to process protocol benchmark result.")
                from xtask.benchmarks.aisbench.aisbench_postprocess import (
                    protocol_result_post_process,
                )

                context.result["benchmark_protocol_result"], output_dir = (
                    protocol_result_post_process(
                        task_config, context.output_file_path, init_entries
                    )
                )
                if output_dir:
                    context.result["benchmark_output_path"] = output_dir
                    _post_process_benchmark_rst(
                        container_manager,
                        container_config,
                        task_config,
                        context.output_file_path,
                        output_dir,
                        service_runtimes=state.get("service_runtimes"),
                    )
                    _record_service_group_archive_path(state, output_dir)
            else:
                logging.debug(
                    "metrics is %s, skip accuracy task.",
                    task_config.get("metrics"),
                )

            if "perf" in task_config.get("metrics") and not _is_asr_task(task_config):
                from xtask.benchmarks.base_benchmark import result_postprocess

                context.result["benchmark_perf_result"], output_dir = (
                    result_postprocess(
                        benchmark_result.stdout,
                        task_config,
                        context.output_file_path,
                        init_entries,
                    )
                )
                if output_dir:
                    context.result["benchmark_output_path"] = output_dir
                _post_process_benchmark_rst(
                    container_manager,
                    container_config,
                    task_config,
                    context.output_file_path,
                    output_dir,
                    service_runtimes=state.get("service_runtimes"),
                )
                _record_service_group_archive_path(state, output_dir)
        else:
            logging.error(
                "perf benchmark command failed with exit code %s",
                benchmark_result.exit_code,
            )
            if benchmark_result.stdout:
                logging.error("Benchmark stdout:\n%s", benchmark_result.stdout)
            if benchmark_result.stderr:
                logging.error("Benchmark stderr:\n%s", benchmark_result.stderr)
    else:
        state["benchmark_result"] = None
        logging.warning("benchmark_cmd is null, skip benchmark task.")

    if benchmark_result is not None and not benchmark_result.success:
        context.result["message"] = (
            f"Benchmark command failed with exit code {benchmark_result.exit_code}"
        )
        logging.error(context.result["message"])

    context.result["logs"] = container_manager.get_container_logs(
        context.result["container_id"], tail=100
    )
    context.result["success"] = False
    if verify_result(context.result, task_config):
        context.result["success"] = True
        context.result["message"] = (
            f"Task scheduled successfully in container {context.result['container_id']}"
        )
    else:
        if benchmark_result is None and not context.result.get("message"):
            context.result["message"] = "Benchmark command was not generated"
        elif benchmark_result is not None and benchmark_result.success and not context.result.get("message"):
            context.result["message"] = "Benchmark result verification failed"
        logging.error("verify result failed.")


def _finalize_result(context):
    task_config = context.result.get("task_config") or context.state.get(
        "task_config"
    )
    if _is_asr_task(task_config or {}):
        _try_archive_asr_execution(context)
    try:
        _ensure_benchmark_date_dir(context)
    except Exception:
        logging.error("Failed to prepare fallback result directory", exc_info=True)
    try:
        written = write_result(
            context.result,
            task_config,
            task_path=context.task_yaml_path,
            args=context.args,
        )
    except Exception as exc:
        if not _is_asr_task(task_config or {}):
            raise
        written = False
        context.result["archive_error"] = f"ASR workbook generation failed: {exc}"
        logging.error(context.result["archive_error"], exc_info=True)
    if _is_asr_task(task_config or {}):
        context.result["report_generated"] = written is True
        if written is not True:
            context.result["archive_error"] = context.result.get("archive_error") or "ASR workbook generation failed"
            context.result["success"] = False
            context.result["message"] = context.result["archive_error"]
        _try_archive_asr_execution(context)


def _freeze_service_group_logs(runtimes):
    """Capture clean runtime logs immediately before service shutdown."""
    snapshots = {}
    for runtime in runtimes or []:
        log_dir = Path(runtime.plan.output_dir) / "log"
        snapshot_dir = Path(runtime.plan.output_dir) / ".log_before_shutdown"
        try:
            if not _make_runtime_path_host_accessible(runtime, log_dir):
                continue
            if not runtime.plan.is_local:
                if not _download_remote_runtime_path(
                    runtime,
                    str(log_dir), str(log_dir)
                ):
                    logging.warning(
                        "Skip runtime log snapshot after download failure: %s",
                        log_dir,
                    )
                    continue
            if not log_dir.is_dir():
                logging.warning(
                    "Skip missing runtime log directory before shutdown: %s",
                    log_dir,
                )
                continue
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            shutil.copytree(log_dir, snapshot_dir)
            snapshots[runtime.plan.instance_name] = snapshot_dir
        except Exception:
            logging.warning(
                "Failed to freeze runtime logs for %s on %s",
                runtime.plan.instance_name,
                runtime.plan.host,
                exc_info=True,
            )
    return snapshots


def _separate_shutdown_logs(runtimes, snapshots) -> None:
    """Preserve shutdown output and restore clean logs on the control host."""
    for runtime in runtimes or []:
        snapshot_dir = snapshots.get(runtime.plan.instance_name)
        if snapshot_dir is None or not snapshot_dir.is_dir():
            continue

        log_dir = Path(runtime.plan.output_dir) / "log"
        shutdown_log_dir = Path(runtime.plan.output_dir) / "shutdown_log"
        try:
            if not _make_runtime_path_host_accessible(
                runtime, runtime.plan.output_dir
            ):
                continue
            if shutdown_log_dir.exists():
                shutil.rmtree(shutdown_log_dir)

            if runtime.plan.is_local:
                if log_dir.is_dir():
                    shutil.copytree(log_dir, shutdown_log_dir)
            else:
                if not _download_remote_runtime_path(
                    runtime,
                    str(log_dir), str(shutdown_log_dir)
                ):
                    logging.warning(
                        "Failed to download shutdown logs for %s from %s",
                        runtime.plan.instance_name,
                        runtime.plan.host,
                    )

            if log_dir.exists():
                shutil.rmtree(log_dir)
            shutil.copytree(snapshot_dir, log_dir)
        except Exception:
            logging.warning(
                "Failed to separate shutdown logs for %s on %s",
                runtime.plan.instance_name,
                runtime.plan.host,
                exc_info=True,
            )
        finally:
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)

def _npu_hbm_probe_cmd(device_ids=None) -> str:
    """npu-smi 命令：输出指定设备 HBM 已用量(MB)的最大值。

    输出形如 "3129"。行结构 ``| NPU Chip | Bus-Id | ... Mem used/total HBM
    used/total |``：Chip 列即 ASCEND_RT_VISIBLE_DEVICES 用的芯片(Phy-ID)
    编号，与 YAML device_ids 同一编号空间。行内取**最后一个** ``N / M``
    （前面还有 AICore 0/0、Memory 0/0），总量不硬编码（32G/64G/96G 各异）。

    device_ids 为空时查全部设备（兼容旧行为/单机无 plan 场景）。只查
    本用例占的卡：共享宿主上其他 xTask 实例/租户占用卡的水位起落不会
    抬高基线或误判释放（全卡模式下，别人恰好在清理窗口退出会用自己的
    低水位淹没本用例的残留）。
    """
    npu_smi = "{ npu-smi info 2>/dev/null || /usr/local/bin/npu-smi info 2>/dev/null; }"
    # NOTE: npu-smi 的 "used/total" 是固定列宽右对齐，斜杠两侧空格随数值
    # 位数变化：4 位数 "3149 / 65536"、5 位数 "59194/ 65536"。正则必须
    # 写成 *\/ *（斜杠两侧零或多个空格），只匹配 " / " 会在高水位
    # (>9999MB，恰是最需要等待释放的时刻)解析失败返回空。
    # awk 正则字面量里 / 必须转义为 \/，否则被当作正则定界符截断。
    if not device_ids:
        # cut 去掉 "/total" 部分；紧贴格式 "59194/ 65536" 下 awk '{print $1}'
        # 会带上斜杠("59194/")，导致 _read_npu_hbm_max_mb 的 isdigit 判假。
        return (
            f"{npu_smi} | grep -oE '[0-9]+ */ *[0-9]+' | cut -d'/' -f1"
            " | tr -d ' ' | sort -n | tail -1"
        )
    devs = ",".join(str(int(d)) for d in device_ids)
    awk_filter = (
        "BEGIN { n=split(devs,D,\",\"); for(i=1;i<=n;i++) want[D[i]]=1 } "
        "/\\/ *[0-9]+ +\\|/ { "
        "  t=$0; gsub(/\\|/,\"\",t); sub(/^ +/,\"\",t); split(t,f,\" +\"); chip=f[2]; "
        "  if (!(chip in want)) next; "
        "  used=\"\"; line=$0; "
        "  while (match(line, /[0-9]+ *\\/ *[0-9]+/)) { "
        "    pair=substr(line,RSTART,RLENGTH); sub(/ *\\/ .*/, \"\", pair); used=pair; "
        "    line=substr(line,RSTART+RLENGTH); "
        "  } "
        "  if (used != \"\") print used "
        "}"
    )
    return (
        f"{npu_smi} | awk -v devs='{devs}' '{awk_filter}' | sort -n | tail -1"
    )


def _runtime_device_ids(runtime) -> Optional[list]:
    """本 runtime 占用的设备编号（= ASCEND_RT_VISIBLE_DEVICES 编号空间）。"""
    plan = getattr(runtime, "plan", None)
    devices = getattr(plan, "device_ids", None)
    if isinstance(devices, (list, tuple)) and devices:
        return list(devices)
    return None


def _read_npu_hbm_max_mb(runtime) -> Optional[int]:
    """读取本 runtime 设备的 HBM 最大占用(MB)；失败返回 None。

    与 _wait_npu_hbm_released 共用同一条宿主侧命令（npu-smi 反映宿主机
    设备状态，不受容器生命周期影响）。device_ids 缺失时查全部设备。
    """
    cmd = _npu_hbm_probe_cmd(_runtime_device_ids(runtime))
    host = getattr(getattr(runtime, "plan", None), "host", "?")
    try:
        r = runtime.container_manager.executor.run(cmd, timeout=20)
    except Exception as exc:
        logging.warning("npu-smi poll error on %s: %s", host, exc)
        return None
    out = (getattr(r, "stdout", None) or "").strip()
    if not isinstance(out, str):
        return None
    # 命令输出已是单行数字；逐行过滤数字行兜底（防御输出末尾多空行/杂质）。
    for ln in reversed(out.splitlines()):
        ln = ln.strip()
        if ln.isdigit():
            return int(ln)
    return None


def _sample_npu_hbm_baseline(runtime) -> int:
    """启动前采样本宿主 HBM 基线（驱动常驻水位），记到 plan 上。

    "什么算干净"由这台机器自己定义，而不是代码里的常数：不同驱动版本/
    固件常驻占用不同（实测本机约 3GB），共享集群上别的租户常驻任务也
    会被如实计入。采样失败返回 0（=未采样），清理时回退到环境变量/默认值。
    在容器创建后、xLLM 启动前调用：此时 HBM 恰为无引擎状态的基线。
    """
    try:
        mb = _read_npu_hbm_max_mb(runtime)
    except Exception:
        mb = None
    if mb is None:
        logging.warning(
            "HBM baseline sampling failed on %s; cleanup will use env/default",
            getattr(getattr(runtime, "plan", None), "host", "?"),
        )
        return 0
    runtime.plan.hbm_baseline_mb = mb
    logging.info("HBM baseline sampled on %s: %sMB", runtime.plan.host, mb)
    return mb


def _single_host_device_ids(task_config) -> Optional[list]:
    """单机路径的设备编号：优先 device_ids，回退 start_device_id/num_devices。

    与 service_group 的 _device_ids_from_instance 语义一致（合并后的
    task_config["device_ids"] 优先），编号空间同为 ASCEND_RT_VISIBLE_DEVICES。
    返回 None 表示无信息（HBM 探测回退为全卡）。
    """
    devices = task_config.get("device_ids")
    if isinstance(devices, (list, tuple)) and devices:
        return [int(d) for d in devices]
    try:
        start = int(task_config.get("start_device_id", 0))
        count = int(task_config.get("num_devices", 1))
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    return list(range(start, start + count))


def _wait_npu_hbm_released(runtime, timeout_s=None, baseline_mb=None, settle_s=None) -> None:
    """轮询 npu-smi，等每卡 HBM 回落到 ~baseline，并在回基线后额外静置 settle_s 秒。
    Why: HCCL communicator 和 HBM 由 NPU 驱动在引擎进程退出**之后**才释放，
    不是进程一死就释放。teardown 杀掉 16 个 xllm 进程后若立即起下一档，
    rank-0 构造 CollectiveService 时 HcclGetRootInfo 会在没释放干净的设备上
    失败(status 7)→ 8773 永不绑 → master 28888 不开 → /health 失败。
    累积型：低 batch 残留少够用，重负载档后下一档首次越阈值，之后毒化全档。
    关键：HBM 先回到基线，但 HCCL communicator 释放**比 HBM 慢且 npu-smi 看不到**。
    实测 acc sg3→sg4：HBM 已干净(max=3141MB)但 sg4 仍 HcclGetRootInfo status 7。
    故 HBM 达到基线后必须再静置 settle_s 秒(默认 30s)，给 communicator 释放窗口，
    否则 HBM 干净而 HCCL 仍脏 → 下一档起不来。

    baseline 优先级：显式参数 > 启动前采样值+容差(见 _sample_npu_hbm_baseline)
    > 环境变量 XTASK_HBM_BASELINE_MB > 默认 5000。采样值容差
    XTASK_HBM_BASELINE_SLACK_MB(默认 512) 覆盖轮询噪声与少量驱动抖动。
    """
    import os
    if timeout_s is None:
        try:
            timeout_s = max(10, int(os.environ.get("XTASK_HBM_RELEASE_TIMEOUT", "60")))
        except ValueError:
            timeout_s = 60
    if settle_s is None:
        try:
            settle_s = max(0, int(os.environ.get("XTASK_HBM_SETTLE_SECONDS", "30")))
        except ValueError:
            settle_s = 30

    host = getattr(getattr(runtime, "plan", None), "host", "?")
    # npu-smi 瞬断重试次数：SIGKILL 大批进程后驱动可能短暂忙不过来，
    # 输出为空并不代表设备真不可读。
    try:
        probe_retries = max(1, int(os.environ.get("XTASK_HBM_PROBE_RETRIES", "3")))
    except ValueError:
        probe_retries = 3
    if baseline_mb is None:
        baseline_mb = _resolve_hbm_baseline_mb(runtime)

    deadline = time.monotonic() + timeout_s
    last = None
    settled_at = None  # HBM 首次回基线的时间戳；None=尚未回基线
    unreadable_streak = 0
    while time.monotonic() < deadline:
        cur = _read_npu_hbm_max_mb(runtime)
        if cur is None:
            # npu-smi 瞬断(SIGKILL 大批进程后驱动忙、SSH 抖动)时输出为空，
            # 立即放行会把"恰好没等到释放"伪装成"探测失败"。先重试；
            # 连续失败才认定真不可读，放行兜底避免卡死流程。
            unreadable_streak += 1
            if unreadable_streak < probe_retries:
                logging.warning(
                    "npu-smi HBM unreadable on %s (attempt %d/%d); retrying",
                    host, unreadable_streak, probe_retries,
                )
                time.sleep(3)
                continue
            logging.warning("npu-smi HBM unreadable on %s; skipping release wait", host)
            return
        unreadable_streak = 0
        last = cur
        if cur <= baseline_mb:
            if settled_at is None:
                settled_at = time.monotonic()
                logging.info(
                    "NPU HBM released on %s (max=%sMB <= %sMB); "
                    "settling %ss for HCCL communicator release",
                    host, cur, baseline_mb, settle_s,
                )
                if settle_s <= 0:
                    return
                # 保证 settle 窗口不被总 deadline 截断。
                deadline = max(deadline, settled_at + settle_s)
            if time.monotonic() - settled_at >= settle_s:
                logging.info(
                    "NPU HBM settle window done on %s (max=%sMB); proceeding",
                    host, cur,
                )
                return
            # 距 settle 结束最多再睡 1s：最后一轮轮询若落在窗口尾部
            # (如 29.7s)，sleep(3) 会越过 deadline，导致窗口明明走完
            # 却在 while 条件处以"超时未释放"误报退出(实测 3149MB 低于
            # 基线仍报 not released)。
            time.sleep(min(3, max(0.2, settled_at + settle_s - time.monotonic())))
            continue
        # HBM 又涨回基线之上(异常，例如残留进程)：重新等回基线并重置 settle。
        if settled_at is not None:
            logging.warning(
                "NPU HBM rose above baseline again on %s (%sMB > %sMB); re-arming settle",
                host, cur, baseline_mb,
            )
            settled_at = None
        time.sleep(3)
    logging.warning("NPU HBM not released on %s after %ss (max=%sMB); proceeding anyway", host, timeout_s, last)


def _resolve_hbm_baseline_mb(runtime) -> int:
    """解析本 runtime 的 HBM 基线阈值(MB)：采样值+容差 > 环境变量 > 默认。"""
    import os
    try:
        slack = max(0, int(os.environ.get("XTASK_HBM_BASELINE_SLACK_MB", "512")))
    except ValueError:
        slack = 512
    sampled = getattr(getattr(runtime, "plan", None), "hbm_baseline_mb", 0) or 0
    if sampled > 0:
        return sampled + slack
    try:
        return int(os.environ.get("XTASK_HBM_BASELINE_MB", "5000"))
    except ValueError:
        return 5000


def _cleanup_xllm_shm(runtimes) -> None:
    """Remove stale xLLM POSIX shared-memory objects left by this case.

    xLLM names its SHM objects ``xllm_<device>_<kind>`` (e.g.
    ``xllm_9590_dpg_0_input`` where 9590 is the Atlas chip id, not a port)
    and validates their size on startup ("incompatible shared memory
    layout").  A previous crashed run therefore poisons the next start on the
    same devices.  The runner uses ``--ipc=host``, so the objects live in the
    host ``/dev/shm`` and must be removed there.

    Called after ``_wait_xllm_processes_exit`` guarantees no xllm process is
    alive, so every remaining ``xllm_*`` object on a local host is stale.
    Remote hosts keep their stale objects; surface that instead of silently
    missing them.  Unrelated namespaces (``xllm_expert_*`` from other teams'
    MoE runs) are left untouched.

    An independent xllm service (not managed by this xTask run) may coexist on
    the same host; deleting its live SHM objects would crash it.  Each target
    is therefore checked with ``fuser`` first and skipped when any process
    still holds it open.

    ``runtimes`` may be empty for the single-host path, where the local host
    is the only cleanup target by definition.
    """
    import subprocess

    # 每个实例容器(含远端宿主，经各自 manager 的 executor 执行)都已先跑过
    # 容器内 root 清理 _cleanup_xllm_shm_in_container，宿主侧这层只是本地
    # 兜底。多机场景下旧版无条件打的"远端可能残留"提示已过时(远端实际已
    # 在容器内清干净)，已删除该误导性警告；本地兜底清理保留。
    has_local = (not runtimes) or any(
        getattr(getattr(r, "plan", None), "is_local", True) for r in runtimes
    )
    if not has_local:
        return

    try:
        result = subprocess.run(
            ["ls", "/dev/shm"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        logging.warning("Failed to list /dev/shm for xLLM SHM cleanup", exc_info=True)
        return

    # Per-case objects carry the chip id (``xllm_9590_dpg_0_input``).  The
    # generation lock is a host-global xLLM lock shared across runs, so it
    # must survive case cleanup.
    targets = [
        name
        for name in result.stdout.split()
        if name.startswith("xllm_")
        and not name.startswith("xllm_expert_")
        and name != "xllm_shm_generation_lock_v2"
    ]
    if not targets:
        return

    def _is_held(name: str) -> bool:
        """True when some live process still has the SHM object open.

        ``fuser`` prints PIDs to stdout (and the object path to stderr) when
        the file is held; a non-zero exit means nothing holds it.  When
        ``fuser`` is unavailable, fall back to deleting (historical behaviour)
        rather than skipping cleanup entirely.
        """
        try:
            probe = subprocess.run(
                ["fuser", f"/dev/shm/{name}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return probe.returncode == 0

    logging.info("Removing %d stale xLLM SHM objects: %s", len(targets), targets[:8])
    for name in targets:
        if _is_held(name):
            # Skip live objects belonging to an xllm service this run does not
            # own (e.g. another team's standalone deployment on shared hosts).
            logging.info(
                "Skipping /dev/shm/%s: still held by a live process", name
            )
            continue
        try:
            os.remove(f"/dev/shm/{name}")
        except FileNotFoundError:
            pass
        except OSError:
            # Root-owned objects from a container run cannot be removed by the
            # current user; the next start still hits the layout error.  Log it
            # so the operator knows to clean up manually.
            logging.warning("Cannot remove /dev/shm/%s (permission?)", name)


def _cleanup_xllm_shm_in_container(container_manager, kill_xllm: bool = True) -> None:
    """Stop xLLM inside the container, then remove its SHM objects as root.

    xLLM creates its POSIX SHM objects as the container user (root); the host
    user running xTask cannot unlink them, and once the container is gone the
    objects can only be removed manually.  The container shares the host IPC
    namespace (``--ipc=host``), so removing them from inside the container
    clears the host objects as well.

    Sequence (mirrors the multi-host ``_wait_xllm_processes_exit`` flow):
      1. pkill xllm inside the container and wait for it to exit, so the
         objects are no longer held;
      2. run tools/clean_xllm_shm.sh inside the container (as root), which
         removes the stale objects while skipping any object still held by a
         live process (a coexisting independent xllm service on the same host
         must not be crashed by deleting its live SHM).

    ``kill_xllm=False`` skips step 1: used for the pre-start cleanup where
    this container has no xllm yet — pkill in a shared PID/IPC namespace
    could hit another team's live xllm process.

    Only per-case objects (``xllm_<chip_id>_*``) are targeted; the
    ``xllm_expert_*`` namespace (other teams' MoE runs) and the host-global
    ``xllm_shm_generation_lock_v2`` are left in place, matching
    ``_cleanup_xllm_shm``.  Must run BEFORE ``container_manager.cleanup()``.
    """
    container_name = getattr(
        getattr(container_manager, "container_config", None)
        or getattr(container_manager, "config", None),
        "container_name",
        None,
    )
    if not container_name:
        return

    def _exec(cmd: str, timeout: int = 30):
        try:
            return container_manager.exec_in_container(
                container_name, cmd, timeout=timeout
            )
        except Exception:
            logging.warning(
                "In-container xLLM cleanup step failed for %s",
                container_name,
                exc_info=True,
            )
            return None

    # 1. Stop xllm and wait until it is gone (bounded by the timeout itself;
    #    the loop exits early once no xllm process remains).  Skip entirely in
    #    pre-start mode: the container's own xllm does not exist yet, and the
    #    pkill pattern is not container-scoped on hosts sharing the PID
    #    namespace — it could kill a coexisting foreign xllm process.
    if kill_xllm:
        _exec(
            "sh -c 'pkill -TERM -x xllm 2>/dev/null || true; "
            "for i in $(seq 1 30); do "
            "pgrep -x xllm >/dev/null 2>&1 || exit 0; sleep 1; "
            "done; pkill -KILL -x xllm 2>/dev/null || true'",
            timeout=45,
        )

    # 2. Remove stale objects as root inside the container, skipping any
    #    object still held by a live process (fuser exit 0 = held).  The
    #    removal logic lives in tools/clean_xllm_shm.sh (mounted at
    #    /workspace/tools/ inside the container) instead of an inline
    #    `sh -c` string: exec_in_container wraps commands in sh -c "...",
    #    and a `$f` inside that double-quoted layer is expanded to the empty
    #    string by the HOST shell before docker sees it — the previous inline
    #    while-loop therefore deleted nothing (observed on real runs).
    #    The script reports "removed: N" / "skip(held): name" lines.
    script = "/workspace/tools/clean_xllm_shm.sh"
    result = _exec(f"sh {script} 2>&1", timeout=30)
    out = getattr(result, "stdout", "") or ""
    if not isinstance(out, str):
        # Mock/fake exec results without a real stdout string
        out = ""
    out = out.strip()
    if getattr(result, "success", False):
        removed = 0
        for line in out.splitlines():
            if line.startswith("removed:"):
                try:
                    removed = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        if "fail(rm):" in out:
            logging.warning(
                "In-container SHM cleanup reported failures for %s:\n%s",
                container_name,
                out,
            )
        logging.info(
            "Cleaned %d stale xLLM SHM objects inside container %s",
            removed,
            container_name,
        )


def _wait_xllm_processes_exit(runtimes, grace_s: int = 15) -> None:
    """Signal xllm on each runtime, then wait until processes are gone.

    Instead of issuing one short ``pgrep`` per second, each runtime gets a single
    long-running shell loop. It exits 0 as soon as no xllm process remains, or exits
    1 after ``grace_s`` seconds. This reduces the cleanup log noise from ~180
    ``exec_in_container`` invocations down to one per runtime.
    """
    from xtask.core.service_group import run_on_runtimes_parallel

    def _term_one(runtime):
        return runtime.container_manager.exec_in_container(
            runtime.container_config.container_name,
            "pkill -TERM -x xllm 2>/dev/null || true",
            timeout=30,
        )

    # The nodes are independent at this point.  A slow SSH/docker exec on one
    # node must not delay TERM delivery to the other nodes.
    term_outcomes = run_on_runtimes_parallel(runtimes, _term_one)
    for runtime, _result, exc in term_outcomes:
        if exc is not None:
            logging.warning(
                "Failed to stop xLLM process for %s: %s",
                runtime.plan.instance_name,
                exc,
            )

    # xLLM children can remain as zombies until the container is removed.
    # Zombies hold no NPU resources and cannot be killed, so do not let them
    # consume the full graceful-shutdown timeout.
    live_process_check = (
        "ps -eo stat=,comm= | grep -v '^[[:space:]]*Z' | "
        "grep -q '[[:space:]]xllm'"
    )
    wait_cmd = (
        f"i=0; while [ \\$i -lt {grace_s} ]; do "
        f"{live_process_check} || exit 0; "
        "sleep 1; "
        "i=\\$((i+1)); "
        "done; exit 1"
    )

    def _wait_one(runtime):
        return runtime.container_manager.exec_in_container(
            runtime.container_config.container_name,
            wait_cmd,
            timeout=grace_s + 15,
        )

    outcomes = run_on_runtimes_parallel(runtimes, _wait_one)
    kill_runtimes = []
    for runtime, result, exc in outcomes:
        still_alive = exc is not None or (result is not None and not result.success)
        if still_alive:
            logging.warning(
                "xLLM processes still alive on %s after %ss; sending SIGKILL",
                runtime.plan.instance_name,
                grace_s,
            )
            kill_runtimes.append(runtime)

    def _kill_one(runtime):
        return runtime.container_manager.exec_in_container(
            runtime.container_config.container_name,
            "pkill -KILL -x xllm 2>/dev/null || true",
            timeout=30,
        )

    if kill_runtimes:
        kill_outcomes = run_on_runtimes_parallel(kill_runtimes, _kill_one)
        for runtime, _result, exc in kill_outcomes:
            if exc is not None:
                logging.warning(
                    "Failed to SIGKILL xLLM for %s: %s",
                    runtime.plan.instance_name,
                    exc,
                )
        # Give the runtime a short opportunity to reap the killed processes
        # before docker stop/rm runs.  Container removal remains the final
        # authority for any non-xLLM helper processes.
        time.sleep(0.2)

    # NPU/HCCL 设备状态(HBM、communicator)由驱动在引擎进程退出后才释放。
    # 并行等 66/68 两宿主 HBM 回基线，**再静置 30s settle 窗口**给 HCCL
    # communicator 释放(HBM 先干净、HCCL 后释放且 npu-smi 看不到)，否则下一档
    # rank-0 的 HcclGetRootInfo 会在脏设备上失败(status 7)→ 28888 不开 → /health 失败。
    run_on_runtimes_parallel(runtimes, _wait_npu_hbm_released)

def _cleanup_scheduler(context):
    from xtask.core.service_group import cleanup_service_group

    state = context.state
    if state.get("external_service"):
        _cleanup_external_client(context)
        return
    if state.get("gedit_specs"):
        _cleanup_gedit_services(context)
        return
    if _is_asr_task(state.get("task_config") or {}):
        _try_archive_asr_execution(context)
    # Archive into a dated directory while containers are still available so
    # root-owned aisbench directories can be chmod'd for host-side copies.  The
    # finalize-stage call is kept as a no-op safety net when this already ran.
    try:
        _ensure_benchmark_date_dir(context)
    except Exception:
        logging.error(
            "Failed to prepare fallback result directory during cleanup",
            exc_info=True,
        )

    engine_config = context.state.get("engine_config")
    should_clean = bool(
        engine_config is not None
        and not engine_config.get("serving_container_name")
        and get_xtask_config_data().get("ENV_CONFIG").get("CLEAN_RESULTS")
    )
    if context.state.get("multi_host"):
        runtimes = context.state.get("service_runtimes") or []
        # Snapshot/separate shutdown logs only when the runtime will actually
        # be stopped.  In keep-alive mode, separating logs would remove the
        # live log directory from underneath still-running xLLM processes.
        snapshots = _freeze_service_group_logs(runtimes) if should_clean else {}
        control_service = context.state.get("xllm_control_service") or (
            context.state.get("pd_control_service")
        )
        deployment = context.state.get("task_config", {}).get("deployment") or {}
        try:
            shutdown_grace_s = max(
                1, int(deployment.get("shutdown_grace_seconds", 15))
            )
        except (TypeError, ValueError):
            shutdown_grace_s = 15
        try:
            # 默认放宽到 15s：多卡 NPU 场景进程退出时的驱动级资源回收
            # （KV 传输链路 unlink、设备上下文销毁）偶发超过 5s，SIGTERM
            # 提前转 SIGKILL 会造成 "did not receive an exit event" 噪声报错。
            pd_stop_timeout = max(
                1, int(deployment.get("container_stop_timeout_seconds", 15))
            )
        except (TypeError, ValueError):
            pd_stop_timeout = 15
        try:
            if should_clean and control_service is not None:
                # Keep etcd until the routed data-plane processes have exited.
                # CLEAN_RESULTS=false deliberately leaves the complete runtime
                # (xLLM, xLLM-Service, ETCD and the containers) available.
                # The synchronous benchmark has returned: xTask sends no more
                # requests. Do not invent a drain endpoint for other clients.
                # Split Service follows Quickstart order, retaining owned etcd
                # until the Engines have released their communication state.
                from xtask.core.xllm_service_cluster import XllmServiceCluster
                if isinstance(control_service, XllmServiceCluster) and not control_service.legacy:
                    try:
                        control_service.drain_after_benchmark()
                        control_service.stop_members()
                    except Exception:
                        logging.error("Failed to stop Service members", exc_info=True)
                try:
                    _wait_xllm_processes_exit(runtimes, grace_s=shutdown_grace_s)
                finally:
                    try:
                        control_service.stop()
                    except Exception:
                        logging.error("Failed to stop xLLM control service", exc_info=True)
            elif should_clean:
                # Non-service-routed deployments never went through
                # _wait_xllm_processes_exit above; stop xllm here so the SHM
                # objects are no longer held before the in-container removal.
                _wait_xllm_processes_exit(runtimes, grace_s=shutdown_grace_s)
            if should_clean:
                # SHM cleanup must not depend on control_service: non-PD,
                # non-service-routed deployments (the common case) have none,
                # and their stale xllm_<chip>_* objects would otherwise poison
                # the next start with "incompatible shared memory layout".
                # The host-side pass only cleans the local host (and fails on
                # root-owned objects), so delete them from inside EVERY
                # instance container (as root, via each manager's own
                # executor — remote hosts included) while it still exists.
                for runtime in runtimes:
                    _cleanup_xllm_shm_in_container(
                        getattr(runtime, "container_manager", None)
                    )
                _cleanup_xllm_shm(runtimes)
        finally:
            if getattr(control_service, "is_split", False):
                try:
                    control_service.collect_logs()
                except Exception:
                    logging.warning("Failed to collect split Service logs", exc_info=True)
            if should_clean:
                _separate_shutdown_logs(runtimes, snapshots)
            for benchmarkpath in context.state.get(
                "service_group_archive_paths", []
            ):
                _archive_service_group_runtime_artifacts(
                    runtimes,
                    context.output_file_path,
                    benchmarkpath,
                )
                if getattr(control_service, "is_split", False):
                    try:
                        control_service.archive_logs(context.output_file_path, benchmarkpath)
                    except Exception:
                        logging.warning("Failed to archive Service logs", exc_info=True)
        if control_service is not None:
            cleanup_service_group(
                runtimes, should_clean, stop_timeout=pd_stop_timeout
            )
        else:
            cleanup_service_group(runtimes, should_clean)
        if should_clean and state.get("vbench_container_manager") is not None:
            state["vbench_container_manager"].cleanup()
        return

    container_manager = context.state.get("container_manager")
    if should_clean and container_manager is not None:
        # Single-host path: xLLM runs --ipc=host, so its POSIX SHM objects
        # live in the host /dev/shm and are created by root (the container
        # user).  The host user cannot unlink them, so stop xllm and remove
        # the objects from inside the container (as root) BEFORE the
        # container is torn down; a leftover poisons the next start with
        # "incompatible shared memory layout".  Multi-host does the same in
        # the service-group branch above.
        _cleanup_xllm_shm_in_container(container_manager)
        # Same driver-async-release window as the service-group path: HBM and
        # HCCL communicator are freed after the engine exits; a case running
        # immediately after on the same devices can fail HcclGetRootInfo.
        # Single-host has no ServiceRuntime, so wrap the manager in a minimal
        # shim exposing the same .container_manager.executor surface the
        # waiter polls; devices come from task_config (device_ids or
        # start_device_id/num_devices), the baseline falls back to env/default
        # (not sampled — the non-deployment path has no launch-plan hook).
        _wait_npu_hbm_released(
            types.SimpleNamespace(
                plan=types.SimpleNamespace(
                    host="local",
                    hbm_baseline_mb=0,
                    device_ids=_single_host_device_ids(
                        state.get("task_config") or {}
                    ),
                ),
                container_manager=container_manager,
            )
        )
        container_manager.cleanup()
        # Host-side fallback for objects the in-container step could not
        # reach (e.g. container already dying).  Harmless when it fails on
        # root-owned leftovers; those surface as a warning.
        _cleanup_xllm_shm([])
    if should_clean and context.state.get("vbench_container_manager") is not None:
        context.state["vbench_container_manager"].cleanup()


def _build_task_scheduler():
    from xtask.core.task_scheduler import SchedulerDependencies, TaskScheduler

    dependencies = SchedulerDependencies(
        load_task_and_engine=_load_task_and_engine,
        prepare_container=_prepare_container,
        prepare_runtime_dependencies=_prepare_runtime_dependencies,
        start_and_wait_service=_start_and_wait_service,
        run_benchmark_and_process=_run_benchmark_and_process,
        finalize_result=_finalize_result,
        cleanup=_cleanup_scheduler,
    )
    return TaskScheduler(dependencies, logging.getLogger(__name__))


@dataclass
class SchedulerSession:
    """Prepared scheduler state reused by auto-assess benchmark trials."""

    scheduler: Any
    context: Any
    base_output_file_path: str
    prepared: bool = False
    closed: bool = False
    failed: bool = False
    trial_count: int = 0


def _set_scheduler_error(context, message: str) -> None:
    context.result["success"] = False
    context.result["scheduler_error"] = True
    context.result["message"] = message
    logging.error(message, exc_info=True)


def _benchmark_only_service_config(task_config: Dict[str, Any]) -> Dict[str, Any]:
    service_config = copy.deepcopy(task_config)
    normalize_xllm_service_config(service_config)
    for key in ("benchmark", "case_name", "metrics"):
        service_config.pop(key, None)
    return service_config


def _load_session_trial_config(context, task_yaml_path: str) -> Dict[str, Any]:
    task_config = TaskConfig.load_task_config(task_yaml_path)
    if not task_config:
        raise ValueError(f"Failed to load task config from {task_yaml_path}")

    if context.args:
        host_addr_override = getattr(context.args, "host_addr", None)
        if host_addr_override:
            from xtask.core.service_group import apply_host_addr_override

            apply_host_addr_override(task_config, host_addr_override)

    from xtask.core.external_service import (
        apply_external_service_to_task_config,
        is_external_service_enabled,
        validate_external_service_config,
    )

    if is_external_service_enabled(task_config):
        validate_external_service_config(task_config)
        apply_external_service_to_task_config(task_config)

    base_config = context.state.get("source_task_config") or {}
    if _benchmark_only_service_config(task_config) != _benchmark_only_service_config(
        base_config
    ):
        raise ValueError(
            "Auto-assess scheduler session can only change benchmark parameters; "
            "service or deployment configuration changed"
        )
    return task_config


def _reset_session_trial(
    session: SchedulerSession,
    task_yaml_path: str,
    output_file_path: str,
) -> None:
    context = session.context
    state = context.state
    task_config = _load_session_trial_config(context, task_yaml_path)
    os.makedirs(output_file_path, exist_ok=True)
    benchmark_cmd = generate_benchmark_command(
        task_config,
        output_file_path,
        service_plans=state.get("service_plans") or [],
    )

    previous_result = context.result
    result = _initial_result()
    result.update(
        container_id=previous_result.get("container_id"),
        task_config=task_config,
        framework_config=state.get("engine_config"),
        startup_command=state.get("startup_cmd", ""),
        benchmark_command=benchmark_cmd,
        output_file_path=output_file_path,
    )
    if previous_result.get("instances") is not None:
        result["instances"] = previous_result["instances"]
    if previous_result.get("external_service"):
        result["external_service"] = True

    context.task_yaml_path = task_yaml_path
    context.output_file_path = output_file_path
    context.result = result
    context.complete = False
    state["task_config"] = task_config
    state["benchmark_cmd"] = benchmark_cmd
    state["init_entries"] = os.listdir(output_file_path)
    state.pop("benchmark_result", None)

    if state.get("multi_host") and output_file_path != session.base_output_file_path:
        from xtask.core.service_group import get_benchmark_runtime

        runner = get_benchmark_runtime(state.get("service_runtimes") or [])
        if runner is not None and not runner.plan.is_local:
            if _is_asr_task(task_config):
                transferred = _sync_asr_trial_inputs(runner, output_file_path)
            else:
                transferred = runner.container_manager.executor.upload(output_file_path, output_file_path)
            if not transferred:
                raise RuntimeError(
                    f"Failed to sync benchmark trial to {runner.plan.host}"
                )
            state["session_remote_benchmark_runner"] = runner
        else:
            state.pop("session_remote_benchmark_runner", None)
    else:
        state.pop("session_remote_benchmark_runner", None)


def _check_session_service_health(context) -> bool:
    state = context.state
    if state.get("external_service"):
        return _probe_external_service(context)
    task_config = state["task_config"]
    manager = state.get("container_manager")
    container_config = state.get("container_config")
    if manager is None or container_config is None:
        context.result["message"] = "Scheduler session has no benchmark runtime"
        state["service_healthy"] = False
        return False

    if state.get("multi_host"):
        from xtask.core.service_group import get_benchmark_endpoint

        endpoint = get_benchmark_endpoint(
            task_config, state.get("service_plans") or []
        )
        health_path = (task_config.get("deployment") or {}).get(
            "benchmark", {}
        ).get("health_path", "/health")
        check_url = endpoint.rstrip("/") + health_path
        check_cmd = f"curl --noproxy '*' -i {check_url}"
    else:
        check_url = (
            f"http://{task_config.get('host_addr', '0.0.0.0')}:"
            f"{task_config.get('port', 14830)}/health"
        )
        check_cmd = f"curl -i {check_url}"

    control_service = state.get("xllm_control_service")
    if getattr(control_service, "is_split", False):
        ready = control_service.wait_for_http_endpoint(timeout=90)
        state["service_healthy"] = ready
        if not ready:
            context.result["message"] = (
                "Reused Router endpoint is not ready: " + control_service.last_error
            )
        return ready

    health_result = manager.exec_in_container(
        container_config.container_name, check_cmd, timeout=30
    )
    state["health_result"] = health_result
    state["service_healthy"] = bool(
        health_result.success and "ok" in (health_result.stdout or "").lower()
    )
    if not state["service_healthy"]:
        context.result["message"] = (
            f"Reused service health check failed for {check_url}: "
            f"{health_result.stderr or health_result.stdout}"
        )
        logging.error(context.result["message"])
    return state["service_healthy"]


def start_scheduler_session(
    task_yaml_path: str, output_file_path: str, args=None
) -> SchedulerSession:
    """Prepare one service lifecycle for multiple auto-assess trials."""
    from xtask.core.task_scheduler import TaskExecutionContext

    session_started_at = time.time()
    context = TaskExecutionContext(
        task_yaml_path,
        output_file_path,
        args,
        result=_initial_result(),
    )
    session = SchedulerSession(
        scheduler=_build_task_scheduler(),
        context=context,
        base_output_file_path=output_file_path,
    )
    try:
        session.scheduler.prepare(context)
        session.prepared = not context.complete
    except Exception as exc:
        session.failed = True
        _set_scheduler_error(context, f"Error preparing scheduler session: {exc}")

    if not session.prepared:
        close_scheduler_session(session)
        context.result["execution_time"] = time.time() - session_started_at
        session.scheduler.finalize_result(context)
    return session


def run_scheduler_session_trial(
    session: SchedulerSession,
    task_yaml_path: str,
    output_file_path: str,
) -> Dict[str, Any]:
    """Run one benchmark trial without recreating containers or services."""
    if session.closed or not session.prepared or session.failed:
        return session.context.result

    trial_started_at = time.time()
    try:
        _reset_session_trial(session, task_yaml_path, output_file_path)
        _check_session_service_health(session.context)
        session.scheduler.run_trial(session.context)
        session.trial_count += 1
    except Exception as exc:
        session.failed = True
        _set_scheduler_error(
            session.context, f"Error running scheduler session trial: {exc}"
        )
    finally:
        session.context.result["execution_time"] = time.time() - trial_started_at
        session.scheduler.finalize_result(session.context)
    return session.context.result


def close_scheduler_session(session: Optional[SchedulerSession]) -> None:
    """Clean one reusable scheduler session exactly once."""
    if session is None or session.closed:
        return
    try:
        session.scheduler.cleanup(session.context)
    except Exception:
        logging.error("Scheduler session cleanup failed", exc_info=True)
    finally:
        session.closed = True


def run_scheduler(task_yaml_path: str, output_file_path: str, args = None) -> Dict[str, Any]:
    """Schedule one local inference task through explicit lifecycle stages."""
    logging.debug("entry function: %s", inspect.currentframe().f_code.co_name)
    from xtask.core.task_scheduler import TaskExecutionContext

    context = TaskExecutionContext(
        task_yaml_path,
        output_file_path,
        args,
        result=_initial_result(),
    )
    return _build_task_scheduler().run(context)

def cleanup_containers(container_names: list, force: bool = False):
    """
    清理容器
    
    Args:
        container_names: 要清理的容器名称列表
        force: 是否强制清理
    """
    for container_name in container_names:
        try:
            # 创建临时容器管理器进行清理
            temp_config = ContainerConfig(
                image_name="dummy",
                container_name=container_name,
                runtime="docker"
            )
            
            manager = ContainerManager(temp_config)
            
            if manager.container_exists(container_name):
                logging.info(f"Cleaning up container: {container_name}")
                
                # 停止容器
                manager.stop_container(container_name)
                
                # 移除容器
                manager.remove_container(container_name, force=force)
                
                logging.info(f"Container {container_name} cleaned up")
            else:
                logging.info(f"Container {container_name} does not exist")
                
        except Exception as e:
            logging.error(f"Failed to cleanup container {container_name}: {e}")


if __name__ == "__main__":
    """测试调度器功能"""
    import logging as test_logging
    
    # 设置日志
    test_logging.basicConfig(
        level=test_logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建测试任务YAML文件
    test_yaml_content = """
    case_name: test-xllm-inference
    model_path: ./models/llama-7b
    served_model_name: xllm-llama-7b
    start_device_id: 0
    max_memory_utiliization: 0.9
    nodes: 1
    world_size: 1
    node_addr: 0.0.0.0
    node_port: 14830
    max_tokens_per_batch: 20000
    max_seqs_per_batch: 256
    enable_mla: false
    block_size: 128
    communication_backend: "hccl"
    enable_prefix_cache: false
    enable_chunked_prefill: false
    enable_schedule_overlap: false
    enable_mtp: false

    benchmark:
    benchmark_type: vllm_ascend_bencmark
    perf:
        case_type: performance
        dataset_path: ./data/test.json
        dataset_conf:
        request_conf:
        num_prompts: 100
        output_len: 128
        batch_size: 16
        request_rate: 10
        baseline:
        threshold: 0.97
    acc:
        case_type: accuracy
        dataset_path: vllm
        dataset_conf:
        request_conf:
        num_prompts: 100
        output_len: 128
        batch_size: 16
        request_rate: 10
        baseline: 95
        threshold: 5
    """
    
    # 保存测试YAML文件
    test_yaml_path = "./test_xllm_task.yaml"
    with open(test_yaml_path, 'w') as f:
        f.write(test_yaml_content)
    
    print("=" * 60)
    print("Testing XLLM scheduler...")
    print("=" * 60)
    
    try:
        # 运行调度器
        result = run_scheduler(test_yaml_path)
        
        print("\nScheduler Result:")
        print(f"Success: {result.get('success', False)}")
        print(f"Message: {result.get('message', '')}")
        print(f"Container ID: {result.get('container_id', 'None')}")
        print(f"Execution Time: {result.get('execution_time', 0):.2f}s")
        
        if result.get('benchmark_success') is not None:
            print(f"Benchmark Success: {result.get('benchmark_success')}")
        
        # 清理测试容器
        if result.get('container_id'):
            print(f"\nCleaning up test container: {result['container_id']}")
            cleanup_containers([result['container_id']], force=True)
        
    finally:
        # 清理测试文件
        if os.path.exists(test_yaml_path):
            os.remove(test_yaml_path)
        
        print("\nTest completed.")
