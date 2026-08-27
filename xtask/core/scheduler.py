import inspect
import logging
import os
import json
import time
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
            "benchmark_timeout": 7200,
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
        volumes = get_xtask_config_data().get("DOCKER_CONFIG").get("volumes")
        
        # 从任务配置中获取模型路径，如果存在则添加到挂载卷
        # model_path = task_config.get("model_path", "")
        # if model_path and os.path.exists(model_path):
        #     volumes[os.path.abspath(model_path)] = f"{work_dir}/model_data"
        
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
    from xtask.core.service_group import is_multi_host_deployment

    def _bench_task_config_with_endpoint():
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
    from xtask.core.service_group import is_multi_host_deployment

    work_dir = engine_config.get("working_dir", "/workspace")
    config_file = f"{work_dir}/config/{os.path.basename(task_yaml_path)}"
    startup_cmd = ""
    benchmark_cmd = ""
    
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

def _build_warmup_command(task_config, output_file_path):
    """构建预热命令：深拷贝配置并合并 warmup 参数"""
    warmup_task_config = copy.deepcopy(task_config)
    warmup_paras = task_config["benchmark"]["perf"].get("warmup_paras", {})
    warmup_task_config["benchmark"]["perf"].update(warmup_paras)

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
                                      run_on_host=False):
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
                        timeout=10,
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
        guard = NpuProcessGuard(
            npu_card_ids=list(npu_card_ids),
            check_interval=engine_config.get("npu_guard_interval", 180.0),
            startup_grace=engine_config.get("npu_guard_startup_grace", 0.0),
            container_name=container_config.container_name,
        )
        guards.append(guard)
        guard.start()
        abort_event = guard.abort_event

    try:
        executor = (
            container_manager.executor.run
            if run_on_host
            else lambda command, **kwargs: container_manager.exec_in_container(
                container_config.container_name, command, **kwargs
            )
        )
        benchmark_result = executor(
            benchmark_cmd,
            timeout=engine_config.get("benchmark_timeout", 3600),
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

    source_task_config = copy.deepcopy(task_config)
    context.result["task_config"] = task_config
    context.result["output_file_path"] = context.output_file_path
    engine = TaskConfig.extract_framework(task_config)
    logging.info(f"Detected engine type: {engine}")

    try:
        config_path = os.path.join(os.getcwd(), "config", f"{engine}_config.json")
        config_data = Json_Reader.load_config(config_path)
        engine_config = Json_Reader.get_container_config(config_data, engine)
    except Exception as exc:
        _fail_stage(context, f"Failed to load XLLM config: {exc}")
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
    multi_host = is_multi_host_deployment(task_config)
    context.state.update(
        task_config=task_config,
        source_task_config=source_task_config,
        engine=engine,
        config_data=config_data,
        engine_config=engine_config,
        startup_cmd=startup_cmd,
        benchmark_cmd=benchmark_cmd,
        multi_host=multi_host,
        service_plans=service_plans or [],
        pd_disaggregated=(
            task_config.get("deployment", {}).get("mode") == "pd_disaggregated"
            or any(
                str(instance.get("role", "")).lower() in {"prefill", "decode"}
                for instance in (task_config.get("deployment", {}).get("instances") or [])
            )
        ),
        service_routed=(
            task_config.get("deployment", {}).get("mode") == "service_routed"
        ),
    )
    if multi_host and not service_plans:
        _fail_stage(
            context,
            "Multi-host deployment detected but service plans were not generated",
        )


def _create_container(context, container_manager, container_config):
    logging.info(f"Creating new container: {container_config.container_name}")
    success, container_id = container_manager.create_container()
    if not success:
        _fail_stage(context, f"Failed to create container: {container_id}")
        return False
    context.result["container_id"] = container_id
    return True


def _prepare_container(context):
    state = context.state
    if state.get("multi_host"):
        _prepare_container_multi_host(context)
        return

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
            sync_workspace_to_hosts(
                [plan.host for plan in plans],
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
        reuse_existing = bool(
            plan.instance_config.get("reuse_existing_container")
            or plan.instance_config.get("container_name")
        )
        if not manager.check_runtime_available():
            _fail_stage(
                context,
                f"Container runtime not available on host {plan.host}",
            )
            state["service_runtimes"] = runtimes
            return

        if reuse_existing:
            name = runtime.container_config.container_name
            if not manager.container_exists(name):
                _fail_stage(context, f"Existing container {name} not found on {plan.host}")
                state["service_runtimes"] = runtimes
                return
            if not manager.container_is_running(name) and not manager.start_container(name):
                _fail_stage(context, f"Failed to start existing container {name}")
                state["service_runtimes"] = runtimes
                return
            inspect_result = manager.run_command(
                f"{runtime.container_config.runtime} inspect -f '{{{{.Id}}}}' {name}",
                timeout=30,
            )
            container_id = inspect_result.stdout.strip() or name
            logging.info("Reusing existing container %s on %s", name, plan.host)
        else:
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

        runtimes.append(runtime)
        instances_meta[plan.instance_name] = {
            "host": plan.host,
            "role": plan.role,
            "container_id": container_id,
            "container_name": runtime.container_config.container_name,
            "rank_start": plan.rank_start,
            "device_ids": plan.device_ids,
            "reused_existing": reuse_existing,
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
    control_key = (
        "service_control"
        if deployment.get("mode") == "service_routed"
        else "pd_control"
    )
    control = deployment.get(control_key) or {}
    host = control.get("host") or deployment.get("master_host")
    return str(host) if host else None


def _download_xllm_command(install_service):
    value = 1 if install_service else 0
    return f"XLLM_INSTALL_SERVICE={value} bash /workspace/tools/download_xllm.sh;"


def _prepare_runtime_dependencies(context):
    state = context.state
    if state.get("multi_host"):
        _prepare_runtime_dependencies_multi_host(context)
        return

    task_config = state["task_config"]
    engine_config = state["engine_config"]
    container_config = state["container_config"]
    container_manager = state["container_manager"]
    timeout = engine_config.get("startup_timeout", 300)

    compilation_cmd = _download_xllm_command(
        _uses_xllm_control_service(task_config)
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
    if benchmark_type == "aisbench":
        down_benchmark_cmd = "bash /workspace/tools/download_ais_bench.sh;"
    elif benchmark_type == "evalscope":
        down_benchmark_cmd = "bash /workspace/tools/download_evalscope.sh;"
    logging.info(f"Starting download the benchmark tool: {down_benchmark_cmd}")
    container_manager.exec_in_container(
        container_config.container_name, down_benchmark_cmd, timeout=timeout
    )

    model_config_path = os.path.join(task_config.get("model_path"), "config.json")
    if not os.path.exists(model_config_path):
        logging.error(f"model config file: {model_config_path} is not exist")
        context.complete = True
        return

    with open(model_config_path, "r") as config_file:
        transformers_version = json.load(config_file).get("transformers_version")
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

    control_runtime = None
    if _uses_xllm_control_service(task_config):
        control_host = _xllm_control_host(task_config)
        control_runtime = next(
            (
                runtime
                for runtime in runtimes
                if str(runtime.plan.host) == str(control_host)
            ),
            None,
        )
        if control_runtime is None:
            _fail_stage(
                context,
                f"xLLM control host {control_host!r} has no service runtime",
            )
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
    if benchmark_type == "aisbench":
        down_benchmark_cmd = "bash /workspace/tools/download_ais_bench.sh;"
    elif benchmark_type == "evalscope":
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
        runtime_cfg = getattr(runtime.plan, "instance_config", {}) or {}
        reuse_existing = bool(
            runtime_cfg.get("reuse_existing_container")
            or runtime_cfg.get("container_name")
        )
        if reuse_existing:
            binary = runtime_cfg.get("xllm_binary", "xllm")
            triton_path = runtime_cfg.get("triton_binary_path")
            checks = [f"test -x '{binary}'"]
            if triton_path:
                checks.append(f"test -d '{triton_path}'")
            result = manager.exec_in_container(
                name, " && ".join(checks), timeout=30
            )
            if not result.success:
                raise RuntimeError(
                    f"existing xLLM artifacts are invalid in {name}: "
                    f"{result.stderr or result.stdout}"
                )
            logging.info("Validated existing xLLM artifacts in %s", name)
            return True
        download_cmd = _download_xllm_command(runtime is control_runtime)
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
        install_result = manager.exec_in_container(
            name,
            f"pip install transformers=={transformers_version} --upgrade",
            timeout=timeout,
        )
        if not install_result.success:
            manager.exec_in_container(
                name, "pip install --upgrade transformers;", timeout=timeout
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


def _start_and_wait_service(context):
    state = context.state
    if state.get("multi_host"):
        _start_and_wait_service_multi_host(context)
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
                plan.instance_config.get("serving_ready_timeout") or 320
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


def _start_and_wait_service_multi_host(context):
    from xtask.core.service_group import (
        get_benchmark_endpoint,
        get_benchmark_runtime,
        run_on_runtimes_parallel,
    )

    state = context.state
    task_config = state["task_config"]
    if state.get("pd_disaggregated") or state.get("service_routed"):
        from xtask.core.pd_control_service import (
            XllmControlService,
            build_pd_control_config,
            build_service_control_config,
        )

        control_config = (
            build_pd_control_config(task_config)
            if state.get("pd_disaggregated")
            else build_service_control_config(task_config)
        )
        control_runtime = next(
            (
                runtime
                for runtime in state.get("service_runtimes") or []
                if str(runtime.plan.host) == str(control_config.host)
            ),
            None,
        )
        if control_runtime is None:
            _fail_stage(
                context,
                f"xLLM control host {control_config.host} has no service runtime",
            )
            return
        control_service = XllmControlService(
            control_runtime,
            control_config,
            task_config.get("model_path", ""),
            context.output_file_path,
        )
        try:
            control_service.start_with_retry()
        except Exception as exc:
            _fail_stage(context, f"xLLM control service startup failed: {exc}")
            state["xllm_control_service"] = control_service
            if state.get("pd_disaggregated"):
                state["pd_control_service"] = control_service
            return
        state["xllm_control_service"] = control_service
        if state.get("pd_disaggregated"):
            state["pd_control_service"] = control_service
    engine_config = state["engine_config"]
    runtimes = state.get("service_runtimes") or []
    serving_wait_time = (
        task_config.get("serving_wait_time")
        if task_config.get("serving_wait_time")
        else state["config_data"].get("XLLM_CONFIG").get("serving_wait_time")
    )

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
                f"xLLM control HTTP endpoint did not open at "
                f"http://{control_service.config.host}:"
                f"{control_service.config.http_server_port}/health",
            )
            state["service_healthy"] = False
            return

    endpoint = get_benchmark_endpoint(task_config, state["service_plans"])
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
    if state["service_healthy"]:
        logging.info("Unified service endpoint is healthy: %s", check_url)
    else:
        context.result["message"] = (
            f"Service health check failed for {check_url}: {health_result.stderr}"
        )
        logging.error(context.result["message"])

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


def _run_benchmark_and_process(context):
    state = context.state
    if not state["service_healthy"]:
        return

    task_config = state["task_config"]
    engine_config = state["engine_config"]
    container_config = state["container_config"]
    container_manager = state["container_manager"]
    benchmark_cmd = state["benchmark_cmd"]
    init_entries = state["init_entries"]
    logging.info(f"Executing benchmark command: {benchmark_cmd}")

    benchmark_result = None
    if benchmark_cmd:
        perf_cfg = task_config.get("benchmark").get("perf", {})
        if (
            task_config.get("benchmark").get("benchmark_type") == "evalscope"
            and perf_cfg.get("warmup_paras")
        ):
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
        run_on_host = bool(
            (task_config.get("deployment") or {})
            .get("benchmark", {})
            .get("run_on_host", False)
        )
        benchmark_result = _execute_benchmark_with_npu_guard(
            container_manager,
            container_config,
            task_config,
            engine_config,
            benchmark_cmd,
            service_runtimes=service_runtimes,
            run_on_host=run_on_host,
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
        if remote_runner is not None:
            if not _download_remote_runtime_path(
                remote_runner,
                context.output_file_path, context.output_file_path
            ):
                raise RuntimeError(
                    f"Failed to download benchmark result from "
                    f"{remote_runner.plan.host}"
                )

        if benchmark_result.success:
            logging.info("perf benchmark completed successfully.")
            if "acc" in task_config.get("metrics"):
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
            else:
                logging.debug(
                    "metrics is %s, skip accuracy task.",
                    task_config.get("metrics"),
                )

            if "perf" in task_config.get("metrics"):
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
        elif benchmark_result is not None and benchmark_result.success:
            context.result["message"] = "Benchmark result verification failed"
        logging.error("verify result failed.")


def _finalize_result(context):
    task_config = context.result.get("task_config") or context.state.get(
        "task_config"
    )
    try:
        _ensure_benchmark_date_dir(context)
    except Exception:
        logging.error("Failed to prepare fallback result directory", exc_info=True)
    write_result(
        context.result,
        task_config,
        task_path=context.task_yaml_path,
        args=context.args,
    )


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


def _cleanup_scheduler(context):
    from xtask.core.service_group import cleanup_service_group

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
        reused_runtimes = [
            runtime for runtime in runtimes
            if (
                getattr(runtime.plan, "instance_config", {}) or {}
            ).get("reuse_existing_container")
            or (
                getattr(runtime.plan, "instance_config", {}) or {}
            ).get("container_name")
        ]
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
            pd_stop_timeout = max(
                1, int(deployment.get("container_stop_timeout_seconds", 5))
            )
        except (TypeError, ValueError):
            pd_stop_timeout = 5
        try:
            # A reused container must survive cleanup, but its xLLM workers must
            # stop so the next para_set trial can bind the same ports/devices.
            if reused_runtimes:
                _wait_xllm_processes_exit(
                    reused_runtimes, grace_s=shutdown_grace_s
                )
            if should_clean and control_service is not None:
                # Stop routed data-plane processes before etcd/xllm-service,
                # otherwise instances keep retrying heartbeat/etcd during cleanup.
                # CLEAN_RESULTS=false deliberately leaves the complete runtime
                # (xLLM, xLLM-Service, ETCD and the containers) available.
                _wait_xllm_processes_exit(runtimes, grace_s=shutdown_grace_s)
                try:
                    control_service.stop()
                except Exception:
                    logging.error("Failed to stop xLLM control service", exc_info=True)
        finally:
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
        if control_service is not None:
            cleanup_service_group(
                runtimes, should_clean, stop_timeout=pd_stop_timeout
            )
        else:
            cleanup_service_group(runtimes, should_clean)
        return

    container_manager = context.state.get("container_manager")
    if should_clean and container_manager is not None:
        container_manager.cleanup()


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
            if not runner.container_manager.executor.upload(
                output_file_path, output_file_path
            ):
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
