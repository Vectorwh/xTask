import inspect
import logging
import os
import json
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from xtask.utils.container_manager import ContainerManager, ContainerConfig
from xtask.utils.config_parser import get_config_data
from xtask.serv_fmk.serv_common import wait_serv_start


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
    def get_container_config(config_data: Dict[str, Any], framework: str = "xllm") -> Dict[str, Any]:
        """
        从XLLM配置中获取容器配置
        
        Args:
            config_data: JSON配置
            framework: 框架名称
            
        Returns:
            容器配置字典
        """
        cfg_data = config_data.get(f"{framework.upper()}_CONFIG", {})
        
        # 构建容器配置
        container_config = {
            "framework": framework,
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
                "HTTPS_PROXY": cfg_data.get("https_proxy_url", ""),
                "HTTP_PROXY": cfg_data.get("http_proxy_url", ""),
                "WORKING_DIR": "/workspace"
            },
            "serving_container_name": cfg_data.get("serving_container_name", ""),
            "serving_script_path": cfg_data.get("serving_script_path", ""),
            "startup_command": "",  # 启动命令（根据任务动态生成）
            "startup_timeout": 300,
            "benchmark_command": "",  # 基准测试命令（根据任务动态生成）
            "benchmark_timeout": 3600,
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
                "HTTPS_PROXY": cfg_data.get("https_proxy_url", ""),
                "HTTP_PROXY": cfg_data.get("http_proxy_url", ""),
            })
        
        logging.info(f"Generated container config for framework: {framework}")
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
            
            logging.info(f"Loaded task config from: {task_yaml_path}")
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
        framework_config: Dict[str, Any],
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
        container_home = framework_config.get("container_home", "xllm/xllm-ai")
        container_image = framework_config.get("container_image", "latest")
        
        # 构建容器名称（使用case_name或生成唯一名称）
        case_name = task_config.get("case_name", "")
        if case_name:
            container_name = f"{case_name}-{int(time.time())}"
        else:
            container_name = f"xllm-task-{int(time.time())}"
        
        # 获取工作目录
        work_dir = framework_config.get("working_dir", "/workspace")
        
        # 构建挂载卷
        volumes = get_config_data().get("DOCKER_CONFIG").get("volumes")
        
        # 从任务配置中获取模型路径，如果存在则添加到挂载卷
        # model_path = task_config.get("model_path", "")
        # if model_path and os.path.exists(model_path):
        #     volumes[os.path.abspath(model_path)] = f"{work_dir}/model_data"
        
        # 构建环境变量
        environment = {
            "TASK_CONFIG_PATH": f"{work_dir}/config/{os.path.basename(task_yaml_path)}",
            "FRAMEWORK": framework_config.get("framework", "xllm"),
            "WORKING_DIR": work_dir,
            "MODEL_PATH": task_config.get("model_path", ""),
            "SERVED_MODEL_NAME": task_config.get("served_model_name", ""),
            "START_DEVICE_ID": str(task_config.get("start_device_id", 0)),
            "MAX_MEMORY_UTILIZATION": str(task_config.get("max_memory_utiliization", 0.9)),
            "NODES": str(task_config.get("nodes", 1)),
            "WORLD_SIZE": str(task_config.get("world_size", 1)),
            "NODE_ADDR": task_config.get("node_addr", "0.0.0.0"),
            "NODE_PORT": str(task_config.get("node_port", 14830))
        }
        
        # 添加框架配置中的环境变量
        environment.update(framework_config.get("environment", {}))
        
        # 构建容器配置
        container_config = ContainerConfig(
            image_home=container_home,
            image_name=container_image,
            container_name=container_name,
            runtime=framework_config.get("runtime", "docker"),
            gpu_ids=framework_config.get("gpu_ids", []),
            ports=framework_config.get("ports", {}),
            volumes=volumes,
            environment=environment,
            working_dir=work_dir,
            entrypoint=framework_config.get("entrypoint"),
            command=framework_config.get("command"),
            network_mode=framework_config.get("network_mode", "host"),
            privileged=framework_config.get("privileged", True),
            auto_remove=framework_config.get("auto_remove", False),
            detach=True,
            stdin_open=framework_config.get("stdin_open", True),
            tty=framework_config.get("tty", True)
        )
        
        logging.info(f"Created container config: {container_name}")
        return container_config


def generate_commands(
    framework_config: Dict[str, Any], 
    task_config: Dict[str, Any],
    task_yaml_path: str
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
    work_dir = framework_config.get("working_dir", "/workspace")
    config_file = f"{work_dir}/config/{os.path.basename(task_yaml_path)}"
    startup_cmd = ""
    benchmark_cmd = ""
    
    # 生成启动命令
    if framework_config.get("framework") == "xllm":
        from xtask.serv_fmk.xllm_serv.generate_xllm_serv import generate_xllm_serv_commands

        startup_cmd = generate_xllm_serv_commands(task_config, framework_config)

    elif framework_config.get("framework") == "vllm-ascend":
        from xtask.serv_fmk.vllm_ascend_serv.generate_vllm_ascend_serv \
            import generate_vllm_ascend_serv_commands

        startup_cmd = generate_vllm_ascend_serv_commands(task_config, framework_config)

    elif framework_config.get("framework") == "mindie":
        from xtask.serv_fmk.mindie_serv.generate_mindie_serv \
            import generate_mindie_serv_commands

        startup_cmd = generate_mindie_serv_commands(task_config, framework_config)

    else:
        logging.error("framework: %s is invalid."% framework_config.get("framework"))
    
    # 生成基准测试命令
    if task_config.get("benchmark").get("benchmark_type") == "aisbench":
        from xtask.benckmarks.aisbench.generate_aisbench_script import generate_aisbench_script
        benchmark_cmd = generate_aisbench_script(
            task_config=task_config,
            output_shell_file="run_aisbench_test.sh",
            python_exec="python"
        )
    elif task_config.get("benchmark").get("benchmark_type") == "vllm_benchmark":
        pass
    else:
        logging.error("benchmark_type: %s is not supported yet."% 
                      task_config.get("benchmark").get("benchmark_type"))
    
    return startup_cmd, benchmark_cmd


def run_scheduler(task_yaml_path: str) -> Dict[str, Any]:
    """
    调度器主函数
    
    Args:
        task_yaml_path: 任务YAML文件路径
        
    Returns:
        执行结果字典
    """
    logging.debug("entry function: %s" % inspect.currentframe().f_code.co_name)
    
    result = {
        "success": False,
        "message": "",
        "container_id": None,
        "task_config": None,
        "framework_config": None,
        "startup_command": "",
        "benchmark_command": "",
        "execution_time": 0,
        "logs": ""
    }
    
    start_time = time.time()
    
    try:
        # 1. 读取任务YAML文件
        logging.info(f"Loading task config from: {task_yaml_path}")
        task_config = TaskConfig.load_task_config(task_yaml_path)
        
        if not task_config:
            result["message"] = f"Failed to load task config from {task_yaml_path}"
            logging.error(result["message"])
            return result
        
        result["task_config"] = task_config
        
        # 2. 提取框架信息
        framework = TaskConfig.extract_framework(task_config)
        logging.info(f"Detected framework: {framework}")
        
        # 3. 加载XLLM配置
        try:
            config_path = os.path.join(os.getcwd(), "config", f"{framework}_config.json")
            config_data = Json_Reader.load_config(config_path)
            framework_config = Json_Reader.get_container_config(config_data, framework)
            result["framework_config"] = framework_config
        except Exception as e:
            result["message"] = f"Failed to load XLLM config: {str(e)}"
            logging.error(result["message"])
            return result
        
        # 4. 生成命令
        startup_cmd, benchmark_cmd = generate_commands(framework_config, task_config, task_yaml_path)
        result["startup_command"] = startup_cmd
        result["accuray_command"] = startup_cmd
        result["benchmark_command"] = benchmark_cmd
        
        # 5. 创建容器配置
        container_config = ContainerConfigBuilder.create_container_config(
            framework_config, 
            task_config, 
            task_yaml_path
        )
        
        # 6. 创建容器管理器
        container_manager = ContainerManager(container_config, logging.getLogger())
        
        # 7. 检查容器运行时
        if not container_manager.check_runtime_available():
            result["message"] = f"Container runtime {container_config.runtime} not available"
            logging.error(result["message"])
            return result
        
        image_url = f"{container_config.image_home}:{container_config.image_name}"
        # 8. 检查指定容器是否已存在
        if framework_config.get("serving_container_name") and \
            container_manager.container_exists(framework_config.get("serving_container_name")):

            logging.info(f"Container {framework_config.get('serving_container_name')} already exists")
            if container_manager.container_is_running(framework_config.get("serving_container_name")):
                logging.info(f"Container logging.inf{framework_config.get('serving_container_name')}o( is already running")
                result["container_id"] = framework_config.get("serving_container_name")
                container_config.container_name = framework_config.get("serving_container_name")

        elif not container_manager.image_exists(image_url):
            # 8. 检查镜像是否存在，不存在则拉取
            logging.info(f"Image {image_url} not found locally, pulling...")
            
            if not container_manager.pull_image(image_url):
                result["message"] = f"Failed to pull image {image_url}"
                logging.error(result["message"])
                return result
        
            # 9. 检查容器是否已存在
            if container_manager.container_exists(container_config.container_name):
                logging.info(f"Container {container_config.container_name} already exists")
                
                if container_manager.container_is_running(container_config.container_name):
                    logging.info(f"Container {container_config.container_name} is already running")
                    result["container_id"] = container_config.container_name
                else:
                    # 启动已存在的容器
                    logging.info(f"Starting existing container {container_config.container_name}")
                    if container_manager.start_container(container_config.container_name):
                        result["container_id"] = container_config.container_name
                    else:
                        result["message"] = f"Failed to start existing container {container_config.container_name}"
                        logging.error(result["message"])
                        return result
            else:
            # 10. 创建并启动新容器
                logging.info(f"Creating new container: {container_config.container_name}")
                success, container_id = container_manager.create_container()
                
                if not success:
                    result["message"] = f"Failed to create container: {container_id}"
                    logging.error(result["message"])
                    return result
                
                result["container_id"] = container_id
        else:
            # 10. 创建并启动新容器
            logging.info(f"Creating new container: {container_config.container_name}")
            success, container_id = container_manager.create_container()
            
            if not success:
                result["message"] = f"Failed to create container: {container_id}"
                logging.error(result["message"])
                return result
            
            result["container_id"] = container_id
        
        # 11. 等待容器就绪
        logging.info(f"Waiting for container {container_config.container_name} to be ready...")
        if not container_manager.wait_for_container(container_config.container_name, timeout=60):
            result["message"] = f"Container {container_config.container_name} failed to start within timeout"
            logging.error(result["message"])
            return result
        
        # 12. 复制任务YAML文件到容器
        container_yaml_path = f"{container_config.working_dir}/outputs/{os.path.basename(task_yaml_path)}"
        logging.info(f"Copying task YAML to container: {container_yaml_path}")
        
        if not container_manager.copy_to_container(
            container_config.container_name,
            task_yaml_path,
            container_yaml_path
        ):
            result["message"] = f"Failed to copy task YAML to container"
            logging.warning(result["message"])
            # 继续执行，可能文件已经通过挂载卷可用
        
        # 13. TODO: 待增加配置：编译/取版本包
        compilation_cmd = "bash /workspace/tools/download_xllm.sh;"
        logging.info(f"Starting the compilation task: {compilation_cmd}")
        
        startup_result = container_manager.exec_in_container(
            container_config.container_name,
            compilation_cmd,
            timeout=framework_config.get("startup_timeout", 300)
        )

        # 读取model/config.json中 transformers的版本号，容器中transformers升级到指定版本
        model_config_path = os.path.join(task_config.get("model_path"), "config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r') as f:
                transformers_version = json.load(f).get("transformers_version")
            install_transformers_cmd = f"pip install transformers=={transformers_version} --upgrade"
            install_result = container_manager.exec_in_container(
                container_config.container_name,
                install_transformers_cmd,
                timeout=framework_config.get("startup_timeout", 300)
            )
            if not install_result.success:
                logging.error("transformer upgrade to %s failed."% transformers_version)
                return result
        else:
            logging.error("model config file: %s is not exist"% model_config_path)
            return result

        # TODO：cleanup 指定npu上的任务

        # 14. 执行启动命令
        logging.info(f"Executing startup command: {startup_cmd}")
        
        startup_result = container_manager.exec_in_container(
            container_config.container_name,
            startup_cmd,
            timeout=framework_config.get("startup_timeout", 300)
        )
        
        if not startup_result.success:
            result["message"] = f"Startup command failed: {startup_result.stderr}"
            logging.error(result["message"])
            # 继续执行，可能服务启动需要时间
        else:
            logging.debug("service startup message is: %s"% {startup_result.stderr})
        
        # 15. 等待服务启动
        logging.info("Waiting for service to start...")
        ret = wait_serv_start(framework, task_config)
        if ret:
            logging.error(f"{framework} serv startup failed.")
        
        # 16. 检查服务是否运行
        time.sleep(15) # 等待服务真正启动
        check_cmd = f"curl -s http://{task_config.get('node_addr', '0.0.0.0')}:{task_config.get('node_port', 14830)}/health"
        health_result = container_manager.exec_in_container(
            container_config.container_name,
            check_cmd,
            timeout=30
        )
        
        if health_result.success and "ok" in health_result.stdout.lower():
            logging.info("Service is healthy and running")
        else:
            result["message"] = f"Service health check failed: {health_result.stderr}"
            logging.error(result["message"])

        # 17. 执行精度测试命令
        accuracy_cmd = ""
        logging.info(f"Executing accuracy command: {accuracy_cmd}")
        if accuracy_cmd:
            accuracy_result = container_manager.exec_in_container(
                container_config.container_name,
                accuracy_cmd,
                timeout=framework_config.get("benchmark_timeout", 3600)
            )
            result["accuracy_success"] = accuracy_result.success
            result["accuracy_exit_code"] = accuracy_result.exit_code
            result["accuracy_stdout"] = accuracy_result.stdout
            result["accuracy_stderr"] = accuracy_result.stderr
            result["accuracy_execution_time"] = accuracy_result.execution_time
        else:
            logging.warning("accuracy_cmd is null, skip accuracy task.")
        
        # 18. 执行基准测试命令
        logging.info(f"Executing benchmark command: {benchmark_cmd}")
        
        if benchmark_cmd:
            benchmark_result = container_manager.exec_in_container(
                container_config.container_name,
                benchmark_cmd,
                timeout=framework_config.get("benchmark_timeout", 3600)
            )
            result["benchmark_success"] = benchmark_result.success
            result["benchmark_exit_code"] = benchmark_result.exit_code
            result["benchmark_stdout"] = benchmark_result.stdout
            result["benchmark_stderr"] = benchmark_result.stderr
            result["benchmark_execution_time"] = benchmark_result.execution_time

            if benchmark_result.success:
                logging.info("Benchmark completed successfully.")
                from xtask.benckmarks.benchmark_common import result_postprocess
                result = result_postprocess(benchmark_result.stdout, task_config.get("benchmark_type"))
            else:
                logging.error(f"Benchmark command failed with exit code {benchmark_result.exit_code}")
        else:
            benchmark_result = ""
            logging.warning("benchmark_cmd is null, skip benchmark task.")

        if not benchmark_result.success:
            result["message"] = f"Benchmark command failed with exit code {benchmark_result.exit_code}"
            logging.error(result["message"])
        
        # 19. 获取容器日志
        logs = container_manager.get_container_logs(result["container_id"], tail=100)
        result["logs"] = logs
        
        # 20. 标记执行成功
        result["success"] = True
        result["message"] = f"Task scheduled successfully in container {result['container_id']}"
        logging.info(result["message"])
        
    except Exception as e:
        result["message"] = f"Error in scheduler: {str(e)}"
        logging.error(result["message"], exc_info=True)
    
    finally:
        # 计算执行时间
        result["execution_time"] = time.time() - start_time
        bool_map = {"true": True, "false": False}
        # 如果启动是通过已存在的容器或者 全局配置 CLEAN_RESULTS是Fasle，则任务退出时不会清除容器实例
        if not framework_config.get("serving_container_name") or \
            bool_map.get(get_config_data().get("ENV_config").get("CLEAN_RESULTS").lower()):
            container_manager.cleanup()
        
        # 记录执行时间
        logging.info(f"Scheduler execution completed in {result['execution_time']:.2f} seconds")
    
    return result


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
        max_out_len: 128
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
        max_out_len: 128
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