import os
import sys
import json
import logging


def get_xtask_config_data():
    try:
        host_dir =os.getcwd()
        config_file = os.path.join(host_dir, "config/xTask_config.json")
        with open(config_file, 'r') as cf:
            xtask_config = json.load(cf)

            #使用main.py执行路径映射为容器里workspace根路径
            if xtask_config.get("DOCKER_CONFIG") and xtask_config.get("DOCKER_CONFIG").get("volumes"):
                xtask_config["DOCKER_CONFIG"]["volumes"].update({host_dir: "/workspace"})

            # 自动将 DEPENDENCE_PACKAGE_PATH 映射到容器内固定挂载点 /xllm_test_060，
            # 免去在 volumes 里显式配置该条映射。容器侧路径 /xllm_test_060 是稳定
            # 约定（aisbench/vbench 等容器内脚本依赖此路径），不随 host 部署位置变化。
            dep_path = (xtask_config.get("ENV_CONFIG") or {}).get(
                "DEPENDENCE_PACKAGE_PATH", ""
            ).strip()
            if dep_path:
                volumes = xtask_config.setdefault("DOCKER_CONFIG", {}).setdefault("volumes", {})
                dep_norm = os.path.normpath(dep_path)
                if not any(
                    os.path.normpath(h) == dep_norm for h in volumes
                ):
                    volumes[dep_norm] = "/xllm_test_060"

            return xtask_config

    except Exception as e:
        logging.error(f"Argument validation failed: {str(e)}")
        sys.exit(1)


def get_config_data(config_file):
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as cf:
                return json.load(cf)
        else:
            logging.error(f"config_file: {config_file} is not exist.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Argument validation failed: {str(e)}")
        sys.exit(1)
