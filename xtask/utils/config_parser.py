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
