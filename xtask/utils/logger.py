import os
import json
import logging


def init_logger():
    config_file = os.path.join(os.getcwd(), "config/xTask_config.json")
    with open(config_file, 'r') as cf:
        config_data = json.load(cf)
        xtask_log_level = config_data.get("ENV_config").get("XTask_LOG_LEVEL", "INFO")
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        logging.basicConfig(format=
                            '%(asctime)s - %(levelname)s - [xTASK](%(filename)s:%(lineno)d): %(message)s',
                            level=xtask_log_level)
        logging.getLogger("paramiko").setLevel(logging.ERROR)
