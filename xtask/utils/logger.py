import os
import json
import logging

from xtask.utils.config_parser import get_config_data

def init_logger():
    config_data = get_config_data()
    xtask_log_level = config_data.get("ENV_CONFIG").get("XTask_LOG_LEVEL", "INFO")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logging.basicConfig(format=
                        '%(asctime)s - %(levelname)s - [xTASK](%(filename)s:%(lineno)d): %(message)s',
                        level=xtask_log_level)
    logging.getLogger("paramiko").setLevel(logging.ERROR)
