import os
import sys
import json
import logging


def get_xtask_config_data():
    try:
        config_file = os.path.join(os.getcwd(), "config/xTask_config.json")
        with open(config_file, 'r') as cf:
            return json.load(cf)
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
