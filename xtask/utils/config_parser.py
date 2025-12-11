import os
import sys
import json
import logging


def get_config_data():
    try:
        config_file = os.path.join(os.getcwd(), "config/xTask_config.json")
        with open(config_file, 'r') as cf:
            return json.load(cf)
    except Exception as e:
        logging.error(f"Argument validation failed: {str(e)}")
        sys.exit(1)

