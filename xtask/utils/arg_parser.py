import os
import json

import argparse


def get_config():
    """解析配置xTask_config文件"""
    config_file = os.path.join(os.getcwd(), 'config/xTask_config.json')
    with open(config_file, 'r') as cf:
        return json.load(cf)


def arg_parser():
    parser = argparse.ArgumentParser(description='Example argparse with groups')

    # 
    framework_group = parser.add_argument_group('framework_group', 'framework group description')
    xllm_group = parser.add_argument_group('xllm_group', 'xllm group description')
    mindie_group = parser.add_argument_group('mindie_group', 'mindie group description')
    vllm = parser.add_argument_group('vllm', 'vllm group description')

    # create framework param group
    framework_group.add_argument('-a', '--arg1', action='store_true', help='Argument 1 in group 1')
    framework_group.add_argument('-b', '--arg2', action='store_true', help='Argument 2 in group 1')

    # 在组2中添加参数
    xllm_group.add_argument('-c', '--arg3', action='store_true', help='Argument 1 in group 2')
    xllm_group.add_argument('-d', '--arg4', action='store_true', help='Argument 2 in group 2')
    return parser
