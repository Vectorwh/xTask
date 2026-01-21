""" 联网状态下 下载源码自动装包函数
如果测试环境未联网，那建议下载wheel文件 离线按照
"""
import os
import sys


def install_aisbench():
    pass


def uninstall_aisbench():
    pass


def get_ais_bench_path():
    """获取ais_bench模块的安装路径"""
    try:
        import ais_bench
        # 检查__file__属性是否存在
        if hasattr(ais_bench, '__file__') and ais_bench.__file__:
            return os.path.dirname(os.path.abspath(ais_bench.__file__))
        else:
            # 尝试通过sys.modules查找
            module_file = sys.modules.get('ais_bench', None)
            if module_file and hasattr(module_file, '__file__'):
                return os.path.dirname(os.path.abspath(module_file.__file__))
            return None
    except ImportError:
        return None
