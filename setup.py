# setup.py 示例
from setuptools import setup, find_packages # type: ignore

setup(
    name="xtask",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0.0",
        "psutil>=7.0.0",
        "openpyxl>=3.1.0",
        # ... 其他依赖
    ],
    extras_require={
        "search": ["optuna>=4.0,<5.0"],
    },
    entry_points={
        'console_scripts': [
            'xtask=xtask.main:main',  # 创建命令行命令
        ],
    },
)
