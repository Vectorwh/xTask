#!/bin/bash
set -e

WHL_FILE="evalscope-1.5.0-py3-none-any.whl"
SOURCE_DIR="/xllm_test_060"
TARGET_DIR="/workspace"

# 检查源文件
if [ ! -f "$SOURCE_DIR/$WHL_FILE" ]; then
    echo "错误: $SOURCE_DIR/$file 不存在"
    exit 1
fi

# 复制文件
rm -f "$TARGET_DIR/$WHL_FILE"
cp -fv "$SOURCE_DIR/$WHL_FILE" "$TARGET_DIR/"

cd "$TARGET_DIR"
pip install "$WHL_FILE"
pip install 'evalscope[perf]'

echo "安装完成！"