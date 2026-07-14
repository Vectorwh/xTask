#!/bin/bash
set -e

# 仅当本地 /workspace/pkg/xllm/ 下存在 xllm_npu*.whl 且 vendors 目录存在时，才执行下载/解压/安装 Lib64 及拷贝操作
XLLM_PKG_DIR="/workspace/pkg/xllm"
XLLM_VENDORS_DIR="/workspace/pkg/xllm/vendors"

if ls "$XLLM_PKG_DIR"/xllm_npu*.whl >/dev/null 2>&1 && [ -d "$XLLM_VENDORS_DIR" ]; then
    echo "检测到 $XLLM_PKG_DIR 下存在 xllm_npu*.whl 且 $XLLM_VENDORS_DIR 存在，开始处理 xLLM 依赖库..."

    # xLLM 依赖所需Lib64从外网连接下载
    wget https://9n-das-tools.s3.cn-north-1.jdcloud-oss.com/xllm-ai/xllm_release/0.10.0/a3-arm/xllm_pack.tar.gz

    tar -zxvf xllm_pack.tar.gz
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract xllm_pack.tar.gz"
        exit 1
    fi

    cd xllm_pack || {
        echo "Error: Failed to enter xllm_pack directory"
        exit 1
    }

    tar -zxvf xllm-lib-*.tar.gz
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract xllm-lib-*.tar.gz"
        exit 1
    fi

    yes | mv xllm-lib/* /usr/lib64/
     
    cd /workspace
    # 删除旧数据
    rm -rf xllm_pack
    rm -rf xllm_pack.tar.gz
    # rm -rf xllm-lib xllm-lib-*.tar.gz vendors-*.tar.gz xllm-*-linux*.whl

    # xLLM 二进制、算子库等从pack目录中获取
    # cp /workspace/pkg/xllm/xllm /usr/local/bin/
    rm -rf /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/*
    yes | cp -rf "$XLLM_VENDORS_DIR"/* /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/

    # 安装 xLLM及其他依赖
    pip install "$XLLM_PKG_DIR"/xllm_npu*.whl --force-reinstall

    # 查找 pip 安装 xllm-npu* 后 xllm 二进制所在路径，并建立软链接到 /usr/local/bin/xllm
    # 注意：whl 安装后的包名为 xllm-npu*（如 xllm-npu-torch2-9-0）
    XLLM_PKG_NAME=$(pip list 2>/dev/null | awk '{print $1}' | grep -i '^xllm[-_]npu' | head -n 1)
    XLLM_INSTALL_DIR=""
    if [ -n "$XLLM_PKG_NAME" ]; then
        XLLM_INSTALL_DIR=$(pip show "$XLLM_PKG_NAME" 2>/dev/null | awk -F': ' '/^Location:/ {print $2}')
    fi
    XLLM_BIN_PATH=""
    if [ -n "$XLLM_INSTALL_DIR" ]; then
        XLLM_BIN_PATH=$(find "$XLLM_INSTALL_DIR" -type f -name xllm 2>/dev/null | head -n 1)
    fi
    if [ -n "$XLLM_BIN_PATH" ] && [ -f "$XLLM_BIN_PATH" ]; then
        echo "找到 xllm 二进制: $XLLM_BIN_PATH，建立软链接到 /usr/local/bin/xllm"
        rm -f /usr/local/bin/xllm
        ln -sf "$XLLM_BIN_PATH" /usr/local/bin/xllm
    else
        echo "警告: 未找到 xllm 二进制，跳过软链接创建。"
    fi
else
    echo "未检测到 $XLLM_PKG_DIR 下的 xllm_npu*.whl 或 $XLLM_VENDORS_DIR，跳过 xLLM 依赖库下载与安装。"
fi

# 从本地安装numctl
RPM_FILE="numactl-2.0.16-9.vo2203sp4.aarch64.rpm"
SOURCE_DIR="/xllm_test_060"
TARGET_DIR="/workspace"

# 检查源文件
if [ ! -f "$SOURCE_DIR/$RPM_FILE" ]; then
    echo "错误: $SOURCE_DIR/$RPM_FILE 不存在"
    exit 1
fi

# 复制文件
rm -f "$TARGET_DIR/$RPM_FILE"
cp -fv "$SOURCE_DIR/$RPM_FILE" "$TARGET_DIR/"

cd "$TARGET_DIR"
rpm -ivh "$RPM_FILE"

echo "get xllm success"