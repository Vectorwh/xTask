#!/bin/bash
set -e

# 仅当本地已存在 xllm 二进制和 vendors 目录时，才执行下载/解压/安装 Lib64 及拷贝操作
XLLM_BIN="/workspace/pkg/xllm/xllm"
XLLM_VENDORS_DIR="/workspace/pkg/xllm/vendors"

if [ -f "$XLLM_BIN" ] && [ -d "$XLLM_VENDORS_DIR" ]; then
    echo "检测到 $XLLM_BIN 和 $XLLM_VENDORS_DIR，开始处理 xLLM 依赖库..."

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
    cp /workspace/pkg/xllm/xllm /usr/local/bin/
    rm -rf /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/*
    yes | cp -r /workspace/pkg/xllm/vendors/* /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/
else
    echo "未检测到 $XLLM_BIN 或 $XLLM_VENDORS_DIR，跳过 xLLM 依赖库下载与安装。"
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