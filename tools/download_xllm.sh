#!/bin/bash
set -e

# 仅当本地 /workspace/pkg/xllm/ 下存在 xllm_npu*.whl 且 vendors 目录存在时，才执行下载/解压/安装 Lib64 及拷贝操作
XLLM_PKG_DIR="/workspace/pkg/xllm"
XLLM_VENDORS_DIR="/workspace/pkg/xllm/vendors"
XLLM_RUNTIME_LIB_DIR="$XLLM_PKG_DIR/runtime_libs"
XLLM_SERVICE_BINARY_DIR="$XLLM_PKG_DIR/service_binary"

RPM_FILE="numactl-2.0.16-9.vo2203sp4.aarch64.rpm"
TORCH_NPU_FILE="torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl"
SOURCE_DIR="/xllm_test_060"
TARGET_DIR="/workspace"
XLLM_PACK_URL="https://9n-das-tools.s3.cn-north-1.jdcloud-oss.com/xllm-ai/xllm_release/0.10.0/a3-arm/xllm_pack.tar.gz"
XLLM_PACK_ARCHIVE="$TARGET_DIR/xllm_pack.tar.gz"
XLLM_PACK_DIR="$TARGET_DIR/xllm_pack"
XLLM_PACK_READY=0

ensure_xllm_pack() {
    if [ "$XLLM_PACK_READY" -eq 1 ]; then
        return 0
    fi

    # 固定下载文件名，并清理上次异常退出留下的压缩包和解压目录，
    # 避免 wget 生成 .1 文件后仍误用旧包。
    rm -rf "$XLLM_PACK_DIR"
    rm -f "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_ARCHIVE".*
    wget --tries=3 -O "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_URL"
    tar -zxvf "$XLLM_PACK_ARCHIVE" -C "$TARGET_DIR"

    if [ ! -d "$XLLM_PACK_DIR" ]; then
        echo "Error: xllm_pack 解压后未生成目录: $XLLM_PACK_DIR" >&2
        exit 1
    fi
    XLLM_PACK_READY=1
}

install_xllm_runtime_libraries() {
    local lib_name
    local source_path

    if [ ! -d "$XLLM_RUNTIME_LIB_DIR" ]; then
        echo "未检测到 $XLLM_RUNTIME_LIB_DIR，跳过额外 xLLM 运行时库安装。"
        return 0
    fi

    for lib_name in libxxhash.so.0 libasio.so; do
        source_path="$XLLM_RUNTIME_LIB_DIR/$lib_name"
        if [ ! -f "$source_path" ]; then
            echo "Error: Required xLLM runtime library not found: $source_path" >&2
            exit 1
        fi
        install -m 0755 "$source_path" "/usr/lib64/$lib_name"
    done

    ldconfig
    echo "xLLM runtime libraries installed to /usr/lib64: libxxhash.so.0 libasio.so"
}

if ls "$XLLM_PKG_DIR"/xllm_npu*.whl >/dev/null 2>&1 && [ -d "$XLLM_VENDORS_DIR" ]; then
    echo "检测到 $XLLM_PKG_DIR 下存在 xllm_npu*.whl 且 $XLLM_VENDORS_DIR 存在，开始处理 xLLM 依赖库..."

    # xLLM 依赖所需 Lib64 从外网连接下载。
    ensure_xllm_pack
    cd "$XLLM_PACK_DIR"

    #for binary in etcd etcdctl xllm_master_serving; do
    #    if [ ! -f "$binary" ]; then
    #        echo "错误: xllm_pack 中缺少 $binary"
    #        exit 1
    #    fi
    #done

    #install -m 0755 etcd etcdctl xllm_master_serving /usr/local/bin/

    tar -zxvf xllm-lib-*.tar.gz
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract xllm-lib-*.tar.gz"
        exit 1
    fi

    yes | mv xllm-lib/* /usr/lib64/
     
    # cd /workspace
    # 删除旧数据
    # rm -rf xllm_pack
    # rm -rf xllm_pack.tar.gz*
    # rm -rf xllm-lib xllm-lib-*.tar.gz vendors-*.tar.gz xllm-*-linux*.whl

    # 安装本次 xLLM 编译时收集的额外运行时动态库。
    install_xllm_runtime_libraries

    # xLLM 二进制、算子库等从pack目录中获取
    # cp /workspace/pkg/xllm/xllm /usr/local/bin/
    rm -rf /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/*
    yes | cp -rf "$XLLM_VENDORS_DIR"/* /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/

    # 使用同一个 Python 完成安装和路径查询，避免 pip/python3 指向不同环境。
    if ! XLLM_PYTHON=$(command -v python3); then
        echo "错误: 未找到 python3，无法安装 xLLM wheel。"
        exit 1
    fi

    rm -f "$TARGET_DIR/$TORCH_NPU_FILE"
    cp -fv "$SOURCE_DIR/$TORCH_NPU_FILE" "$TARGET_DIR/"

    "$XLLM_PYTHON" -m pip install \
        --force-reinstall \
        --no-deps \
        "$TARGET_DIR/$TORCH_NPU_FILE"

    echo "torch_npu 安装完成: $TORCH_NPU_FILE"

    # 安装 xLLM wheel。
    "$XLLM_PYTHON" -m pip install "$XLLM_PKG_DIR"/xllm_npu*.whl --force-reinstall

    # 优先使用 wheel 安装的 console script。部分旧 wheel 将 xllm 二进制放在
    # site-packages 内，因此保留原来的包目录搜索方式作为兼容回退。
    XLLM_SCRIPTS_DIR=$(
        "$XLLM_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts") or "")'
    )
    XLLM_BIN_PATH="$XLLM_SCRIPTS_DIR/xllm"
    if [ ! -f "$XLLM_BIN_PATH" ]; then
        XLLM_PKG_NAME=$(
            "$XLLM_PYTHON" -m pip list --format=freeze 2>/dev/null |
                awk -F'==' 'tolower($1) ~ /^xllm[-_]npu/ {print $1; exit}'
        )
        XLLM_INSTALL_DIR=""
        if [ -n "$XLLM_PKG_NAME" ]; then
            XLLM_INSTALL_DIR=$(
                "$XLLM_PYTHON" -m pip show "$XLLM_PKG_NAME" 2>/dev/null |
                    awk -F': ' '/^Location:/ {print $2; exit}'
            )
        fi
        if [ -n "$XLLM_INSTALL_DIR" ]; then
            XLLM_BIN_PATH=$(find "$XLLM_INSTALL_DIR" -type f -name xllm 2>/dev/null | head -n 1)
        fi
    fi

    if [ -z "$XLLM_BIN_PATH" ] || [ ! -f "$XLLM_BIN_PATH" ] || [ ! -x "$XLLM_BIN_PATH" ]; then
        echo "错误: xLLM wheel 安装后未找到可执行的 xllm 命令。"
        exit 1
    fi

    XLLM_LINK_PATH="/usr/local/bin/xllm"
    if [ "$XLLM_BIN_PATH" = "$XLLM_LINK_PATH" ]; then
        echo "xllm 已安装在 $XLLM_LINK_PATH，无需创建软链接。"
    else
        echo "找到 xllm 二进制: $XLLM_BIN_PATH，建立软链接到 $XLLM_LINK_PATH"
        ln -sfn "$XLLM_BIN_PATH" "$XLLM_LINK_PATH"
    fi

    if [ ! -x "$XLLM_LINK_PATH" ]; then
        echo "错误: $XLLM_LINK_PATH 不可执行。"
        exit 1
    fi
    echo "xllm 命令准备完成: $XLLM_LINK_PATH -> $(readlink "$XLLM_LINK_PATH" 2>/dev/null || echo "$XLLM_LINK_PATH")"
else
    echo "未检测到 $XLLM_PKG_DIR 下的 xllm_npu*.whl 或 $XLLM_VENDORS_DIR，跳过 xLLM 依赖库下载与安装。"
fi

# etcd、xllm-service 的安装不依赖 xLLM 编译产物存在与否。
if [ -f "$XLLM_SERVICE_BINARY_DIR/etcd" ] && \
    [ -f "$XLLM_SERVICE_BINARY_DIR/etcdctl" ] && \
    [ -f "$XLLM_SERVICE_BINARY_DIR/xllm_master_serving" ]; then
    echo "service_binary下找到二进制文件，使用本地二进制。"
    install -m 0755 \
        "$XLLM_SERVICE_BINARY_DIR/etcd" \
        "$XLLM_SERVICE_BINARY_DIR/etcdctl" \
        "$XLLM_SERVICE_BINARY_DIR/xllm_master_serving" \
        /usr/local/bin/
else
    echo "service_binary下未找到二进制文件，使用wget的二进制。"
    ensure_xllm_pack
    for binary in etcd etcdctl xllm_master_serving; do
        if [ ! -f "$XLLM_PACK_DIR/$binary" ]; then
            echo "错误: xllm_pack 中缺少 $binary" >&2
            exit 1
        fi
    done
    install -m 0755 \
        "$XLLM_PACK_DIR/etcd" \
        "$XLLM_PACK_DIR/etcdctl" \
        "$XLLM_PACK_DIR/xllm_master_serving" \
        /usr/local/bin/
fi

cd "$TARGET_DIR"
# 删除旧数据
rm -rf "$XLLM_PACK_DIR"
rm -f "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_ARCHIVE".*

# 从本地安装numctl
if command -v numactl >/dev/null 2>&1; then
    echo "numactl 已安装，跳过 RPM 检查和安装。"
else
    # 仅在系统尚未安装 numactl 时，才要求挂载目录提供 RPM。
    if [ ! -f "$SOURCE_DIR/$RPM_FILE" ]; then
        echo "错误: $SOURCE_DIR/$RPM_FILE 不存在"
        exit 1
    fi

    rm -f "$TARGET_DIR/$RPM_FILE"
    cp -fv "$SOURCE_DIR/$RPM_FILE" "$TARGET_DIR/"

    cd "$TARGET_DIR"
    rpm -ivh "$RPM_FILE"
fi

echo "get xllm success"
