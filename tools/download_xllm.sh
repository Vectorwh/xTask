#!/bin/bash
set -e

# 仅当本地 /workspace/pkg/xllm/ 下存在 xllm_npu*.whl 且 vendors 目录存在时，才执行下载/解压/安装 Lib64 及拷贝操作
XLLM_PKG_DIR="/workspace/pkg/xllm"
XLLM_VENDORS_DIR="/workspace/pkg/xllm/vendors"
XLLM_RUNTIME_LIB_DIR="$XLLM_PKG_DIR/runtime_libs"
XLLM_THIRD_WHL_DIR="$XLLM_PKG_DIR/third_whl"
XLLM_SERVICE_BINARY_DIR="$XLLM_PKG_DIR/service_binary"
XLLM_SERVICE_CURRENT_DIR="$XLLM_SERVICE_BINARY_DIR/current"
XLLM_INSTALL_SERVICE="${XLLM_INSTALL_SERVICE:-1}"

case "$XLLM_INSTALL_SERVICE" in
    0|1) ;;
    *)
        echo "错误: XLLM_INSTALL_SERVICE 必须为 0 或 1，当前值: $XLLM_INSTALL_SERVICE" >&2
        exit 1
        ;;
esac

RPM_FILE="numactl-2.0.16-9.vo2203sp4.aarch64.rpm"
TORCH_NPU_FILE="torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl"
SOURCE_DIR="/xllm_test_060"
TARGET_DIR="/workspace"
XLLM_PACK_URL="https://9n-das-tools.s3.cn-north-1.jdcloud-oss.com/xllm-ai/xllm_release/0.10.0/a3-arm/xllm_pack.tar.gz"
XLLM_PACK_ARCHIVE="$TARGET_DIR/xllm_pack.tar.gz"
XLLM_PACK_DIR="$TARGET_DIR/xllm_pack"
XLLM_PACK_READY=0
# 宿主机 /export/home/models/xllm_test_060 会归档 xllm_pack.tar.gz。该目录在
# 容器内可能是相同路径直挂，也可能是 /xllm_test_060；两个候选都探测，也可用
# XLLM_PACK_LOCAL_DIR 显式指定归档目录。
XLLM_PACK_LOCAL_DIR="${XLLM_PACK_LOCAL_DIR:-}"
XLLM_PACK_LOCAL_CANDIDATES=(
    "/export/home/models/xllm_test_060"
    "/xllm_test_060"
)
XLLM_PACK_LOCAL_SOURCE=""

# Opt-in new release pack, or a locally published new pack when no separate
# Engine artifacts exist. Never silently combine a selected new pack with old
# pkg wheels/runtime libraries or download the old pack to supply its etcd.
XLLM_RELEASE_PACK="${XLLM_RELEASE_PACK:-}"
XLLM_PACK_LAYOUT=legacy
RELEASE_STAGE=""
MOONCAKE_RUNTIME=""
if [ -z "$XLLM_RELEASE_PACK" ] && ! compgen -G "$XLLM_PKG_DIR/xllm_npu*.whl" >/dev/null && \
   [ -f "$XLLM_PKG_DIR/release_pack/current/xllm_pack.tar.gz" ]; then
    XLLM_RELEASE_PACK="$XLLM_PKG_DIR/release_pack/current/xllm_pack.tar.gz"
fi
if [ -n "$XLLM_RELEASE_PACK" ]; then
    RELEASE_STAGE="$(mktemp -d "$TARGET_DIR/.runtime-release.XXXXXX")"
    rmdir "$RELEASE_STAGE"
    trap 'rm -rf "$RELEASE_STAGE"' EXIT
    python3 /workspace/tools/xllm_release_pack.py prepare "$XLLM_RELEASE_PACK" "$RELEASE_STAGE"
    XLLM_PKG_DIR="$RELEASE_STAGE/xllm_pack"
    XLLM_VENDORS_DIR="$XLLM_PKG_DIR/vendors"
    XLLM_THIRD_WHL_DIR="$XLLM_PKG_DIR/third_whl"
    XLLM_RUNTIME_LIB_DIR="$XLLM_PKG_DIR/runtime_libs"
    XLLM_SERVICE_BINARY_DIR="$XLLM_PKG_DIR/service_binary"
    XLLM_SERVICE_CURRENT_DIR="$XLLM_SERVICE_BINARY_DIR/current"
    XLLM_PACK_LAYOUT=release-v1
    MOONCAKE_RUNTIME="$(find "$XLLM_PKG_DIR" -maxdepth 1 -type d -name 'mooncake-v*' -print -quit)"
fi

service_archive_ready() {
    local directory="$1" info
    [ -f "$directory/xllm_master_serving" ] || return 1
    # Inspect DT_NEEDED: missing tokenizer is only legal for static builds.
    info="$(readelf -d "$directory/xllm_master_serving" 2>&1)" || return 1
    if grep -q 'Shared library: \[librust_tokenizers.so\]' <<< "$info"; then
        [ -f "$directory/librust_tokenizers.so" ] || return 1
    fi
    if [ -f "$directory/SHA256SUMS" ]; then
        (cd "$directory" && sha256sum -c SHA256SUMS) || return 1
    fi
}

install_mooncake_runtime() {
    [ -n "$MOONCAKE_RUNTIME" ] || return 0
    # A dedicated directory retains metadata/check scripts; active libraries
    # are installed alongside the legacy /usr/lib64 loader contract.
    local destination="/opt/xllm/mooncake/$(basename "$MOONCAKE_RUNTIME")"
    mkdir -p "$destination"
    cp -a "$MOONCAKE_RUNTIME/." "$destination/"
    install -m 0755 "$MOONCAKE_RUNTIME"/lib/* /usr/lib64/
    install -m 0755 "$MOONCAKE_RUNTIME"/bin/* /usr/local/bin/
    ldconfig
}

prepare_release_etcd() {
    local directory="${XLLM_ETCD_BIN_DIR:-}"
    if [ -n "$directory" ]; then
        if [ ! -x "$directory/etcd" ] || [ ! -x "$directory/etcdctl" ]; then
            echo "错误: XLLM_ETCD_BIN_DIR 必须包含可执行的 etcd/etcdctl。" >&2
            return 1
        fi
        install -m 0755 "$directory/etcd" "$directory/etcdctl" /usr/local/bin/
    elif ! command -v etcd >/dev/null || ! command -v etcdctl >/dev/null; then
        echo "错误: 新 pack 不含 etcd。预装 etcd/etcdctl 或设置 XLLM_ETCD_BIN_DIR。" >&2
        return 1
    fi
}

# Validate control-plane components before touching Engine packages/vendors.
if [ "$XLLM_PACK_LAYOUT" = release-v1 ] && [ "$XLLM_INSTALL_SERVICE" -eq 1 ]; then
    if ! service_archive_ready "$XLLM_SERVICE_BINARY_DIR"; then
        echo "错误: 新 pack 未包含有效 Service；请先 append/publish Service。" >&2
        exit 1
    fi
    prepare_release_etcd
fi

ensure_xllm_pack() {
    if [ "$XLLM_PACK_READY" -eq 1 ]; then
        return 0
    fi

    # 固定下载文件名，并清理上次异常退出留下的压缩包和解压目录，
    # 避免 wget 生成 .1 文件后仍误用旧包。
    rm -rf "$XLLM_PACK_DIR"
    rm -f "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_ARCHIVE".*

    # 优先使用服务器本地归档，避免每次从外网 wget ~585MB。
    if [ -z "$XLLM_PACK_LOCAL_SOURCE" ]; then
        local candidate
        for candidate in "$XLLM_PACK_LOCAL_DIR" "${XLLM_PACK_LOCAL_CANDIDATES[@]}"; do
            [ -n "$candidate" ] || continue
            if [ -f "$candidate/xllm_pack.tar.gz" ]; then
                XLLM_PACK_LOCAL_SOURCE="$candidate"
                break
            fi
        done
    fi
    if [ -n "$XLLM_PACK_LOCAL_SOURCE" ]; then
        echo "使用本地归档: $XLLM_PACK_LOCAL_SOURCE/xllm_pack.tar.gz"
        if tar -zxf "$XLLM_PACK_LOCAL_SOURCE/xllm_pack.tar.gz" -C "$TARGET_DIR"; then
            if [ ! -d "$XLLM_PACK_DIR" ]; then
                echo "Error: 本地归档解压后未生成目录: $XLLM_PACK_DIR，回退到外网下载。" >&2
            else
                if compgen -G "$XLLM_PACK_DIR/mooncake-runtime-v*.tar.gz" >/dev/null; then
                    echo "错误: 检测到新版 pack；请用 XLLM_RELEASE_PACK 或 deployment.xllm_package.release_pack 选择完整安装，不能混用旧 pkg。" >&2
                    exit 1
                fi
                XLLM_PACK_READY=1
                return 0
            fi
        else
            echo "Error: 本地归档校验失败，回退到外网下载: $XLLM_PACK_LOCAL_SOURCE/xllm_pack.tar.gz" >&2
            rm -rf "$XLLM_PACK_DIR"
            rm -f "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_ARCHIVE".*
        fi
        XLLM_PACK_LOCAL_SOURCE=""
    fi

    wget --tries=3 -O "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_URL"
    tar -zxvf "$XLLM_PACK_ARCHIVE" -C "$TARGET_DIR"

    if [ ! -d "$XLLM_PACK_DIR" ]; then
        echo "Error: xllm_pack 解压后未生成目录: $XLLM_PACK_DIR" >&2
        exit 1
    fi
    if compgen -G "$XLLM_PACK_DIR/mooncake-runtime-v*.tar.gz" >/dev/null; then
        echo "错误: 下载到新版 pack；请使用 XLLM_RELEASE_PACK 完整安装，不能按旧 xllm-lib 格式安装。" >&2
        exit 1
    fi
    XLLM_PACK_READY=1
}

install_optional_third_party_wheels() {
    local -a third_wheels=()
    local wheel
    shopt -s nullglob
    for wheel in "$XLLM_THIRD_WHL_DIR"/*.whl; do
        [ -f "$wheel" ] || continue
        third_wheels+=("$wheel")
    done
    shopt -u nullglob

    if [ "${#third_wheels[@]}" -eq 0 ]; then
        echo "未检测到 $XLLM_THIRD_WHL_DIR 下的三方 wheel，跳过可选三方包安装。"
        return 0
    fi

    echo "安装三方 wheel（${#third_wheels[@]} 个）:"
    printf '  %s\n' "${third_wheels[@]}"
    # 用 xLLM 同一解释器一次安装全部本地 wheel，不依赖文件名顺序。
    # 不自动下载依赖或升级镜像中的 torch/torch_npu；冲突或安装失败明确报错。
    if ! "$XLLM_PYTHON" -m pip install --force-reinstall --no-deps --no-index "${third_wheels[@]}"; then
        echo "错误: $XLLM_THIRD_WHL_DIR 下的三方 wheel 安装失败。" >&2
        return 1
    fi
    echo "三方 wheel 安装完成（${#third_wheels[@]} 个）: $XLLM_THIRD_WHL_DIR"
}

install_xllm_runtime_libraries() {
    local lib_name
    local source_path

    if [ ! -d "$XLLM_RUNTIME_LIB_DIR" ]; then
        echo "未检测到 $XLLM_RUNTIME_LIB_DIR，跳过额外 xLLM 运行时库安装。"
        return 0
    fi

    for lib_name in libxxhash.so.0 libasio.so liburing.so.2; do
        source_path="$XLLM_RUNTIME_LIB_DIR/$lib_name"
        if [ ! -f "$source_path" ]; then
            echo "Error: Required xLLM runtime library not found: $source_path" >&2
            exit 1
        fi
        # 归档端把 SONAME 库解引用成普通文件；而 ldconfig 要求 /usr/lib64
        # 下的 SONAME 路径必须是符号链接，否则报 "is not a symbolic link"。
        # 先落真实文件（版本化后缀），再建 SONAME 软链接指向它。
        if [ "$lib_name" = "libasio.so" ]; then
            install -m 0755 "$source_path" "/usr/lib64/$lib_name"
            continue
        fi
        local real_version
        real_version="$(readlink -f "$source_path" 2>/dev/null || true)"
        if [ -n "$real_version" ] && [ "$real_version" != "$source_path" ]; then
            install -m 0755 "$real_version" "/usr/lib64/$lib_name"
        else
            install -m 0755 "$source_path" "/usr/lib64/${lib_name}.xllm"
            ln -sfn "${lib_name}.xllm" "/usr/lib64/$lib_name"
        fi
    done

    ldconfig
    echo "xLLM runtime libraries installed to /usr/lib64: libxxhash.so.0 libasio.so liburing.so.2"
}

if { [ "$XLLM_PACK_LAYOUT" = release-v1 ] || ls "$XLLM_PKG_DIR"/xllm_npu*.whl >/dev/null 2>&1; } && [ -d "$XLLM_VENDORS_DIR" ]; then
    echo "检测到 $XLLM_PKG_DIR 下存在 xllm_npu*.whl 且 $XLLM_VENDORS_DIR 存在，开始处理 xLLM 依赖库..."

    # xLLM 依赖所需 Lib64 从外网连接下载。
    if [ "$XLLM_PACK_LAYOUT" = legacy ]; then
        ensure_xllm_pack
        cd "$XLLM_PACK_DIR"
    fi

    #for binary in etcd etcdctl xllm_master_serving; do
    #    if [ ! -f "$binary" ]; then
    #        echo "错误: xllm_pack 中缺少 $binary"
    #        exit 1
    #    fi
    #done

    #install -m 0755 etcd etcdctl xllm_master_serving /usr/local/bin/

    if [ "$XLLM_PACK_LAYOUT" = legacy ]; then
        tar -zxvf xllm-lib-*.tar.gz
        if [ $? -ne 0 ]; then
            echo "Error: Failed to extract xllm-lib-*.tar.gz"
            exit 1
        fi
        yes | mv xllm-lib/* /usr/lib64/
    else
        install_mooncake_runtime
        install -m 0644 "$XLLM_PKG_DIR/xtask-pack-manifest.json" /opt/xllm/installed-pack.json
    fi
     
    # cd /workspace
    # 删除旧数据
    # rm -rf xllm_pack
    # rm -rf xllm_pack.tar.gz*
    # rm -rf xllm-lib xllm-lib-*.tar.gz vendors-*.tar.gz xllm-*-linux*.whl

    # 安装本次 xLLM 编译时收集的额外运行时动态库。
    install_xllm_runtime_libraries

    # xLLM 二进制、算子库等从pack目录中获取
    # cp /workspace/pkg/xllm/xllm /usr/local/bin/
    # vendors 的清理与拷贝必须对 bind mount 安全：gedit 自服务容器会把宿主
    # vendors 的子目录（custom_xllm_math）直接 bind mount 到安装目标下，此时
    # 目标内该子目录与 cp 源 $XLLM_VENDORS_DIR 下同名目录是同一份文件（同一
    # inode），逐条目处理：
    #   1) 清理时跳过挂载点（rm 会命中挂载点报 Device busy 并在 set -e 下
    #      退出，且穿透删除宿主源内容 —— 2026-11-21 踩中）；
    #   2) 拷贝时跳过与源同 inode 的子目录（GNU cp 对 self->self 报
    #      "are the same file" 并以非零退出 —— 2026-11-22 踩中）。
    VENDORS_TARGET="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors"
    for entry in "$VENDORS_TARGET"/*; do
        [ -e "$entry" ] || continue
        if mountpoint -q "$entry"; then
            continue
        fi
        rm -rf "$entry"
    done
    for entry in "$XLLM_VENDORS_DIR"/*; do
        [ -e "$entry" ] || continue
        name="$(basename "$entry")"
        target="$VENDORS_TARGET/$name"
        if [ -e "$target" ] && mountpoint -q "$target"; then
            # bind-mounted target overlays the same host content: skip copy
            continue
        fi
        yes | cp -rf "$entry" "$target"
    done

    # 使用同一个 Python 完成安装和路径查询，避免 pip/python3 指向不同环境。
    if ! XLLM_PYTHON=$(command -v python3); then
        echo "错误: 未找到 python3，无法安装 xLLM wheel。"
        exit 1
    fi

    if [ "$XLLM_PACK_LAYOUT" = legacy ]; then
        rm -f "$TARGET_DIR/$TORCH_NPU_FILE"
        cp -fv "$SOURCE_DIR/$TORCH_NPU_FILE" "$TARGET_DIR/"
    fi

    # 最新镜像已经预装torch_npu，不再单独安装
    #"$XLLM_PYTHON" -m pip install \
    #    --force-reinstall \
    #    --no-deps \
    #    "$TARGET_DIR/$TORCH_NPU_FILE"

    #echo "torch_npu 安装完成: $TORCH_NPU_FILE"

    # 安装 xLLM wheel。
    if [ "$XLLM_PACK_LAYOUT" = legacy ]; then
        "$XLLM_PYTHON" -m pip install "$XLLM_PKG_DIR"/xllm_npu*.whl --force-reinstall
    else
        "$XLLM_PYTHON" -m pip install "$XLLM_PKG_DIR"/xllm*.whl --force-reinstall --no-deps --no-index
    fi

    # 目录不存在或没有 wheel 时仅跳过，不影响原有 xLLM/service 安装。
    install_optional_third_party_wheels

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

# xTask 对普通 runtime 显式传 0；PD/service_routed 只在控制面 runtime 传 1。
# 默认值 1 保留外部直接执行该脚本时的历史行为。
if [ "$XLLM_INSTALL_SERVICE" -eq 1 ]; then
    # etcd、xllm-service 的安装不依赖 xLLM 编译产物存在与否。
    # xLLM-Service 编译归档使用 service_binary/current；保留直接放置文件的
    # 旧布局兼容。Service 的 etcd_cpp_apiv3 是客户端库，不替代 etcd 服务端。
    SERVICE_RUNTIME_DIR=""
    if service_archive_ready "$XLLM_SERVICE_CURRENT_DIR"; then
        SERVICE_RUNTIME_DIR="$XLLM_SERVICE_CURRENT_DIR"
        echo "service_binary/current下找到xLLM-Service二进制，使用本地二进制。"
    elif [ -e "$XLLM_SERVICE_CURRENT_DIR" ] || [ -L "$XLLM_SERVICE_CURRENT_DIR" ]; then
        echo "错误: service_binary/current 归档不完整或 ELF/校验失败，拒绝静默回退旧 Service。" >&2
        exit 1
    fi
    if [ -z "$SERVICE_RUNTIME_DIR" ] && [ -f "$XLLM_SERVICE_BINARY_DIR/xllm_master_serving" ]; then
        if ! service_archive_ready "$XLLM_SERVICE_BINARY_DIR"; then
            echo "错误: 本地 Service 归档不完整或 ELF/校验失败，拒绝回退。" >&2
            exit 1
        fi
        SERVICE_RUNTIME_DIR="$XLLM_SERVICE_BINARY_DIR"
        echo "service_binary下找到xLLM-Service二进制，使用本地二进制。"
    fi

    if [ -n "$SERVICE_RUNTIME_DIR" ]; then
        if [ -f "$SERVICE_RUNTIME_DIR/librust_tokenizers.so" ]; then
            install -m 0755 "$SERVICE_RUNTIME_DIR/librust_tokenizers.so" /usr/lib64/librust_tokenizers.so
            ldconfig
        fi
        install -m 0755 "$SERVICE_RUNTIME_DIR/xllm_master_serving" /usr/local/bin/xllm_master_serving
    fi

    # ETCD remains independently sourced. A Service source tree only contains
    # the etcd C++ client library, not the etcd/etcdctl server binaries.
    LOCAL_ETCD_READY=0
    if [ -f "$XLLM_SERVICE_BINARY_DIR/etcd" ] && \
        [ -f "$XLLM_SERVICE_BINARY_DIR/etcdctl" ]; then
        echo "service_binary下找到本地etcd/etcdctl，使用本地二进制。"
        install -m 0755 \
            "$XLLM_SERVICE_BINARY_DIR/etcd" \
            "$XLLM_SERVICE_BINARY_DIR/etcdctl" \
            /usr/local/bin/
        LOCAL_ETCD_READY=1
    else
        echo "service_binary下未找到本地etcd/etcdctl。"
    fi

    if [ "$XLLM_PACK_LAYOUT" = release-v1 ]; then
        if [ -z "$SERVICE_RUNTIME_DIR" ]; then
            echo "错误: 新 pack 未包含有效 Service；请先 append/publish Service。" >&2
            exit 1
        fi
        if [ "$LOCAL_ETCD_READY" -eq 0 ]; then
            prepare_release_etcd
            LOCAL_ETCD_READY=1
        fi
    fi

    # Download/extract the legacy pack only when one of its independent runtime
    # components is missing. This preserves the old all-in-one fallback while
    # allowing a newly compiled Service to coexist with pack-provided ETCD.
    if [ "$LOCAL_ETCD_READY" -eq 0 ] || [ -z "$SERVICE_RUNTIME_DIR" ]; then
        ensure_xllm_pack
    fi
    if [ "$LOCAL_ETCD_READY" -eq 0 ]; then
        for binary in etcd etcdctl; do
            if [ ! -f "$XLLM_PACK_DIR/$binary" ]; then
                echo "错误: xllm_pack 中缺少 $binary" >&2
                exit 1
            fi
        done
        install -m 0755 "$XLLM_PACK_DIR/etcd" "$XLLM_PACK_DIR/etcdctl" /usr/local/bin/
    fi
    if [ -z "$SERVICE_RUNTIME_DIR" ]; then
        if [ ! -f "$XLLM_PACK_DIR/xllm_master_serving" ]; then
            echo "错误: xllm_pack 中缺少 xllm_master_serving" >&2
            exit 1
        fi
        install -m 0755 "$XLLM_PACK_DIR/xllm_master_serving" /usr/local/bin/
    fi
else
    echo "当前 runtime 不需要 xLLM-Service 控制面，跳过 xllm_master_serving 和 ETCD 安装。"
fi

cd "$TARGET_DIR"
# Release input is immutable/user-owned, even when selected from /workspace.
# Only legacy downloads/extractions are temporary to this installer.
if [ "$XLLM_PACK_LAYOUT" = legacy ]; then
    rm -rf "$XLLM_PACK_DIR"
    rm -f "$XLLM_PACK_ARCHIVE" "$XLLM_PACK_ARCHIVE".*
fi

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
