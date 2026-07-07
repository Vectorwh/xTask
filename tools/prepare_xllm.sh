#!/bin/bash
# xllm 内网编译下载部署（容器外编排层）
# 负责：路径计算、docker 检查、容器参数配置，并调用容器内脚本执行实际编译
set -e

# 以脚本所在目录的上一层作为工作根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
XLLM_DIR="${WORK_ROOT}/xllm"
IMAGE="quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605"

export http_proxy="http://bamboo-proxy.jd.com:80"
export https_proxy="http://bamboo-proxy.jd.com:80"
export no_proxy=".jd.com,.jd.local,.local"

cd "${WORK_ROOT}"

# 通过代理访问 GitHub 时，HTTP/2 易触发 "Error in the HTTP2 framing layer" /
# "expected flush after ref listing"。强制 git 使用 HTTP/1.1 可规避此类协议层错误。
git config --global http.version HTTP/1.1
# 大仓库/弱网下降低单次数据缓冲导致的 RPC 失败概率
git config --global http.postBuffer 524288000

# 网络操作重试参数：最大尝试次数与每次失败后的等待秒数（可通过环境变量覆盖）
GIT_MAX_RETRIES="${GIT_MAX_RETRIES:-10}"
GIT_RETRY_DELAY="${GIT_RETRY_DELAY:-10}"

# 通用重试封装：弱网/代理下 git clone/pull/submodule 易随机失败，重试可显著提高成功率
# 用法：retry <描述> <命令...>
retry() {
  local desc="$1"; shift
  local attempt=1
  until "$@"; do
    if (( attempt >= GIT_MAX_RETRIES )); then
      echo "ERROR: ${desc} failed after ${GIT_MAX_RETRIES} attempts." >&2
      return 1
    fi
    echo "WARN: ${desc} failed (attempt ${attempt}/${GIT_MAX_RETRIES}), retrying in ${GIT_RETRY_DELAY}s..." >&2
    sleep "${GIT_RETRY_DELAY}"
    (( attempt++ ))
  done
}

# 若 xLLM 已存在则仅更新（git pull），否则克隆
if [[ -d "${XLLM_DIR}/.git" ]]; then
  echo "xLLM already exists at '${XLLM_DIR}', pulling latest changes..."
  retry "git pull" git -C "${XLLM_DIR}" pull --ff-only
elif [[ -d "${XLLM_DIR}" ]]; then
  echo "ERROR: '${XLLM_DIR}' exists but is not a git repository." && exit 1
else
  echo "Cloning xLLM into '${XLLM_DIR}'..."
  retry "git clone" git clone https://github.com/jd-opensource/xllm "${XLLM_DIR}"
fi

cd "${XLLM_DIR}"

# 将 xLLM 及其所有第三方子模块标记为 git 安全目录
git config --global --add safe.directory "${XLLM_DIR}"
THIRD_PARTY_DIRS=(
  xllm_ops brpc etcd_cpp_apiv3 minja smhasher torch_npu_ops Mooncake
  cpprestsdk hccl_transfer sentencepiece spdlog cutlass xllm_atb_layers xxHash
)
for dep in "${THIRD_PARTY_DIRS[@]}"; do
  git config --global --add safe.directory "${XLLM_DIR}/third_party/${dep}"
done

# 初始化并递归更新所有子模块
retry "git submodule update" git submodule update --init --recursive

# 容器内执行脚本（与本脚本同目录）
IN_CONTAINER_SCRIPT="${SCRIPT_DIR}/build_xllm_in_container.sh"

[[ ! -f "${IN_CONTAINER_SCRIPT}" ]] && echo "ERROR: '${IN_CONTAINER_SCRIPT}' not found." && exit 1
command -v docker >/dev/null 2>&1 || { echo "ERROR: 'docker' command is missing." && exit 1; }

RUN_OPTS=(
  --rm
  -t
  -u 0
  --name xllm_main_daily_build
  --privileged
  --ipc=host
  --network=host
  --device=/dev/davinci0
  --device=/dev/davinci_manager
  --device=/dev/devmm_svm
  --device=/dev/hisi_hdc
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi
  -v /usr/local/sbin/:/usr/local/sbin/
  -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf
  -v /var/log/npu/slog/:/var/log/npu/slog
  -v /var/log/npu/profiling/:/var/log/npu/profiling
  -v /var/log/npu/dump/:/var/log/npu/dump
  -v /export/home:/export/home
  -v /export/home/npu_vcpkg_cache_abi_1:/root/.cache/vcpkg # cached vcpkg installed dir
  -v /etc/hccn.conf:/etc/hccn.conf
  -e WORK_ROOT="${WORK_ROOT}"
  # 传入宿主机启动用户的 uid/gid，供容器内归档后 chown，避免产物被 root 占有
  -e HOST_UID="$(id -u)"
  -e HOST_GID="$(id -g)"
  -w /export/home
)

# 在容器内执行 build_in_container.sh（路径已通过 -v /export/home 挂载，保持一致）
docker run "${RUN_OPTS[@]}" "${IMAGE}" bash "${IN_CONTAINER_SCRIPT}"

echo "xllm prepare successfully!"