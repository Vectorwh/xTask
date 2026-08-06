#!/bin/bash
# 容器内执行：代理设置、编译、生成件归档
# 由外层 prepare_xllm.sh 通过 docker run 调用，WORK_ROOT 经环境变量传入
# 文件名：build_xllm_in_container.sh
set -euo pipefail

# 代理设置（内网编译需要）
export http_proxy="http://bamboo-proxy.jd.com:80"
export https_proxy="http://bamboo-proxy.jd.com:80"
export no_proxy=".jd.com,.jd.local,.local"

# 工作根目录：由外层 prepare_xllm.sh 通过 -e WORK_ROOT 传入（/export/home 已挂载，路径一致）
if [[ -z "${WORK_ROOT:-}" ]]; then
  echo "ERROR: WORK_ROOT is not set. This script must be launched by prepare_xllm.sh." >&2
  exit 1
fi

XLLM_DIR="${WORK_ROOT}/xllm"
XLLM_PKG_DIR="${WORK_ROOT}/pkg/xllm"
RUNTIME_LIB_DIR="${XLLM_PKG_DIR}/runtime_libs"

if [[ ! -f "${XLLM_DIR}/setup.py" ]]; then
  echo "ERROR: '${XLLM_DIR}/setup.py' not found. Is the xLLM source prepared?" >&2
  exit 1
fi

# 进入源码目录后再编译（setup.py 位于 XLLM_DIR）
cd "${XLLM_DIR}"

# 容器以 root 运行，宿主机的 safe.directory 配置对容器内 root 无效，
# 且 setup.py 会调用 `git submodule status`，故在容器内为 root 配置 safe.directory。
# 用通配符一次性覆盖主仓库与所有（含嵌套）子模块目录，避免逐个枚举遗漏。
git config --global --add safe.directory '*'

# 编译：安装 pre-commit 后执行 setup.py 构建
pip install pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple
# python setup.py build --device npu
python setup.py bdist_wheel --device npu

# 编译完执行生成件归档
mkdir -p "${XLLM_PKG_DIR}"
cp -rf /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ "${XLLM_PKG_DIR}/"
#cp "${XLLM_DIR}/build/xllm/core/server/xllm" "${WORK_ROOT}/pkg/xllm/"
cp -f "${XLLM_DIR}"/dist/* "${XLLM_PKG_DIR}/"

# 收集编译环境中 xLLM 启动所需、但尚未打入 wheel 的动态库。
# libxxhash.so.0 通常是版本化文件的软链接，解引用后归档，避免测试容器拿到断链。
XXHASH_SONAME_PATH="/usr/lib64/libxxhash.so.0"
if [[ ! -e "${XXHASH_SONAME_PATH}" ]]; then
  echo "ERROR: '${XXHASH_SONAME_PATH}' not found after xLLM build." >&2
  exit 1
fi
XXHASH_REAL_PATH="$(readlink -f "${XXHASH_SONAME_PATH}")"
if [[ -z "${XXHASH_REAL_PATH}" || ! -f "${XXHASH_REAL_PATH}" ]]; then
  echo "ERROR: failed to resolve '${XXHASH_SONAME_PATH}'." >&2
  exit 1
fi

ASIO_SO_PATH="$(find "${XLLM_DIR}/build" -type f -path '*/mooncake-common/libasio.so' -print -quit)"
if [[ -z "${ASIO_SO_PATH}" || ! -f "${ASIO_SO_PATH}" ]]; then
  echo "ERROR: libasio.so not found under '${XLLM_DIR}/build'." >&2
  exit 1
fi

rm -rf "${RUNTIME_LIB_DIR}"
mkdir -p "${RUNTIME_LIB_DIR}"
install -m 0755 "${XXHASH_REAL_PATH}" "${RUNTIME_LIB_DIR}/libxxhash.so.0"
install -m 0755 "${ASIO_SO_PATH}" "${RUNTIME_LIB_DIR}/libasio.so"
echo "Collected runtime libraries in '${RUNTIME_LIB_DIR}':"
ls -l "${RUNTIME_LIB_DIR}/libxxhash.so.0" "${RUNTIME_LIB_DIR}/libasio.so"

# 由外层 prepare_xllm.sh 通过 -e HOST_UID / -e HOST_GID 传入宿主机启动用户的 uid/gid，
# 归档后把 pkg/xllm 目录及其子文件所有者改回宿主机用户。
if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
  echo "chown ${XLLM_PKG_DIR} to host user ${HOST_UID}:${HOST_GID}"
  chown -R "${HOST_UID}:${HOST_GID}" "${XLLM_PKG_DIR}/"
else
  echo "WARN: HOST_UID/HOST_GID not set, skip chown; files may be owned by root on host."
fi

echo "xllm build (in container) finished!"