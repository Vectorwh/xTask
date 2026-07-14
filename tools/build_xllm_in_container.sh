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
cp -rf /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ "${WORK_ROOT}/pkg/xllm/"
#cp "${XLLM_DIR}/build/xllm/core/server/xllm" "${WORK_ROOT}/pkg/xllm/"
cp -f "${XLLM_DIR}"/dist/* "${WORK_ROOT}/pkg/xllm/"

# 由外层 prepare_xllm.sh 通过 -e HOST_UID / -e HOST_GID 传入宿主机启动用户的 uid/gid，
# 归档后把 pkg/xllm 目录及其子文件所有者改回宿主机用户。
if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
  echo "chown ${WORK_ROOT}/pkg/xllm to host user ${HOST_UID}:${HOST_GID}"
  chown -R "${HOST_UID}:${HOST_GID}" "${WORK_ROOT}/pkg/xllm/"
else
  echo "WARN: HOST_UID/HOST_GID not set, skip chown; files may be owned by root on host."
fi

echo "xllm build (in container) finished!"