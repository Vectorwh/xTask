#!/bin/bash
# 容器内执行：代理设置、编译、生成件归档
# 由外层 prepare_xllm.sh 通过 docker run 调用，WORK_ROOT 经环境变量传入
# 文件名：build_xllm_in_container.sh
set -euo pipefail

# 代理设置（内网编译需要）
export http_proxy="http://bamboo-proxy.jd.com:80"
export https_proxy="http://bamboo-proxy.jd.com:80"
export no_proxy=".jd.com,.jd.local,.local"
#export no_proxy=".jd.com,.jd.local,.local,repo.openeuler.org,mirrors.openeuler.org,dl-cdn.openeuler.openatom.cn"

# 工作根目录：由外层 prepare_xllm.sh 通过 -e WORK_ROOT 传入（/export/home 已挂载，路径一致）
if [[ -z "${WORK_ROOT:-}" ]]; then
  echo "ERROR: WORK_ROOT is not set. This script must be launched by prepare_xllm.sh." >&2
  exit 1
fi

XLLM_DIR="${WORK_ROOT}/xllm"
XLLM_PKG_DIR="${WORK_ROOT}/pkg/xllm"
XLLM_DIST_DIR="${XLLM_DIR}/dist"
RUNTIME_LIB_DIR="${XLLM_PKG_DIR}/runtime_libs"

# root 容器中的 git submodule sync/config 会替换宿主机 .git 内的文件。
# 成功和失败退出都归还这些元数据，避免下一轮普通用户无法继续增量更新。
restore_git_metadata_ownership() {
  local git_dir="${XLLM_DIR}/.git"
  if [[ -z "${HOST_UID:-}" || -z "${HOST_GID:-}" ]]; then
    echo "WARN: HOST_UID/HOST_GID not set; cannot restore Git metadata ownership." >&2
    return 0
  fi
  if [[ ! "${HOST_UID}" =~ ^[0-9]+$ || ! "${HOST_GID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: HOST_UID/HOST_GID must be numeric; refusing ownership changes." >&2
    return 1
  fi
  if [[ -L "${XLLM_DIR}" || -L "${git_dir}" ]]; then
    echo "ERROR: refusing Git ownership cleanup through a symlink: ${git_dir}" >&2
    return 1
  fi
  if [[ ! -e "${git_dir}" ]]; then
    return 0
  fi
  if [[ ! -d "${git_dir}" ]]; then
    echo "ERROR: expected a .git directory; refusing external Git metadata cleanup: ${git_dir}" >&2
    return 1
  fi
  echo "[git-ownership] Restore root-owned Git metadata to ${HOST_UID}:${HOST_GID}: ${git_dir}"
  # 不跟随符号链接、不跨文件系统、不修改非 root 属主，也不调整文件权限。
  # --from 在 chown 时再次检查属主，避免覆盖并发发生的属主变更。
  find -P "${git_dir}" -xdev -uid 0 \
    -exec chown -h --from=0 -- "${HOST_UID}:${HOST_GID}" {} +
}

on_build_exit() {
  local build_status=$?
  trap - EXIT
  if ! restore_git_metadata_ownership; then
    echo "ERROR: Git metadata ownership cleanup failed; repair it before the next build." >&2
    if (( build_status == 0 )); then
      build_status=1
    fi
  fi
  exit "${build_status}"
}
trap on_build_exit EXIT

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
# 只清理源码目录中的历史 wheel。pkg/xllm 中的上一份可用产物会保留到新
# wheel 成功生成并完成暂存，避免构建失败时丢失可用包。
rm -f "${XLLM_DIST_DIR}"/xllm_npu*.whl
python setup.py bdist_wheel --device npu

# 本轮构建必须且只能生成一个 xLLM wheel，避免通配符把多个版本归档。
shopt -s nullglob
XLLM_WHEELS=("${XLLM_DIST_DIR}"/xllm_npu*.whl)
shopt -u nullglob
if [[ ${#XLLM_WHEELS[@]} -ne 1 ]]; then
  echo "ERROR: expected exactly one xLLM wheel in '${XLLM_DIST_DIR}', found ${#XLLM_WHEELS[@]}." >&2
  exit 1
fi

# 编译完执行生成件归档。先完整暂存新 wheel，再清理历史版本并替换，避免
# 复制失败时破坏 pkg/xllm 中的上一份可用产物。
mkdir -p "${XLLM_PKG_DIR}"
cp -rf /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ "${XLLM_PKG_DIR}/"
#cp "${XLLM_DIR}/build/xllm/core/server/xllm" "${WORK_ROOT}/pkg/xllm/"
XLLM_WHEEL_NAME="$(basename "${XLLM_WHEELS[0]}")"
XLLM_WHEEL_TMP="${XLLM_PKG_DIR}/.${XLLM_WHEEL_NAME}.tmp"
cp -f "${XLLM_WHEELS[0]}" "${XLLM_WHEEL_TMP}"
rm -f "${XLLM_PKG_DIR}"/xllm_npu*.whl
mv -f "${XLLM_WHEEL_TMP}" "${XLLM_PKG_DIR}/${XLLM_WHEEL_NAME}"
echo "Archived xLLM wheel: ${XLLM_PKG_DIR}/${XLLM_WHEEL_NAME}"

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

# Mooncake Store/TENT 启用 io_uring 后，xllm 二进制对 liburing.so.2 存在硬链接
# 依赖（DT_NEEDED），但该库既未打入 wheel 也未随基础镜像分发，需一并归档。
# /usr/lib64/liburing.so.2 可能是软链接，与 libxxhash.so.0 相同，解引用后归档，
# 避免测试容器拿到断链。
URING_SONAME_PATH="/usr/lib64/liburing.so.2"
if [[ ! -e "${URING_SONAME_PATH}" ]]; then
  echo "ERROR: '${URING_SONAME_PATH}' not found after xLLM build." >&2
  echo "Hint: xllm links against liburing (io_uring) when the build environment" >&2
  echo "has liburing-devel installed; the runtime library must be collectable here." >&2
  exit 1
fi
URING_REAL_PATH="$(readlink -f "${URING_SONAME_PATH}")"
if [[ -z "${URING_REAL_PATH}" || ! -f "${URING_REAL_PATH}" ]]; then
  echo "ERROR: failed to resolve '${URING_SONAME_PATH}'." >&2
  exit 1
fi

rm -rf "${RUNTIME_LIB_DIR}"
mkdir -p "${RUNTIME_LIB_DIR}"
install -m 0755 "${XXHASH_REAL_PATH}" "${RUNTIME_LIB_DIR}/libxxhash.so.0"
install -m 0755 "${ASIO_SO_PATH}" "${RUNTIME_LIB_DIR}/libasio.so"
install -m 0755 "${URING_REAL_PATH}" "${RUNTIME_LIB_DIR}/liburing.so.2"
echo "Collected runtime libraries in '${RUNTIME_LIB_DIR}':"
ls -l "${RUNTIME_LIB_DIR}/libxxhash.so.0" "${RUNTIME_LIB_DIR}/libasio.so" "${RUNTIME_LIB_DIR}/liburing.so.2"

# 由外层 prepare_xllm.sh 通过 -e HOST_UID / -e HOST_GID 传入宿主机启动用户的 uid/gid，
# 归档后把 pkg/xllm 目录及其子文件所有者改回宿主机用户。
if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
  echo "chown ${XLLM_PKG_DIR} to host user ${HOST_UID}:${HOST_GID}"
  chown -R "${HOST_UID}:${HOST_GID}" "${XLLM_PKG_DIR}/"
else
  echo "WARN: HOST_UID/HOST_GID not set, skip chown; files may be owned by root on host."
fi

echo "xllm build (in container) finished!"
