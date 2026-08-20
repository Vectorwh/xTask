#!/bin/bash
# xLLM-Service 编译下载部署（容器外编排层）。
#
# 这是独立于 prepare_xllm.sh 的可选编译入口。默认不参与原有 xLLM
# wheel 编译；需要 Service 产物时显式执行本脚本，或使用 cron 的
# service_build 阶段。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVICE_SOURCE_DIR="${XLLM_SERVICE_SOURCE_DIR:-${WORK_ROOT}/xllm-service}"
SERVICE_REPO_URL="${XLLM_SERVICE_REPO_URL:-git@coding.jd.com:xLLM_AI/xllm-service.git}"
IMAGE="${XLLM_SERVICE_BUILD_IMAGE:-quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605}"
IN_CONTAINER_SCRIPT="${SCRIPT_DIR}/build_xllm_service_in_container.sh"
UPDATE_SOURCE="${XLLM_SERVICE_UPDATE_SOURCE:-1}"

case "${UPDATE_SOURCE}" in
  0|1) ;;
  *)
    echo "ERROR: XLLM_SERVICE_UPDATE_SOURCE must be 0 or 1, got '${UPDATE_SOURCE}'." >&2
    exit 1
    ;;
esac

export http_proxy="${http_proxy:-http://bamboo-proxy.jd.com:80}"
export https_proxy="${https_proxy:-http://bamboo-proxy.jd.com:80}"
export no_proxy="${no_proxy:-.jd.com,.jd.local,.local}"

GIT_MAX_RETRIES="${GIT_MAX_RETRIES:-5}"
GIT_RETRY_DELAY="${GIT_RETRY_DELAY:-10}"
retry() {
  local description="$1"
  shift
  local attempt=1
  until "$@"; do
    if (( attempt >= GIT_MAX_RETRIES )); then
      echo "ERROR: ${description} failed after ${GIT_MAX_RETRIES} attempts." >&2
      return 1
    fi
    echo "WARN: ${description} failed (attempt ${attempt}/${GIT_MAX_RETRIES}), retrying..." >&2
    sleep "${GIT_RETRY_DELAY}"
    ((attempt++))
  done
}

if [[ -d "${SERVICE_SOURCE_DIR}/.git" ]]; then
  git config --global --add safe.directory "${SERVICE_SOURCE_DIR}" 2>/dev/null || true
  if [[ "${UPDATE_SOURCE}" == "1" ]]; then
    CPPREST_DIR="${SERVICE_SOURCE_DIR}/third_party/cpprestsdk"
    PATCH_FILE="${SERVICE_SOURCE_DIR}/third_party/custom_cache/cpprestsdk.patch"
    if [[ -d "${CPPREST_DIR}/.git" ]] && ! git -C "${CPPREST_DIR}" diff --quiet; then
      if [[ -f "${PATCH_FILE}" ]] && git -C "${CPPREST_DIR}" apply --reverse --check "${PATCH_FILE}"; then
        echo "Temporarily reversing the known xLLM-Service cpprest patch before source update."
        git -C "${CPPREST_DIR}" apply --reverse "${PATCH_FILE}"
        if ! git -C "${CPPREST_DIR}" diff --quiet; then
          # Restore the official patch before stopping so an unrelated local
          # edit never leaves the user's checkout in a different state.
          git -C "${CPPREST_DIR}" apply "${PATCH_FILE}" || true
          echo "ERROR: xLLM-Service cpprest submodule also has changes outside the known prepare.sh patch." >&2
          exit 1
        fi
      else
        echo "ERROR: xLLM-Service cpprest submodule has changes other than the known prepare.sh patch." >&2
        exit 1
      fi
    fi
    echo "xLLM-Service already exists at '${SERVICE_SOURCE_DIR}', pulling latest changes..."
    retry "xLLM-Service git pull" git -C "${SERVICE_SOURCE_DIR}" pull --ff-only
  else
    echo "Using existing xLLM-Service checkout without pulling: ${SERVICE_SOURCE_DIR}"
  fi
elif [[ -e "${SERVICE_SOURCE_DIR}" ]]; then
  echo "ERROR: '${SERVICE_SOURCE_DIR}' exists but is not a git repository." >&2
  exit 1
else
  mkdir -p "$(dirname "${SERVICE_SOURCE_DIR}")"
  echo "Cloning xLLM-Service into '${SERVICE_SOURCE_DIR}'..."
  retry "xLLM-Service git clone" git clone "${SERVICE_REPO_URL}" "${SERVICE_SOURCE_DIR}"
fi

git config --global --add safe.directory "${SERVICE_SOURCE_DIR}" 2>/dev/null || true
# The upstream build instructions initialize the Service repository's direct
# submodules. Several of those dependencies contain their own optional
# submodules (for example cpprestsdk websocket support) which this build does
# not use; recursively fetching them adds unrelated network failure points.
retry "xLLM-Service submodule update" git -C "${SERVICE_SOURCE_DIR}" submodule update --init

if [[ ! -f "${SERVICE_SOURCE_DIR}/CMakeLists.txt" || ! -f "${SERVICE_SOURCE_DIR}/prepare.sh" ]]; then
  echo "ERROR: xLLM-Service source is incomplete: ${SERVICE_SOURCE_DIR}" >&2
  exit 1
fi
REQUIRED_SOURCE_FILES=(
  third_party/brpc/CMakeLists.txt
  third_party/cpprestsdk/Release/CMakeLists.txt
  third_party/etcd_cpp_apiv3/CMakeLists.txt
  third_party/minja/CMakeLists.txt
  third_party/sentencepiece/CMakeLists.txt
  third_party/smhasher/src/CMakeLists.txt
  third_party/xllm/CMakeLists.txt
  third_party/xxHash/CMakeLists.txt
)
for source_file in "${REQUIRED_SOURCE_FILES[@]}"; do
  if [[ ! -f "${SERVICE_SOURCE_DIR}/${source_file}" ]]; then
    echo "ERROR: xLLM-Service submodule is incomplete: ${source_file}" >&2
    exit 1
  fi
done
if [[ ! -f "${IN_CONTAINER_SCRIPT}" ]]; then
  echo "ERROR: '${IN_CONTAINER_SCRIPT}' not found." >&2
  exit 1
fi
command -v docker >/dev/null 2>&1 || {
  echo "ERROR: 'docker' command is missing." >&2
  exit 1
}

# The build container sees the same absolute paths through /export/home.
case "${SERVICE_SOURCE_DIR}" in
  /export/home/*) ;;
  *)
    echo "ERROR: XLLM_SERVICE_SOURCE_DIR must be under /export/home so the build container can see it: ${SERVICE_SOURCE_DIR}" >&2
    exit 1
    ;;
esac

RUN_OPTS=(
  --rm
  -t
  -u 0
  --name xllm_service_build
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
  -v /export/home/npu_vcpkg_cache_abi_1:/root/.cache/vcpkg
  -v /etc/hccn.conf:/etc/hccn.conf
  -e WORK_ROOT="${WORK_ROOT}"
  -e XLLM_SERVICE_SOURCE_DIR="${SERVICE_SOURCE_DIR}"
  -e XLLM_SERVICE_BUILD_JOBS="${XLLM_SERVICE_BUILD_JOBS:-8}"
  -e XLLM_SERVICE_BUILD_TYPE="${XLLM_SERVICE_BUILD_TYPE:-Release}"
  -e XLLM_SERVICE_CXX_STANDARD="${XLLM_SERVICE_CXX_STANDARD:-20}"
  -e XLLM_SERVICE_BUILD_IMAGE="${IMAGE}"
  -e HOST_UID="$(id -u)"
  -e HOST_GID="$(id -g)"
  -w /export/home
)

docker run "${RUN_OPTS[@]}" "${IMAGE}" bash "${IN_CONTAINER_SCRIPT}"
echo "xLLM-Service prepare successfully!"
