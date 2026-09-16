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
IN_CONTAINER_SCRIPT="${SCRIPT_DIR}/build_xllm_service_in_container.sh"
UPDATE_SOURCE="${XLLM_SERVICE_UPDATE_SOURCE:-1}"

# 分支可由第一个位置参数或 XLLM_SERVICE_BRANCH 指定，位置参数优先。
# 不指定时保持历史行为：已有仓库更新当前分支，新仓库克隆远端默认分支。
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash tools/prepare_xllm_service.sh [branch]"
  echo "Environment: XLLM_SERVICE_BRANCH, XLLM_SERVICE_REPO_URL"
  exit 0
fi
if [[ $# -gt 1 ]]; then
  echo "ERROR: expected at most one branch argument, got $# arguments." >&2
  exit 2
fi
SERVICE_TARGET_BRANCH="${1:-${XLLM_SERVICE_BRANCH:-}}"
if [[ -n "${SERVICE_TARGET_BRANCH}" ]] \
  && ! git check-ref-format --branch "${SERVICE_TARGET_BRANCH}" >/dev/null 2>&1; then
  echo "ERROR: invalid xLLM-Service branch name: '${SERVICE_TARGET_BRANCH}'." >&2
  exit 2
fi

IMAGE="${XLLM_SERVICE_BUILD_IMAGE:-}"
if [[ -z "${IMAGE}" ]]; then
  IMAGE="$("${PYTHON_EXEC:-python}" -c \
    'import json, sys; cfg = json.load(open(sys.argv[1], encoding="utf-8"))["XLLM_CONFIG"]; print("{}:{}".format(cfg["image_home"], cfg["default_image_name"]))' \
    "${WORK_ROOT}/config/xllm_config.json")"
fi
echo "[build-image] xLLM-Service image: ${IMAGE}"

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

GIT_MAX_RETRIES="${GIT_MAX_RETRIES:-50}"
GIT_RETRY_DELAY="${GIT_RETRY_DELAY:-30}"
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

update_service_checkout() {
  if [[ -z "${SERVICE_TARGET_BRANCH}" ]]; then
    echo "xLLM-Service already exists at '${SERVICE_SOURCE_DIR}', pulling its current branch..."
    retry "xLLM-Service git pull" git -C "${SERVICE_SOURCE_DIR}" pull --ff-only
    return
  fi

  echo "Updating xLLM-Service branch '${SERVICE_TARGET_BRANCH}' in '${SERVICE_SOURCE_DIR}'..."
  if git -C "${SERVICE_SOURCE_DIR}" show-ref --verify --quiet \
    "refs/heads/${SERVICE_TARGET_BRANCH}"; then
    git -C "${SERVICE_SOURCE_DIR}" checkout "${SERVICE_TARGET_BRANCH}"
    retry "xLLM-Service git pull branch ${SERVICE_TARGET_BRANCH}" \
      git -C "${SERVICE_SOURCE_DIR}" pull --ff-only origin "${SERVICE_TARGET_BRANCH}"
  else
    retry "xLLM-Service git fetch branch ${SERVICE_TARGET_BRANCH}" \
      git -C "${SERVICE_SOURCE_DIR}" fetch origin \
      "refs/heads/${SERVICE_TARGET_BRANCH}:refs/remotes/origin/${SERVICE_TARGET_BRANCH}"
    git -C "${SERVICE_SOURCE_DIR}" checkout -b "${SERVICE_TARGET_BRANCH}" \
      --track "origin/${SERVICE_TARGET_BRANCH}"
  fi
}

validate_service_repository() {
  local target_ref="HEAD"
  if [[ -n "${SERVICE_TARGET_BRANCH}" ]]; then
    target_ref="refs/heads/${SERVICE_TARGET_BRANCH}"
  fi
  echo "Validating requested xLLM-Service repository before removing the existing checkout..."
  retry "validate xLLM-Service repository and ref ${target_ref}" \
    git ls-remote --exit-code "${SERVICE_REPO_URL}" "${target_ref}" >/dev/null
}

# 仓库地址切换时不复用旧目录，避免旧分支、对象、子模块和未跟踪文件
# 混入新仓库的构建。这里只允许删除当前 xTask 工程根目录下固定的 xllm-service 路径。
if [[ -d "${SERVICE_SOURCE_DIR}/.git" ]]; then
  CURRENT_SERVICE_REPO_URL="$(git -C "${SERVICE_SOURCE_DIR}" remote get-url origin 2>/dev/null || true)"
  if [[ "${CURRENT_SERVICE_REPO_URL}" != "${SERVICE_REPO_URL}" ]]; then
    echo "WARN: existing xLLM-Service origin differs from the requested repository; recloning source." >&2
    echo "  existing: ${CURRENT_SERVICE_REPO_URL:-<missing origin>}" >&2
    echo "  requested: ${SERVICE_REPO_URL}" >&2
    if [[ "${UPDATE_SOURCE}" != "1" ]]; then
      echo "ERROR: XLLM_SERVICE_UPDATE_SOURCE=0 cannot replace the existing xLLM-Service checkout." >&2
      exit 1
    fi
    validate_service_repository
    if [[ "${SERVICE_SOURCE_DIR}" != "${WORK_ROOT}/xllm-service" || "${SERVICE_SOURCE_DIR}" == "/" ]]; then
      echo "ERROR: refusing to remove unexpected xLLM-Service path: '${SERVICE_SOURCE_DIR}'." >&2
      exit 1
    fi
    rm -rf -- "${SERVICE_SOURCE_DIR}"
  fi
fi

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
    update_service_checkout
  else
    if [[ -n "${SERVICE_TARGET_BRANCH}" ]]; then
      CURRENT_SERVICE_BRANCH="$(git -C "${SERVICE_SOURCE_DIR}" symbolic-ref --quiet --short HEAD || true)"
      if [[ "${CURRENT_SERVICE_BRANCH}" != "${SERVICE_TARGET_BRANCH}" ]]; then
        echo "ERROR: XLLM_SERVICE_UPDATE_SOURCE=0 cannot switch from branch '${CURRENT_SERVICE_BRANCH:-detached HEAD}' to '${SERVICE_TARGET_BRANCH}'." >&2
        exit 1
      fi
    fi
    echo "Using existing xLLM-Service checkout without pulling: ${SERVICE_SOURCE_DIR}"
  fi
elif [[ -e "${SERVICE_SOURCE_DIR}" ]]; then
  echo "ERROR: '${SERVICE_SOURCE_DIR}' exists but is not a git repository." >&2
  exit 1
else
  mkdir -p "$(dirname "${SERVICE_SOURCE_DIR}")"
  if [[ -n "${SERVICE_TARGET_BRANCH}" ]]; then
    echo "Cloning xLLM-Service branch '${SERVICE_TARGET_BRANCH}' into '${SERVICE_SOURCE_DIR}'..."
    retry "xLLM-Service git clone branch ${SERVICE_TARGET_BRANCH}" \
      git clone --branch "${SERVICE_TARGET_BRANCH}" --single-branch \
      "${SERVICE_REPO_URL}" "${SERVICE_SOURCE_DIR}"
  else
    echo "Cloning xLLM-Service default branch into '${SERVICE_SOURCE_DIR}'..."
    retry "xLLM-Service git clone" git clone "${SERVICE_REPO_URL}" "${SERVICE_SOURCE_DIR}"
  fi
fi

git config --global --add safe.directory "${SERVICE_SOURCE_DIR}" 2>/dev/null || true
# Initialize direct Service submodules first. Do not recurse the whole tree:
# cpprestsdk carries optional nested submodules (websocketpp/vcpkg) that this
# build does not need and that add unrelated network failure points.
retry "xLLM-Service submodule update" git -C "${SERVICE_SOURCE_DIR}" submodule update --init
# xLLM-Service links against xLLM's pinned spdlog via
# third_party/xllm/third_party/spdlog. That nested submodule must be present
# before CMake configure, but only under third_party/xllm.
retry "xLLM-Service xllm nested submodule update" \
  git -C "${SERVICE_SOURCE_DIR}" submodule update --init --recursive third_party/xllm

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
  third_party/xllm/third_party/spdlog/CMakeLists.txt
  third_party/xxHash/CMakeLists.txt
)

submodule_path_for_required_file() {
  local source_file="$1"
  if [[ "${source_file}" != third_party/* ]]; then
    return 1
  fi
  local top="${source_file#third_party/}"
  echo "third_party/${top%%/*}"
}

nested_submodule_within_xllm() {
  local source_file="$1"
  if [[ "${source_file}" != third_party/xllm/third_party/* ]]; then
    return 1
  fi
  local nested="${source_file#third_party/xllm/third_party/}"
  echo "third_party/${nested%%/*}"
}

repair_required_submodules() {
  local source_file submodule_path submodule_dir nested_submodule repair_key
  local -a repaired_keys=()
  local already_repaired

  for source_file in "${REQUIRED_SOURCE_FILES[@]}"; do
    if [[ -f "${SERVICE_SOURCE_DIR}/${source_file}" ]]; then
      continue
    fi

    if nested_submodule="$(nested_submodule_within_xllm "${source_file}")"; then
      repair_key="xllm-nested:${nested_submodule}"
      for already_repaired in "${repaired_keys[@]}"; do
        if [[ "${already_repaired}" == "${repair_key}" ]]; then
          continue 2
        fi
      done
      repaired_keys+=("${repair_key}")

      submodule_dir="${SERVICE_SOURCE_DIR}/third_party/xllm/${nested_submodule}"
      echo "WARN: missing required source '${source_file}', repairing nested submodule '${nested_submodule}'..." >&2
      if [[ -e "${submodule_dir}" ]]; then
        rm -rf -- "${submodule_dir}"
      fi
      if [[ -f "${SERVICE_SOURCE_DIR}/third_party/xllm/CMakeLists.txt" ]]; then
        retry "xLLM-Service nested submodule repair ${nested_submodule}" \
          git -C "${SERVICE_SOURCE_DIR}/third_party/xllm" submodule update --init "${nested_submodule}"
      else
        retry "xLLM-Service submodule repair third_party/xllm" \
          git -C "${SERVICE_SOURCE_DIR}" submodule update --init --recursive third_party/xllm
      fi
      continue
    fi

    if ! submodule_path="$(submodule_path_for_required_file "${source_file}")"; then
      continue
    fi
    repair_key="top:${submodule_path}"
    for already_repaired in "${repaired_keys[@]}"; do
      if [[ "${already_repaired}" == "${repair_key}" ]]; then
        continue 2
      fi
    done
    repaired_keys+=("${repair_key}")

    submodule_dir="${SERVICE_SOURCE_DIR}/${submodule_path}"
    echo "WARN: missing required source '${source_file}', repairing submodule '${submodule_path}'..." >&2
    # A half-initialized submodule can look present to git while the working tree
    # is empty (for example when a sibling submodule clone aborts the batch).
    if [[ -e "${submodule_dir}" ]]; then
      rm -rf -- "${submodule_dir}"
    fi
    if [[ "${submodule_path}" == "third_party/xllm" ]]; then
      retry "xLLM-Service submodule repair ${submodule_path}" \
        git -C "${SERVICE_SOURCE_DIR}" submodule update --init --recursive third_party/xllm
    else
      retry "xLLM-Service submodule repair ${submodule_path}" \
        git -C "${SERVICE_SOURCE_DIR}" submodule update --init "${submodule_path}"
    fi
  done
}

repair_required_submodules
for source_file in "${REQUIRED_SOURCE_FILES[@]}"; do
  if [[ ! -f "${SERVICE_SOURCE_DIR}/${source_file}" ]]; then
    echo "ERROR: xLLM-Service submodule is incomplete: ${source_file}" >&2
    exit 1
  fi
done

# 记录实际进入容器编译的源码版本。该输出会由 cron 写入 Service 编译日志
# 和 case_framework.log，便于将归档产物与源码提交精确对应。
SERVICE_COMPILE_COMMIT="$(git -C "${SERVICE_SOURCE_DIR}" rev-parse HEAD)"
SERVICE_COMPILE_BRANCH="$(git -C "${SERVICE_SOURCE_DIR}" symbolic-ref --quiet --short HEAD || echo detached-HEAD)"
echo "[source-version] xLLM-Service repo: ${SERVICE_REPO_URL}"
echo "[source-version] xLLM-Service branch: ${SERVICE_COMPILE_BRANCH}"
echo "[source-version] xLLM-Service commit: ${SERVICE_COMPILE_COMMIT}"
if git -C "${SERVICE_SOURCE_DIR}/third_party/xllm" rev-parse HEAD >/dev/null 2>&1; then
  SERVICE_XLLM_COMPILE_COMMIT="$(git -C "${SERVICE_SOURCE_DIR}/third_party/xllm" rev-parse HEAD)"
  echo "[source-version] xLLM-Service third_party/xllm commit: ${SERVICE_XLLM_COMPILE_COMMIT}"
fi

if [[ ! -f "${IN_CONTAINER_SCRIPT}" ]]; then
  echo "ERROR: '${IN_CONTAINER_SCRIPT}' not found." >&2
  exit 1
fi
command -v docker >/dev/null 2>&1 || {
  echo "ERROR: 'docker' command is missing." >&2
  exit 1
}

# 容器名带编译时间，避免多人共用机器时 --name 冲突。
CONTAINER_NAME="xllm_svc_daily_build_$(date +%Y%m%d_%H%M%S)"
echo "[build-container] name: ${CONTAINER_NAME}"

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
  --name "${CONTAINER_NAME}"
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
