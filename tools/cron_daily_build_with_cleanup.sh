#!/bin/bash
# 在 cron_daily_build.sh 之前清理指定服务器的 NPU 占用与运行中容器。
# 不修改 cron_daily_build.sh / install_crontab.sh，通过本脚本配合使用。
#
# 用法（与 cron_daily_build.sh 相同，额外支持清理相关环境变量）：
#   bash tools/cron_daily_build_with_cleanup.sh build test
#   CLEANUP_HOSTS="11.87.191.75 11.87.191.121" bash tools/cron_daily_build_with_cleanup.sh test
#   CLEANUP_SKIP=1 bash tools/cron_daily_build_with_cleanup.sh test   # 跳过清理
#
# 定时任务请使用 tools/install_crontab_with_cleanup.sh 安装。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CRON_SCRIPT="${SCRIPT_DIR}/cron_daily_build.sh"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup_npu_hosts.py"

CLEANUP_HOSTS="${CLEANUP_HOSTS:-11.87.191.75 11.87.191.121}"
CLEANUP_SKIP="${CLEANUP_SKIP:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUTPUT_DIR="${PROJECT_ROOT}/outputs"
LOG_DIR="${OUTPUT_DIR}/log"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
CLEANUP_LOG="${LOG_DIR}/npu_cleanup_${TS}.log"
ln -sf "log/npu_cleanup_${TS}.log" "${OUTPUT_DIR}/npu_cleanup.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_cleanup() {
  if [[ "${CLEANUP_SKIP}" == "1" ]]; then
    log "[cleanup] skipped (CLEANUP_SKIP=1)"
    return 0
  fi
  if [[ -z "${CLEANUP_HOSTS}" ]]; then
    log "[cleanup] skipped (CLEANUP_HOSTS is empty)"
    return 0
  fi
  if [[ ! -f "${CLEANUP_SCRIPT}" ]]; then
    log "[cleanup] ERROR: '${CLEANUP_SCRIPT}' not found."
    return 1
  fi

  # shellcheck disable=SC2206
  local hosts=( ${CLEANUP_HOSTS} )
  log "[cleanup] running: ${PYTHON_BIN} ${CLEANUP_SCRIPT} ${hosts[*]}"
  set +e
  ( cd "${PROJECT_ROOT}" && ${PYTHON_BIN} "${CLEANUP_SCRIPT}" "${hosts[@]}" )
  local rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    log "[cleanup] WARNING: cleanup exited with code ${rc}; continue daily build anyway"
  fi
  return 0
}

if [[ ! -f "${CRON_SCRIPT}" ]]; then
  log "ERROR: '${CRON_SCRIPT}' not found."
  exit 1
fi
chmod +x "${CRON_SCRIPT}" "${CLEANUP_SCRIPT}" 2>/dev/null || true

{
  log "==== pre-cleanup start, hosts: ${CLEANUP_HOSTS:-<none>} ===="
  run_cleanup
  cleanup_rc=$?
  if [[ ${cleanup_rc} -ne 0 ]]; then
    log "==== pre-cleanup FAILED (exit ${cleanup_rc}), continue daily build anyway ===="
  else
    log "==== pre-cleanup SUCCESS ===="
  fi

  log "==== delegate to cron_daily_build.sh: $* ===="
  exec /bin/bash "${CRON_SCRIPT}" "$@"
} >>"${CLEANUP_LOG}" 2>&1
