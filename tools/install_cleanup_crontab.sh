#!/bin/bash
# 单独安装 NPU/容器清理定时任务，与现有 install_crontab.sh 并存，不修改原脚本。
# 默认在每日构建开始前 5 分钟执行 cleanup_npu_hosts.py。
#
# 用法：
#   bash tools/install_cleanup_crontab.sh
#   CLEANUP_HOSTS="11.87.191.75 11.87.191.121" bash tools/install_cleanup_crontab.sh
#   CRON_TIME="55 1 * * *" bash tools/install_cleanup_crontab.sh
# 卸载：
#   bash tools/install_cleanup_crontab.sh --uninstall

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup_npu_hosts.py"

CRON_TIME="${CRON_TIME:-55 1 * * *}"
CLEANUP_HOSTS="${CLEANUP_HOSTS:-11.87.191.75 11.87.191.121}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TAG="# xllm_npu_cleanup"

quote_cron_value() {
    local value="$1"
    value="${value//\'/\'\\\'\'}"
    printf "'%s'" "${value}"
}

CRON_CLEANUP_HOSTS="$(quote_cron_value "${CLEANUP_HOSTS}")"
CRON_LINE="${CRON_TIME} CLEANUP_HOSTS=${CRON_CLEANUP_HOSTS} cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${CLEANUP_SCRIPT} ${CLEANUP_HOSTS} >> ${PROJECT_ROOT}/outputs/log/npu_cleanup_cron.log 2>&1 ${TAG}"

CURRENT_USER="$(whoami)"
if crontab -l >/dev/null 2>&1; then
    CRONTAB_CMD="crontab"
elif sudo crontab -u "${CURRENT_USER}" -l >/dev/null 2>&1; then
    CRONTAB_CMD="sudo crontab -u ${CURRENT_USER}"
    echo "注意：当前用户无 crontab 权限，将使用 ${CRONTAB_CMD}"
else
    echo "ERROR: 当前用户无 crontab 权限，且 sudo crontab 也不可用" >&2
    exit 1
fi

if [[ "${1:-}" == "--uninstall" ]]; then
    existing=$(${CRONTAB_CMD} -l 2>/dev/null | grep -v "${TAG}" || true)
    if [[ -n "${existing}" ]]; then
        echo "${existing}" | ${CRONTAB_CMD} -
    else
        echo "" | ${CRONTAB_CMD} -
    fi
    echo "已移除 NPU 清理定时任务。"
    exit 0
fi

if [[ ! -f "${CLEANUP_SCRIPT}" ]]; then
  echo "ERROR: 找不到 ${CLEANUP_SCRIPT}" >&2
  exit 1
fi
chmod +x "${CLEANUP_SCRIPT}"
mkdir -p "${PROJECT_ROOT}/outputs/log"

existing=$(${CRONTAB_CMD} -l 2>/dev/null | grep -v "${TAG}" || true)
if [[ -n "${existing}" ]]; then
    printf '%s\n%s\n' "${existing}" "${CRON_LINE}" | ${CRONTAB_CMD} -
else
    echo "${CRON_LINE}" | ${CRONTAB_CMD} -
fi

echo "已安装独立 NPU 清理定时任务："
echo "  ${CRON_LINE}"
echo
echo "可与 install_crontab.sh 已安装的构建任务并存。"
echo "查看当前 crontab： ${CRONTAB_CMD} -l"
