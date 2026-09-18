#!/bin/bash
# 安装带 NPU/容器预清理的每日定时任务（不修改 install_crontab.sh / cron_daily_build.sh）。
# 实际调用 cron_daily_build_with_cleanup.sh，测试前会先执行 cleanup_npu_hosts.py。
#
# 用法：
#   bash tools/install_crontab_with_cleanup.sh
#   CLEANUP_HOSTS="11.87.191.75 11.87.191.121" bash tools/install_crontab_with_cleanup.sh
#   STAGES="build test" MAIN_ARGS="--case_group gp_dsv4_glm52_qwen356_0916" \
#     bash tools/install_crontab_with_cleanup.sh
#   XLLM_REPO_URL=git@coding.jd.com:xLLM_AI/xllm.git XLLM_BRANCH=main \
#     XLLM_SERVICE_REPO_URL=https://github.com/xLLM-AI/xllm-service.git XLLM_SERVICE_BRANCH=main \
#     STAGES="build service_build test" bash tools/install_crontab_with_cleanup.sh
#   # 基准结果上报（可选）：XTASK_BENCH_REPORT=1 开启，BENCH_REPORTER_URL 指定上报服务
#   XTASK_BENCH_REPORT=1 BENCH_REPORTER_URL=http://124.222.187.33:8080 \
#     bash tools/install_crontab_with_cleanup.sh
#   CRON_TIME="30 3 * * *" bash tools/install_crontab_with_cleanup.sh
# 卸载：
#   bash tools/install_crontab_with_cleanup.sh --uninstall

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_SCRIPT="${SCRIPT_DIR}/cron_daily_build_with_cleanup.sh"

CRON_TIME="${CRON_TIME:-0 2 * * *}"
STAGES="${STAGES:-build test}"
MAIN_ARGS="${MAIN_ARGS:-}"
MAIN_ARGS_LIST="${MAIN_ARGS_LIST:-}"
XLLM_REPO_URL="${XLLM_REPO_URL:-}"
XLLM_BRANCH="${XLLM_BRANCH:-}"
XLLM_SERVICE_REPO_URL="${XLLM_SERVICE_REPO_URL:-}"
XLLM_SERVICE_BRANCH="${XLLM_SERVICE_BRANCH:-}"
CLEANUP_HOSTS="${CLEANUP_HOSTS:-11.87.191.75 11.87.191.121}"
# 基准结果上报（可选）：XTASK_BENCH_REPORT 非空且为 1/true/on/yes 时 main.py 会上报；
# BENCH_REPORTER_URL 为上报服务地址。两者默认留空，保持历史"不上报"行为。
XTASK_BENCH_REPORT="${XTASK_BENCH_REPORT:-}"
BENCH_REPORTER_URL="${BENCH_REPORTER_URL:-}"

TAG="# xllm_daily_build"

quote_cron_value() {
    local value="$1"
    value="${value//\'/\'\\\'\'}"
    printf "'%s'" "${value}"
}

CRON_XLLM_REPO_URL="$(quote_cron_value "${XLLM_REPO_URL}")"
CRON_XLLM_BRANCH="$(quote_cron_value "${XLLM_BRANCH}")"
CRON_XLLM_SERVICE_REPO_URL="$(quote_cron_value "${XLLM_SERVICE_REPO_URL}")"
CRON_XLLM_SERVICE_BRANCH="$(quote_cron_value "${XLLM_SERVICE_BRANCH}")"
CRON_MAIN_ARGS="$(quote_cron_value "${MAIN_ARGS}")"
CRON_MAIN_ARGS_LIST="$(quote_cron_value "${MAIN_ARGS_LIST}")"
CRON_CLEANUP_HOSTS="$(quote_cron_value "${CLEANUP_HOSTS}")"
CRON_XTASK_BENCH_REPORT="$(quote_cron_value "${XTASK_BENCH_REPORT}")"
CRON_BENCH_REPORTER_URL="$(quote_cron_value "${BENCH_REPORTER_URL}")"

CRON_LINE="${CRON_TIME} CLEANUP_HOSTS=${CRON_CLEANUP_HOSTS} XTASK_BENCH_REPORT=${CRON_XTASK_BENCH_REPORT} BENCH_REPORTER_URL=${CRON_BENCH_REPORTER_URL} XLLM_REPO_URL=${CRON_XLLM_REPO_URL} XLLM_BRANCH=${CRON_XLLM_BRANCH} XLLM_SERVICE_REPO_URL=${CRON_XLLM_SERVICE_REPO_URL} XLLM_SERVICE_BRANCH=${CRON_XLLM_SERVICE_BRANCH} MAIN_ARGS=${CRON_MAIN_ARGS} MAIN_ARGS_LIST=${CRON_MAIN_ARGS_LIST} /bin/bash ${CRON_SCRIPT} ${STAGES} ${TAG}"

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
    echo "已移除 xllm 每日定时任务（含预清理）。"
    exit 0
fi

if [[ ! -f "${CRON_SCRIPT}" ]]; then
  echo "ERROR: 找不到 ${CRON_SCRIPT}" >&2
  exit 1
fi
chmod +x "${CRON_SCRIPT}" "${SCRIPT_DIR}/cleanup_npu_hosts.py" "${SCRIPT_DIR}/cron_daily_build.sh"

existing=$(${CRONTAB_CMD} -l 2>/dev/null | grep -v "${TAG}" || true)
if [[ -n "${existing}" ]]; then
    printf '%s\n%s\n' "${existing}" "${CRON_LINE}" | ${CRONTAB_CMD} -
else
    echo "${CRON_LINE}" | ${CRONTAB_CMD} -
fi

echo "已安装定时任务（测试前自动清理 NPU/容器）："
echo "  ${CRON_LINE}"
echo
echo "查看当前 crontab： ${CRONTAB_CMD} -l"
echo "清理日志：         ${SCRIPT_DIR}/../outputs/npu_cleanup.log"
echo "框架日志：         ${SCRIPT_DIR}/../outputs/case_framework.log"
