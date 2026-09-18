#!/bin/bash
# 一键安装/更新 xllm 每日定时任务到当前用户的 crontab。
# 定时任务实际调用 cron_daily_build.sh，可指定要依次执行的分段任务（build/service_build/test）。
#
# 用法：
#   bash tools/install_crontab.sh                       # 默认每天 02:00 执行 build + test
#   STAGES="build" bash tools/install_crontab.sh        # 只编译
#   STAGES="build test" bash tools/install_crontab.sh   # 先编译后测试
#   STAGES="build service_build test" bash tools/install_crontab.sh  # 编译 xLLM 与 Service 后测试
#   CRON_TIME="30 3 * * *" bash tools/install_crontab.sh            # 自定义时间
#   MAIN_ARGS="--case_group base_case_group" bash tools/install_crontab.sh   # 指定 main.py 参数
#   XLLM_BRANCH=main XLLM_SERVICE_BRANCH=main STAGES="build service_build test" \
#     bash tools/install_crontab.sh                               # 固化两个编译分支
#   # 基准结果上报（可选）：XTASK_BENCH_REPORT=1 开启，BENCH_REPORTER_URL 指定上报服务
#   XTASK_BENCH_REPORT=1 BENCH_REPORTER_URL=http://124.222.187.33:8080 \
#     STAGES="build test" bash tools/install_crontab.sh
#   # 定时依次执行多组用例（分号分隔，每组各跑一次 main.py）：
#   MAIN_ARGS_LIST="--case_name case1 ; --case_name case2" bash tools/install_crontab.sh
# 卸载：
#   bash tools/install_crontab.sh --uninstall

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_SCRIPT="${SCRIPT_DIR}/cron_daily_build.sh"

# cron 时间表达式（分 时 日 月 周），默认每天 02:00
CRON_TIME="${CRON_TIME:-0 2 * * *}"
# 依次执行的分段任务，默认 build + test
STAGES="${STAGES:-build test}"
# 传给 main.py 的参数（test 阶段使用）
MAIN_ARGS="${MAIN_ARGS:-}"
# 多组用例参数列表（分号分隔，每组各执行一次 main.py；设置后优先于 MAIN_ARGS）
MAIN_ARGS_LIST="${MAIN_ARGS_LIST:-}"
# 两个编译阶段各自使用的源码分支；为空时保持 prepare 脚本原有默认行为。
XLLM_BRANCH="${XLLM_BRANCH:-}"
XLLM_SERVICE_BRANCH="${XLLM_SERVICE_BRANCH:-}"
# 基准结果上报（可选）：XTASK_BENCH_REPORT 非空且为 1/true/on/yes 时 main.py 会上报；
# BENCH_REPORTER_URL 为上报服务地址。两者默认留空，保持历史"不上报"行为。
XTASK_BENCH_REPORT="${XTASK_BENCH_REPORT:-}"
BENCH_REPORTER_URL="${BENCH_REPORTER_URL:-}"
# 用于识别/去重的标记
TAG="# xllm_daily_build"

quote_cron_value() {
    local value="$1"
    value="${value//\'/\'\\\'\'}"
    printf "'%s'" "${value}"
}

CRON_XLLM_BRANCH="$(quote_cron_value "${XLLM_BRANCH}")"
CRON_XLLM_SERVICE_BRANCH="$(quote_cron_value "${XLLM_SERVICE_BRANCH}")"
CRON_XTASK_BENCH_REPORT="$(quote_cron_value "${XTASK_BENCH_REPORT}")"
CRON_BENCH_REPORTER_URL="$(quote_cron_value "${BENCH_REPORTER_URL}")"
CRON_MAIN_ARGS="$(quote_cron_value "${MAIN_ARGS}")"
CRON_MAIN_ARGS_LIST="$(quote_cron_value "${MAIN_ARGS_LIST}")"
# 把测试参数和构建分支写入 cron 行，保证 cron 的精简环境下仍能拿到配置。
CRON_LINE="${CRON_TIME} XTASK_BENCH_REPORT=${CRON_XTASK_BENCH_REPORT} BENCH_REPORTER_URL=${CRON_BENCH_REPORTER_URL} XLLM_BRANCH=${CRON_XLLM_BRANCH} XLLM_SERVICE_BRANCH=${CRON_XLLM_SERVICE_BRANCH} MAIN_ARGS=${CRON_MAIN_ARGS} MAIN_ARGS_LIST=${CRON_MAIN_ARGS_LIST} /bin/bash ${CRON_SCRIPT} ${STAGES} ${TAG}"

# 检测当前用户是否有 crontab 权限，自动选择 crontab 命令
# 优先尝试普通 crontab，失败则尝试 sudo crontab -u $USER
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

# 卸载
if [[ "${1:-}" == "--uninstall" ]]; then
    # 读取当前 crontab，过滤掉标记行，写回；空结果则删除
    existing=$(${CRONTAB_CMD} -l 2>/dev/null | grep -v "${TAG}" || true)
    if [[ -n "${existing}" ]]; then
        echo "${existing}" | ${CRONTAB_CMD} -
    else
        echo "" | ${CRONTAB_CMD} -
    fi
    echo "已移除 xllm 每日定时任务。"
    exit 0
fi

if [[ ! -f "${CRON_SCRIPT}" ]]; then
  echo "ERROR: 找不到 ${CRON_SCRIPT}" >&2
  exit 1
fi
chmod +x "${CRON_SCRIPT}"

# 先删除旧的同标记任务，再追加新的，实现幂等安装
# 使用 || true 避免 grep -v 对空输入返回非零导致 set -e 退出
existing=$(${CRONTAB_CMD} -l 2>/dev/null | grep -v "${TAG}" || true)
if [[ -n "${existing}" ]]; then
    printf '%s\n%s\n' "${existing}" "${CRON_LINE}" | ${CRONTAB_CMD} -
else
    echo "${CRON_LINE}" | ${CRONTAB_CMD} -
fi

echo "已安装定时任务："
echo "  ${CRON_LINE}"
echo
echo "查看当前 crontab： ${CRONTAB_CMD} -l"
echo "框架日志（run+test）：${SCRIPT_DIR}/../outputs/case_framework.log"
echo "编译日志：           ${SCRIPT_DIR}/../outputs/build.log"
echo "Service编译日志：    ${SCRIPT_DIR}/../outputs/log/service_build_*.log"
