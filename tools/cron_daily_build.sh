#!/bin/bash
# 定时任务包装脚本：由 crontab 调用，负责日志记录、并发锁、分段任务编排。
# 支持两个分段任务（stage），按入参顺序依次执行：
#   build : 编译 xllm，即执行 prepare_xllm.sh
#   test  : 运行用例测试，即执行 python main.py
#
# 用法：
#   bash tools/cron_daily_build.sh                 # 不带参数：默认依次执行 build test
#   bash tools/cron_daily_build.sh build           # 只编译
#   bash tools/cron_daily_build.sh test            # 只测试
#   bash tools/cron_daily_build.sh build test      # 先编译后测试（按入参顺序执行）
#   bash tools/cron_daily_build.sh test build      # 也可调整顺序
#
# test 阶段传给 main.py 的参数通过环境变量 MAIN_ARGS 指定（默认按用例组执行）：
#   MAIN_ARGS="--case_group base_case_group" bash tools/cron_daily_build.sh test
#   MAIN_ARGS="--case_name qwen3_8b_xllm_001"  bash tools/cron_daily_build.sh build test
#
# 日志查看：
#   带时间戳的真实日志统一归档在 outputs/log/ 下：
#     框架日志 case_framework_<ts>.log ：总编排信息 + test 阶段输出（run 与 test 合并）
#     编译日志 build_<ts>.log          ：build 阶段独立日志
#   outputs/ 下保留软链指向最新一次运行：case_framework.log / build.log
#   实时跟踪：tail -f outputs/case_framework.log
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_xllm.sh"
MAIN_PY="${PROJECT_ROOT}/main.py"

# test 阶段 python 解释器与 main.py 参数（可用环境变量覆）
PYTHON_BIN="${PYTHON_BIN:-python}"
MAIN_ARGS="${MAIN_ARGS:-}"

# 入参即需要依次执行的分段任务；未指定时默认 build + test
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(build test)
fi

# 日志目录与文件
# - 带时间戳的真实日志统一归档到 outputs/log/ 下：
#     case_framework_<ts>.log : 框架日志（总编排信息 + test 阶段输出，run 与 test 合并）
#     build_<ts>.log          : build 阶段独立日志
# - outputs/ 下保留 case_framework.log / build.log 软链，指向最新一次运行的真实日志
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
LOG_DIR="${OUTPUT_DIR}/log"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/case_framework_${TS}.log"
BUILD_LOG="${LOG_DIR}/build_${TS}.log"

# outputs/ 下始终指向最新一次运行的软链接，方便直接 tail -f，无需查时间戳
# 使用相对路径 log/xxx，保证移动 outputs 目录后软链仍有效
ln -sf "log/case_framework_${TS}.log" "${OUTPUT_DIR}/case_framework.log"
ln -sf "log/build_${TS}.log" "${OUTPUT_DIR}/build.log"

# 并发锁：避免上一次任务未完成时被再次触发
LOCK_FILE="/tmp/xllm_daily_build.lock"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 分段任务：编译（输出同时进总日志与 build 独立日志）
run_build() {
  if [[ ! -f "${PREPARE_SCRIPT}" ]]; then
    log "ERROR: '${PREPARE_SCRIPT}' not found."
    return 1
  fi
  log "[stage:build] running: bash ${PREPARE_SCRIPT}  (log: ${BUILD_LOG})"
  bash "${PREPARE_SCRIPT}" 2>&1 | tee "${BUILD_LOG}"
  return "${PIPESTATUS[0]}"
}

# 分段任务：用例测试（输出直接并入框架日志 case_framework，由主体重定向统一落盘）
run_test() {
  if [[ ! -f "${MAIN_PY}" ]]; then
    log "ERROR: '${MAIN_PY}' not found."
    return 1
  fi
  log "[stage:test] running: ${PYTHON_BIN} ${MAIN_PY} ${MAIN_ARGS}  (log: ${LOG_FILE})"
  ( cd "${PROJECT_ROOT}" && ${PYTHON_BIN} "${MAIN_PY}" ${MAIN_ARGS} ) 2>&1
  return $?
}

{
  log "==== daily job start, stages: ${STAGES[*]} ===="

  # flock 保证同一时刻只有一个任务在跑；获取不到锁则直接跳过本次
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    log "another job is already running, skip this run."
    exit 0
  fi

  OVERALL_RC=0
  for stage in "${STAGES[@]}"; do
    case "${stage}" in
      build) run_build ;;
      test)  run_test ;;
      *)
        log "ERROR: unknown stage '${stage}' (allowed: build test)."
        OVERALL_RC=2
        break
        ;;
    esac
    RC=$?
    if [[ ${RC} -ne 0 ]]; then
      log "==== stage '${stage}' FAILED (exit ${RC}), abort remaining stages ===="
      OVERALL_RC=${RC}
      break
    fi
    log "==== stage '${stage}' SUCCESS ===="
  done

  if [[ ${OVERALL_RC} -eq 0 ]]; then
    log "==== daily job ALL SUCCESS ===="
  else
    log "==== daily job FAILED (exit ${OVERALL_RC}) ===="
  fi
  exit ${OVERALL_RC}
} >>"${LOG_FILE}" 2>&1