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
# 一次命令依次执行多组用例：通过 MAIN_ARGS_LIST 指定，用分号 ; 分隔各组参数，
# 每组会单独执行一次 python main.py，按顺序先后执行（某组失败不影响后续组，最终返回非0）：
#   MAIN_ARGS_LIST="--case_name case1 ; --case_name case2" bash tools/cron_daily_build.sh test
#   MAIN_ARGS_LIST="--case_group g1 --host_addr 11.87.191.75 ; --case_group g2 --host_addr 11.87.191.76" \
#     bash tools/cron_daily_build.sh test
#   注：同时设置 MAIN_ARGS_LIST 时优先使用它；否则回退到 MAIN_ARGS 的单组执行。
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
# 多组用例参数列表：用分号 ; 分隔，每组单独执行一次 main.py（设置后优先于 MAIN_ARGS）
MAIN_ARGS_LIST="${MAIN_ARGS_LIST:-}"
# 多组之间的等待秒数（仅 MAIN_ARGS_LIST 多组执行时生效，最后一组后不等待），默认 30s
GROUP_INTERVAL="${GROUP_INTERVAL:-30}"

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
# - 若设置了 MAIN_ARGS_LIST：按分号 ; 拆成多组，按顺序各执行一次 main.py；
#   某组失败仅记录并继续执行后续组，全部结束后若有失败返回最后一个非0码。
# - 否则回退为 MAIN_ARGS 单组执行（向后兼容）。
run_test() {
  if [[ ! -f "${MAIN_PY}" ]]; then
    log "ERROR: '${MAIN_PY}' not found."
    return 1
  fi

  # 单组执行辅助函数
  run_one_case() {
    local args="$1"
    log "[stage:test] running: ${PYTHON_BIN} ${MAIN_PY} ${args}  (log: ${LOG_FILE})"
    ( cd "${PROJECT_ROOT}" && ${PYTHON_BIN} "${MAIN_PY}" ${args} ) 2>&1
    return $?
  }

  # 无多组列表：按原 MAIN_ARGS 单次执行
  if [[ -z "${MAIN_ARGS_LIST}" ]]; then
    run_one_case "${MAIN_ARGS}"
    return $?
  fi

  # 多组列表：按分号拆分，依次执行
  local test_rc=0
  local idx=0
  local old_ifs="${IFS}"
  IFS=';'
  # 关闭通配符展开，避免参数里的 * 被误扩展
  set -f
  # shellcheck disable=SC2086
  for group in ${MAIN_ARGS_LIST}; do
    # 去掉每组首尾空白（纯 bash 实现，避免依赖外部 sed）
    group="${group#"${group%%[![:space:]]*}"}"
    group="${group%"${group##*[![:space:]]}"}"
    [[ -z "${group}" ]] && continue
    idx=$((idx + 1))
    # 从第 2 组起，先等待 GROUP_INTERVAL 秒再执行；最后一组之后不会有等待
    if [[ ${idx} -gt 1 && ${GROUP_INTERVAL} -gt 0 ]]; then
      log "==== [stage:test] waiting ${GROUP_INTERVAL}s before group #${idx} ===="
      sleep "${GROUP_INTERVAL}"
    fi
    log "==== [stage:test] group #${idx} start: ${group} ===="
    IFS="${old_ifs}"
    set +f
    run_one_case "${group}"
    local rc=$?
    IFS=';'
    set -f
    if [[ ${rc} -ne 0 ]]; then
      log "==== [stage:test] group #${idx} FAILED (exit ${rc}): ${group} ===="
      test_rc=${rc}
    else
      log "==== [stage:test] group #${idx} SUCCESS: ${group} ===="
    fi
  done
  IFS="${old_ifs}"
  set +f

  return ${test_rc}
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