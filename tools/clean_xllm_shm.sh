#!/bin/sh
# Remove stale xLLM POSIX shared-memory objects from /dev/shm.
#
# Run INSIDE the instance container (docker exec ... bash /workspace/tools/
# clean_xllm_shm.sh): containers share the host IPC namespace (--ipc=host),
# and the SHM objects are root-owned, so only an in-container root process
# can unlink them.
#
# Safety rules (must match xtask/core/scheduler.py::_cleanup_xllm_shm):
#   * only per-case objects  xllm_<chip_id>_*  are targeted
#   * xllm_expert_*          (other teams' MoE runs)         -> untouched
#   * xllm_shm_generation_lock_v2 (host-global xLLM lock)   -> untouched
#   * objects still held by a live process (fuser exit 0)   -> skipped, so a
#     coexisting independent xllm service on the same host is not crashed;
#     when fuser is unavailable, fall back to deleting (historical behaviour)
#
# This is a standalone script (not an inline `sh -c` string) on purpose:
# exec_in_container wraps commands in `sh -c "..."`, and a `$f` inside that
# double-quoted layer is expanded to the empty string by the HOST shell
# before docker ever sees it — an inline while-loop silently deleted nothing.

count=0
for f in /dev/shm/xllm_*; do
    [ -e "$f" ] || continue
    name=${f##*/}
    case "$name" in
        xllm_expert_*) continue ;;
        xllm_shm_generation_lock_v2) continue ;;
    esac
    if command -v fuser >/dev/null 2>&1 && fuser "$f" >/dev/null 2>&1; then
        echo "skip(held): $name"
        continue
    fi
    if rm -f "$f" 2>/dev/null; then
        count=$((count + 1))
    else
        echo "fail(rm): $name" >&2
    fi
done
echo "removed: $count"
