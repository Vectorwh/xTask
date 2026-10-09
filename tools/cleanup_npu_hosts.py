#!/usr/bin/env python3
"""Query and cleanup NPU occupancy / containers on remote hosts.

Reads SSH credentials from ~/.xtask.env (XTASK_SSH_USER / XTASK_SSH_PASS /
XTASK_SSH_PORT) or config/ssh_hosts.json, same as xTask.

Examples:
  # Query only
  python3 tools/cleanup_npu_hosts.py --query-only 11.87.191.75 11.87.191.121

  # Stop running containers, then kill remaining NPU processes
  python3 tools/cleanup_npu_hosts.py 11.87.191.75 11.87.191.121

  # Dry run (show what would be done)
  python3 tools/cleanup_npu_hosts.py --dry-run 11.87.191.75
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

XTASK_ROOT = Path(__file__).resolve().parents[1]
if str(XTASK_ROOT) not in sys.path:
    sys.path.insert(0, str(XTASK_ROOT))

from xtask.utils.host_executor import create_host_executor, load_ssh_credentials

def npu_smi_cmd(args: str) -> str:
    return (
        f"{{ npu-smi {args} 2>/dev/null || "
        f"/usr/local/bin/npu-smi {args} 2>/dev/null || "
        f"/usr/local/sbin/npu-smi {args} 2>/dev/null; }}"
    )
PROC_ID_RE = re.compile(r"Process\s+id\s*:\s*(\d+)", re.IGNORECASE)
PROC_NAME_RE = re.compile(
    r"Process\s+name\s*:\s*(.*?)\s*(?:Process\s+memory|$)", re.IGNORECASE
)
PROC_MEM_RE = re.compile(r"Process\s+memory\s*\(MB\)\s*:\s*(\d+)", re.IGNORECASE)


@dataclass
class NpuProcess:
    logic_id: int
    npu_id: int
    chip_id: int
    pid: int
    name: str
    memory_mb: str = ""


@dataclass
class HostReport:
    host: str
    devices: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    processes: List[NpuProcess] = field(default_factory=list)
    containers: List[str] = field(default_factory=list)
    hbm_max_mb: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    cleanup_failures: List[str] = field(default_factory=list)


def _run(executor, command: str, timeout: int = 60):
    result = executor.run(command, timeout=timeout, capture_output=True)
    return result


def _parse_npu_mapping(text: str) -> Dict[int, Tuple[int, int]]:
    """logic_id -> (npu_id, chip_id) from `npu-smi info -m`."""
    mapping: Dict[int, Tuple[int, int]] = {}
    for line in (text or "").splitlines():
        cols = line.split()
        if len(cols) < 5:
            continue
        if not cols[0].isdigit() or not cols[1].isdigit():
            continue
        if cols[2] == "-" or not cols[2].isdigit():
            continue
        if len(cols) >= 5 and cols[4] != "Ascend910":
            continue
        mapping[int(cols[2])] = (int(cols[0]), int(cols[1]))
    return dict(sorted(mapping.items()))


def _parse_proc_mem_line(line: str) -> Optional[NpuProcess]:
    if "No process in device" in line or "Process id" not in line:
        return None
    pid_match = PROC_ID_RE.search(line)
    if not pid_match:
        return None
    name_match = PROC_NAME_RE.search(line)
    mem_match = PROC_MEM_RE.search(line)
    return NpuProcess(
        logic_id=-1,
        npu_id=-1,
        chip_id=-1,
        pid=int(pid_match.group(1)),
        name=(name_match.group(1).strip() if name_match else ""),
        memory_mb=(mem_match.group(1) if mem_match else ""),
    )


def discover_devices(executor) -> Dict[int, Tuple[int, int]]:
    result = _run(executor, npu_smi_cmd("info -m"))
    mapping = _parse_npu_mapping(result.stdout)
    if mapping:
        return mapping

    # Fallback: probe card_id 0..15 using watchdog's chip/core mapping.
    fallback: Dict[int, Tuple[int, int]] = {}
    for card_id in range(16):
        chip_id = card_id // 2
        core_id = card_id % 2
        probe = _run(
            executor,
            npu_smi_cmd(f"info -t proc-mem -i {chip_id} -c {core_id}"),
            timeout=20,
        )
        if probe.exit_code != 0 and not probe.stdout.strip():
            continue
        if "not found" in (probe.stderr or "").lower():
            continue
        fallback[card_id] = (chip_id, core_id)
    return fallback


def query_npu_processes(
    executor, devices: Dict[int, Tuple[int, int]]
) -> List[NpuProcess]:
    processes: List[NpuProcess] = []
    for logic_id, (npu_id, chip_id) in devices.items():
        cmd = npu_smi_cmd(f"info -t proc-mem -i {npu_id} -c {chip_id}")
        result = _run(executor, cmd, timeout=20)
        if result.exit_code != 0 and not result.stdout.strip():
            continue
        for line in (result.stdout or "").splitlines():
            proc = _parse_proc_mem_line(line.strip())
            if proc is None:
                continue
            proc.logic_id = logic_id
            proc.npu_id = npu_id
            proc.chip_id = chip_id
            processes.append(proc)
    return processes


def query_hbm_max_mb(executor) -> Optional[int]:
    cmd = (
        f"{npu_smi_cmd('info')} | grep -oE '[0-9]+ */ *[0-9]+' | cut -d'/' -f1"
        " | tr -d ' ' | sort -n | tail -1"
    )
    result = _run(executor, cmd, timeout=20)
    text = (result.stdout or "").strip()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def list_running_containers(executor) -> List[str]:
    result = _run(
        executor,
        "docker ps --format '{{.Names}}' 2>/dev/null || true",
        timeout=30,
    )
    names = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return sorted(names)


def collect_host_report(executor, host: str) -> HostReport:
    report = HostReport(host=host)
    try:
        report.devices = discover_devices(executor)
        if not report.devices:
            report.errors.append("no NPU devices discovered")
        report.processes = query_npu_processes(executor, report.devices)
        report.hbm_max_mb = query_hbm_max_mb(executor)
        report.containers = list_running_containers(executor)
    except Exception as exc:
        report.errors.append(str(exc))
    return report


def _command_error(result) -> str:
    detail = (getattr(result, "stderr", None) or "").strip()
    if not detail:
        detail = (getattr(result, "stdout", None) or "").strip()
    if detail:
        return detail
    exit_code = getattr(result, "exit_code", None)
    if exit_code is not None:
        return f"exit code {exit_code}"
    return "unknown error"


def _signal_npu_pids(executor, pids: List[int], signal_name: str):
    if signal_name not in {"TERM", "KILL"}:
        raise ValueError(f"Unsupported cleanup signal: {signal_name}")
    if not pids or any(type(pid) is not int or pid <= 1 for pid in pids):
        raise ValueError("Cleanup signals require explicit PIDs greater than 1")
    pid_list = " ".join(str(pid) for pid in pids)
    # One remote shell invocation avoids opening an SSH connection per PID.
    # Try the current identity first. sudo checks authorization for the exact
    # kill command; -n must never prompt or consume an SSH password in cron.
    # /proc visibility is not a liveness test (e.g. hidepid can hide root PIDs).
    # Only a C-locale diagnostic from kill itself confirms an absent process;
    # a sudo authorization/execution error must never be mistaken for that.
    command = f"""failed=0
cleanup_pid_missing() {{
    case "$1" in
        "kill: "*"No such process"|/*/"kill: "*"No such process") return 0 ;;
        *) return 1 ;;
    esac
}}
for pid in {pid_list}; do
    if kill_output=$(LC_ALL=C /bin/kill -{signal_name} -- "$pid" 2>&1); then
        if [ -n "$kill_output" ]; then printf '%s\\n' "$kill_output"; fi
        continue
    fi
    if [ -n "$kill_output" ]; then printf '%s\\n' "$kill_output" >&2; fi
    if cleanup_pid_missing "$kill_output"; then
        continue
    fi
    printf '[SUDO] retry SIG{signal_name} for PID %s\\n' "$pid"
    if sudo_output=$(LC_ALL=C sudo -n -u root -- /bin/kill -{signal_name} -- "$pid" 2>&1); then
        if [ -n "$sudo_output" ]; then printf '%s\\n' "$sudo_output"; fi
        continue
    fi
    if [ -n "$sudo_output" ]; then printf '%s\\n' "$sudo_output" >&2; fi
    if cleanup_pid_missing "$sudo_output"; then
        continue
    fi
    printf '[FAILED] unable to signal PID %s with SIG{signal_name}\\n' "$pid" >&2
    failed=1
done
exit "$failed"
"""
    return _run(executor, command, timeout=30)


def kill_npu_processes(
    executor,
    host: str,
    processes: Iterable[NpuProcess],
    dry_run: bool,
) -> Tuple[List[int], List[str]]:
    process_list = list(processes)
    pids = sorted({proc.pid for proc in process_list if proc.pid > 1})
    protected = sorted({proc.pid for proc in process_list if proc.pid <= 1})
    failures = [f"refusing to signal protected PID {pid}" for pid in protected]
    if not pids or dry_run:
        return pids, failures

    def send_signal(targets: List[int], signal_name: str) -> None:
        result = _signal_npu_pids(executor, targets, signal_name)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            print(f"[{host}] {stdout}")
        if result.success:
            # A normal kill may fail before an authorized sudo retry succeeds.
            if stderr:
                print(
                    f"[WARN] {host}: SIG{signal_name} diagnostics: {stderr}",
                    file=sys.stderr,
                )
        else:
            message = f"signal NPU processes ({signal_name}) failed: {_command_error(result)}"
            failures.append(message)
            print(f"[FAILED] {host}: {message}", file=sys.stderr)

    send_signal(pids, "TERM")
    time.sleep(3)

    remaining = query_npu_processes(executor, discover_devices(executor))
    remaining_pids = sorted({proc.pid for proc in remaining if proc.pid > 1})
    if remaining_pids:
        send_signal(remaining_pids, "KILL")
        # Allow NPU runtime teardown before declaring the process still alive.
        time.sleep(3)
        remaining = query_npu_processes(executor, discover_devices(executor))
        remaining_pids = sorted({proc.pid for proc in remaining})
        if remaining_pids:
            failures.append(
                f"NPU processes still alive after kill: {remaining_pids}"
            )
            print(
                f"[FAILED] {host}: NPU processes still alive: {remaining_pids}",
                file=sys.stderr,
            )

    return pids, failures


def stop_running_containers(
    executor,
    host: str,
    names: Iterable[str],
    dry_run: bool,
) -> Tuple[List[str], List[str]]:
    stopped: List[str] = []
    failures: List[str] = []
    for name in names:
        if dry_run:
            stopped.append(name)
            continue
        result = _run(executor, f"docker stop --time=20 {name} 2>&1", timeout=60)
        if result.success:
            stopped.append(name)
            continue

        stop_reason = _command_error(result)
        kill_result = _run(executor, f"docker kill {name} 2>&1", timeout=30)
        if kill_result.success:
            stopped.append(name)
            print(
                f"[WARN] {host}: docker stop failed for {name}, "
                f"docker kill succeeded ({stop_reason})",
                file=sys.stderr,
            )
            continue

        kill_reason = _command_error(kill_result)
        reason = f"docker stop failed ({stop_reason}); docker kill failed ({kill_reason})"
        failures.append(f"stop container {name}: {reason}")
        print(f"[FAILED] {host}: {reason}", file=sys.stderr)
    return stopped, failures


def print_report(report: HostReport, title: str) -> None:
    print(f"\n=== {title}: {report.host} ===")
    if report.errors:
        for err in report.errors:
            print(f"  [ERROR] {err}")
    if report.cleanup_failures:
        for err in report.cleanup_failures:
            print(f"  [CLEANUP-FAILED] {err}")

    print(f"  NPU devices: {len(report.devices)}")
    if report.hbm_max_mb is not None:
        print(f"  HBM max used: {report.hbm_max_mb} MB")

    if report.processes:
        print("  NPU processes:")
        for proc in report.processes:
            mem = f", {proc.memory_mb}MB" if proc.memory_mb else ""
            print(
                f"    - logic={proc.logic_id} npu={proc.npu_id} chip={proc.chip_id} "
                f"pid={proc.pid} name={proc.name}{mem}"
            )
    else:
        print("  NPU processes: none")

    if report.containers:
        print("  running containers:")
        for name in report.containers:
            print(f"    - {name}")
    else:
        print("  running containers: none")


def process_host(
    host: str,
    *,
    query_only: bool,
    dry_run: bool,
    kill_processes: bool,
    stop_containers: bool,
) -> HostReport:
    load_ssh_credentials()
    executor = create_host_executor(host)

    before = collect_host_report(executor, host)
    print_report(before, "BEFORE")

    if query_only:
        return before

    action = "DRY-RUN" if dry_run else "ACTION"
    containers = before.containers
    if stop_containers and containers:
        try:
            stopped, stop_failures = stop_running_containers(
                executor, host, containers, dry_run=dry_run
            )
            print(f"[{action}] {host}: stop containers -> {stopped}")
            if stop_failures:
                print(
                    f"[{action}] {host}: stop containers failed -> {stop_failures}",
                    file=sys.stderr,
                )
            before.cleanup_failures.extend(stop_failures)
        except Exception as exc:
            msg = f"stop containers raised: {exc}"
            before.cleanup_failures.append(msg)
            print(f"[FAILED] {host}: {msg}", file=sys.stderr)

    if kill_processes:
        try:
            # Container shutdown can remove most PIDs. Never signal the stale
            # pre-stop list; re-query every device immediately before cleanup.
            # Dry-run does not actually stop containers, so retain its snapshot.
            remaining = (
                before.processes
                if dry_run
                else query_npu_processes(executor, discover_devices(executor))
            )
            if remaining:
                targeted, kill_failures = kill_npu_processes(
                    executor, host, remaining, dry_run=dry_run
                )
                print(f"\n[{action}] {host}: signal NPU processes -> {targeted}")
                before.cleanup_failures.extend(kill_failures)
        except Exception as exc:
            msg = f"kill NPU processes raised: {exc}"
            before.cleanup_failures.append(msg)
            print(f"[FAILED] {host}: {msg}", file=sys.stderr)

    after = collect_host_report(executor, host)
    after.cleanup_failures = list(before.cleanup_failures)
    print_report(after, "AFTER")
    if after.cleanup_failures:
        print(
            f"[SUMMARY] {host}: cleanup completed with "
            f"{len(after.cleanup_failures)} failure(s); continuing.",
            file=sys.stderr,
        )
    return after


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stop running containers on remote hosts, then clear remaining NPU "
            "processes; retry unauthorized signals with non-interactive sudo."
        )
    )
    parser.add_argument(
        "hosts",
        nargs="+",
        help="Target host IPs, e.g. 11.87.191.75 11.87.191.121",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Only query and print NPU/container status; do not kill or stop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without killing processes or stopping containers.",
    )
    parser.add_argument(
        "--no-kill-processes",
        action="store_true",
        help="Skip killing NPU-occupying processes.",
    )
    parser.add_argument(
        "--no-stop-containers",
        action="store_true",
        help="Skip stopping all running containers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    creds = load_ssh_credentials()
    if not creds and not args.query_only:
        print(
            "WARNING: no SSH credentials found in ~/.xtask.env or config/ssh_hosts.json",
            file=sys.stderr,
        )

    exit_code = 0
    cleanup_failures = 0
    for host in args.hosts:
        try:
            report = process_host(
                host,
                query_only=args.query_only,
                dry_run=args.dry_run,
                kill_processes=not args.no_kill_processes,
                stop_containers=not args.no_stop_containers,
            )
            if report.errors:
                exit_code = 1
            if report.cleanup_failures:
                cleanup_failures += len(report.cleanup_failures)
                if not args.query_only:
                    print(
                        f"[WARN] {host}: recorded {len(report.cleanup_failures)} "
                        "cleanup failure(s); not aborting remaining hosts.",
                        file=sys.stderr,
                    )
        except Exception as exc:
            print(f"\n[FAILED] {host}: {exc}", file=sys.stderr)
            if args.query_only:
                exit_code = 1
            else:
                print(
                    f"[WARN] {host}: cleanup error ignored; "
                    "continuing with remaining hosts.",
                    file=sys.stderr,
                )

    if cleanup_failures and not args.query_only:
        print(
            f"[SUMMARY] cleanup finished with {cleanup_failures} failure(s); "
            "exit 0 so downstream jobs can continue.",
            file=sys.stderr,
        )
        return 0
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
