#!/usr/bin/env python3
"""Own one detached Service/etcd process inside its Engine container.

PID namespace, Linux start time and executable fence PID reuse. No process-name
kill or host-wide cleanup is allowed. State is private and locked per instance.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_time": stat[19],
            "executable": os.readlink(f"/proc/{pid}/exe"),
            "pid_namespace": os.readlink("/proc/self/ns/pid"),
        }
    except (OSError, IndexError):
        return None


def load_record(path):
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("pid"), int)
        or record["pid"] <= 1
    ):
        raise ValueError("Invalid process ownership record")
    return record


def alive(record):
    return record is not None and identity(record["pid"]) == record


def start(path, log, argv):
    record = load_record(path)
    if alive(record):
        raise RuntimeError("Owned process is already running; stop it before restarting")
    if not argv:
        raise ValueError("Missing executable")
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        # Preserve failed-attempt evidence while the active health check reads
        # a fresh log; old fatal lines must not poison a successful retry.
        log.replace(log.with_name(log.stem + "." + str(time.time_ns()) + log.suffix))
    with log.open("wb") as stream:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    record = identity(process.pid)
    if record is None:
        raise RuntimeError("Process exited before its ownership record was established")
    temporary = path.with_suffix(".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream)
        temporary.replace(path)
    except Exception:
        if identity(process.pid) == record:
            process.terminate()
            process.wait(timeout=5)
        raise


def stop(path, grace):
    record = load_record(path)
    descriptor = None
    if alive(record):
        # pidfd targets the original task even if it exits and its numeric PID
        # is reused between validation and signal delivery.
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Safe process cleanup requires Linux pidfd support")
        try:
            descriptor = os.pidfd_open(record["pid"])
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            return
        if not alive(record):
            os.close(descriptor)
            path.unlink(missing_ok=True)
            return
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            deadline = time.monotonic() + grace
            while alive(record) and time.monotonic() < deadline:
                time.sleep(0.1)
            if alive(record):
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while alive(record) and time.monotonic() < deadline:
                    time.sleep(0.1)
            if alive(record):
                raise RuntimeError("Owned process did not exit after SIGKILL")
        except ProcessLookupError:
            pass  # The fenced child exited between the liveness check and signal.
        finally:
            os.close(descriptor)
    path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status", "check"))
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--grace", type=float, default=15)
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    argv = arguments[separator + 1 :]
    args = parser.parse_args(arguments[:separator])
    path = Path(args.pid_file)
    if args.action == "check":
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise RuntimeError("Safe process cleanup requires Linux pidfd support")
        descriptor = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(descriptor, 0)
        finally:
            os.close(descriptor)
        return 0
    if not math.isfinite(args.grace) or args.grace < 0 or args.grace > 300:
        raise ValueError("Invalid finite shutdown grace")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "start":
            if not args.log_file:
                raise ValueError("start requires --log-file")
            start(path, Path(args.log_file), argv)
        elif args.action == "stop":
            stop(path, args.grace)
        elif not alive(load_record(path)):
            return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Managed process error: {error}", file=sys.stderr)
        sys.exit(1)
