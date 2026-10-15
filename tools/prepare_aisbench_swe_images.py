#!/usr/bin/env python3
"""Preload a SWE-bench case's instance images on its x86 runner in one command.

Generate the filtered image list on the control host, upload it together with
SWE-only prefetch helpers, and execute prefill -> prefetch on the runner. This
never starts xLLM or changes Docker daemon configuration.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

import yaml

from tools.generate_aisbench_swe_images import image_list_for_case, write_image_list
from xtask.benchmarks.aisbench.swebench import (
    is_aisbench_swebench_task,
    swebench_runner,
)
from xtask.utils.host_executor import create_host_executor, is_local_host


_HELPERS = (
    "prefill_aisbench_swe_layers.py",
    "resume_aisbench_swe_blob.py",
    "prefetch_aisbench_swe_images.py",
)


def _require_run(executor, command: str, label: str, timeout: int = 30) -> str:
    result = executor.run(command, timeout=timeout)
    if not result.success:
        raise RuntimeError(
            f"x86 {label} failed (exit {result.exit_code}): "
            f"{(result.stderr or result.stdout or '')[-400:]}"
        )
    return result.stdout.strip()


def _runner_home(executor) -> Path:
    home = _require_run(executor, 'printf %s "$HOME"', "HOME lookup", timeout=15)
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", home) or ".." in Path(home).parts:
        raise ValueError(f"Invalid x86 runner HOME: {home!r}")
    return Path(home)


def _proxy_exports(task: Mapping) -> str:
    runtime = swebench_runner(task).get("runtime_container") or {}
    proxy = runtime.get("proxy") or {} if isinstance(runtime, Mapping) else {}
    if not isinstance(proxy, Mapping):
        raise ValueError("benchmark.swebench.runner.runtime_container.proxy must be a mapping")
    exports = []
    for key in ("http", "https"):
        value = proxy.get(key)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise ValueError(f"SWE-bench {key} proxy must be a URL")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"SWE-bench {key} proxy must be an HTTP(S) URL")
        exports.append(f"export {key.upper()}_PROXY={shlex.quote(value)};")
    no_proxy = str(proxy.get("no_proxy") or "").strip()
    hosts = [host.strip() for host in no_proxy.split(",") if host.strip()]
    if "docker.1ms.run" not in hosts:
        hosts.append("docker.1ms.run")
    exports.append(f"export NO_PROXY={shlex.quote(','.join(hosts))};")
    return " ".join(exports)


def prepare_images(
    task: Mapping,
    images: list[str],
    executor,
    *,
    crane: str | None = None,
    min_free_gb: float = 100,
    timeout: int = 360,
    retries: int = 3,
    total_timeout: int = 86400,
) -> tuple[str, str]:
    """Run on a selected x86 host; only the supplied executor does remote I/O."""
    if not is_aisbench_swebench_task(task):
        raise ValueError("YAML must be an explicit AISBench SWE-bench case")
    if not images or min_free_gb < 0 or timeout <= 0 or retries <= 0 or total_timeout <= 0:
        raise ValueError("Image list, timeouts, retries and disk minimum must be valid")
    case = str(task.get("case_name") or "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", case):
        raise ValueError("SWE-bench case_name must contain only letters, digits, _, . or -")

    home = _runner_home(executor)
    root = home / "xtask_swe" / "swe_image_prefetch"
    run_dir = root / "runs" / f"{case}-{time.time_ns()}"
    docker_root = _require_run(
        executor, "docker info --format '{{.DockerRootDir}}'", "Docker root lookup"
    )
    if not docker_root.startswith("/") or "\n" in docker_root:
        raise ValueError(f"Invalid x86 Docker root: {docker_root!r}")
    crane_path = Path(crane.replace("${HOME}", str(home)).replace("$HOME", str(home))) if crane else (
        home / "xtask_swe" / "tools" / "crane-v0.22.1"
    )
    if not crane_path.is_absolute():
        raise ValueError("--crane must be an absolute path on the x86 runner")
    _require_run(executor, f"test -x {shlex.quote(str(crane_path))}", "crane executable check")
    _require_run(executor, "command -v flock >/dev/null", "flock check")
    _require_run(
        executor,
        f"mkdir -p {shlex.quote(str(run_dir))} && "
        f"chmod 700 {shlex.quote(str(root))} {shlex.quote(str(run_dir.parent))} "
        f"{shlex.quote(str(run_dir))}",
        "user workspace preparation",
    )

    with tempfile.TemporaryDirectory(prefix="xtask-swe-images-") as tmp:
        local_list = Path(tmp) / "images.txt"
        write_image_list(local_list, images)
        for local, name in ((local_list, "images.txt"), *(
            (Path(__file__).parent / helper, helper) for helper in _HELPERS
        )):
            destination = run_dir / name
            if not executor.upload(str(local), str(destination)):
                raise RuntimeError(f"Failed to upload SWE-bench {name} to x86 runner")

    quoted = lambda path: shlex.quote(str(path))
    prefill = (
        f"python3 {quoted(run_dir / _HELPERS[0])} "
        f"--images-file {quoted(run_dir / 'images.txt')} "
        f"--cache {quoted(root / 'cache')} "
        f"--blob-helper {quoted(run_dir / _HELPERS[1])}"
    )
    prefetch = (
        f"python3 {quoted(run_dir / _HELPERS[2])} "
        f"--images-file {quoted(run_dir / 'images.txt')} "
        f"--crane {quoted(crane_path)} --root {quoted(root)} "
        f"--docker-root {quoted(docker_root)} --min-free-gb {min_free_gb} "
        f"--timeout {timeout} --retries {retries}"
    )
    script = (
        "set -e; " + _proxy_exports(task) + " "
        + f"chmod 600 {quoted(run_dir / 'images.txt')} "
        + " ".join(quoted(run_dir / name) for name in _HELPERS)
        + "; " + prefill + "; " + prefetch
    )
    result = executor.run(
        f"flock -n {quoted(root / '.prepare.lock')} sh -c {shlex.quote(script)}",
        timeout=total_timeout,
    )
    if not result.success:
        raise RuntimeError(
            f"SWE-bench image preparation failed on x86 (exit {result.exit_code}); "
            f"runner workdir: {run_dir}; {(result.stderr or result.stdout or '')[-600:]}"
        )
    return str(run_dir / "images.txt"), result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-yaml", required=True, type=Path)
    parser.add_argument("--dataset-path", type=Path,
                        help="Override the local parquet snapshot on the control host")
    parser.add_argument("--crane", help="Verified crane binary path on x86 (defaults to ~/xtask_swe/tools/crane-v0.22.1)")
    parser.add_argument("--min-free-gb", type=float, default=100)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--total-timeout", type=int, default=86400)
    args = parser.parse_args(argv)
    task = yaml.safe_load(args.case_yaml.read_text(encoding="utf-8"))
    images = image_list_for_case(args.case_yaml, args.dataset_path)
    runner = swebench_runner(task)
    host = str(runner.get("host") or "").strip()
    if not host or is_local_host(host):
        raise ValueError("SWE-bench image preparation requires a dedicated remote x86 runner host")
    user = runner.get("user")
    port = int(runner.get("port") or 22)
    executor = create_host_executor(host, user=user, port=port)
    print(f"Preparing {len(images)} SWE-bench images on {host} without starting xLLM", flush=True)
    images_path, details = prepare_images(
        task, images, executor, crane=args.crane, min_free_gb=args.min_free_gb,
        timeout=args.timeout, retries=args.retries, total_timeout=args.total_timeout,
    )
    print(details[-3000:])
    print(f"SWE-bench images ready on {host}; list: {images_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"SWE-bench image preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
