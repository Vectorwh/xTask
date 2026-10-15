"""Optional, explicitly configured Docker driver for x86 SWE-bench runs.

The driver runs AISBench/mini-swe-agent/harness Python. It is NOT one of the
per-instance Docker containers created by mini-swe-agent or the harness.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from xtask.benchmarks.aisbench.swebench import (
    swebench_agent_container_timeout,
    swebench_config,
    swebench_runner,
    swebench_setup,
)
from xtask.utils.host_executor import HostExecutor

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def runtime_container_config(task_config: Mapping[str, Any]) -> Optional[dict]:
    raw = swebench_setup(task_config).get("runtime_container")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark.swebench.runner.runtime_container must be a mapping")
    return dict(raw)


@dataclass
class SweRuntimeContainerState:
    """Only containers created by this xTask run are eligible for cleanup."""

    container_id: str = ""
    name: str = ""
    wrappers: Tuple[str, str] = ("", "")
    remove_on_exit: bool = False


def cleanup_swebench_runtime_container(
    executor: HostExecutor, state: SweRuntimeContainerState
) -> bool:
    """Remove this run's driver by ID, never by an existing container's name."""
    if not state.remove_on_exit or not state.container_id:
        return True
    try:
        result = executor.run(f"docker rm -f {shlex.quote(state.container_id)}", timeout=120)
    except Exception as exc:
        logging.warning("Failed to remove SWE-bench x86 driver %s: %s", state.name, exc)
        return False
    if not result.success and not any(
        marker in (result.stderr or "") for marker in ("No such container", "No such object")
    ):
        logging.warning("Failed to remove SWE-bench x86 driver %s: %s", state.name, result.stderr or result.stdout)
        return False
    state.remove_on_exit = False
    python_wrapper, bench_wrapper = state.wrappers
    if python_wrapper and bench_wrapper:
        try:
            removed = executor.run(
                f"rm -f {shlex.quote(python_wrapper)} {shlex.quote(bench_wrapper)}",
                timeout=30,
            )
        except Exception as exc:
            logging.warning("Failed to remove SWE-bench driver wrappers for %s: %s", state.name, exc)
        else:
            if not removed.success:
                logging.warning("Failed to remove SWE-bench driver wrappers for %s: %s", state.name, removed.stderr)
    logging.info("SWE-bench: removed per-case x86 driver %s", state.name)
    return True


def _runtime_proxy_args(raw: Mapping[str, Any]) -> str:
    """Proxy only AISBench's runtime process, not x86 dependency installation."""
    proxy = raw.get("proxy") or {}
    if not isinstance(proxy, Mapping):
        raise ValueError("benchmark.swebench.runner.runtime_container.proxy must be a mapping")
    env = []
    for field, names in (
        ("http", ("HTTP_PROXY", "http_proxy")),
        ("https", ("HTTPS_PROXY", "https_proxy")),
    ):
        value = str(proxy.get(field) or "").strip()
        if not value:
            continue
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"Invalid SWE-bench runtime_container.proxy.{field}: {value!r}")
        env.extend(f"-e {shlex.quote(f'{name}={value}')}" for name in names)
    no_proxy = str(proxy.get("no_proxy") or "").strip()
    if no_proxy:
        if any(char in no_proxy for char in "\n\r"):
            raise ValueError("SWE-bench runtime_container.proxy.no_proxy contains a newline")
        env.extend(f"-e {shlex.quote(f'{name}={no_proxy}')}" for name in ("NO_PROXY", "no_proxy"))
    return " ".join(env)


def _required(raw: Mapping[str, Any], field: str, *, absolute: bool = False) -> str:
    value = str(raw.get(field) or "").strip()
    if not value or any(char.isspace() for char in value) or value.startswith("-"):
        raise ValueError(f"benchmark.swebench.runtime_container.{field} is required and must be a single token")
    if absolute and not os.path.isabs(value):
        raise ValueError(f"benchmark.swebench.runtime_container.{field} must be an absolute path")
    return value


def _in_mount(path: str, mount_root: Path, field: str) -> None:
    candidate = Path(os.path.normpath(path))
    if not candidate.is_absolute() or not candidate.is_relative_to(mount_root):
        raise ValueError(
            f"SWE-bench {field} must be under {mount_root} when runtime_container is enabled: {path}"
        )


def _run_or_raise(executor: HostExecutor, command: str, label: str, timeout: int = 60) -> str:
    result = executor.run(command, timeout=timeout)
    if not result.success:
        detail = (result.stderr or result.stdout or f"exit {result.exit_code}").strip()
        raise RuntimeError(f"SWE-bench x86 {label} failed: {detail}")
    return (result.stdout or "").strip()


def _parse_dns(text: str) -> list[str]:
    servers = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] != "nameserver":
            continue
        try:
            server = str(ipaddress.ip_address(parts[1]))
        except ValueError:
            continue
        if server not in servers:
            servers.append(server)
    return servers


def _dns_servers(raw: Mapping[str, Any], executor: HostExecutor) -> list[str]:
    configured = raw.get("dns")
    if configured is None:
        text = _run_or_raise(executor, "cat /etc/resolv.conf", "host DNS lookup", timeout=15)
        servers = _parse_dns(text)
    else:
        if not isinstance(configured, (list, tuple)):
            raise ValueError("benchmark.swebench.runner.runtime_container.dns must be a list of IP addresses")
        servers = []
        for value in configured:
            try:
                server = str(ipaddress.ip_address(value))
            except ValueError as exc:
                raise ValueError(f"Invalid SWE-bench runtime_container DNS server: {value!r}") from exc
            if server not in servers:
                servers.append(server)
    if not servers:
        raise RuntimeError("SWE-bench x86 runner has no usable DNS servers; set runtime_container.dns")
    return servers


def ensure_swebench_runtime_container(
    task_config: Mapping[str, Any],
    executor: HostExecutor,
    remote_home: str,
    work_dir: str,
    install_dir: str,
    state: Optional[SweRuntimeContainerState] = None,
) -> Optional[Tuple[str, str]]:
    """Prepare the opted-in driver, leaving legacy setup containers reusable.

    New runner.runtime_container creates a uniquely named per-case driver. Its
    caller removes it in a finally block, including on benchmark failure.
    """
    raw = runtime_container_config(task_config)
    if raw is None:
        return None
    per_case = "runtime_container" in swebench_runner(task_config)
    case_name = str(task_config.get("case_name") or "swebench")
    if per_case and state is None:
        raise ValueError("Per-case SWE-bench runtime container requires lifecycle state")
    if per_case and not _NAME.fullmatch(case_name):
        raise ValueError(f"Invalid SWE-bench case_name for runtime container: {case_name!r}")
    name = str(raw.get("name") or (
        f"{case_name}-swebench_rt-{int(time.time())}" if per_case else "xtask-swe-python"
    )).strip()
    if not _NAME.fullmatch(name):
        raise ValueError("benchmark.swebench.runner.runtime_container.name is invalid")
    image = _required(raw, "image")
    python_bin = _required(raw, "python", absolute=True)
    ais_bench_bin = _required(raw, "ais_bench_bin", absolute=True)
    proxy_args = _runtime_proxy_args(raw)
    mount_root = Path(os.path.normpath(os.path.join(remote_home, "xtask_swe")))
    _in_mount(work_dir, mount_root, "runner.work_dir")
    _in_mount(install_dir, mount_root, "setup.install_dir")
    dataset_path = str(swebench_config(task_config).get("dataset_path") or "").strip()
    if dataset_path:
        _in_mount(dataset_path, mount_root, "dataset_path")

    cli = _run_or_raise(executor, "command -v docker", "Docker CLI lookup", timeout=15)
    docker_bin = cli.splitlines()[-1].strip()
    if not docker_bin.startswith("/"):
        raise RuntimeError(f"SWE-bench x86 Docker CLI path is not absolute: {docker_bin!r}")
    _run_or_raise(executor, "docker ps >/dev/null && test -S /var/run/docker.sock", "Docker daemon check", timeout=30)
    dns_servers = _dns_servers(raw, executor)
    _run_or_raise(executor, f"mkdir -p {shlex.quote(str(mount_root))} {shlex.quote(install_dir)}", "runtime directory setup")

    quoted_name = shlex.quote(name)
    inspect = executor.run(
        f"docker inspect --type container -f '{{{{.State.Running}}}} {{{{.Config.Image}}}}' {quoted_name}",
        timeout=15,
    )
    if inspect.success:
        if per_case:
            raise RuntimeError(f"SWE-bench per-case driver {name} already exists; refusing to reuse it")
        parts = inspect.stdout.strip().split(maxsplit=1)
        if len(parts) != 2 or parts[1] != image:
            raise RuntimeError(
                f"SWE-bench x86 container {name} already exists with an unexpected image: {inspect.stdout.strip()}"
            )
        if parts[0] == "false":
            _run_or_raise(executor, f"docker start {quoted_name}", "driver start")
        elif parts[0] != "true":
            raise RuntimeError(f"SWE-bench x86 container {name} has unknown state: {parts[0]}")
    elif "No such object" in inspect.stderr or "No such container" in inspect.stderr:
        _run_or_raise(executor, f"docker image inspect {shlex.quote(image)} >/dev/null", "driver image check")
        command = (
            f"docker run -d --name {quoted_name} --label xtask.swebench.driver=true "
            f"--network host -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 "
            f"--mount {shlex.quote('type=bind,src=' + str(mount_root) + ',dst=' + str(mount_root))} "
            "--mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock "
            f"--mount {shlex.quote('type=bind,src=' + docker_bin + ',dst=/usr/local/bin/docker,readonly')} "
            + " ".join(f"--dns {shlex.quote(server)}" for server in dns_servers)
            + f" {shlex.quote(image)} sleep infinity"
        )
        created_id = _run_or_raise(executor, command, "driver creation", timeout=120).splitlines()[-1]
        if per_case:
            if not re.fullmatch(r"[0-9a-f]{12,64}", created_id):
                raise RuntimeError(f"SWE-bench driver {name} returned an invalid container ID")
            if state is not None:
                state.container_id = created_id
                state.name = name
                state.remove_on_exit = True
        logging.info("SWE-bench: created x86 driver %s from %s", name, image)
    else:
        raise RuntimeError(f"SWE-bench x86 container inspect failed: {inspect.stderr or inspect.stdout}")

    mounts = _run_or_raise(
        executor,
        f"docker inspect --type container -f '{{{{range .Mounts}}}}{{{{.Source}}}}:{{{{.Destination}}}};{{{{end}}}}' {quoted_name}",
        "driver mount inspection",
    )
    for required in (f"{mount_root}:{mount_root};", "/var/run/docker.sock:/var/run/docker.sock;"):
        if required not in mounts:
            raise RuntimeError(f"SWE-bench x86 container {name} is missing required mount: {required}")
    _run_or_raise(executor, f"docker exec -i {quoted_name} {shlex.quote(python_bin)} --version", "driver Python probe")
    _run_or_raise(executor, f"docker exec -i {quoted_name} /usr/local/bin/docker ps >/dev/null", "driver Docker probe")
    current_dns = _run_or_raise(executor, f"docker exec -i {quoted_name} cat /etc/resolv.conf", "driver DNS check")
    if _parse_dns(current_dns) != dns_servers:
        label = _run_or_raise(
            executor,
            f"docker inspect --type container -f '{{{{index .Config.Labels \"xtask.swebench.driver\"}}}}' {quoted_name}",
            "driver ownership check",
        )
        if label != "true":
            raise RuntimeError(f"SWE-bench x86 container {name} has different DNS; refusing to modify an unmanaged container")
        resolv = "".join(f"nameserver {server}\n" for server in dns_servers)
        fix = f"printf %s {shlex.quote(resolv)} > /etc/resolv.conf"
        _run_or_raise(executor, f"docker exec -i {quoted_name} sh -c {shlex.quote(fix)}", "driver DNS setup")
        logging.info("SWE-bench: updated x86 driver %s DNS from host", name)

    python_wrapper = os.path.join(install_dir, f"xtask_{name}_python")
    bench_wrapper = os.path.join(install_dir, f"xtask_{name}_ais_bench")
    proxy_flags = f" {proxy_args}" if proxy_args else ""
    bench_script = (
        f'#!/bin/sh\nexec docker exec -i -w "$PWD" '
        f'-e TORCH_DEVICE_BACKEND_AUTOLOAD=0{proxy_flags} {quoted_name} '
        f'{shlex.quote(ais_bench_bin)} "$@"\n'
    )
    runner_config = swebench_runner(task_config)
    copy_fallback = runner_config.get("docker_copy_fallback") is True
    run_retry = runner_config.get("docker_run_retry") is True
    clean_infer = runner_config.get("clean_infer_container") is True
    github_raw_retry = runner_config.get("github_raw_retry") is True
    agent_container_timeout = swebench_agent_container_timeout(task_config)
    if copy_fallback or run_retry or clean_infer or github_raw_retry or agent_container_timeout:
        # Preserve any PYTHONPATH baked into the runner image. Only this
        # benchmark command and its children see the per-run sitecustomize.
        shim_flags = ""
        if copy_fallback:
            shim_flags += " -e XTASK_SWEBENCH_DOCKER_COPY_FALLBACK=1"
        if run_retry:
            shim_flags += " -e XTASK_SWEBENCH_DOCKER_RUN_RETRY=1"
        if clean_infer:
            shim_flags += " -e XTASK_SWEBENCH_CLEAN_INFER_CONTAINER=1"
        if github_raw_retry:
            shim_flags += " -e XTASK_SWEBENCH_GITHUB_RAW_RETRY=1"
        if agent_container_timeout:
            shim_flags += (
                " -e XTASK_SWEBENCH_AGENT_CONTAINER_TIMEOUT="
                + shlex.quote(agent_container_timeout)
            )
        bench_script = (
            '#!/bin/sh\n: "${XTASK_SWE_PYTHONPATH:?SWE-bench shim path is missing}"\n'
            f'exec docker exec -i -w "$PWD" -e TORCH_DEVICE_BACKEND_AUTOLOAD=0'
            f'{shim_flags} -e PYTHONDONTWRITEBYTECODE=1{proxy_flags} '
            f'-e "XTASK_SWE_PYTHONPATH=$XTASK_SWE_PYTHONPATH" {quoted_name} '
            "sh -c 'export PYTHONPATH=\"$XTASK_SWE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}\"; "
            "exec \"$@\"' "
            f'sh {shlex.quote(ais_bench_bin)} "$@"\n'
        )
    scripts = (
        (python_wrapper, f'#!/bin/sh\nexec docker exec -i -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 {quoted_name} {shlex.quote(python_bin)} "$@"\n'),
        (bench_wrapper, bench_script),
    )
    if state is not None and state.remove_on_exit:
        state.wrappers = python_wrapper, bench_wrapper
    for path, content in scripts:
        _run_or_raise(
            executor,
            f"printf %s {shlex.quote(content)} > {shlex.quote(path)} && chmod 755 {shlex.quote(path)}",
            "driver wrapper preparation",
        )
    return python_wrapper, bench_wrapper
