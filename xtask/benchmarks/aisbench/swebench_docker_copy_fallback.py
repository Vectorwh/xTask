"""Narrow SWE-bench Docker archive fallback for a broken x86 Docker daemon.

This module is copied as ``sitecustomize.py`` into one opted-in SWE-bench run.
It is not installed globally or enabled for ordinary AISBench/VBench tasks.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit


def install() -> None:
    """Retry only Docker's bind-mount ENOSPC put_archive via docker exec tar."""
    from docker.errors import APIError
    from docker.models.containers import Container

    original = Container.put_archive
    if getattr(original, "_xtask_swe_copy_fallback", False):
        return

    @functools.wraps(original)
    def put_archive(self, path, data):
        try:
            return original(self, path, data)
        except APIError as exc:
            error = str(exc).lower()
            if "no space left on device" not in error or "mount " not in error:
                raise
            if not isinstance(data, (bytes, bytearray)) or not os.path.isabs(str(path)):
                raise
            print(
                f"xTask SWE-bench: Docker archive mount failed; retrying "
                f"container {self.id} via docker exec tar",
                file=sys.stderr,
            )
            try:
                result = subprocess.run(
                    ["docker", "exec", "-i", self.id, "tar", "-xf", "-", "-C", str(path)],
                    input=bytes(data),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as fallback_error:
                raise RuntimeError(
                    f"SWE-bench archive fallback failed after Docker mount error: {fallback_error}"
                ) from exc
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace")[-1000:]
                raise RuntimeError(
                    f"SWE-bench archive fallback failed (exit {result.returncode}): {detail}"
                ) from exc
            return True

    put_archive._xtask_swe_copy_fallback = True
    Container.put_archive = put_archive


def install_minisweagent_docker_retry() -> None:
    """Retry only transient overlay ENOSPC during SWE-agent container creation."""
    from minisweagent.environments.docker import DockerEnvironment

    original = DockerEnvironment._start_container
    if getattr(original, "_xtask_swe_docker_run_retry", False):
        return

    @functools.wraps(original)
    def start_container(self):
        for attempt in range(3):
            try:
                return original(self)
            except subprocess.CalledProcessError as exc:
                detail = str(exc.stderr or "").lower()
                if (
                    attempt == 2
                    or exc.returncode != 125
                    or "error creating overlay mount" not in detail
                    or "no space left on device" not in detail
                ):
                    raise
                delay = 2 * (attempt + 1)
                print(
                    f"xTask SWE-bench: Docker overlay mount ENOSPC starting "
                    f"mini-swe-agent instance; retrying in {delay}s "
                    f"({attempt + 1}/2)",
                    file=sys.stderr,
                )
                time.sleep(delay)

    start_container._xtask_swe_docker_run_retry = True
    DockerEnvironment._start_container = start_container


def install_minisweagent_cleanup_each_instance() -> None:
    """Stop this thread's completed mini-agent environment before the next case."""
    from minisweagent.run.benchmarks import swebench as mini_swebench

    original_process = mini_swebench.process_instance
    if getattr(original_process, "_xtask_swe_clean_infer_container", False):
        return
    original_get_environment = mini_swebench.get_sb_environment
    current = threading.local()

    @functools.wraps(original_get_environment)
    def tracked_environment(*args, **kwargs):
        env = original_get_environment(*args, **kwargs)
        current.environment = env
        return env

    @functools.wraps(original_process)
    def process_instance(*args, **kwargs):
        current.environment = None
        try:
            return original_process(*args, **kwargs)
        finally:
            env = getattr(current, "environment", None)
            current.environment = None
            container_id = getattr(env, "container_id", None)
            if container_id:
                try:
                    result = subprocess.run(
                        ["docker", "rm", "-f", container_id],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise RuntimeError(
                        f"SWE-bench mini-agent container {container_id} cleanup failed: {exc}"
                    ) from exc
                if result.returncode != 0 and "No such container" not in result.stderr:
                    raise RuntimeError(
                        f"SWE-bench mini-agent container {container_id} cleanup failed: {result.stderr[-500:]}"
                    )
                env.container_id = None  # Avoid the mini-agent destructor's async cleanup.
                print(
                    f"xTask SWE-bench: cleaned completed mini-agent container {container_id}",
                    file=sys.stderr,
                )

    process_instance._xtask_swe_clean_infer_container = True
    mini_swebench.get_sb_environment = tracked_environment
    mini_swebench.process_instance = process_instance


def install_minisweagent_container_timeout(timeout: str) -> None:
    """Extend only this SWE run's mini-agent Docker lifetime for hard tasks."""
    if not re.fullmatch(r"[1-9][0-9]*[smh]", timeout):
        raise ValueError("Invalid SWE-bench agent container timeout")
    from minisweagent.environments.docker import DockerEnvironment

    original = DockerEnvironment.__init__
    if getattr(original, "_xtask_swe_agent_container_timeout", False):
        return

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        kwargs["container_timeout"] = timeout
        return original(self, *args, **kwargs)

    init._xtask_swe_agent_container_timeout = True
    DockerEnvironment.__init__ = init


def install_github_raw_retry() -> None:
    """Retry only idempotent GitHub Raw GETs used by SWE-bench harness.

    The x86 proxy and direct path both occasionally drop a connection. Keep
    this opt-in and out of xTask's own requests and the model API traffic.
    """
    import requests

    original = requests.sessions.Session.request
    if getattr(original, "_xtask_swe_github_raw_retry", False):
        return

    @functools.wraps(original)
    def request(self, method, url, **kwargs):
        if str(method).upper() != "GET" or urlsplit(str(url)).hostname != "raw.githubusercontent.com":
            return original(self, method, url, **kwargs)
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (5, 20)

        for attempt in range(4):
            try:
                response = original(self, method, url, **kwargs)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                # A certificate failure cannot be repaired by a retry.
                if isinstance(exc, requests.exceptions.SSLError) or attempt == 3:
                    raise
                print(
                    f"xTask SWE-bench: retry GitHub Raw GET after {type(exc).__name__} "
                    f"({attempt + 1}/3)", file=sys.stderr,
                )
            else:
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == 3:
                    return response
                response.close()
                print(
                    f"xTask SWE-bench: retry GitHub Raw GET after HTTP {response.status_code} "
                    f"({attempt + 1}/3)", file=sys.stderr,
                )
            time.sleep(2 ** attempt)

    request._xtask_swe_github_raw_retry = True
    requests.sessions.Session.request = request


if __name__ == "sitecustomize":
    if timeout := os.environ.get("XTASK_SWEBENCH_AGENT_CONTAINER_TIMEOUT"):
        install_minisweagent_container_timeout(timeout)
    if os.environ.get("XTASK_SWEBENCH_DOCKER_COPY_FALLBACK") == "1":
        install()
    if os.environ.get("XTASK_SWEBENCH_DOCKER_RUN_RETRY") == "1":
        install_minisweagent_docker_retry()
    if os.environ.get("XTASK_SWEBENCH_CLEAN_INFER_CONTAINER") == "1":
        install_minisweagent_cleanup_each_instance()
    if os.environ.get("XTASK_SWEBENCH_GITHUB_RAW_RETRY") == "1":
        install_github_raw_retry()
