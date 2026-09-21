"""Prepare AISBench SWE dependencies on the x86 runner host.

Syncs AISBench source from the control/NPU node (same tree used by
``download_ais_bench.sh``), then pip-installs AISBench, mini-swe-agent and the
SWE-bench harness on the runner. Docker itself is not installed.

This path is gated by ``is_aisbench_swebench_task`` and never runs inside the
xLLM/NPU serving container.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from xtask.benchmarks.aisbench.swebench import swebench_config, swebench_runner
from xtask.utils.container_manager import CommandResult
from xtask.utils.host_executor import HostExecutor, is_local_host
from xtask.utils.resource_sync import get_dependence_package_path


DEFAULT_INSTALL_ROOT = "/export/home/xtask_swe/runtime"
DEFAULT_SETUP_TIMEOUT_S = 1800
SETUP_SCRIPT_NAME = "setup_aisbench_swe_x86.sh"
MINI_SWE_DIR_NAMES = ("mini-swe-agent", "mini_swe_agent", "minisweagent")
HARNESS_DIR_NAMES = ("SWE-bench", "swe-bench", "swebench", "SWE_bench")


def _setup_config(task_config: Mapping[str, Any]) -> dict:
    raw = swebench_config(task_config).get("setup") or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _nonempty(value: Any) -> str:
    return str(value or "").strip()


def _has_content(path: Path) -> bool:
    try:
        if not path.is_dir():
            return False
        next(path.iterdir())
        return True
    except (OSError, PermissionError, StopIteration):
        return False


def _is_aisbench_tree(path: Path) -> bool:
    try:
        if not path.is_dir() or not (path / "ais_bench").is_dir():
            return False
        return (path / "setup.py").is_file() or (path / "pyproject.toml").is_file()
    except PermissionError:
        return False


def _is_mini_swe_tree(path: Path) -> bool:
    try:
        return path.is_dir() and (
            (path / "src" / "minisweagent").is_dir() or (path / "minisweagent").is_dir()
        )
    except PermissionError:
        return False


def _is_harness_tree(path: Path) -> bool:
    try:
        if not path.is_dir() or not (path / "swebench").is_dir():
            return False
        return any(
            (path / name).is_file()
            for name in ("pyproject.toml", "setup.py", "setup.cfg")
        )
    except PermissionError:
        return False


def _first_matching(roots: Sequence[Path], names: Sequence[str], predicate) -> Optional[Path]:
    for root in roots:
        for name in names:
            candidate = root / name
            if predicate(candidate):
                return candidate
    return None


def _dependence_roots(task_config: Mapping[str, Any]) -> List[Path]:
    roots = []
    setup = _setup_config(task_config)
    explicit_root = _nonempty(setup.get("source_root"))
    if explicit_root:
        roots.append(Path(explicit_root))
    try:
        roots.append(Path(get_dependence_package_path()))
    except Exception:
        pass
    roots.append(Path("/export/home/models/xllm_test_060"))
    xtask_root = Path(__file__).resolve().parents[3]
    roots.append(xtask_root)
    roots.append(xtask_root / "benchmark")
    seen = []
    for root in roots:
        resolved = root
        if resolved not in seen:
            seen.append(resolved)
    return seen


def resolve_aisbench_source(task_config: Mapping[str, Any]) -> Path:
    setup = _setup_config(task_config)
    explicit = _nonempty(setup.get("aisbench_source"))
    if explicit:
        path = Path(explicit)
        if not _is_aisbench_tree(path):
            raise FileNotFoundError(
                f"benchmark.swebench.setup.aisbench_source is not an AISBench tree: {path}"
            )
        return path
    for root in _dependence_roots(task_config):
        for candidate in (root / "benchmark", root):
            if _is_aisbench_tree(candidate):
                return candidate
    raise FileNotFoundError(
        "AISBench source was not found on the control node. Expected "
        "DEPENDENCE_PACKAGE_PATH/benchmark (the same tree download_ais_bench.sh uses)."
    )


def resolve_optional_source(
    task_config: Mapping[str, Any],
    field: str,
    names: Sequence[str],
    predicate,
) -> Optional[Path]:
    setup = _setup_config(task_config)
    explicit = _nonempty(setup.get(field))
    if explicit:
        path = Path(explicit)
        if not predicate(path):
            raise FileNotFoundError(
                f"benchmark.swebench.setup.{field} is not a valid source tree: {path}"
            )
        return path
    return _first_matching(_dependence_roots(task_config), names, predicate)


def setup_script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / SETUP_SCRIPT_NAME


def _remote_dir_exists(executor: HostExecutor, path: Path) -> bool:
    result = executor.run(
        f"test -d {shlex.quote(str(path))} && find {shlex.quote(str(path))} "
        f"-mindepth 1 -print -quit | grep -q .",
        timeout=60,
    )
    return bool(result.success)


def _sync_tree(executor: HostExecutor, host: str, local: Path, remote: Path) -> None:
    if is_local_host(host):
        if local.resolve() == remote.resolve():
            return
        copied = executor.upload(str(local), str(remote))
        if not copied:
            raise RuntimeError(f"Failed to copy {local} to local runner path {remote}")
        return
    if str(local) == str(remote) and _remote_dir_exists(executor, remote):
        logging.info("SWE-bench: reuse shared path on x86 runner %s: %s", host, remote)
        return
    if _remote_dir_exists(executor, local) and str(local) == str(remote):
        return
    logging.info("SWE-bench: sync %s -> %s:%s", local, host, remote)
    ok = executor.sync_path(str(local), str(remote), timeout=DEFAULT_SETUP_TIMEOUT_S)
    if not ok:
        raise RuntimeError(f"Failed to sync {local} to {host}:{remote}")


@dataclass
class SweX86SetupResult:
    command: CommandResult
    ais_bench_bin: str = "ais_bench"
    python_bin: str = "python3"


def setup_aisbench_swe_runner(
    task_config: Mapping[str, Any],
    executor: HostExecutor,
    host: str,
    work_dir: str,
    timeout: Optional[int] = None,
    install_dir: Optional[str] = None,
) -> SweX86SetupResult:
    """Sync sources and pip-install SWE runtime on the x86 runner."""
    setup = _setup_config(task_config)
    if setup.get("skip") in {True, "true", "True", 1, "1"}:
        logging.info("SWE-bench: skip x86 AISBench/SWE dependency setup")
        return SweX86SetupResult(
            command=CommandResult(
                success=True,
                exit_code=0,
                stdout="skip setup",
                stderr="",
                execution_time=0.0,
            ),
            ais_bench_bin=_nonempty(swebench_config(task_config).get("ais_bench_bin"))
            or "ais_bench",
            python_bin=_nonempty(setup.get("python")) or "python3",
        )

    python_bin = _nonempty(setup.get("python")) or "python3"
    runner = swebench_runner(task_config)
    install_root = (
        _nonempty(install_dir)
        or _nonempty(setup.get("install_dir"))
        or _nonempty(runner.get("install_dir"))
        or DEFAULT_INSTALL_ROOT
    )
    timeout_s = timeout
    if timeout_s is None:
        try:
            timeout_s = int(setup.get("timeout") or DEFAULT_SETUP_TIMEOUT_S)
        except (TypeError, ValueError):
            timeout_s = DEFAULT_SETUP_TIMEOUT_S

    prepared = executor.run(
        f"mkdir -p {shlex.quote(work_dir)} {shlex.quote(install_root)}",
        timeout=60,
    )
    if not prepared.success:
        return SweX86SetupResult(
            command=CommandResult(
                success=False,
                exit_code=prepared.exit_code,
                stdout=prepared.stdout,
                stderr=prepared.stderr or f"Failed to create {install_root} on {host}",
                execution_time=prepared.execution_time,
            ),
            python_bin=python_bin,
        )

    try:
        return _install_swe_runtime(
            task_config,
            setup,
            executor,
            host,
            work_dir,
            python_bin,
            install_root,
            timeout_s,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return SweX86SetupResult(
            command=CommandResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                execution_time=0.0,
            ),
            python_bin=python_bin,
        )


def _install_swe_runtime(
    task_config: Mapping[str, Any],
    setup: Mapping[str, Any],
    executor: HostExecutor,
    host: str,
    work_dir: str,
    python_bin: str,
    install_root: str,
    timeout_s: int,
) -> SweX86SetupResult:
    aisbench_source = resolve_aisbench_source(task_config)
    mini_source = resolve_optional_source(
        task_config, "mini_swe_agent_source", MINI_SWE_DIR_NAMES, _is_mini_swe_tree
    )
    harness_source = resolve_optional_source(
        task_config, "swebench_harness_source", HARNESS_DIR_NAMES, _is_harness_tree
    )

    remote_aisbench = Path(install_root) / "benchmark"
    if is_local_host(host) or _remote_dir_exists(executor, aisbench_source):
        remote_aisbench = aisbench_source
    else:
        _sync_tree(executor, host, aisbench_source, remote_aisbench)

    remote_mini = ""
    if mini_source is not None:
        remote_mini_path = Path(install_root) / "mini-swe-agent"
        if is_local_host(host) or _remote_dir_exists(executor, mini_source):
            remote_mini_path = mini_source
        else:
            _sync_tree(executor, host, mini_source, remote_mini_path)
        remote_mini = str(remote_mini_path)

    remote_harness = ""
    if harness_source is not None:
        remote_harness_path = Path(install_root) / "SWE-bench"
        if is_local_host(host) or _remote_dir_exists(executor, harness_source):
            remote_harness_path = harness_source
        else:
            _sync_tree(executor, host, harness_source, remote_harness_path)
        remote_harness = str(remote_harness_path)

    script_src = setup_script_path()
    if not script_src.is_file():
        raise FileNotFoundError(f"SWE-bench x86 setup script is missing: {script_src}")
    remote_script = str(Path(work_dir) / SETUP_SCRIPT_NAME)
    if is_local_host(host):
        remote_script = str(script_src)
    else:
        _sync_tree(executor, host, script_src, Path(remote_script))

    env_parts = [
        f"PYTHON_BIN={shlex.quote(python_bin)}",
        f"INSTALL_ROOT={shlex.quote(install_root)}",
        f"AISBENCH_DIR={shlex.quote(str(remote_aisbench))}",
        f"MINI_SWE_DIR={shlex.quote(remote_mini)}",
        f"SWE_HARNESS_DIR={shlex.quote(remote_harness)}",
    ]
    git_mini = _nonempty(setup.get("mini_swe_agent_git"))
    git_harness = _nonempty(setup.get("swebench_harness_git"))
    if git_mini:
        env_parts.append(f"MINI_SWE_GIT_URL={shlex.quote(git_mini)}")
    if git_harness:
        env_parts.append(f"SWE_HARNESS_GIT_URL={shlex.quote(git_harness)}")
    command = " ".join(env_parts + ["bash", shlex.quote(remote_script)])
    logging.info("SWE-bench: installing AISBench/SWE runtime on x86 host %s", host)
    result = executor.run(command, timeout=timeout_s)
    ais_bench_bin = "ais_bench"
    if result.success:
        for line in reversed((result.stdout or "").splitlines()):
            if line.startswith("AIS_BENCH_BIN="):
                ais_bench_bin = line.split("=", 1)[1].strip() or ais_bench_bin
                break
        configured = _nonempty(swebench_config(task_config).get("ais_bench_bin"))
        if configured:
            ais_bench_bin = configured
    else:
        logging.error(
            "SWE-bench x86 runtime setup failed: %s",
            result.stderr or result.stdout,
        )
    return SweX86SetupResult(
        command=result,
        ais_bench_bin=ais_bench_bin,
        python_bin=python_bin,
    )
