"""Fast, requirement-driven synchronization of shared xLLM test resources."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shlex
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from xtask.utils.host_executor import HostExecutor, create_host_executor
from xtask.utils.config_parser import get_xtask_config_data


DEFAULT_RESOURCE_ROOT = "/export/home/models/xllm_test_060"
CONTAINER_RESOURCE_ROOT = "/xllm_test_060"
NUMACTL_RPM = "numactl-2.0.16-9.vo2203sp4.aarch64.rpm"
EVALSCOPE_OFFLINE_DIR = "evalscope_offline"
EVALSCOPE_MAIN_WHEEL = "evalscope-1.14.0-py3-none-any.whl"
TORCH_NPU_WHEEL = "torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl"


def get_dependence_package_path() -> str:
    """Return the environment-level shared resource root."""
    value = (get_xtask_config_data().get("ENV_CONFIG") or {}).get(
        "DEPENDENCE_PACKAGE_PATH", DEFAULT_RESOURCE_ROOT
    )
    text = str(value or "").strip()
    return text or DEFAULT_RESOURCE_ROOT


DEPENDENCE_PLACEHOLDER = "${DEPENDENCE_PACKAGE_PATH}"


def expand_dependence_placeholder(value: Any) -> Any:
    """Expand ``${DEPENDENCE_PACKAGE_PATH}`` in a YAML path field.

    Host-side paths become ``DEPENDENCE_PACKAGE_PATH/...``; container-side
    paths (those already under the fixed ``/xllm_test_060`` mount) become
    ``/xllm_test_060/...``.  Callers pass the resolved string to
    ``path_mapping_in_container`` when the field is read inside a container.

    Non-string values and strings without the placeholder pass through
    unchanged, so this is safe to apply to every task-config string.
    """
    if not isinstance(value, str) or DEPENDENCE_PLACEHOLDER not in value:
        return value
    return value.replace(DEPENDENCE_PLACEHOLDER, get_dependence_package_path())


def expand_dependence_placeholders(node: Any) -> Any:
    """Recursively expand the placeholder through a task-config mapping."""
    if isinstance(node, str):
        return expand_dependence_placeholder(node)
    if isinstance(node, dict):
        return {key: expand_dependence_placeholders(item) for key, item in node.items()}
    if isinstance(node, list):
        return [expand_dependence_placeholders(item) for item in node]
    return node


def _nonempty_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_resource_sync_roots(task_config: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Resolve local sync source and remote sync target independently.

    YAML ``deployment.resource_sync.source`` / ``target`` override the
    environment default when present.  Missing fields fall back to
    ``DEPENDENCE_PACKAGE_PATH`` separately — target does not inherit source.
    """
    default_root = get_dependence_package_path()
    raw_config = ((task_config or {}).get("deployment") or {}).get("resource_sync", {})
    if not isinstance(raw_config, dict):
        return default_root, default_root
    source = _nonempty_path(raw_config.get("source")) or default_root
    target = _nonempty_path(raw_config.get("target")) or default_root
    return source, target


def apply_dependence_package_mount(volumes: Dict[str, str], host_path: str) -> None:
    """Bind ``host_path`` to the fixed container mount ``/xllm_test_060``."""
    normalized = os.path.normpath(str(host_path or "").strip())
    if not normalized or normalized == ".":
        return
    container_root = CONTAINER_RESOURCE_ROOT.rstrip("/")
    stale = [
        existing
        for existing, inside in list(volumes.items())
        if os.path.normpath(str(inside).rstrip("/")) == container_root
    ]
    for existing in stale:
        volumes.pop(existing, None)
    volumes[normalized] = CONTAINER_RESOURCE_ROOT


class ResourceSyncError(RuntimeError):
    """Raised when required resources cannot be checked or synchronized."""


@dataclass(frozen=True)
class ResourceComponent:
    name: str
    relative_path: str
    probes: Tuple[str, ...]
    excludes: Tuple[str, ...] = ()


def _runtime_component() -> ResourceComponent:
    return ResourceComponent("runtime-numactl", NUMACTL_RPM, (NUMACTL_RPM,))


def _torch_npu_component() -> ResourceComponent:
    return ResourceComponent(
        "runtime-torch-npu",
        TORCH_NPU_WHEEL,
        (TORCH_NPU_WHEEL,),
    )


def _aisbench_component() -> ResourceComponent:
    return ResourceComponent(
        "aisbench-core",
        "benchmark",
        (
            "benchmark/setup.py",
            "benchmark/ais_bench/benchmark/cli/main.py",
        ),
        (
            ".git/",
            "__pycache__/",
            "*.pyc",
            "ais_bench/datasets/textvqa/",
            "LOG_PLACE/",
        ),
    )


def _evalscope_component() -> ResourceComponent:
    return ResourceComponent(
        "evalscope-offline",
        EVALSCOPE_OFFLINE_DIR,
        (
            EVALSCOPE_OFFLINE_DIR,
            f"{EVALSCOPE_OFFLINE_DIR}/{EVALSCOPE_MAIN_WHEEL}",
        ),
    )


def _safe_component_name(relative_path: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", relative_path).strip("-")
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    return f"path-{readable[-48:] or 'root'}-{digest}"


def _first_file(path: Path) -> Optional[Path]:
    for root, directories, files in os.walk(path):
        directories[:] = sorted(item for item in directories if not item.startswith("."))
        files = sorted(item for item in files if not item.startswith("."))
        if files:
            return Path(root) / files[0]
    return None


def _directory_probes(source_root: Path, relative_path: str) -> Tuple[str, ...]:
    """Pick a few stable files so manually provisioned directories skip quickly."""
    local = source_root / relative_path
    probes = [relative_path]
    children = sorted(
        (item for item in local.iterdir() if not item.name.startswith(".")),
        key=lambda item: item.name,
    )
    for child in children:
        candidate = child if child.is_file() else _first_file(child)
        if candidate is not None:
            probes.append(candidate.relative_to(source_root).as_posix())
        if len(probes) >= 4:
            break
    if len(probes) == 1:
        candidate = _first_file(local)
        if candidate is not None:
            probes.append(candidate.relative_to(source_root).as_posix())
    return tuple(probes)


def _path_component(source_root: Path, relative_path: str) -> ResourceComponent:
    relative_path = relative_path.strip("/")
    local = source_root / relative_path
    if not local.exists():
        raise ResourceSyncError(f"Required local resource does not exist: {local}")
    probes = (
        (relative_path,)
        if local.is_file()
        else _directory_probes(source_root, relative_path)
    )
    return ResourceComponent(_safe_component_name(relative_path), relative_path, probes)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _walk_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_strings(nested)


def _referenced_resource_paths(
    task_config: Dict[str, Any], source_root: str = DEFAULT_RESOURCE_ROOT
) -> List[str]:
    prefixes = dict.fromkeys(
        (
            CONTAINER_RESOURCE_ROOT.rstrip("/") + "/",
            DEFAULT_RESOURCE_ROOT.rstrip("/") + "/",
            str(source_root).rstrip("/") + "/",
        )
    )
    found = []
    for value in _walk_strings(task_config):
        for prefix in prefixes:
            if value.startswith(prefix):
                relative_path = value[len(prefix):].strip("/")
                if relative_path and relative_path not in found:
                    found.append(relative_path)
                break
    return found


def _append_component(
    components: List[ResourceComponent], component: ResourceComponent
) -> None:
    if component.relative_path not in {item.relative_path for item in components}:
        components.append(component)


def build_host_requirements(
    task_config: Dict[str, Any],
    plans: Sequence[Any],
    source_root: str = DEFAULT_RESOURCE_ROOT,
) -> Dict[str, List[ResourceComponent]]:
    """Map each non-local deployment host to only the resources it needs."""
    root = Path(source_root).resolve()
    requirements: Dict[str, List[ResourceComponent]] = {}
    for plan in plans:
        if plan.is_local:
            continue
        host_components = requirements.setdefault(str(plan.host), [])
        _append_component(host_components, _runtime_component())
        # Every non-local xLLM runtime needs the pinned torch_npu wheel.  Keep
        # this separate from benchmark resources so it is checked and copied
        # on every service node, not only on the benchmark runner.
        _append_component(host_components, _torch_npu_component())

    runner = next((plan for plan in plans if plan.is_benchmark_runner), None)
    if runner is None and plans:
        runner = plans[0]
    if runner is None or runner.is_local:
        return requirements

    runner_components = requirements.setdefault(str(runner.host), [])
    benchmark_type = str(
        (task_config.get("benchmark") or {}).get("benchmark_type", "")
    ).lower()
    dataset_name = str(task_config.get("dataset_name", "")).lower()
    # Benchmark code and datasets are runner-only resources.  Service workers
    # must not receive these potentially large directories.
    if benchmark_type == "aisbench":
        _append_component(runner_components, _aisbench_component())
        if dataset_name == "textvqa":
            _append_component(
                runner_components,
                _path_component(root, "benchmark/ais_bench/datasets/textvqa"),
            )
    elif benchmark_type == "evalscope":
        _append_component(runner_components, _evalscope_component())
    elif benchmark_type == "image_edit":
        # image_edit installs from the same evalscope_offline wheel closure.
        _append_component(runner_components, _evalscope_component())

    for relative_path in _referenced_resource_paths(task_config, root):
        _append_component(
            runner_components,
            _path_component(root, relative_path),
        )
    return requirements


def _component_marker(remote_root: str, component: ResourceComponent, suffix: str) -> str:
    return str(Path(remote_root) / ".xtask_resource_sync" / f"{component.name}.{suffix}")


def _probe_expression(remote_root: str, component: ResourceComponent) -> str:
    checks = [
        f"test -e {shlex.quote(str(Path(remote_root) / probe))}"
        for probe in component.probes
    ]
    incomplete = _component_marker(remote_root, component, "incomplete")
    checks.append(f"test ! -e {shlex.quote(incomplete)}")
    return " && ".join(checks)


def _probe_components(
    executor: HostExecutor,
    remote_root: str,
    components: Sequence[ResourceComponent],
) -> List[ResourceComponent]:
    commands = []
    for component in components:
        commands.append(
            f"if {_probe_expression(remote_root, component)}; "
            f"then echo PRESENT:{component.name}; else echo MISSING:{component.name}; fi"
        )
    result = executor.run("; ".join(commands), timeout=60)
    if not result.success:
        raise ResourceSyncError(
            f"Resource probe failed on {getattr(executor, 'host', 'remote')}: "
            f"{result.stderr or result.stdout}"
        )
    present = {
        line.split(":", 1)[1].strip()
        for line in (result.stdout or "").splitlines()
        if line.startswith("PRESENT:")
    }
    return [component for component in components if component.name not in present]


def _verify_component(
    executor: HostExecutor,
    remote_root: str,
    component: ResourceComponent,
) -> bool:
    checks = " && ".join(
        f"test -e {shlex.quote(str(Path(remote_root) / probe))}"
        for probe in component.probes
    )
    result = executor.run(checks, timeout=60)
    return result.success


def _sync_host(
    host: str,
    components: Sequence[ResourceComponent],
    source_root: str,
    remote_root: str,
    timeout: int,
    executor_factory,
) -> Dict[str, Any]:
    started = time.time()
    executor = executor_factory(host)
    missing = _probe_components(executor, remote_root, components)
    if not missing:
        logging.info("Required xLLM resources already exist on %s", host)
        return {"status": "ready", "synced": [], "elapsed": time.time() - started}

    synced = []
    marker_dir = str(Path(remote_root) / ".xtask_resource_sync")
    for component in missing:
        local_path = str(Path(source_root) / component.relative_path)
        remote_path = str(Path(remote_root) / component.relative_path)
        if not Path(local_path).exists():
            raise ResourceSyncError(f"Required local resource does not exist: {local_path}")

        incomplete = _component_marker(remote_root, component, "incomplete")
        ready = _component_marker(remote_root, component, "ready")
        begin = executor.run(
            f"mkdir -p {shlex.quote(marker_dir)} && "
            f": > {shlex.quote(incomplete)}",
            timeout=60,
        )
        if not begin.success:
            raise ResourceSyncError(
                f"Cannot prepare resource sync marker on {host}: "
                f"{begin.stderr or begin.stdout}"
            )

        logging.info(
            "Synchronizing xLLM resource %s to %s:%s",
            component.name,
            host,
            remote_path,
        )
        if not executor.sync_path(
            local_path,
            remote_path,
            excludes=component.excludes,
            timeout=timeout,
        ):
            raise ResourceSyncError(
                f"Failed to sync resource {component.name} to {host}"
            )
        if not _verify_component(executor, remote_root, component):
            raise ResourceSyncError(
                f"Resource {component.name} is incomplete on {host} after sync"
            )

        finish = executor.run(
            f"rm -f {shlex.quote(incomplete)} && "
            f"printf '%s\\n' ready > {shlex.quote(ready)}",
            timeout=60,
        )
        if not finish.success:
            raise ResourceSyncError(
                f"Cannot finalize resource sync marker on {host}: "
                f"{finish.stderr or finish.stdout}"
            )
        synced.append(component.name)

    logging.info(
        "Required xLLM resources prepared on %s in %.2fs",
        host,
        time.time() - started,
    )
    return {"status": "synced", "synced": synced, "elapsed": time.time() - started}


def sync_required_resources_to_hosts(
    task_config: Dict[str, Any],
    plans: Sequence[Any],
    executor_factory=None,
) -> Dict[str, Dict[str, Any]]:
    """Check and incrementally fill resources on unique non-local hosts."""
    deployment = task_config.get("deployment") or {}
    raw_config = deployment.get("resource_sync", {})
    if raw_config is False:
        return {}
    if raw_config is True or raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ResourceSyncError("deployment.resource_sync must be a mapping or boolean")
    if not raw_config.get("enabled", True):
        return {}

    configured_source, configured_target = resolve_resource_sync_roots(task_config)
    source_root = str(Path(configured_source).resolve())
    # Preserve the configured host path for remote mounts even if the local path
    # happens to be a symlink to another storage location.
    remote_root = configured_target
    timeout = int(raw_config.get("timeout", 3600))
    max_workers = max(1, int(raw_config.get("max_workers", 4)))
    requirements = build_host_requirements(task_config, plans, source_root)
    if not requirements:
        return {}
    if executor_factory is None:
        executor_factory = create_host_executor

    results: Dict[str, Dict[str, Any]] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(requirements))) as pool:
        futures = {
            pool.submit(
                _sync_host,
                host,
                components,
                source_root,
                remote_root,
                timeout,
                executor_factory,
            ): host
            for host, components in requirements.items()
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                results[host] = future.result()
            except Exception as exc:
                failures.append(f"{host}: {exc}")
    if failures:
        raise ResourceSyncError("; ".join(failures))
    return results
