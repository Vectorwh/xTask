"""Multi-host HCCL ranktable generate / merge / inject for xTask."""

from __future__ import annotations

import logging
import os
import shlex
from typing import Any, Dict, List, Optional, Sequence, Tuple

from xtask.core.service_group import (
    ServiceLaunchPlan,
    is_multi_host_deployment,
    normalize_pd_deployment,
)
from xtask.utils.container_manager import path_mapping_in_container
from xtask.utils.hccl_ranktable import (
    RanktableError,
    format_device_ids,
    load_ranktable,
    merge_ranktables,
    write_ranktable,
)
from xtask.utils.host_executor import create_host_executor, is_local_host
from xtask.utils.workspace_sync import resolve_workspace_root


def _is_pd_mode(task_config: Dict[str, Any]) -> bool:
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    if deployment.get("mode") == "pd_disaggregated":
        return True
    for instance in deployment.get("instances") or []:
        role = str(instance.get("role", "")).lower()
        if role in {"prefill", "decode"}:
            return True
    return False


def _group_key_for_plan(plan: ServiceLaunchPlan, task_config: Dict[str, Any]) -> str:
    """Resolve ranktable group name for one plan (yaml order preserved by caller)."""
    normalize_pd_deployment(task_config)
    deployment = task_config.get("deployment") or {}
    instance_meta = None
    for item in deployment.get("instances") or []:
        if item.get("name") == plan.instance_name:
            instance_meta = item
            break
    if instance_meta:
        if instance_meta.get("pd_group"):
            return str(instance_meta["pd_group"])
    if _is_pd_mode(task_config):
        raise ValueError(
            f"PD runtime instance {plan.instance_name!r} is missing pd_group; "
            "author PD YAML with deployment.pd_instances"
        )
    return "default"


def group_instances_for_ranktable(
    task_config: Dict[str, Any],
    plans: Optional[Sequence[ServiceLaunchPlan]] = None,
) -> Dict[str, List[ServiceLaunchPlan]]:
    """Group service plans for ranktable generation.

    - Non-PD multi-host: one group ``default`` with all instances.
    - PD: group by the logical ``pd_group`` generated from ``pd_instances[].name``.
      A logical group maps to one P/D instance; its members are physical nodes.
    """
    normalize_pd_deployment(task_config)
    if plans is None:
        from xtask.core.service_group import build_service_plans

        plans = build_service_plans(task_config, output_file_path="/tmp/xtask_ranktable_plans")

    groups: Dict[str, List[ServiceLaunchPlan]] = {}
    pd = _is_pd_mode(task_config)
    for plan in plans:
        key = _group_key_for_plan(plan, task_config) if pd else "default"
        groups.setdefault(key, []).append(plan)

    # Stable order within each group: rank_start, then original plan order.
    for key, members in groups.items():
        indexed = list(enumerate(members))
        indexed.sort(key=lambda item: (item[1].rank_start, item[0]))
        groups[key] = [item[1] for item in indexed]
    return groups


def _manual_rank_tablefile(plan: ServiceLaunchPlan) -> Optional[str]:
    ext = (plan.instance_config or {}).get("serving_extensiable_param") or {}
    value = ext.get("rank_tablefile")
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _ensure_hccn_tool_in_runtime(runtime) -> None:
    """Make the host's hccn_tool available in a prepared serving container."""
    manager = runtime.container_manager
    container_name = runtime.container_config.container_name
    probe = manager.exec_in_container(
        container_name,
        "command -v hccn_tool",
        timeout=30,
    )
    if probe.success:
        return

    executor = manager.executor
    host_probe = executor.run("command -v hccn_tool", timeout=30)
    host_tool = (host_probe.stdout or "").strip().splitlines()
    host_tool = next((line.strip() for line in host_tool if line.startswith("/")), "")
    if not host_probe.success or not host_tool:
        raise RanktableError(
            f"hccn_tool is unavailable on host {runtime.plan.host} and inside "
            f"container {container_name}"
        )

    destination = "/usr/bin/hccn_tool"
    copy_result = executor.run(
        "docker cp "
        f"{shlex.quote(host_tool)} "
        f"{shlex.quote(container_name + ':' + destination)}",
        timeout=60,
    )
    if not copy_result.success:
        raise RanktableError(
            f"failed to copy hccn_tool into container {container_name} on "
            f"{runtime.plan.host}: {copy_result.stderr or copy_result.stdout}"
        )
    logging.info(
        "Copied host hccn_tool into ranktable runtime %s on %s",
        container_name,
        runtime.plan.host,
    )


def _remote_generate_host_table(
    plan: ServiceLaunchPlan,
    workspace_root: str,
    staging_dir: str,
    runtime=None,
) -> str:
    """Run HCCL generator on the plan's host; download JSON to staging_dir."""
    device_spec = format_device_ids(plan.device_ids)
    local_path = os.path.join(staging_dir, f"{plan.instance_name}.json")
    executor = (
        runtime.container_manager.executor
        if runtime is not None
        else create_host_executor(plan.host)
    )

    if runtime is not None:
        # Some hosts expose DCMI only inside the privileged serving container.
        # The workspace bind mount uses the same host path on every machine, so
        # generate there and then download the JSON to the control host.
        _ensure_hccn_tool_in_runtime(runtime)
        remote_output = local_path
        container_output = path_mapping_in_container(remote_output)
        container_staging_dir = os.path.dirname(container_output)
        container_ranktable_dir = os.path.dirname(container_staging_dir)
        container_script = path_mapping_in_container(
            os.path.join(workspace_root, "tools", "generate_hccl_ranktable.py")
        )
        cmd_parts = [
            "python3",
            shlex.quote(container_script),
            "--server-ip",
            shlex.quote(plan.host),
            "-o",
            shlex.quote(container_output),
            "--include-super-pod",
        ]
        if device_spec:
            cmd_parts.extend(["--device-ids", shlex.quote(device_spec)])
        command = (
            f"mkdir -p {shlex.quote(container_staging_dir)} && "
            f"chmod -R a+rwX {shlex.quote(container_ranktable_dir)} && "
            + " ".join(cmd_parts)
            + f" && chmod 0644 {shlex.quote(container_output)}"
        )
        logging.info(
            "Generating host ranktable for %s inside container on %s: %s",
            plan.instance_name,
            plan.host,
            command,
        )
        result = runtime.container_manager.exec_in_container(
            runtime.container_config.container_name,
            command,
            timeout=120,
        )
        if not result.success:
            raise RanktableError(
                f"ranktable generation failed in container on {plan.host} "
                f"({plan.instance_name}): {result.stderr or result.stdout}"
            )

        os.makedirs(staging_dir, exist_ok=True)
        if not plan.is_local and not executor.download(remote_output, local_path):
            raise RanktableError(
                f"failed to download ranktable from {plan.host}:{remote_output}"
            )
        if not os.path.isfile(local_path):
            raise RanktableError(
                f"ranktable generator did not create {local_path} for "
                f"{plan.instance_name}"
            )
        if not plan.is_local:
            executor.run(f"rm -f {shlex.quote(remote_output)}", timeout=30)
        return local_path

    remote_tmp = f"/tmp/xtask_ranktable_{plan.instance_name}.json"
    script = os.path.join(workspace_root, "tools", "generate_hccl_ranktable.py")
    cmd_parts = [
        "python3",
        shlex.quote(script),
        "--server-ip",
        shlex.quote(plan.host),
        "-o",
        shlex.quote(remote_tmp),
        "--include-super-pod",
    ]
    if device_spec:
        cmd_parts.extend(["--device-ids", shlex.quote(device_spec)])
    cmd = " ".join(cmd_parts)
    logging.info(
        "Generating host ranktable for %s on %s: %s",
        plan.instance_name,
        plan.host,
        cmd,
    )
    result = executor.run(cmd, timeout=120)
    if not result.success:
        raise RanktableError(
            f"ranktable generation failed on {plan.host} ({plan.instance_name}): "
            f"{result.stderr or result.stdout}"
        )
    os.makedirs(staging_dir, exist_ok=True)
    if not executor.download(remote_tmp, local_path):
        raise RanktableError(
            f"failed to download ranktable from {plan.host}:{remote_tmp}"
        )
    executor.run(f"rm -f {shlex.quote(remote_tmp)}", timeout=30)
    return local_path


def generate_and_merge_for_group(
    members: Sequence[ServiceLaunchPlan],
    output_dir: str,
    group_name: str = "default",
    workspace_root: Optional[str] = None,
    runtimes=None,
) -> Tuple[str, str]:
    """Generate per-host tables, merge, write local file, upload to all members.

    Returns:
        (host_path, container_path) of the merged ranktable.
    """
    if not members:
        raise RanktableError("generate_and_merge_for_group requires members")

    root = str(workspace_root or resolve_workspace_root())
    os.makedirs(output_dir, exist_ok=True)
    staging = os.path.join(output_dir, f".staging_{group_name}")
    os.makedirs(staging, exist_ok=True)
    runtime_by_name = {
        runtime.plan.instance_name: runtime for runtime in (runtimes or [])
    }

    host_tables = []
    for plan in members:
        local_json = _remote_generate_host_table(
            plan,
            root,
            staging,
            runtime=runtime_by_name.get(plan.instance_name),
        )
        host_tables.append(load_ranktable(local_json))

    merged = merge_ranktables(host_tables)
    safe_name = group_name.replace("/", "_").replace(" ", "_") or "default"
    host_path = os.path.join(output_dir, f"{safe_name}_merged.json")
    write_ranktable(host_path, merged)
    logging.info(
        "Merged ranktable for group %s (%d hosts) -> %s",
        group_name,
        len(members),
        host_path,
    )

    # Ensure parent exists on each host, then upload (skip no-op on control host).
    for plan in members:
        if is_local_host(plan.host) and os.path.isfile(host_path):
            continue
        executor = create_host_executor(plan.host)
        parent = os.path.dirname(host_path)
        mkdir = executor.run(f"mkdir -p {shlex.quote(parent)}", timeout=60)
        if not mkdir.success:
            raise RanktableError(
                f"failed to mkdir {parent} on {plan.host}: {mkdir.stderr}"
            )
        if not executor.upload(host_path, host_path):
            raise RanktableError(
                f"failed to upload ranktable to {plan.host}:{host_path}"
            )

    container_path = path_mapping_in_container(host_path)
    return host_path, container_path


def inject_rank_tablefile(plan: ServiceLaunchPlan, container_path: str) -> None:
    """Set serving_extensiable_param.rank_tablefile on the plan's instance_config."""
    cfg = plan.instance_config
    ext = dict(cfg.get("serving_extensiable_param") or {})
    ext["rank_tablefile"] = container_path
    cfg["serving_extensiable_param"] = ext


def prepare_multihost_ranktables(
    task_config: Dict[str, Any],
    plans: Sequence[ServiceLaunchPlan],
    output_file_path: str,
    runtimes=None,
) -> Dict[str, str]:
    """Generate/merge/inject ranktables for all groups that need them.

    Manual ``rank_tablefile`` on an instance is preserved (not overwritten).
    Groups where every member already has a manual path are skipped.

    Returns:
        Mapping group_name -> container path of merged ranktable (injected groups only).
    """
    if not is_multi_host_deployment(task_config):
        return {}

    groups = group_instances_for_ranktable(task_config, plans)
    ranktable_dir = os.path.join(output_file_path, "ranktable")
    injected: Dict[str, str] = {}

    for group_name, members in groups.items():
        need_inject = [m for m in members if not _manual_rank_tablefile(m)]
        if not need_inject:
            logging.info(
                "Ranktable group %s: all members have manual rank_tablefile, skip",
                group_name,
            )
            continue

        # Experiment: single-node groups may not need an auto-generated ranktable.
        # Skip generation for groups with only one member; manual paths are still honored.
        if len(members) == 1:
            logging.info(
                "Ranktable group %s: single-node group, skip auto ranktable generation",
                group_name,
            )
            continue

        _host_path, container_path = generate_and_merge_for_group(
            members,
            output_dir=ranktable_dir,
            group_name=group_name,
            runtimes=runtimes,
        )
        for plan in need_inject:
            inject_rank_tablefile(plan, container_path)
            logging.info(
                "Injected rank_tablefile=%s for instance %s (group=%s)",
                container_path,
                plan.instance_name,
                group_name,
            )
        injected[group_name] = container_path

    return injected
